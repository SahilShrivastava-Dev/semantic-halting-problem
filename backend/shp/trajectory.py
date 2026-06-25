"""
trajectory.py

Generation of a *full-depth* Writer→Critic trajectory for one question,
decoupled from both quality evaluation (RAGAS) and the halting decision.

Why this exists
---------------
The production graph in ``agent_workflow.py`` interleaves three concerns in one
loop: generation (Writer/Critic), quality scoring (RAGAS), and halting. For a
*fair, cheap* comparison of halt policies that coupling is fatal:

  * If each policy re-ran the loop, every policy would see DIFFERENT drafts
    (LLM stochasticity), so any rounds/quality difference would be confounded
    by generation noise rather than the policy itself.
  * RAGAS (the binding cost on Groq free tier) would be re-paid per policy.

Instead we generate each question's trajectory **once**, to a fixed maximum
depth, recording per round the draft, critic feedback, embedding, and cosine
distance — but NOT calling RAGAS. The evaluation harness then:

  * lets every policy *replay* over this single cached trajectory and pick its
    own stop round (strictly paired comparison — all policies see identical
    drafts), and
  * computes RAGAS quality lazily per (question, round) and caches it, so a
    draft is judged at most once no matter how many policies stop there.

Generation runs to full depth regardless of critic approval, so a policy such as
``FixedK(12)`` always has a round-12 draft to stop on. Approval is recorded per
round (``is_approved``) for the critic-halt policy to consult during replay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings

from shp.config import EMBEDDING_MODEL_NAME, DEFAULT_AGENT_MODELS, DEFAULT_PROVIDER
from shp.semantic_entropy import SemanticEntropyCalculator
from shp.agents import make_writer_node, make_critic_node
from shp.providers import get_llm
from shp.token_meter import METER

logger = logging.getLogger(__name__)


@dataclass
class RoundRecord:
    """One Writer→Critic round, everything needed to replay any halt policy."""
    round: int                      # 1-indexed round number
    draft: str
    feedback: str                   # critic feedback for this round's draft
    is_approved: bool               # critic returned APPROVED
    word_count: int
    distance: Optional[float]       # cosine distance to previous draft (None on round 1)
    embedding: List[float]          # draft embedding (cached; judging is separate)
    writer_tokens: int
    critic_tokens: int

    def public(self) -> Dict[str, Any]:
        """Serialisable view WITHOUT the bulky embedding (kept in a side file)."""
        return {
            "round": self.round,
            "draft": self.draft,
            "feedback": self.feedback,
            "is_approved": self.is_approved,
            "word_count": self.word_count,
            "distance": self.distance,
            "writer_tokens": self.writer_tokens,
            "critic_tokens": self.critic_tokens,
        }


@dataclass
class Trajectory:
    """A question's full-depth trajectory plus identifying metadata."""
    scenario_id: str
    topic: str
    question: str
    contexts: List[str]
    ground_truth: str
    rounds: List[RoundRecord] = field(default_factory=list)

    @property
    def distance_history(self) -> List[float]:
        return [r.distance for r in self.rounds if r.distance is not None]

    def drafts_up_to(self, k: int) -> List[str]:
        return [r.draft for r in self.rounds[:k]]


class TrajectoryGenerator:
    """
    Builds the Writer/Critic LLMs + embedding model ONCE, then generates a
    full-depth trajectory per scenario via ``generate()``. Reuses the exact node
    factories the production graph uses (``make_writer_node`` /
    ``make_critic_node``) so generation semantics are identical to deployment.
    """

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        agent_model: str | None = None,
        emit: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        agent_model = agent_model or DEFAULT_AGENT_MODELS[provider]
        self.provider = provider
        self.agent_model = agent_model
        self._emit = emit or (lambda e: None)

        logger.info("TrajectoryGenerator: provider=%s agent=%s", provider, agent_model)
        agent_llm = get_llm(provider, agent_model)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.calculator = SemanticEntropyCalculator(embedding_model=self.embeddings)
        self.writer_fn = make_writer_node(agent_llm, self._emit)
        self.critic_fn = make_critic_node(agent_llm, self._emit)

    def _role_total(self, role: str) -> int:
        return METER.snapshot().get(role, {}).get("total_tokens", 0)

    def generate(self, scenario: dict, max_rounds: int) -> Trajectory:
        """
        Run Writer→Critic for ``max_rounds`` rounds (no early halt) and return a
        Trajectory. Token usage per round is attributed by diffing the global
        meter around each Writer/Critic call.
        """
        traj = Trajectory(
            scenario_id=scenario["id"],
            topic=scenario.get("topic", ""),
            question=scenario["question"],
            contexts=scenario["contexts"],
            ground_truth=scenario["ground_truth"],
        )

        # Minimal state the node factories expect (a subset of WorkflowState).
        state: Dict[str, Any] = {
            "scenario": scenario,
            "current_draft": "",
            "history": [],
            "loop_count": 0,
            "current_embedding": None,
        }

        for i in range(max_rounds):
            # ── Writer ───────────────────────────────────────────────
            w_before = self._role_total("writer")
            with METER.scope("writer"):
                state.update(self.writer_fn(state))
            writer_tokens = self._role_total("writer") - w_before

            draft = state["current_draft"]
            new_emb = self.calculator.get_embedding(draft)
            prev_emb = state.get("current_embedding")
            distance = (
                self.calculator.calculate_distance(new_emb, prev_emb)
                if prev_emb is not None else None
            )
            state["current_embedding"] = new_emb

            # ── Critic ───────────────────────────────────────────────
            c_before = self._role_total("critic")
            with METER.scope("critic"):
                state.update(self.critic_fn(state))   # appends to history, bumps loop_count
            critic_tokens = self._role_total("critic") - c_before

            feedback = state["history"][-1]["feedback"]
            is_approved = feedback.strip().upper().startswith("APPROVED")

            traj.rounds.append(RoundRecord(
                round=i + 1,
                draft=draft,
                feedback=feedback,
                is_approved=is_approved,
                word_count=len(draft.split()),
                distance=distance,
                embedding=new_emb,
                writer_tokens=writer_tokens,
                critic_tokens=critic_tokens,
            ))

            logger.info(
                "[Trajectory %s] round %d/%d | dist=%s | approved=%s | w_tok=%d c_tok=%d",
                scenario["id"], i + 1, max_rounds,
                f"{distance:.4f}" if distance is not None else "—",
                is_approved, writer_tokens, critic_tokens,
            )

        return traj
