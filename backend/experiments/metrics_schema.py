"""
metrics_schema.py

Typed result rows for the SHP efficiency-vs-quality experiments.

The key modelling decision is the split between two kinds of token cost:

  * OPERATIONAL tokens — what a policy spends to actually RUN to its stop round:
    Writer + Critic every round, plus the RAGAS judge ONLY if the policy needs a
    quality score at runtime to decide when to halt. Full SHP consults the
    Information-Score-gain signal, so it pays per-round judge cost; an
    entropy-only variant, FixedK, CriticOnly and RandomStop do NOT call the judge
    to run, so they pay no operational judge cost. Charging SHP for its own judge
    overhead is what makes the efficiency comparison honest (and motivates the
    judge-free entropy-only ablation).

  * EVALUATION tokens — judge calls WE make to measure each policy's final-draft
    quality for the paper. These are a measurement instrument, not a cost of the
    policy, so they are recorded separately and never added to operational cost.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class ResultRow:
    """One (scenario × policy × seed) outcome."""
    scenario_id: str
    policy: str
    seed: int
    stop_round: int                 # 1-indexed round the policy halted on
    halt_reason: str

    # Operational cost (charged to the policy).
    writer_tokens: int
    critic_tokens: int
    judge_tokens_operational: int
    total_operational_tokens: int

    # Final-draft quality at the stop round (measured; evaluation instrument).
    final_is: float
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    # Provenance.
    n_contexts: int
    max_rounds: int

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "ResultRow":
        return ResultRow(**d)

    @staticmethod
    def key(scenario_id: str, policy: str, seed: int) -> str:
        """Stable identity used to dedupe rows on resume."""
        return f"{scenario_id}|{policy}|{seed}"

    @property
    def row_key(self) -> str:
        return ResultRow.key(self.scenario_id, self.policy, self.seed)


@dataclass
class RoundScore:
    """Cached RAGAS result for one (scenario, round) — judged at most once."""
    metrics: Dict[str, float]
    information_score: float
    judge_tokens: int

    def to_dict(self) -> Dict:
        return {
            "metrics": self.metrics,
            "information_score": self.information_score,
            "judge_tokens": self.judge_tokens,
        }

    @staticmethod
    def from_dict(d: Dict) -> "RoundScore":
        return RoundScore(
            metrics=d["metrics"],
            information_score=d["information_score"],
            judge_tokens=d.get("judge_tokens", 0),
        )
