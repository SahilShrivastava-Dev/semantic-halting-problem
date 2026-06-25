"""
halting.py

Single source of truth for the SHP halting decision.

Historically the halt cascade was implemented twice — once in
``agent_workflow.check_convergence`` (the *live* decision that actually drives
the LangGraph loop) and once again in ``agent_workflow.run_scenario`` (a
*post-hoc* re-derivation of ``halt_reason``). The two used DIFFERENT priority
orders (the live cascade tests entropy convergence before the failsafe, while
the post-hoc copy tested the failsafe second), so when several signals fired in
the same round the reported reason could disagree with the reason that actually
stopped the loop. That latent inconsistency is exactly the kind of bug a
reviewer catches.

This module collapses both into one pure, total function ``shp_should_halt``
plus a typed ``HaltConfig``. The live graph, the post-hoc reason derivation, and
the offline policy-replay harness all call the same function, so they cannot
drift. The function is side-effect-free and makes no LLM/API calls, which lets
``theory_checks.py`` machine-check its guarantees (see Theorem 1 / Lemma 2).

Canonical priority order (matches the live cascade):

    1. critic_approved      — critic LLM returned "APPROVED"
    2. entropy_convergence  — cosine distance < threshold for k consecutive rounds
    3. no_information_gain   — Information Score Δ ≤ 0 after warm-up
    4. failsafe             — hard cap at MAX_ROUNDS  (NEVER disablable — it is the
                              termination guarantee of Theorem 1)

Each non-failsafe signal can be toggled off for ablation studies via HaltConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence

from shp.config import (
    CONVERGENCE_PATIENCE,
    CONVERGENCE_THRESHOLD,
    ENABLE_CRITIC_HALT,
    ENABLE_ENTROPY_HALT,
    ENABLE_IS_GAIN_HALT,
    MAX_ROUNDS,
    MIN_ROUNDS_FOR_GAIN_CHECK,
)

# Canonical halt-reason labels (use these constants everywhere — no string typos).
REASON_CRITIC = "critic_approved"
REASON_ENTROPY = "entropy_convergence"
REASON_NO_GAIN = "no_information_gain"
REASON_FAILSAFE = "failsafe"
REASON_CONTINUE = None  # sentinel: do not halt yet


@dataclass(frozen=True)
class HaltConfig:
    """
    Immutable bundle of every threshold + ablation flag the cascade reads.

    Defaults mirror config.py so production behaviour is unchanged. The offline
    harness constructs variants (e.g. ``enable_entropy_halt=False``) to ablate a
    single signal while holding the recorded trajectory fixed.
    """
    convergence_threshold: float = CONVERGENCE_THRESHOLD
    convergence_patience: int = CONVERGENCE_PATIENCE
    max_rounds: int = MAX_ROUNDS
    min_rounds_for_gain_check: int = MIN_ROUNDS_FOR_GAIN_CHECK
    enable_critic_halt: bool = ENABLE_CRITIC_HALT
    enable_entropy_halt: bool = ENABLE_ENTROPY_HALT
    enable_is_gain_halt: bool = ENABLE_IS_GAIN_HALT
    # The failsafe is intentionally NOT a flag: removing it would break the
    # deterministic-termination guarantee (Theorem 1).


class HaltDecision(NamedTuple):
    """Result of one halt check: whether to stop, and the canonical reason."""
    halt: bool
    reason: Optional[str]


def _critic_approved(last_feedback: str) -> bool:
    """A critic verdict halts the loop iff it begins with the token APPROVED."""
    return last_feedback.strip().upper().startswith("APPROVED")


def shp_should_halt(
    *,
    loop_count: int,
    distance_history: Sequence[float],
    is_score_history: Sequence[float],
    last_feedback: str = "",
    config: HaltConfig = HaltConfig(),
) -> HaltDecision:
    """
    Decide whether the Writer→Critic loop should halt after the current round.

    This is a *pure* function: it reads only its arguments and returns a
    decision. It is called in three places, which is the whole point — there is
    exactly one implementation of the cascade:

      * the live LangGraph conditional edge (``check_convergence``),
      * the post-hoc ``halt_reason`` derivation in ``run_scenario``,
      * every offline policy in ``experiments/policies.py``.

    Args:
        loop_count:        Completed Writer→Critic iterations so far.
        distance_history:  Cosine distances between consecutive drafts, oldest→newest.
        is_score_history:  Information Score per round, oldest→newest.
        last_feedback:     Most recent critic feedback string ("" if none yet).
        config:            Thresholds + ablation flags.

    Returns:
        HaltDecision(halt, reason). ``reason`` is one of the REASON_* constants
        when halting, or None (REASON_CONTINUE) when the loop should continue.

    Guarantees (machine-checked in theory_checks.py):
        * Total: defined for every input — never raises, never returns an
          undefined state.
        * Terminating (Theorem 1): whenever loop_count >= config.max_rounds it
          returns halt=True, so the loop cannot run forever regardless of the
          other signals or ablation flags.
    """
    # Initial pass: with no distance recorded yet there is nothing to converge
    # on. Continue. (Mirrors the live cascade's "no distance yet" guard.)
    if not distance_history:
        return HaltDecision(False, REASON_CONTINUE)

    # Signal 1: Critic APPROVED.
    if config.enable_critic_halt and _critic_approved(last_feedback):
        return HaltDecision(True, REASON_CRITIC)

    # Signal 2: Semantic-entropy convergence (k-patience window).
    if config.enable_entropy_halt and len(distance_history) >= config.convergence_patience:
        recent = distance_history[-config.convergence_patience:]
        if all(d < config.convergence_threshold for d in recent):
            return HaltDecision(True, REASON_ENTROPY)

    # Signal 3: No Information Gain (after warm-up).
    if (
        config.enable_is_gain_halt
        and loop_count >= config.min_rounds_for_gain_check
        and len(is_score_history) >= 2
    ):
        gain = is_score_history[-1] - is_score_history[-2]
        if gain <= 0.0:
            return HaltDecision(True, REASON_NO_GAIN)

    # Signal 4: Hard failsafe — the termination guarantee. Never disablable.
    if loop_count >= config.max_rounds:
        return HaltDecision(True, REASON_FAILSAFE)

    return HaltDecision(False, REASON_CONTINUE)


def derive_halt_reason(
    *,
    rounds: int,
    distance_history: Sequence[float],
    is_score_history: Sequence[float],
    last_feedback: str = "",
    config: HaltConfig = HaltConfig(),
) -> str:
    """
    Re-derive the halt reason for a *completed* trajectory.

    Replays ``shp_should_halt`` at the final round so the post-hoc reason is, by
    construction, identical to the live decision (Lemma 2: halt-priority
    consistency). Returns REASON_FAILSAFE as the defensive default if — only in a
    degenerate/empty trajectory — no signal fires.
    """
    decision = shp_should_halt(
        loop_count=rounds,
        distance_history=distance_history,
        is_score_history=is_score_history,
        last_feedback=last_feedback,
        config=config,
    )
    return decision.reason if decision.halt else REASON_FAILSAFE
