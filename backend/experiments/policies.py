"""
policies.py

Halt policies, each replayed over a single cached trajectory so all policies see
identical drafts (strictly paired comparison). A policy answers one question:
"given the trajectory prefix observed through round k, do we stop at k?"

``uses_judge`` flags whether the policy needs a per-round Information Score to
*run* (not merely to be evaluated). Only that cost is charged as operational
judge tokens — see metrics_schema for the operational/evaluation split.

Policies
--------
  SHPPolicy(halt_config)  — the full cascade via the shared halting.shp_should_halt.
                            uses_judge = enable_is_gain_halt (the only judge-needing signal).
  EntropyOnlyPolicy       — SHP with ONLY the free entropy signal → judge-free at runtime.
  FixedKPolicy(k)         — halt at round k (k=max_rounds == the max_iterations baseline).
  CriticOnlyPolicy        — halt when the critic returns APPROVED (else run to max).
  RandomStopPolicy(seed)  — halt at a deterministic pseudo-random round (lower-bound sanity).
  OraclePolicy            — halt at the round with the highest measured IS (quality upper bound).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence

from shp.halting import HaltConfig, shp_should_halt, REASON_FAILSAFE


@dataclass
class ReplayInputs:
    """Everything a policy may inspect, precomputed for one trajectory."""
    distance_history: List[float]        # d_2 .. d_T  (len = T-1)
    is_history: List[float]              # IS_1 .. IS_T (len = T), or [] if no judge
    feedbacks: List[str]                 # critic feedback per round (len = T)
    approved_flags: List[bool]           # is_approved per round (len = T)
    max_rounds: int                      # T (full trajectory depth)


class Policy:
    name: str = "base"
    uses_judge: bool = False

    def stop_round(self, rp: ReplayInputs) -> tuple[int, str]:
        """Return (1-indexed stop round, halt_reason)."""
        raise NotImplementedError


class SHPPolicy(Policy):
    """Full SHP cascade. Judge cost applies iff the IS-gain signal is enabled."""

    def __init__(self, halt_config: Optional[HaltConfig] = None, name: str = "shp"):
        self.cfg = halt_config or HaltConfig()
        self.name = name
        self.uses_judge = self.cfg.enable_is_gain_halt

    def stop_round(self, rp: ReplayInputs) -> tuple[int, str]:
        T = rp.max_rounds
        for r in range(1, T + 1):
            decision = shp_should_halt(
                loop_count=r,
                distance_history=rp.distance_history[:max(0, r - 1)],
                is_score_history=(rp.is_history[:r] if rp.is_history else []),
                last_feedback=rp.feedbacks[r - 1] if r - 1 < len(rp.feedbacks) else "",
                config=self.cfg,
            )
            if decision.halt:
                return r, decision.reason
        return T, REASON_FAILSAFE


class EntropyOnlyPolicy(SHPPolicy):
    """Judge-free SHP: only the cosine-distance patience signal + failsafe."""

    def __init__(self, base: Optional[HaltConfig] = None):
        base = base or HaltConfig()
        cfg = HaltConfig(
            convergence_threshold=base.convergence_threshold,
            convergence_patience=base.convergence_patience,
            max_rounds=base.max_rounds,
            min_rounds_for_gain_check=base.min_rounds_for_gain_check,
            enable_critic_halt=False,
            enable_entropy_halt=True,
            enable_is_gain_halt=False,
        )
        super().__init__(cfg, name="entropy_only")


class FixedKPolicy(Policy):
    """Fixed iteration budget — the classic max_iterations kill-switch baseline."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"fixed_k{k}"
        self.uses_judge = False

    def stop_round(self, rp: ReplayInputs) -> tuple[int, str]:
        r = min(self.k, rp.max_rounds)
        return r, "failsafe"


class CriticOnlyPolicy(Policy):
    """Halt the first round the critic says APPROVED; else run to the cap."""
    name = "critic_only"
    uses_judge = False

    def stop_round(self, rp: ReplayInputs) -> tuple[int, str]:
        for r in range(1, rp.max_rounds + 1):
            if r - 1 < len(rp.approved_flags) and rp.approved_flags[r - 1]:
                return r, "critic_approved"
        return rp.max_rounds, "failsafe"


class RandomStopPolicy(Policy):
    """Deterministic pseudo-random stop in [1, max_rounds] (seeded by scenario)."""

    def __init__(self, seed: int):
        self.seed = seed
        self.name = f"random_s{seed}"
        self.uses_judge = False

    def stop_round(self, rp: ReplayInputs) -> tuple[int, str]:
        # Deterministic per scenario+seed via hashing the distance signature.
        sig = f"{self.seed}:" + ",".join(f"{d:.4f}" for d in rp.distance_history)
        h = int(hashlib.sha1(sig.encode()).hexdigest(), 16)
        r = 1 + (h % rp.max_rounds)
        return r, "random"


class OraclePolicy(Policy):
    """Upper bound: stop at the round of maximum measured IS (needs full judging)."""
    name = "oracle_is"
    uses_judge = True

    def stop_round(self, rp: ReplayInputs) -> tuple[int, str]:
        if not rp.is_history:
            return rp.max_rounds, "failsafe"
        best = max(range(len(rp.is_history)), key=lambda i: rp.is_history[i])
        return best + 1, "oracle_max_is"


def default_policy_suite(halt_config: Optional[HaltConfig] = None,
                         random_seeds: Sequence[int] = (0, 1, 2)) -> List[Policy]:
    """The standard comparison set used by the main experiment."""
    cfg = halt_config or HaltConfig()
    T = cfg.max_rounds
    suite: List[Policy] = [
        SHPPolicy(cfg),
        EntropyOnlyPolicy(cfg),
        CriticOnlyPolicy(),
        FixedKPolicy(1),
        FixedKPolicy(3),
        FixedKPolicy(max(1, T // 2)),
        FixedKPolicy(T),                 # == max_iterations baseline
        OraclePolicy(),
    ]
    suite += [RandomStopPolicy(s) for s in random_seeds]
    return suite


def _toggle(base: HaltConfig, *, critic: bool, entropy: bool, gain: bool) -> HaltConfig:
    return HaltConfig(
        convergence_threshold=base.convergence_threshold,
        convergence_patience=base.convergence_patience,
        max_rounds=base.max_rounds,
        min_rounds_for_gain_check=base.min_rounds_for_gain_check,
        enable_critic_halt=critic, enable_entropy_halt=entropy, enable_is_gain_halt=gain,
    )


def ablation_policy_suite(halt_config: Optional[HaltConfig] = None) -> List[Policy]:
    """
    Per-signal ablation matrix for the SHP cascade. Replays on the same cached
    trajectories, so the only incremental cost is judge tokens for variants whose
    IS-gain signal is enabled. Variant names encode which signals are ON.
    """
    cfg = halt_config or HaltConfig()
    matrix = [
        ("abl_full",        True,  True,  True),
        ("abl_no_critic",   False, True,  True),
        ("abl_no_entropy",  True,  False, True),
        ("abl_no_isgain",   True,  True,  False),
        ("abl_critic_only", True,  False, False),
        ("abl_entropy_only", False, True, False),
        ("abl_isgain_only", False, False, True),
    ]
    return [SHPPolicy(_toggle(cfg, critic=c, entropy=e, gain=g), name=name)
            for (name, c, e, g) in matrix]
