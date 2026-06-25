"""
run_experiment.py

Main efficiency-vs-quality experiment runner for SHP.

Pipeline per scenario:
    1. Generate (or load cached) a full-depth Writer→Critic trajectory — ONE
       generation pass shared by every policy (trajectory replay).
    2. Judge each round's draft with RAGAS (cached per draft hash) to get the
       Information Score history — the binding cost, paid at most once per draft.
    3. Replay every halt policy over the shared trajectory + IS history, recording
       stop round, halt reason, operational tokens, and final-draft quality.

Resumable: cached trajectories, cached judge scores, and an append-only rows.jsonl
mean an interrupted run (e.g. Groq daily-token cap) restarts with zero repeated
paid calls. A --daily-budget cap exits cleanly when the judge-token ceiling is hit.

Offline self-test: --mock fabricates deterministic trajectories + scores so the
whole harness (replay, costing, checkpointing, stats) can be validated without an
API key. Mock rows are clearly flagged and must never be reported as real results.

Usage:
    python experiments/run_experiment.py --mock --split dev          # offline test
    python experiments/run_experiment.py --split dev --max-rounds 6   # real (needs GROQ_API_KEY)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shp.config import DEFAULT_PROVIDER, MAX_ROUNDS, METRIC_COLS, DEFAULT_IS_WEIGHTS
from shp.halting import HaltConfig
from shp.trajectory import Trajectory, RoundRecord, TrajectoryGenerator
from shp.agent_workflow import _load_is_weights

from experiments.metrics_schema import ResultRow, RoundScore
from experiments.checkpoint import CheckpointStore
from experiments.policies import ReplayInputs, default_policy_suite, ablation_policy_suite
from experiments import stats as S
from shp.logging_utils import quiet_third_party_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
quiet_third_party_logs()   # silence httpx / HF / provider transport chatter
logger = logging.getLogger("run_experiment")


# ─────────────────────────────────────────────────────────────
# Mock generators (offline self-test only)
# ─────────────────────────────────────────────────────────────
def _seeded(text: str) -> float:
    """Deterministic float in [0,1) from text."""
    return (int(hashlib.sha1(text.encode()).hexdigest(), 16) % 10_000) / 10_000.0


def mock_trajectory(scenario: dict, max_rounds: int) -> Trajectory:
    """Fabricate a plausible converging trajectory (distances ↓, occasional approval)."""
    traj = Trajectory(
        scenario_id=scenario["id"], topic=scenario.get("topic", ""),
        question=scenario["question"], contexts=scenario["contexts"],
        ground_truth=scenario["ground_truth"],
    )
    base = _seeded(scenario["id"])
    for i in range(max_rounds):
        draft = f"[mock answer {scenario['id']} round {i+1}] base={base:.3f}"
        # Distance decays geometrically toward 0 with mild noise.
        dist = None if i == 0 else round(0.30 * (0.45 ** (i - 1)) + 0.01 * _seeded(draft), 6)
        approved = (i + 1) >= max(2, int(2 + 4 * base))  # approves partway through
        traj.rounds.append(RoundRecord(
            round=i + 1, draft=draft, feedback=("APPROVED" if approved else f"improve point {i+1}"),
            is_approved=approved, word_count=20, distance=dist,
            embedding=[base, 1 - base], writer_tokens=200 + i * 10, critic_tokens=80 + i * 5,
        ))
    return traj


def mock_score(draft: str, weights: dict) -> RoundScore:
    """Fabricate metrics that rise then plateau with the round index in the draft."""
    s = _seeded(draft)
    # Extract round number to make IS rise then plateau.
    try:
        rnd = int(draft.split("round ")[1].split("]")[0])
    except Exception:
        rnd = 1
    plateau = min(1.0, 0.55 + 0.12 * (1 - 0.6 ** rnd))
    metrics = {m: round(min(1.0, plateau + 0.05 * (_seeded(draft + m) - 0.5)), 4) for m in METRIC_COLS}
    is_score = round(sum(weights.get(m, 0.0) * metrics[m] for m in METRIC_COLS), 4)
    return RoundScore(metrics=metrics, information_score=is_score, judge_tokens=1500)


# ─────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────
def load_scenarios(path: str, split: str, limit: int | None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if split != "all":
        data = [s for s in data if s.get("split") == split]
    if limit:
        data = data[:limit]
    return data


def git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(args) -> None:
    scenarios = load_scenarios(args.scenarios, args.split, args.limit)
    if not scenarios:
        logger.error("No scenarios for split=%s in %s", args.split, args.scenarios)
        sys.exit(1)

    weights = _load_is_weights() if not args.mock else dict(DEFAULT_IS_WEIGHTS)
    halt_cfg = HaltConfig(max_rounds=args.max_rounds)
    policies = default_policy_suite(halt_cfg, random_seeds=tuple(args.random_seeds))
    if args.ablations:
        # Dedupe by name (the suite already has some toggle variants).
        existing = {p.name for p in policies}
        policies += [p for p in ablation_policy_suite(halt_cfg) if p.name not in existing]

    run_id = args.run_id or f"{args.split}_{args.provider}_mr{args.max_rounds}{'_mock' if args.mock else ''}"
    root = os.path.join("results", run_id)
    store = CheckpointStore(root)
    store.save_config_snapshot({
        "args": vars(args), "git_sha": git_sha(), "weights": weights,
        "halt_config": halt_cfg.__dict__, "n_scenarios": len(scenarios),
        "policies": [p.name for p in policies], "mock": args.mock,
    })
    logger.info("Run '%s': %d scenarios, %d policies, max_rounds=%d, mock=%s",
                run_id, len(scenarios), len(policies), args.max_rounds, args.mock)

    generator = None  # built lazily only for real runs
    judge = None
    judge_tokens_spent = 0

    for si, scenario in enumerate(scenarios):
        qid = scenario["id"]
        logger.info("[%d/%d] %s", si + 1, len(scenarios), qid)

        # 1. Trajectory (generate once, cache).
        if store.has_trajectory(qid):
            traj = store.load_trajectory(qid)
        else:
            if args.mock:
                traj = mock_trajectory(scenario, args.max_rounds)
            else:
                if generator is None:
                    generator = TrajectoryGenerator(args.provider, args.agent_model)
                traj = generator.generate(scenario, args.max_rounds)
            store.save_trajectory(traj)

        # 2. Judge each round (cache per draft hash) → IS history.
        is_history: list[float] = []
        judge_tokens_per_round: list[int] = []
        for rr in traj.rounds:
            cached = store.get_score(rr.draft)
            if cached is None:
                if args.mock:
                    cached = mock_score(rr.draft, weights)
                else:
                    if judge is None:
                        from experiments.judge import RagasJudge
                        judge = RagasJudge(args.provider, args.eval_model, weights)
                    res = judge.score(traj.question, rr.draft, traj.contexts, traj.ground_truth)
                    cached = RoundScore(res["metrics"], res["information_score"], res["judge_tokens"])
                store.put_score(rr.draft, cached)
                judge_tokens_spent += cached.judge_tokens
            is_history.append(cached.information_score)
            judge_tokens_per_round.append(cached.judge_tokens)

            if args.daily_budget and judge_tokens_spent >= args.daily_budget:
                logger.warning("Daily judge-token budget %d reached — exiting cleanly. "
                               "Re-run to resume.", args.daily_budget)
                _summarize(store, halt_cfg)
                return

        # 3. Replay every policy.
        rp = ReplayInputs(
            distance_history=traj.distance_history,
            is_history=is_history,
            feedbacks=[r.feedback for r in traj.rounds],
            approved_flags=[r.is_approved for r in traj.rounds],
            max_rounds=len(traj.rounds),
        )
        for policy in policies:
            if store.has_row(qid, policy.name, 0):
                continue
            stop_round, reason = policy.stop_round(rp)
            rounds_run = traj.rounds[:stop_round]
            writer_tok = sum(r.writer_tokens for r in rounds_run)
            critic_tok = sum(r.critic_tokens for r in rounds_run)
            judge_op = sum(judge_tokens_per_round[:stop_round]) if policy.uses_judge else 0

            final_score = store.get_score(traj.rounds[stop_round - 1].draft)
            metrics = final_score.metrics if final_score else {m: 0.0 for m in METRIC_COLS}

            store.append_row(ResultRow(
                scenario_id=qid, policy=policy.name, seed=0,
                stop_round=stop_round, halt_reason=reason,
                writer_tokens=writer_tok, critic_tokens=critic_tok,
                judge_tokens_operational=judge_op,
                total_operational_tokens=writer_tok + critic_tok + judge_op,
                final_is=final_score.information_score if final_score else 0.0,
                faithfulness=metrics.get("faithfulness", 0.0),
                answer_relevancy=metrics.get("answer_relevancy", 0.0),
                context_precision=metrics.get("context_precision", 0.0),
                context_recall=metrics.get("context_recall", 0.0),
                n_contexts=len(traj.contexts), max_rounds=args.max_rounds,
            ))

    _summarize(store, halt_cfg)


def _summarize(store: CheckpointStore, halt_cfg: HaltConfig) -> None:
    """Aggregate rows.jsonl → summary.json/csv with paired stats vs the max_iterations baseline."""
    rows = store.load_rows()
    if not rows:
        logger.warning("No rows to summarize yet.")
        return

    by_policy: dict[str, dict[str, ResultRow]] = {}
    for r in rows:
        by_policy.setdefault(r.policy, {})[r.scenario_id] = r

    baseline_name = f"fixed_k{halt_cfg.max_rounds}"
    baseline = by_policy.get(baseline_name, {})

    summary = {"baseline": baseline_name, "policies": {}}
    for policy, cells in sorted(by_policy.items()):
        shared = sorted(set(cells) & set(baseline)) if baseline else sorted(cells)
        n = len(shared)
        mean_rounds = sum(cells[s].stop_round for s in cells) / len(cells)
        mean_tokens = sum(cells[s].total_operational_tokens for s in cells) / len(cells)
        mean_is = sum(cells[s].final_is for s in cells) / len(cells)
        entry = {
            "n": len(cells), "mean_rounds": round(mean_rounds, 3),
            "mean_operational_tokens": round(mean_tokens, 1), "mean_final_is": round(mean_is, 4),
        }
        if baseline and policy != baseline_name and n >= 2:
            t_rounds = [cells[s].stop_round for s in shared]
            r_rounds = [baseline[s].stop_round for s in shared]
            t_tok = [cells[s].total_operational_tokens for s in shared]
            r_tok = [baseline[s].total_operational_tokens for s in shared]
            t_is = [cells[s].final_is for s in shared]
            r_is = [baseline[s].final_is for s in shared]
            entry["vs_baseline"] = {
                "rounds": S.paired_compare(t_rounds, r_rounds),
                "operational_tokens": S.paired_compare(t_tok, r_tok),
                "final_is": S.paired_compare(t_is, r_is),
                "is_noninferiority_delta0.02": S.tost_noninferiority(t_is, r_is, margin=0.02),
            }
        summary["policies"][policy] = entry

    with open(os.path.join(store.root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Flat CSV for figures / quick scan.
    csv_path = os.path.join(store.root, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("policy,n,mean_rounds,mean_operational_tokens,mean_final_is\n")
        for policy, e in summary["policies"].items():
            fh.write(f"{policy},{e['n']},{e['mean_rounds']},{e['mean_operational_tokens']},{e['mean_final_is']}\n")

    logger.info("Summary written → %s", csv_path)
    logger.info("Baseline = %s", baseline_name)
    for policy, e in summary["policies"].items():
        tail = ""
        if "vs_baseline" in e:
            ni = e["vs_baseline"]["is_noninferiority_delta0.02"]
            tail = f" | ΔIS={e['vs_baseline']['final_is']['mean_diff']:+.4f} non_inferior={ni['non_inferior']}"
        logger.info("  %-14s rounds=%.2f tok=%.0f IS=%.4f%s",
                    policy, e["mean_rounds"], e["mean_operational_tokens"], e["mean_final_is"], tail)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SHP efficiency-vs-quality experiment")
    ap.add_argument("--scenarios", default="data/scenarios_hotpot.json")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["groq", "openai", "nvidia"])
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--eval-model", default=None)
    ap.add_argument("--max-rounds", type=int, default=min(6, MAX_ROUNDS),
                    help="Trajectory depth = failsafe cap for the experiment (budget knob).")
    ap.add_argument("--limit", type=int, default=None, help="Cap #scenarios (debug).")
    ap.add_argument("--daily-budget", type=int, default=None,
                    help="Exit cleanly after this many judge tokens (Groq daily cap).")
    ap.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--ablations", action="store_true",
                    help="Also run the per-signal ablation matrix (free on cached trajectories).")
    ap.add_argument("--mock", action="store_true", help="Offline synthetic self-test (no API).")
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
