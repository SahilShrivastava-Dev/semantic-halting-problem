"""
make_figures.py

Turn a completed run (results/<run_id>/rows.jsonl + cached trajectories) into the
paper's figures and tables. Reads only on-disk artifacts — no LLM calls — so it
is safe to run repeatedly and offline.

Outputs into results/<run_id>/figures/:
    fig1_distance_trajectory.png   mean cosine distance vs round (±95% CI) + Conjecture 1 report
    fig2_pareto_rounds_is.png      rounds (x) vs final IS (y), one point per policy (headline)
    fig3_tokens_saved.png          % operational tokens saved vs max_iterations baseline (±bootstrap CI)
    table1_policies.csv            per-policy rounds/tokens/IS + paired p, d, TOST
    table2_ablations.csv           ablation matrix (abl_* policies)
    table3_theory_claims.csv       machine-checked theory claims (from theory_checks)

Usage:
    python experiments/make_figures.py --run-id dev_groq_mr6
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shp.halting import HaltConfig
from experiments.checkpoint import CheckpointStore
from experiments.metrics_schema import ResultRow
from experiments import stats as S
import shp.theory_checks as TC
from shp.logging_utils import quiet_third_party_logs

logging.basicConfig(level=logging.INFO, format="%(message)s")
quiet_third_party_logs()
logger = logging.getLogger("make_figures")


def _load(store: CheckpointStore):
    rows = store.load_rows()
    by_policy: dict[str, dict[str, ResultRow]] = {}
    for r in rows:
        by_policy.setdefault(r.policy, {})[r.scenario_id] = r
    return rows, by_policy


def fig1_distance_trajectory(store: CheckpointStore, out_dir: str) -> dict:
    """Mean distance vs round with 95% CI; also returns Conjecture 1 report."""
    dist_histories = []
    for qid_file in os.listdir(store.traj_dir):
        if not qid_file.endswith(".json") or qid_file.endswith(".emb.json"):
            continue
        with open(os.path.join(store.traj_dir, qid_file), encoding="utf-8") as fh:
            meta = json.load(fh)
        dh = [r["distance"] for r in meta["rounds"] if r["distance"] is not None]
        if dh:
            dist_histories.append(dh)

    if not dist_histories:
        logger.warning("fig1: no trajectories found.")
        return {}

    max_len = max(len(d) for d in dist_histories)
    # round index here is 2..(max_len+1) because distance starts at round 2.
    means, los, his, xs = [], [], [], []
    for j in range(max_len):
        col = [d[j] for d in dist_histories if j < len(d)]
        if len(col) < 2:
            continue
        m = float(np.mean(col))
        lo, hi = S.bootstrap_ci(col, seed=1)
        xs.append(j + 2)
        means.append(m); los.append(lo); his.append(hi)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, means, marker="o", color="#1f77b4", label="mean cosine distance")
    plt.fill_between(xs, los, his, alpha=0.2, color="#1f77b4", label="95% CI")
    plt.axhline(HaltConfig().convergence_threshold, ls="--", color="crimson",
                label=f"ε = {HaltConfig().convergence_threshold}")
    plt.xlabel("round t"); plt.ylabel("cosine distance d_t")
    plt.title("Semantic distance across rounds (Conjecture 1)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_distance_trajectory.png"), dpi=150)
    plt.close()

    report = TC.empirical_monotonicity_report(dist_histories)
    with open(os.path.join(out_dir, "conjecture1_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("fig1: %d trajectories | frac_monotone=%.2f mean_slope=%.4f supports=%s",
                report.get("n_usable", 0), report.get("frac_monotone_nonincreasing", float('nan')),
                report.get("mean_slope", float('nan')), report.get("supports_conjecture"))
    return report


def fig2_pareto(by_policy: dict, out_dir: str) -> None:
    plt.figure(figsize=(6.5, 4.5))
    for policy, cells in sorted(by_policy.items()):
        if policy.startswith("abl_"):
            continue
        rounds = np.mean([c.stop_round for c in cells.values()])
        is_mean = np.mean([c.final_is for c in cells.values()])
        marker = "*" if policy == "shp" else ("D" if policy == "entropy_only" else "o")
        size = 240 if policy in ("shp", "entropy_only") else 90
        plt.scatter(rounds, is_mean, s=size, marker=marker, label=policy, zorder=3)
        plt.annotate(policy, (rounds, is_mean), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    plt.xlabel("mean rounds (lower = cheaper)")
    plt.ylabel("mean final Information Score (higher = better)")
    plt.title("Efficiency–quality Pareto (top-left is best)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_pareto_rounds_is.png"), dpi=150)
    plt.close()
    logger.info("fig2: Pareto written.")


def fig3_tokens_saved(by_policy: dict, baseline_name: str, out_dir: str) -> None:
    baseline = by_policy.get(baseline_name, {})
    if not baseline:
        logger.warning("fig3: baseline %s missing.", baseline_name)
        return
    names, pcts, los, his = [], [], [], []
    for policy, cells in sorted(by_policy.items()):
        if policy in (baseline_name,) or policy.startswith("abl_"):
            continue
        shared = sorted(set(cells) & set(baseline))
        if len(shared) < 2:
            continue
        saved = [100.0 * (1 - cells[s].total_operational_tokens /
                          max(1, baseline[s].total_operational_tokens)) for s in shared]
        lo, hi = S.bootstrap_ci(saved, seed=2)
        names.append(policy); pcts.append(float(np.mean(saved))); los.append(lo); his.append(hi)

    if not names:
        return
    y = np.arange(len(names))
    errs = [np.array(pcts) - np.array(los), np.array(his) - np.array(pcts)]
    plt.figure(figsize=(6.5, 0.5 * len(names) + 1.5))
    colors = ["#2ca02c" if p >= 0 else "#d62728" for p in pcts]
    plt.barh(y, pcts, xerr=errs, color=colors, alpha=0.8, capsize=3)
    plt.yticks(y, names); plt.axvline(0, color="k", lw=0.8)
    plt.xlabel(f"% operational tokens saved vs {baseline_name} (95% CI)")
    plt.title("Token savings (negative = more expensive)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_tokens_saved.png"), dpi=150)
    plt.close()
    logger.info("fig3: token-savings written.")


def table1_policies(by_policy: dict, baseline_name: str, out_dir: str) -> None:
    baseline = by_policy.get(baseline_name, {})
    path = os.path.join(out_dir, "table1_policies.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "n", "mean_rounds", "mean_tokens", "mean_IS",
                    "dIS_vs_base", "rounds_p", "rounds_dz", "IS_TOST_noninferior(d=0.02)"])
        for policy, cells in sorted(by_policy.items()):
            if policy.startswith("abl_"):
                continue
            mr = np.mean([c.stop_round for c in cells.values()])
            mt = np.mean([c.total_operational_tokens for c in cells.values()])
            mi = np.mean([c.final_is for c in cells.values()])
            dis = rp = dz = ni = ""
            if baseline and policy != baseline_name:
                shared = sorted(set(cells) & set(baseline))
                if len(shared) >= 2:
                    t_is = [cells[s].final_is for s in shared]
                    r_is = [baseline[s].final_is for s in shared]
                    t_r = [cells[s].stop_round for s in shared]
                    r_r = [baseline[s].stop_round for s in shared]
                    cmp_r = S.paired_compare(t_r, r_r)
                    tost = S.tost_noninferiority(t_is, r_is, margin=0.02)
                    dis = round(np.mean(t_is) - np.mean(r_is), 4)
                    rp = cmp_r["t_p_value"]; dz = round(cmp_r["cohens_dz"], 3)
                    ni = tost["non_inferior"]
            w.writerow([policy, len(cells), round(mr, 3), round(mt, 1), round(mi, 4),
                        dis, rp, dz, ni])
    logger.info("table1 → %s", path)


def table2_ablations(by_policy: dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "table2_ablations.csv")
    abl = {k: v for k, v in by_policy.items() if k.startswith("abl_")}
    if not abl:
        logger.info("table2: no ablation rows (run with --ablations).")
        return
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ablation", "n", "mean_rounds", "mean_tokens", "mean_IS"])
        for policy, cells in sorted(abl.items()):
            w.writerow([policy, len(cells),
                        round(np.mean([c.stop_round for c in cells.values()]), 3),
                        round(np.mean([c.total_operational_tokens for c in cells.values()]), 1),
                        round(np.mean([c.final_is for c in cells.values()]), 4)])
    logger.info("table2 → %s", path)


def table3_theory(out_dir: str) -> None:
    """Run the proven checks and record pass/fail as a table for the paper."""
    path = os.path.join(out_dir, "table3_theory_claims.csv")
    claims = [
        ("Theorem 1: termination ≤ MAX_ROUNDS", TC.assert_termination),
        ("Lemma 1a: IS ∈ [0,1]", TC.assert_is_bounds),
        ("Lemma 1c: distance total & bounded", TC.assert_distance_total),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claim", "status"])
        for name, fn in claims:
            try:
                fn(); status = "PASS"
            except Exception as e:  # pragma: no cover
                status = f"FAIL: {e}"
            w.writerow([name, status])
    logger.info("table3 → %s", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline", default=None, help="Baseline policy (default fixed_k<max_rounds>).")
    args = ap.parse_args()

    root = os.path.join("results", args.run_id)
    if not os.path.isdir(root):
        logger.error("No run at %s", root); sys.exit(1)
    store = CheckpointStore(root)
    out_dir = os.path.join(root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Infer baseline from config snapshot if not given.
    baseline = args.baseline
    if baseline is None:
        snap_path = os.path.join(root, "config_snapshot.json")
        mr = 6
        if os.path.exists(snap_path):
            with open(snap_path) as fh:
                mr = json.load(fh).get("halt_config", {}).get("max_rounds", 6)
        baseline = f"fixed_k{mr}"

    rows, by_policy = _load(store)
    logger.info("Loaded %d rows across %d policies. Baseline=%s", len(rows), len(by_policy), baseline)

    fig1_distance_trajectory(store, out_dir)
    fig2_pareto(by_policy, out_dir)
    fig3_tokens_saved(by_policy, baseline, out_dir)
    table1_policies(by_policy, baseline, out_dir)
    table2_ablations(by_policy, out_dir)
    table3_theory(out_dir)
    logger.info("All figures/tables → %s", out_dir)


if __name__ == "__main__":
    main()
