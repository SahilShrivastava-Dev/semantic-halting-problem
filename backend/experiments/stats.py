"""
stats.py

Significance machinery for the efficiency-vs-quality comparison. Everything is
PAIRED: each scenario is a block evaluated under every policy on the identical
trajectory, so we compare policies within-scenario and never across noisy
re-generations.

Provides:
  * paired_compare       — paired t-test + Wilcoxon signed-rank + Cohen's d_z
                           + bootstrap CI on the mean paired difference.
  * tost_noninferiority  — two one-sided tests that a quality metric is NOT worse
                           than a reference by more than margin δ (the rigorous
                           way to claim "no quality loss").
  * holm_correction      — Holm–Bonferroni across a family of p-values.
  * bootstrap_ci         — percentile bootstrap CI for any statistic.

Scipy is used where available; pure-numpy fallbacks keep the module importable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _as_pairs(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    return a, b


def cohens_dz(diff: np.ndarray) -> float:
    """Paired effect size: mean(diff) / sd(diff)."""
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def bootstrap_ci(values: Sequence[float], stat=np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for ``stat`` over ``values``."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [float(stat(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def paired_compare(treatment: Sequence[float], reference: Sequence[float],
                   seed: int = 0) -> Dict:
    """
    Compare ``treatment`` vs ``reference`` (paired). Difference = treatment − reference.
    Returns means, mean diff + bootstrap CI, paired t-test p, Wilcoxon p, Cohen's d_z.
    """
    t, r = _as_pairs(treatment, reference)
    diff = t - r
    out: Dict = {
        "n": int(len(diff)),
        "mean_treatment": float(np.mean(t)),
        "mean_reference": float(np.mean(r)),
        "mean_diff": float(np.mean(diff)),
        "mean_diff_ci95": bootstrap_ci(diff, seed=seed),
        "cohens_dz": cohens_dz(diff),
    }
    out["t_p_value"] = _paired_t_p(diff)
    out["wilcoxon_p_value"] = _wilcoxon_p(diff)
    return out


def _paired_t_p(diff: np.ndarray) -> Optional[float]:
    if len(diff) < 2 or np.std(diff) == 0:
        return None
    try:
        from scipy.stats import ttest_1samp
        return float(ttest_1samp(diff, 0.0).pvalue)
    except Exception:
        # Normal-approx fallback.
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        z = np.mean(diff) / se if se > 0 else 0.0
        from math import erf, sqrt
        return float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))))


def _wilcoxon_p(diff: np.ndarray) -> Optional[float]:
    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) < 6:
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(nz).pvalue)
    except Exception:
        return None


def tost_noninferiority(treatment: Sequence[float], reference: Sequence[float],
                        margin: float, seed: int = 0) -> Dict:
    """
    Non-inferiority via TOST: is treatment NOT worse than reference by more than
    ``margin`` on a higher-is-better quality metric?

    H0: mean(treatment − reference) <= −margin   (treatment is meaningfully worse)
    Reject H0 (one-sided, lower bound) ⇒ non-inferior. We report the lower one-sided
    p-value; non_inferior=True when p_lower < 0.05.
    """
    t, r = _as_pairs(treatment, reference)
    diff = t - r
    n = len(diff)
    result: Dict = {
        "n": int(n),
        "margin": margin,
        "mean_diff": float(np.mean(diff)),
        "mean_diff_ci95": bootstrap_ci(diff, seed=seed),
    }
    if n < 2 or np.std(diff, ddof=1) == 0:
        result["p_lower"] = None
        result["non_inferior"] = bool(np.mean(diff) > -margin)
        return result
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    # Lower test: H0 mean_diff <= -margin ; t-stat = (mean_diff + margin)/se
    t_lower = (float(np.mean(diff)) + margin) / se
    try:
        from scipy.stats import t as tdist
        p_lower = float(1 - tdist.cdf(t_lower, df=n - 1))
    except Exception:
        from math import erf, sqrt
        p_lower = float(1 - 0.5 * (1 + erf(t_lower / sqrt(2))))
    result["p_lower"] = p_lower
    result["non_inferior"] = bool(p_lower < 0.05)
    return result


def holm_correction(pvals: Dict[str, Optional[float]], alpha: float = 0.05) -> Dict[str, Dict]:
    """
    Holm–Bonferroni across a family of named p-values. Entries with p=None are
    passed through as untested. Returns per-name {p, adjusted_alpha, reject}.
    """
    named = [(k, v) for k, v in pvals.items() if v is not None]
    named.sort(key=lambda kv: kv[1])
    m = len(named)
    out: Dict[str, Dict] = {k: {"p": v, "adjusted_alpha": None, "reject": None}
                            for k, v in pvals.items()}
    still_rejecting = True
    for i, (k, p) in enumerate(named):
        adj = alpha / (m - i)
        reject = bool(still_rejecting and p < adj)
        if not reject:
            still_rejecting = False
        out[k] = {"p": p, "adjusted_alpha": adj, "reject": reject}
    return out
