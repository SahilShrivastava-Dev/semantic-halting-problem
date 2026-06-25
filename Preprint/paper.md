# Semantic Early-Stopping for Iterative LLM Agent Loops: A Judge-Efficient Study of When to Halt

> Honest, evidence-backed manuscript scaffold. Every quantitative claim is filled
> from `backend/results/<run>/figures/` (figures + tables); every theoretical
> claim is machine-checked by `backend/theory_checks.py`. Placeholders read
> `‹from …›`. Theory section text is in `Preprint/theory.md`.

## Abstract

Iterative multi-agent LLM loops (e.g. Writer→Critic) typically terminate on a
fixed `max_iterations` cap — a syntactic kill-switch blind to whether the output
has stopped improving. We study **semantic early stopping**: halting when
consecutive draft embeddings stop changing (cosine-distance patience), optionally
combined with critic approval and an information-gain signal. We make three
contributions. (1) An **honest theoretical footing**: we prove deterministic
termination and well-definedness, and treat the convergence of the distance
sequence as an empirically tested *conjecture* rather than a Banach contraction.
(2) A **judge-efficient evaluation protocol** — generate each trajectory once,
replay all halt policies over it, and cache every LLM-judge call — that makes a
paired efficiency-vs-quality comparison feasible on free-tier compute. (3) An
empirical study on multi-hop RAG QA (HotpotQA) showing ‹SHP/entropy-only saves
X% rounds and Y% operational tokens vs `max_iterations` with non-inferior answer
quality (TOST, δ=0.02, p=‹·›)›, and an ablation isolating which halt signal
drives the savings — including the finding that the judge-based information-gain
signal can cost more than it saves, motivating a **judge-free entropy-only**
variant.

## 1. Introduction
- Problem: `max_iterations` is wasteful (over-runs easy cases, under-runs hard
  ones) and blind to output semantics.
- Idea: stop when the *meaning* of the draft converges.
- Contributions (above). Emphasise honesty: termination is guaranteed,
  contraction is not claimed; quality is measured with a noisy judge and tested
  for non-inferiority, not asserted.

## 2. Related Work
Distilled from `Preprint/` (verify IDs first; see `preprint.md`). Fixed-point /
semantic-convergence theory; uncertainty in multi-LLM systems; multi-agent
orchestration and termination; token-efficiency in MAS. Position SHP as: *when to
exit a fixed sequential loop*, judged on quality, evaluated cheaply.

## 3. Method
- Writer→Critic loop (RAG-grounded: both agents condition on retrieved contexts).
- Semantic distance `d_t = 1 − cos(e_t, e_{t-1})` on frozen `bge-small-en-v1.5`.
- Information Score `IS_t = Σ w_m · RAGAS_m`; weights from `optimize_score.py`
  (entropy / AHP / constrained-LS / equal — already implemented).
- Halt cascade (one shared function, `backend/halting.py`): critic_approved →
  entropy_convergence (k-patience) → no_information_gain → failsafe.

## 4. Theory (honest)
Verbatim from `Preprint/theory.md`: Theorem 1 (termination), Lemma 1
(well-definedness), Lemma 2 (halt-priority consistency), Conjecture 1 (semantic
non-expansiveness — empirical). Table 3 (`table3_theory_claims.csv`) reports the
machine-checked status of each proven claim.

## 5. Experimental Setup
- Data: HotpotQA distractor, multi-hop `hard` questions; N=‹80›, 20 dev / 60 test
  (`backend/build_dataset.py`). Thresholds tuned on dev; test frozen.
- Compute: Groq free tier (`llama-3.1-8b-instant` agents; judge ‹model›).
  State the constraint openly. Embeddings local/free.
- Protocol: trajectory replay + cached judging (`backend/experiments/`). Paired
  statistics across scenarios; TOST non-inferiority (δ∈{0.01,0.02,0.05}); Holm
  correction; bootstrap CIs. Seeds recorded in `config_snapshot.json`.
- Cost accounting: operational vs evaluation tokens (Method §metrics_schema).

## 6. Results
- **Fig 2 (headline)** efficiency–quality Pareto: ‹SHP/entropy-only dominates
  fixed-k baselines›.
- **Fig 3** operational tokens saved vs `max_iterations` (±bootstrap CI): ‹Y%›.
- **Table 1** per-policy rounds/tokens/IS + paired p, Cohen's d_z, TOST.
- Non-inferiority on IS: ‹non_inferior = True/False, p_lower = ·›.

## 7. Analysis
- **Fig 1** distance trajectory + Conjecture 1: ‹frac_monotone=·, slope CI=·›.
- **Table 2** ablations: which signal drives savings; the judge-cost of the
  information-gain signal; entropy-only as the efficient sweet spot.
- Failure cases: questions that converge in 1 round; judge-noise sensitivity.

## 8. Limitations & Threats to Validity
- Free-tier scale ceiling (N≈80); designed to scale on budget.
- LLM-judge (8B) is a noisy quality proxy → judge-reliability analysis; stronger
  judge on the frozen test split if budget allows.
- δ for non-inferiority is a modelling choice → sensitivity reported.
- Conjecture 1 may fail per-trajectory; Theorem 1 guarantees safety regardless.

## 9. Reproducibility
Seeds, git SHA, resolved config in every `config_snapshot.json`. Commands:
```
python build_dataset.py --n 80 --dev 20
python experiments/run_experiment.py --split dev  --max-rounds 6 --ablations   # tune
python experiments/run_experiment.py --split test --max-rounds 6 --ablations   # frozen
python experiments/make_figures.py --run-id test_groq_mr6
python theory_checks.py
```
