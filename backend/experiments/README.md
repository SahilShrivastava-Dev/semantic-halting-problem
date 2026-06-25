# SHP Experiment Harness

Judge-efficient, resumable efficiency-vs-quality evaluation of halt policies.

## Pipeline

1. **Generate** each question's full-depth Writer→Critic trajectory **once**
   (`trajectory.TrajectoryGenerator`) and cache it.
2. **Judge** each round's draft with RAGAS, cached per draft hash (judged at most
   once, ever).
3. **Replay** every halt policy over the shared trajectory + IS history. All
   policies see identical drafts → strictly paired comparison.
4. **Summarise** with paired stats (t-test, Wilcoxon, Cohen's d_z, TOST
   non-inferiority, bootstrap CIs) vs the `max_iterations` baseline.

Why this design: only the RAGAS judge costs real money on Groq free tier.
Generating once and caching judging makes the comparison both *fair* (no
generation-noise confound) and *cheap* (each draft judged once).

## Prerequisites

```bash
pip install -r ../requirements.txt
cp ../.env.example ../.env          # set GROQ_API_KEY
python ../build_dataset.py --n 80 --dev 20   # writes ../data/scenarios_hotpot.json
```

## Run (real)

```bash
# Tune on dev, then freeze test. --max-rounds is the budget knob (failsafe cap).
python run_experiment.py --split dev  --max-rounds 6 --ablations
python run_experiment.py --split test --max-rounds 6 --ablations
python make_figures.py --run-id test_groq_mr6
```

## Offline self-test (no API key)

```bash
python run_experiment.py --mock --split dev --max-rounds 6 --ablations --run-id mocktest
python make_figures.py --run-id mocktest
```
Mock runs fabricate deterministic trajectories + scores to validate the harness.
**Mock outputs are flagged in `config_snapshot.json` (`"mock": true`) and must
never be reported as results.**

## Resume & budget

Interrupted runs restart with zero repeated paid calls:
- cached trajectory → generation skipped
- cached judge score (per draft hash) → draft not re-judged
- `(scenario, policy)` already in `rows.jsonl` → cell skipped

`--daily-budget <tokens>` exits cleanly when the judge-token ceiling is hit
(Groq daily cap); just re-run the same command to continue.

## Outputs (`results/<run_id>/`)

| Path | Versioned? | Contents |
|---|---|---|
| `config_snapshot.json` | yes | git SHA, args, seeds, weights, halt config |
| `summary.json` / `summary.csv` | yes | per-policy aggregates + paired stats |
| `figures/` | yes | Fig 1–3, Table 1–3, Conjecture-1 report |
| `trajectories/` | no (bulky) | per-question drafts + embeddings |
| `judge_cache/` | no (bulky) | per-draft RAGAS results |
| `rows.jsonl` | no (regenerable) | raw per-cell result rows |
