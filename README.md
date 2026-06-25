# Semantic Halting Problem (SHP)

**When should a multi-agent LLM loop stop iterating?** Most systems stop after a
fixed number of rounds (`max_iterations`) — a blunt counter that has no idea
whether the answer is still improving. SHP replaces that counter with a
**semantic** stopping rule: stop when the answer stops changing *in meaning* and
stops getting *better in quality*.

This repository contains a working implementation, an honest theoretical
treatment, and a reproducible evaluation that measures whether semantic halting
actually saves work without sacrificing answer quality.

---

## Table of contents
1. [The problem, in one minute](#the-problem-in-one-minute)
2. [The idea](#the-idea)
3. [How it decides to stop (the halt cascade)](#how-it-decides-to-stop-the-halt-cascade)
4. [What we claim — honestly](#what-we-claim--honestly)
5. [How we prove it works (the experiment)](#how-we-prove-it-works-the-experiment)
6. [Repository structure](#repository-structure)
7. [Setup](#setup)
8. [Running things](#running-things)
9. [Where each result in the paper comes from](#where-each-result-in-the-paper-comes-from)
10. [Honest limitations](#honest-limitations)

---

## The problem, in one minute

A common LLM pattern is a **Writer → Critic loop**: a Writer drafts an answer, a
Critic critiques it, the Writer revises, and so on. You need a rule for when to
stop. The usual rule is "stop after N rounds." That is wasteful two ways:

- **Easy questions** are solved in 1–2 rounds but the loop keeps spinning to N.
- **Hard questions** might need more than N, but get cut off.

Worse, the counter is *blind to meaning*: it cannot tell that rounds 4, 5, and 6
all said essentially the same thing.

## The idea

Embed every draft into a vector (using a local sentence-embedding model) and
measure the **cosine distance** between consecutive drafts. When that distance
stays tiny for a couple of rounds, the answer has **converged in meaning** —
further iteration is just churn. Stop there.

We pair that geometric signal with a **quality signal** (the Information Score, a
weighted blend of four RAG-quality metrics) so the loop only stops when the
answer has *both* stopped changing *and* stopped improving.

```
draft₁ ──► draft₂ ──► draft₃ ──► draft₄
         d=0.21     d=0.05     d=0.01     ← cosine distance shrinking
                                ▲
                        meaning has converged → halt
```

## How it decides to stop (the halt cascade)

Four signals, checked in priority order (one shared function,
`backend/shp/halting.py`):

| Priority | Signal | Stops when… | Analogy |
|---|---|---|---|
| 1 | **Critic approval** | the Critic replies `APPROVED` | reviewer signs off |
| 2 | **Entropy convergence** | cosine distance `< ε` for `k` consecutive rounds | loss curve flattens |
| 3 | **No information gain** | the Information Score stops rising | gradient vanishes |
| 4 | **Failsafe** | a hard cap `MAX_ROUNDS` is hit | hard deadline |

Signal 4 can never be disabled — it is what guarantees the loop always
terminates (see below). Signals 1–3 can each be toggled off for ablation studies.

## What we claim — honestly

An earlier version of this project claimed the loop was a **Banach
contraction** with a guaranteed unique fixed point. **We no longer claim that** —
LLM generation has no proven contraction constant and isn't deterministic across
API calls, so that theorem would be unsupported. Instead:

- **Proven (machine-checked in `backend/shp/theory_checks.py`):**
  - *Termination* — the loop always halts in `≤ MAX_ROUNDS` rounds, for any input.
  - *Well-definedness* — the Information Score is always in `[0,1]`; weights form
    a valid probability distribution; the distance is always finite.
  - *Halt-priority consistency* — the reason reported after a run is exactly the
    reason that actually stopped it (one shared cascade, no drift).
- **Conjecture (measured, not assumed):** that cosine distance tends to *decrease*
  across rounds. We report the real fraction of runs where this holds, with a
  confidence interval — including if it sometimes fails.

The full write-up is in [`Preprint/theory.md`](Preprint/theory.md) (theory) and
[`Preprint/paper.md`](Preprint/paper.md) (manuscript scaffold).

## How we prove it works (the experiment)

Claiming "semantic halting saves effort without hurting quality" means nothing
without measurement. The harness in `backend/experiments/` does exactly that, on
the real multi-hop **HotpotQA** benchmark, with a design built for cheap, fair,
reproducible comparison:

- **Trajectory replay** — generate each question's full Writer→Critic trajectory
  **once**, then let every stopping policy *replay* over the same drafts. All
  policies see identical answers → a strictly **paired** comparison, and the
  expensive generation is paid once.
- **Cached judging** — the RAG-quality judge (RAGAS) is the costly part, so every
  draft is judged at most once and cached on disk.
- **Honest cost accounting** — we separate *operational* tokens (what a policy
  spends to run) from *evaluation* tokens (what we spend to measure it). Notably,
  full SHP pays for its own per-round quality checks; a judge-free *entropy-only*
  variant does not — and we charge each fairly.
- **Real statistics** — paired t-test, Wilcoxon, Cohen's *d*, **TOST
  non-inferiority** (the rigorous way to claim "no quality loss"), Holm
  correction, bootstrap confidence intervals.

Baselines compared against SHP: `fixed_k` (the `max_iterations` baseline),
`critic_only`, `random_stop`, and an `oracle` upper bound.

## Repository structure

```
semantic-halting-problem/
├── requirements.txt            # convenience: installs backend/requirements.txt
├── backend/
│   ├── shp/                    # ← core library package
│   │   ├── config.py           #   all constants, thresholds, ablation flags
│   │   ├── halting.py          #   the single shared halt cascade
│   │   ├── semantic_entropy.py #   cosine-distance convergence signal
│   │   ├── agents.py           #   RAG-grounded Writer & Critic
│   │   ├── agent_workflow.py   #   the LangGraph Writer→Critic loop
│   │   ├── trajectory.py       #   generate-once full trajectory (for replay)
│   │   ├── providers.py        #   Groq / OpenAI / NVIDIA model factory (metered)
│   │   ├── token_meter.py      #   process-wide token accounting
│   │   ├── optimize_score.py   #   IS-weight strategies (entropy/AHP/…)
│   │   ├── ragas_eval.py       #   batch RAGAS evaluation
│   │   ├── theory_checks.py    #   machine-checked theorems & lemmas
│   │   └── logging_utils.py    #   quiet noisy third-party logs
│   ├── experiments/            # ← the research study
│   │   ├── run_experiment.py   #   orchestrator (replay + policies + stats)
│   │   ├── policies.py         #   SHP / fixed-k / critic-only / random / oracle
│   │   ├── judge.py            #   cached RAGAS quality judge
│   │   ├── checkpoint.py       #   resumable cache store
│   │   ├── stats.py            #   paired tests, TOST, bootstrap
│   │   ├── metrics_schema.py   #   typed result rows (what we record)
│   │   └── make_figures.py     #   rows → figures & tables for the paper
│   ├── api/app.py              # FastAPI + WebSocket dashboard server
│   ├── scripts/                # build_dataset.py, pipeline.py
│   ├── tests/                  # test_information_score.py
│   ├── data/                   # scenario sets (HotpotQA + legacy toy set)
│   └── results/                # experiment outputs (figures, summaries)
├── Preprint/                   # paper.md, theory.md, reference PDFs
└── frontend/                   # React + TypeScript live dashboard
```

> **Convention:** run all backend commands from the `backend/` directory.

## Setup

```bash
cd backend
python -m venv ../venv && source ../venv/bin/activate    # or use the existing venv
pip install -r requirements.txt

cp .env.example .env     # then add your key(s) — see below
```

`.env` keys (only the one you use is required; `.env` is git-ignored):

| Provider | Env var | Notes |
|---|---|---|
| **NVIDIA build** | `NVIDIA_API_KEY` | OpenAI-compatible, generous limits — **recommended for experiments** |
| **Groq** | `GROQ_API_KEY` | free tier, but heavily rate-limited |
| **OpenAI** | `OPENAI_API_KEY` | paid |
| HuggingFace | `HF_TOKEN` | optional, faster dataset downloads |

Embeddings always run **locally** (`BAAI/bge-small-en-v1.5`) — no API key needed.

## Running things

```bash
cd backend
source ../venv/bin/activate

# 1) Build the real RAG benchmark (HotpotQA multi-hop)
python scripts/build_dataset.py --n 80 --dev 20

# 2) Verify the proven theoretical claims
python -m shp.theory_checks

# 3) Run the study (resumable; --mock runs offline with no API key)
python experiments/run_experiment.py --split dev  --provider nvidia --ablations
python experiments/run_experiment.py --split test --provider nvidia --ablations

# 4) Generate the figures & tables
python experiments/make_figures.py --run-id test_nvidia_mr6

# 5) (Optional) live dashboard
uvicorn api.app:app --reload --port 8000        # + cd ../frontend && npm run dev
```

See [`backend/experiments/README.md`](backend/experiments/README.md) for resume
and budgeting details.

## Where each result in the paper comes from

| Paper artifact | Produced by | From |
|---|---|---|
| Termination / well-definedness table | `shp/theory_checks.py` | proofs, machine-checked |
| Distance-trajectory figure + Conjecture 1 | `make_figures.py` → `fig1` | cached trajectories |
| Efficiency–quality Pareto (headline) | `make_figures.py` → `fig2` | result rows |
| Tokens-saved bar chart | `make_figures.py` → `fig3` | result rows |
| Per-policy stats + non-inferiority | `make_figures.py` → `table1` | `stats.py` |
| Ablation matrix | `make_figures.py` → `table2` | `--ablations` rows |

## Honest limitations

- **Compute ceiling.** Experiments were run on free / credit-based APIs; the
  benchmark size (N≈80) is modest. The harness is built to scale trivially when
  more budget is available.
- **The judge is an LLM.** The quality metric (RAGAS) is itself produced by a
  model, so it is a noisy proxy; we use a stronger model for the judge than for
  the agents and test for non-inferiority rather than asserting equality.
- **Conjecture 1 may fail per-run.** Distances are not guaranteed monotone; the
  *termination* guarantee (Theorem 1) is what keeps the system safe regardless.

---

*Working notes on related work live in `preprint.md` (AI-assisted, advocacy-leaning
— verify its citations before use). The honest manuscript is `Preprint/paper.md`.*
