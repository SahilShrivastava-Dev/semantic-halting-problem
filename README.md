# Semantic Halting Problem (SHP) — Production Full-Stack

A research implementation of **Semantic Entropy Convergence** for multi-agent LangGraph workflows, with a real-time ML-style dashboard.

## The Core Idea

Traditional multi-agent loops use `max_iterations` as a kill-switch — syntactic, blind to meaning. This project replaces it with **mathematical halting** based on the Banach Fixed-Point Theorem applied to text embedding space:

> When consecutive draft embeddings converge (cosine distance → 0), the system has reached a semantic fixed point. Halt.

## Halting Signals (priority order)

| Signal | Analogy | Description |
|---|---|---|
| Critic Approval | — | LLM critic returns `APPROVED` |
| Entropy Convergence | Loss → 0 | k=2 patience: cosine distance < 0.06 for 2 consecutive rounds |
| No IS Gain | Gradient vanished | Information Score Δ ≤ 0 after warm-up |
| Failsafe | Max epochs | Hard cap at `MAX_ROUNDS = 12` |

## ML Visualisation Analogy

| ML Training | SHP Dashboard |
|---|---|
| Loss curve (↓) | Cosine distance per round |
| Accuracy curve (↑) | Information Score per round |
| Gradient magnitude | IS Gain per round |
| Learned weights | IS metric weights (w₁…w₄ from linear regression) |
| Early stopping | Semantic halt signal |

## Project Structure

```
semantic-halting-problem/
├── backend/               # Python FastAPI + LangGraph
│   ├── app.py             # FastAPI server + WebSocket
│   ├── agent_workflow.py  # LangGraph graph (Writer→Critic loop)
│   ├── agents.py          # Writer & Critic LLM nodes
│   ├── providers.py       # Groq / OpenAI abstraction
│   ├── semantic_entropy.py # Cosine distance calculator
│   ├── ragas_eval.py      # Batch Ragas evaluation
│   ├── optimize_score.py  # IS weight learning (LinearRegression)
│   ├── pipeline.py        # Full 5-phase pipeline runner
│   ├── config.py          # All constants & thresholds
│   └── requirements.txt
└── frontend/              # React + TypeScript + Recharts
    └── src/
        ├── App.tsx         # Main app + state management
        ├── components/
        │   ├── ConvergenceChart.tsx  # IS score curve
        │   ├── DistanceChart.tsx     # Cosine distance curve
        │   ├── ISGainChart.tsx       # IS gain bars
        │   ├── WeightsChart.tsx      # Learned weight bars
        │   ├── MetricsBreakdown.tsx  # Per-metric Ragas lines
        │   ├── LogStream.tsx         # Live agent log
        │   ├── ProviderSelector.tsx
        │   ├── ScenarioPanel.tsx
        │   └── HaltBadge.tsx
        └── hooks/
            └── useWebSocket.ts
```

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt

# Create .env from the example
cp .env.example .env
# Edit .env: set GROQ_API_KEY or OPENAI_API_KEY

# Start the server
uvicorn app:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

### 3. Run the full scientific pipeline (optional, CLI)

```bash
cd backend
python pipeline.py                          # uses Groq by default
python pipeline.py --provider openai --agent-model gpt-4o-mini
```

## Supported LLM Providers

| Provider | Models | API Key |
|---|---|---|
| **Groq** (free) | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `mixtral-8x7b-32768` | `GROQ_API_KEY` |
| **OpenAI** | `gpt-4o-mini`, `gpt-4o` | `OPENAI_API_KEY` |

Embeddings always use `BAAI/bge-small-en-v1.5` locally (no API key needed).

## Why Cosine Distance?

- Measures **angular distance** (semantic direction) not magnitude — correct for text
- BAAI/bge is trained with contrastive InfoNCE loss specifically for semantic similarity
- Threshold 0.06 calibrated to catch synonym-level paraphrases (~0.04) with margin
- k=2 patience window prevents false-positive halts on single noisy rounds

## Mathematical Framework

```
SPS(t) = w₁·Faithfulness + w₂·AnswerRelevancy + w₃·ContextPrecision + w₄·ContextRecall

Halt when:
    CosSim(State_t, State_{t-1}) < ε  for k consecutive rounds
    OR  SPS(t) - SPS(t-1) ≤ 0  (after warm-up)
    OR  Critic returns APPROVED
    OR  t ≥ MAX_ROUNDS
```

Weights w₁…w₄ are learned via `LinearRegression` on Ragas scores from training scenarios.
