"""
ragas_eval.py

Evaluates the ACTUAL final drafts produced by agent_workflow.py using the
Ragas v0.4 framework. Reads agent_results.json (written by agent_workflow.py)
so that we are grading real agent outputs — not hardcoded dummy data.

Saves scores to ragas_scores.json for downstream use by optimize_score.py
and test_information_score.py.

Metrics computed:
    - Faithfulness       (did the answer hallucinate?)
    - Answer Relevancy   (did the answer address the question?)
    - Context Precision  (were the right contexts retrieved?)
    - Context Recall     (was any important context missed?)
"""
import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Model Setup
# Judge LLM: Groq (free tier, fast) — no HF credits needed.
# Embeddings: BAAI/bge-small-en-v1.5 (local, no API required).
# ─────────────────────────────────────────────────────────────
print("[Setup] Initialising Ragas judge (Groq) and embedding models...")

ragas_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=2048,
))
ragas_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

# ─────────────────────────────────────────────────────────────
# Load agent results
# ─────────────────────────────────────────────────────────────
AGENT_RESULTS_FILE = "agent_results.json"

if not os.path.exists(AGENT_RESULTS_FILE):
    print(f"❌ ERROR: {AGENT_RESULTS_FILE} not found.")
    print("   Please run agent_workflow.py first to generate agent outputs.")
    exit(1)

with open(AGENT_RESULTS_FILE, "r") as f:
    agent_results = json.load(f)

print(f"[Info] Loaded {len(agent_results)} scenario result(s) from {AGENT_RESULTS_FILE}")
print("[Info] Evaluating the ACTUAL final drafts produced by the agents...\n")

# ─────────────────────────────────────────────────────────────
# Evaluate each scenario's final agent output
# ─────────────────────────────────────────────────────────────
all_scores = []

for entry in agent_results:
    scenario_id = entry["scenario_id"]
    print(f"\n{'━'*55}")
    print(f"📋 Evaluating: {scenario_id.upper().replace('_', ' ')}")
    print(f"   Rounds until halt: {entry['rounds']}")
    print(f"   Question:          {entry['question']}")
    print(f"   Final answer:      ...{entry['answer'][-80:]}")
    print(f"{'━'*55}")

    dataset = Dataset.from_dict({
        "question":     [entry["question"]],
        "answer":       [entry["answer"]],
        "contexts":     [entry["contexts"]],
        "ground_truth": [entry["ground_truth"]]
    })

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    scores = {}
    for metric in METRICS:
        val = result[metric.name]
        raw = float(val[0]) if isinstance(val, list) else float(val)
        # nan means the LLM couldn't finish the evaluation — treat as 0
        scores[metric.name] = round(0.0 if raw != raw else raw, 4)

    print(f"\n   Faithfulness:      {scores.get('faithfulness', 0):.4f}")
    print(f"   Answer Relevancy:  {scores.get('answer_relevancy', 0):.4f}")
    print(f"   Context Precision: {scores.get('context_precision', 0):.4f}")
    print(f"   Context Recall:    {scores.get('context_recall', 0):.4f}")

    all_scores.append({
        "scenario_id":        scenario_id,
        "rounds":             entry["rounds"],
        "faithfulness":       scores.get("faithfulness", 0),
        "answer_relevancy":   scores.get("answer_relevancy", 0),
        "context_precision":  scores.get("context_precision", 0),
        "context_recall":     scores.get("context_recall", 0),
    })

# ─────────────────────────────────────────────────────────────
# Save for optimize_score.py and test_information_score.py
# ─────────────────────────────────────────────────────────────
with open("ragas_scores.json", "w") as f:
    json.dump(all_scores, f, indent=2)

print(f"\n\n{'═'*55}")
print("📊  RAGAS EVALUATION SUMMARY")
print(f"{'═'*55}")
for s in all_scores:
    avg = (s['faithfulness'] + s['answer_relevancy'] +
           s['context_precision'] + s['context_recall']) / 4
    print(f"  [{s['scenario_id']}]  avg={avg:.3f}  halted@round={s['rounds']}")
print(f"{'═'*55}")
print(f"\n💾 Scores saved to ragas_scores.json")