"""
build_dataset.py

Replace the toy scenarios (16 items, 2 topics) with a real public RAG-QA
benchmark: HotpotQA (distractor setting). HotpotQA is multi-hop, so the
Writer→Critic loop has genuine room to iterate — single-fact questions converge
in one round and make the efficiency comparison vacuous, which is exactly the
weakness of the original toy set.

Design choices (all defensible + budget-aware for Groq free tier):
  * Filter to ``level == "hard"`` multi-hop questions (type ∈ {bridge, comparison}).
  * Context = the gold supporting paragraphs + a few distractor paragraphs, in a
    deterministic shuffled order. Keeping ~4 contexts (not all 10) bounds the
    RAGAS judge cost while preserving a realistic retrieval-with-distractors
    setting. Use --max-contexts to widen.
  * Deterministic dev/test split by hashing the example id (no RNG, reproducible).
    Thresholds are tuned on DEV only; TEST is frozen and report-only.

Output: backend/data/scenarios_hotpot.json — same schema as test_scenarios.json
plus an ``initial_brief`` field (fixes the writer's missing-brief fallback) and a
``meta`` block recording provenance.

Usage:
    python build_dataset.py                       # N=80, 20 dev / 60 test, hard, 4 ctx
    python build_dataset.py --n 60 --dev 15 --max-contexts 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from typing import Dict, List

# Make the backend/ dir importable so `shp` resolves when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()   # picks up HF_TOKEN for authenticated, faster HF downloads

from datasets import load_dataset

from shp.logging_utils import quiet_third_party_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
quiet_third_party_logs()   # silence httpx / huggingface_hub transport chatter
logger = logging.getLogger(__name__)

OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "scenarios_hotpot.json")


def _paragraphs(context: dict) -> Dict[str, str]:
    """Map each context title → its joined paragraph text."""
    titles = context["title"]
    sentences = context["sentences"]
    return {t: " ".join(s).strip() for t, s in zip(titles, sentences)}


def _stable_bucket(example_id: str, dev_fraction: float) -> str:
    """Deterministic dev/test assignment from a hash of the id (no RNG)."""
    h = int(hashlib.sha1(example_id.encode("utf-8")).hexdigest(), 16)
    return "dev" if (h % 10_000) / 10_000.0 < dev_fraction else "test"


def _stable_order(example_id: str, items: List[str]) -> List[str]:
    """Deterministically reorder a list using a per-item hash seeded by the id."""
    return sorted(items, key=lambda t: hashlib.sha1((example_id + t).encode()).hexdigest())


def build_scenario(ex: dict, max_contexts: int) -> dict:
    """Convert one HotpotQA example into the SHP scenario schema."""
    para = _paragraphs(ex["context"])
    gold_titles = list(dict.fromkeys(ex["supporting_facts"]["title"]))  # unique, ordered
    gold_titles = [t for t in gold_titles if t in para]

    distractor_titles = [t for t in para if t not in gold_titles]
    n_distract = max(0, max_contexts - len(gold_titles))
    chosen_distractors = _stable_order(ex["id"], distractor_titles)[:n_distract]

    selected = _stable_order(ex["id"], gold_titles + chosen_distractors)
    contexts = [para[t] for t in selected if para[t]]

    topic = gold_titles[0] if gold_titles else (selected[0] if selected else ex["question"][:40])
    question = ex["question"].strip()

    return {
        "id": f"hotpot_{ex['id']}",
        "topic_id": ex.get("type", "unknown"),
        "split": "",  # filled by caller
        "topic": topic,
        "question": question,
        "initial_brief": question,            # fixes agents.py missing-brief fallback
        "contexts": contexts,
        "ground_truth": ex["answer"].strip(),
        "meta": {
            "source": "hotpot_qa/distractor/validation",
            "type": ex.get("type"),
            "level": ex.get("level"),
            "n_gold": len(gold_titles),
            "n_contexts": len(contexts),
        },
    }


def _valid(s: dict) -> bool:
    return bool(s["question"]) and bool(s["ground_truth"]) and len(s["contexts"]) >= 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a HotpotQA RAG scenario set for SHP.")
    ap.add_argument("--n", type=int, default=80, help="Total scenarios to emit.")
    ap.add_argument("--dev", type=int, default=20, help="How many go to the dev split.")
    ap.add_argument("--level", type=str, default="hard",
                    choices=["easy", "medium", "hard", "any"],
                    help="HotpotQA difficulty filter (hard = most multi-hop room).")
    ap.add_argument("--max-contexts", type=int, default=4,
                    help="Gold + distractor paragraphs per question (judge-cost knob).")
    ap.add_argument("--scan-limit", type=int, default=4000,
                    help="Max streamed examples to scan while collecting N matches.")
    args = ap.parse_args()

    if args.dev >= args.n:
        ap.error("--dev must be smaller than --n")

    logger.info("Streaming HotpotQA distractor (validation); collecting %d level=%s scenarios…",
                args.n, args.level)
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)

    collected: List[dict] = []
    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > args.scan_limit:
            break
        if args.level != "any" and ex.get("level") != args.level:
            continue
        scenario = build_scenario(ex, args.max_contexts)
        if _valid(scenario):
            collected.append(scenario)
        if len(collected) >= args.n:
            break

    if len(collected) < args.n:
        logger.warning("Only collected %d/%d after scanning %d examples.",
                       len(collected), args.n, scanned)

    # Deterministic split: bucket by id, then enforce exact dev/test counts by
    # taking the dev-bucketed ones first (stable by id) up to args.dev.
    dev_fraction = args.dev / args.n
    for s in collected:
        s["split"] = _stable_bucket(s["id"], dev_fraction)

    dev = [s for s in collected if s["split"] == "dev"]
    test = [s for s in collected if s["split"] == "test"]
    # Rebalance to exact counts deterministically (hash-ordered) if buckets drift.
    ordered = sorted(collected, key=lambda s: hashlib.sha1(s["id"].encode()).hexdigest())
    for i, s in enumerate(ordered):
        s["split"] = "dev" if i < args.dev else "test"
    dev = [s for s in ordered if s["split"] == "dev"]
    test = [s for s in ordered if s["split"] == "test"]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, indent=2, ensure_ascii=False)

    from collections import Counter
    logger.info("Wrote %d scenarios → %s", len(ordered), OUT_FILE)
    logger.info("  splits: %s", Counter(s["split"] for s in ordered))
    logger.info("  types:  %s", Counter(s["meta"]["type"] for s in ordered))
    logger.info("  avg contexts: %.1f", sum(len(s["contexts"]) for s in ordered) / max(1, len(ordered)))
    logger.info("  dev=%d test=%d (dev tuned, test frozen)", len(dev), len(test))


if __name__ == "__main__":
    main()
