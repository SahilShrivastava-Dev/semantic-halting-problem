"""
pipeline.py

Automated end-to-end SHP scientific pipeline.

Phases
------
    1. TRAIN      — agent_workflow --split train    → agent_results.json
    2. EVALUATE   — ragas_eval                       → ragas_scores.json
    3. CALIBRATE  — optimize_score                   → optimized_weights.json
    4. VALIDATE   — agent_workflow --split val       → agent_results.json (overwritten)
    5. IS TEST    — test_information_score           → doc/information_score_test_results.csv

Usage
-----
    python pipeline.py
    python pipeline.py --provider openai --agent-model gpt-4o-mini
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run_phase(cmd: list[str], title: str) -> None:
    """Execute a pipeline phase as a subprocess; abort on non-zero exit code."""
    sep = "═" * 70
    logger.info(sep)
    logger.info("🚀 %s", title)
    logger.info("   Command: %s", " ".join(cmd))
    logger.info(sep)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error(
            "❌ Phase '%s' failed (exit %d). Aborting pipeline.",
            title, result.returncode,
        )
        sys.exit(result.returncode)

    logger.info("✔  Phase '%s' completed.", title)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHP Scientific Pipeline")
    parser.add_argument("--provider", type=str, default=None, choices=["groq", "openai"])
    parser.add_argument("--agent-model", type=str, default=None)
    parser.add_argument("--eval-model",  type=str, default=None)
    args = parser.parse_args()

    py = sys.executable

    # Build optional provider/model args to pass through
    extra: list[str] = []
    if args.provider:
        extra += ["--provider", args.provider]
    if args.agent_model:
        extra += ["--agent-model", args.agent_model]
    if args.eval_model:
        extra += ["--eval-model", args.eval_model]

    logger.info("🌟 SHP SCIENTIFIC PIPELINE [2 Train | 2 Val | 2 Test]")

    _run_phase(
        [py, "agent_workflow.py", "--split", "train"] + extra,
        "PHASE 1: Training Data Generation",
    )

    eval_extra = []
    if args.provider:
        eval_extra += ["--provider", args.provider]
    if args.eval_model:
        eval_extra += ["--eval-model", args.eval_model]

    _run_phase(
        [py, "ragas_eval.py"] + eval_extra,
        "PHASE 2: Ragas Evaluation (Faithfulness / Relevancy / Precision / Recall)",
    )

    _run_phase(
        [py, "optimize_score.py"],
        "PHASE 3: IS Weight Optimisation (entropy/AHP/constrained_ls → optimized_weights.json)",
    )

    logger.info("═" * 70)
    logger.info("⚖️  ENTERING VALIDATION PHASE — using optimised weights from optimized_weights.json")
    logger.info("   Strategy: set IS_WEIGHT_STRATEGY in .env or pass --strategy to optimize_score.py")
    logger.info("═" * 70)

    _run_phase(
        [py, "agent_workflow.py", "--split", "val"] + extra,
        "PHASE 4: Validation (Live IS-Gain Tracking with learned weights)",
    )

    _run_phase(
        [py, "test_information_score.py"] + eval_extra,
        "PHASE 5: IS Formula Validation (good > convergent > poor)",
    )

    logger.info("")
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("   → agent_results.json | ragas_scores.json | optimized_weights.json")
    logger.info("   → doc/information_score_test_results.csv")


if __name__ == "__main__":
    main()
