"""
main.py

Automated end-to-end pipeline for the Semantic Halting Problem (SHP).

Implements a scientifically sound Train / Validate / Test workflow:

    ┌───────────────────────────────────────────────────────────┐
    │  PHASE 1 — TRAIN      agent_workflow.py  --split train    │
    │  PHASE 2 — EVALUATE   ragas_eval.py                       │
    │  PHASE 3 — CALIBRATE  optimize_score.py                   │
    │  PHASE 4 — VALIDATE   agent_workflow.py  --split val      │
    │  PHASE 5 — IS TEST    test_information_score.py           │
    └───────────────────────────────────────────────────────────┘

Each phase is a subprocess invocation using the same Python interpreter
that started this script (guaranteeing the correct virtual-env is used).
If any phase exits with a non-zero return code the pipeline aborts
immediately and reports the failing phase.

Usage
-----
    python main.py

Optional: run individual phases directly:
    python agent_workflow.py --split train
    python ragas_eval.py
    python optimize_score.py
    python agent_workflow.py --split val
    python test_information_score.py
"""

import logging
import subprocess
import sys

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────
def _run_phase(cmd: list[str], title: str) -> None:
    """
    Execute a pipeline phase as a subprocess and halt on failure.

    Streams stdout/stderr directly to the terminal (``capture_output=False``)
    so the user sees real-time progress from each sub-script.

    Args:
        cmd (list[str]): Command and arguments, e.g.
            ``[sys.executable, "agent_workflow.py", "--split", "train"]``.
        title (str): Human-readable phase label printed as a banner.

    Raises:
        SystemExit: If the subprocess exits with a non-zero return code.
    """
    separator = "═" * 70
    logger.info(separator)
    logger.info("🚀 %s", title)
    logger.info("   Command: %s", " ".join(cmd))
    logger.info(separator)

    result = subprocess.run(cmd, capture_output=False)  # noqa: S603

    if result.returncode != 0:
        logger.error(
            "❌ Phase '%s' failed with exit code %d. Aborting pipeline.",
            title, result.returncode,
        )
        sys.exit(result.returncode)

    logger.info("✔  Phase '%s' completed successfully.", title)


# ─────────────────────────────────────────────────────────────
# Pipeline definition
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """
    Execute the full SHP scientific pipeline in five sequential phases.

    Phase 1 — TRAIN
        Runs ``agent_workflow.py --split train``.
        Writer and Critic agents iterate on training scenarios;
        results are saved to ``agent_results.json``.

    Phase 2 — EVALUATE
        Runs ``ragas_eval.py``.
        Reads ``agent_results.json`` and scores each final draft using
        Ragas (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall).
        Saves scores to ``ragas_scores.json``.

    Phase 3 — CALIBRATE
        Runs ``optimize_score.py``.
        Fits Linear Regression to ``ragas_scores.json`` to learn the
        optimal IS weights.  Saves weights to ``optimized_weights.json``.

    Phase 4 — VALIDATE (live halting with learned weights)
        Runs ``agent_workflow.py --split val``.
        Validation scenarios are run using the newly learned IS weights
        for real-time Information Gain halting.  The log should show
        IS Gain converging toward 0 as the loop halts.

    Phase 5 — IS VALIDATION
        Runs ``test_information_score.py``.
        Confirms the IS formula correctly discriminates between
        good / convergent / poor answer quality levels.
    """
    logger.info("🌟 INITIALISING SHP SCIENTIFIC PIPELINE")
    logger.info("   Data split: [2 Train | 2 Validation | 2 Test]")

    py = sys.executable  # Ensures the active venv interpreter is used

    # ── Phase 1: Training data generation ────────────────────────────────────
    _run_phase(
        [py, "agent_workflow.py", "--split", "train"],
        "PHASE 1: Generating Training Data (Writer → Critic iterations)",
    )

    # ── Phase 2: Ragas evaluation on training outputs ─────────────────────────
    _run_phase(
        [py, "ragas_eval.py"],
        "PHASE 2: Ragas Evaluation (Faithfulness / Relevancy / Precision / Recall)",
    )

    # ── Phase 3: IS weight optimisation ──────────────────────────────────────
    _run_phase(
        [py, "optimize_score.py"],
        "PHASE 3: IS Weight Optimisation (Linear Regression → optimized_weights.json)",
    )

    # ── Phase 4: Validation with live IS-Gain halting ─────────────────────────
    logger.info("═" * 70)
    logger.info("⚖️  ENTERING VALIDATION PHASE")
    logger.info("   Workflow now uses learned weights from optimized_weights.json.")
    logger.info("   Watch for 'IS Gain' convergence in the logs below.")
    logger.info("═" * 70)

    _run_phase(
        [py, "agent_workflow.py", "--split", "val"],
        "PHASE 4: Validation (Live Information Gain Tracking with learned weights)",
    )

    # ── Phase 5: IS formula validation ───────────────────────────────────────
    _run_phase(
        [py, "test_information_score.py"],
        "PHASE 5: Information Score Validation (good > convergent > poor)",
    )

    logger.info("")
    logger.info("✅ SCIENTIFIC PIPELINE COMPLETE")
    logger.info(
        "   → Review Phase 4 logs for 'IS Gain' convergence evidence."
    )
    logger.info(
        "   → Review Phase 5 output for IS formula validation verdict."
    )
    logger.info(
        "   → Results artefacts: agent_results.json | ragas_scores.json | "
        "optimized_weights.json | doc/information_score_test_results.csv"
    )


if __name__ == "__main__":
    main()
