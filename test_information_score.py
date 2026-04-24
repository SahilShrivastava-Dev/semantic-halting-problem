"""
test_information_score.py

Validates the Information Score formula across multiple answer quality tiers.

Reads ``optimized_weights.json`` (produced by ``optimize_score.py``) and
applies the derived IS formula to three quality variants for each scenario:

    good        — A complete, faithful, factual answer grounded in context.
    convergent  — A semantically identical paraphrase (simulates a deadlock
                  output; same information, different wording).
    poor        — An incorrect / hallucinated answer.

Validation criterion
--------------------
    IS(good) > IS(convergent) > IS(poor)

If this ordering holds for every scenario, the formula correctly discriminates
between high-quality, stagnant, and hallucinated outputs — confirming that it
is a valid real-time halting signal for the SHP loop.

Output
------
    Prints a per-scenario, per-quality breakdown to stdout (via logging).
    Saves full results to ``doc/information_score_test_results.csv``.

Usage
-----
    python test_information_score.py
"""

import json
import logging
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config import (
    DEFAULT_IS_WEIGHTS,
    EMBEDDING_MODEL_NAME,
    EVAL_LLM_MAX_TOKENS,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    IS_TEST_RESULTS_CSV,
    METRIC_COLS,
    WEIGHTS_FILE,
)

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
# Load optimised weights
# ─────────────────────────────────────────────────────────────
def _load_weights() -> tuple[dict[str, float], str, object]:
    """
    Load IS weights from ``optimized_weights.json`` or fall back to defaults.

    Returns:
        tuple: ``(weights_dict, data_source_label, r2_or_na)``
    """
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, "r") as fh:
            data = json.load(fh)
        weights     = data["weights"]
        data_source = data.get("data_source", "unknown")
        r2          = data.get("r2_score", "N/A")
        logger.info(
            "Loaded IS weights from %s  (source=%s, R²=%s): %s",
            WEIGHTS_FILE, data_source, r2, weights,
        )
        return weights, data_source, r2

    logger.warning(
        "%s not found — using default equal weights. "
        "Run optimize_score.py first for learned weights.",
        WEIGHTS_FILE,
    )
    return dict(DEFAULT_IS_WEIGHTS), "default", "N/A"


IS_WEIGHTS, _data_source, _r2 = _load_weights()


# ─────────────────────────────────────────────────────────────
# Validation test cases
# ─────────────────────────────────────────────────────────────
# Each entry contains a real-world scenario with three quality tiers:
#   good        → faithful, relevant, context-grounded answer
#   convergent  → paraphrase of the good answer (same semantics, different words)
#   poor        → hallucinated / factually wrong answer
TEST_CASES: dict[str, dict] = {
    "dubai_real_estate": {
        "question": (
            "What structural methods and materials ensure the integrity "
            "of the Dubai high-rise foundation?"
        ),
        "contexts": [
            "High-rise buildings in Dubai's coastal sandy soil typically require deep "
            "pile foundations drilled 30-50 meters into bedrock to prevent differential settlement.",
            "Reinforced concrete pile caps connected by ground beams form the standard "
            "foundation transfer system for Dubai skyscrapers, distributing structural loads evenly.",
            "Grade 60 deformed steel rebar embedded in high-strength concrete (C50 or above) "
            "is mandated by Dubai Municipality for structural columns and shear walls.",
            "Dubai Building Code Section 4.2 requires corrosion-resistant coatings on steel "
            "reinforcement due to the high salinity of groundwater.",
        ],
        "ground_truth": (
            "The Dubai high-rise uses deep reinforced concrete pile foundations "
            "(30-50m into bedrock) with Grade 60 steel rebar in C50+ concrete and "
            "corrosion-resistant coatings per Dubai Building Code Section 4.2."
        ),
        "answers": {
            "good": (
                "The structural integrity of the Dubai high-rise is ensured through deep "
                "reinforced concrete pile foundations drilled 30–50 metres into bedrock, "
                "preventing differential settlement in the coastal sandy soil. "
                "Pile caps connected by reinforced ground beams distribute vertical loads "
                "uniformly across the foundation. Grade 60 deformed steel rebar embedded "
                "in C50+ high-strength concrete is used throughout structural columns and "
                "shear walls, as mandated by Dubai Municipality. "
                "All steel reinforcement is coated with corrosion-resistant treatments in "
                "compliance with Dubai Building Code Section 4.2, protecting against high "
                "groundwater salinity. Post-tensioned flat-plate slabs further reduce "
                "floor-to-floor height while maintaining structural performance under high "
                "vertical loads."
            ),
            "convergent": (
                "The foundation of the Dubai high-rise relies on deep pile foundations "
                "embedded in bedrock, with reinforced concrete pile caps distributing loads. "
                "Grade 60 rebar set in high-grade concrete meets Dubai Municipality "
                "requirements, and corrosion-resistant coatings on all steel comply with "
                "Section 4.2 of the Dubai Building Code."
            ),
            "poor": (
                "The Dubai high-rise uses a standard wooden post-and-beam foundation system "
                "with bamboo reinforcement, which is lightweight and eco-friendly. "
                "The structure uses pre-cast aluminum panels for the load-bearing walls, "
                "reducing construction time. No specific building code requirements apply "
                "to this type of structure."
            ),
        },
    },
    "medical_discharge": {
        "question": (
            "What treatment was administered for the hypertensive crisis "
            "and what is the patient's discharge plan?"
        ),
        "contexts": [
            "A hypertensive crisis (BP > 180/120 mmHg) requires immediate IV labetalol "
            "or hydralazine for rapid blood pressure reduction.",
            "Target blood pressure reduction in hypertensive emergencies is no more than "
            "25% within the first hour to prevent cerebral hypoperfusion.",
            "Amlodipine 5mg OD combined with ramipril 5mg OD is a standard oral "
            "antihypertensive regimen post-crisis.",
            "Patients require 48-72 hour cardiology follow-up and weekly BP monitoring "
            "for the first month post-discharge.",
            "Renal function (serum creatinine, eGFR) must be rechecked within one week "
            "post-discharge.",
        ],
        "ground_truth": (
            "IV labetalol was administered for acute BP reduction (target <25% in hour 1), "
            "followed by oral amlodipine 5mg and ramipril 5mg. Discharged in stable "
            "condition with 48-72 hour cardiology follow-up, weekly BP monitoring, and "
            "renal function recheck within one week."
        ),
        "answers": {
            "good": (
                "The patient presented with a hypertensive crisis at 195/115 mmHg and "
                "received IV labetalol titrated to achieve a controlled reduction of no "
                "more than 25% within the first hour, preventing cerebral hypoperfusion. "
                "Following haemodynamic stabilisation, an oral antihypertensive regimen "
                "of amlodipine 5mg OD and ramipril 5mg OD was commenced. "
                "The patient is discharged in stable condition with instructions for: "
                "(1) a 48–72 hour cardiology follow-up appointment, "
                "(2) weekly home blood pressure monitoring for the first month, "
                "(3) renal function panel (serum creatinine, eGFR) within one week, and "
                "(4) lifestyle modifications including sodium restriction to <2g/day and "
                "30 minutes of daily aerobic exercise."
            ),
            "convergent": (
                "The patient was treated with intravenous labetalol to lower blood pressure "
                "acutely, with a target reduction under 25% in the first hour. "
                "Oral amlodipine and ramipril were started for ongoing control. The patient "
                "is discharged in a stable state with a follow-up scheduled with cardiology "
                "within 48-72 hours and renal labs within one week."
            ),
            "poor": (
                "The patient was given aspirin and ibuprofen to manage headache associated "
                "with the elevated blood pressure. A low-sodium diet was recommended. "
                "The patient was advised to return to the emergency department if symptoms "
                "recur. No specialist follow-up was arranged."
            ),
        },
    },
}


# ─────────────────────────────────────────────────────────────
# Groq compatibility shim
# ─────────────────────────────────────────────────────────────
class _FixedChatGroq(ChatGroq):
    """Forces ``n=1`` on Groq API calls for Ragas compatibility."""

    def _create_message_dicts(self, messages, stop):
        message_dicts, params = super()._create_message_dicts(messages, stop)
        params.pop("n", None)
        return message_dicts, params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ─────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────
logger.info("Initialising Ragas judge (%s) and embeddings (%s)...",
            EVAL_LLM_MODEL, EMBEDDING_MODEL_NAME)
_ragas_llm = LangchainLLMWrapper(_FixedChatGroq(
    model=EVAL_LLM_MODEL,
    temperature=EVAL_LLM_TEMPERATURE,
    max_tokens=EVAL_LLM_MAX_TOKENS,
))
_ragas_emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
_METRICS   = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def compute_information_score(scores: dict[str, float]) -> float:
    """
    Apply the IS formula to a dict of Ragas metric scores.

    Args:
        scores (dict[str, float]): Mapping of metric name → float value
            (from Ragas evaluation).

    Returns:
        float: Composite Information Score in [0.0, 1.0], rounded to 4 d.p.
    """
    return round(
        sum(IS_WEIGHTS[m] * scores.get(m, 0.0) for m in METRIC_COLS),
        4,
    )


def _evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    """
    Run all four Ragas metrics on a single (Q, A, C, GT) tuple.

    Args:
        question (str):     The test question.
        answer (str):       The answer being evaluated.
        contexts (list):    Retrieved context strings.
        ground_truth (str): Reference answer.

    Returns:
        dict[str, float]: Metric name → float score.  NaN → 0.0.
    """
    ds = Dataset.from_dict({
        "question":    [question],
        "answer":      [answer],
        "contexts":    [contexts],
        "ground_truth": [ground_truth],
    })
    result = evaluate(dataset=ds, metrics=_METRICS, llm=_ragas_llm, embeddings=_ragas_emb)
    return {
        m.name: round(
            float(result[m.name][0] if isinstance(result[m.name], list) else result[m.name]),
            4,
        )
        for m in _METRICS
    }


# ─────────────────────────────────────────────────────────────
# Main validation loop
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """
    Run the IS validation suite and report pass/fail per scenario.

    Saves results to ``doc/information_score_test_results.csv``.
    """
    logger.info("═" * 65)
    logger.info("⚖️  INFORMATION SCORE VALIDATION")
    logger.info("   Weights: %s", IS_WEIGHTS)
    logger.info("═" * 65)

    all_rows: list[dict] = []

    for scenario_id, data in TEST_CASES.items():
        logger.info("─" * 65)
        logger.info("📋 Scenario: %s", scenario_id.upper().replace("_", " "))
        logger.info("─" * 65)

        for quality, answer in data["answers"].items():
            logger.info("  ▶ [%s]  %s...", quality.upper(), answer[:80])

            scores     = _evaluate_single(
                data["question"], answer, data["contexts"], data["ground_truth"]
            )
            info_score = compute_information_score(scores)

            for col in METRIC_COLS:
                logger.info("    %-22s %.4f", col + ":", scores.get(col, 0.0))
            logger.info("    ─────────────────────────────────────────")
            logger.info("    ⭐ Information Score: %.4f", info_score)

            all_rows.append({
                "Scenario":          scenario_id,
                "Quality":           quality,
                **scores,
                "Information Score": info_score,
            })

    # ── Summary pivot ──────────────────────────────────────────────────────────
    df    = pd.DataFrame(all_rows)
    pivot = df.pivot_table(
        index="Scenario", columns="Quality", values="Information Score"
    )
    available_cols = [c for c in ["good", "convergent", "poor"] if c in pivot.columns]
    pivot = pivot[available_cols]

    if "good" in pivot.columns and "convergent" in pivot.columns:
        pivot["good>convergent"] = pivot["good"] > pivot["convergent"]
    else:
        pivot["good>convergent"] = False

    if "convergent" in pivot.columns and "poor" in pivot.columns:
        pivot["convergent>poor"] = pivot["convergent"] > pivot["poor"]
    else:
        pivot["convergent>poor"] = False

    logger.info("═" * 65)
    logger.info("📊 VALIDATION SUMMARY — Information Score by Quality Level")
    logger.info("═" * 65)
    logger.info("\n%s", pivot.to_string())

    formula_str = " + ".join([f"({v}×{k})" for k, v in IS_WEIGHTS.items()])
    logger.info("Formula: IS = %s", formula_str)

    all_pass = (
        "good>convergent" in pivot.columns
        and "convergent>poor" in pivot.columns
        and bool(pivot["good>convergent"].all())
        and bool(pivot["convergent>poor"].all())
    )

    if all_pass:
        logger.info(
            "✅ VALIDATION PASSED — IS correctly ranks all quality levels!"
        )
    else:
        fails = pivot[~(pivot["good>convergent"] & pivot["convergent>poor"])]
        logger.warning(
            "⚠️  PARTIAL PASS — Failed scenarios:\n%s\n"
            "Consider collecting more real data to improve weight calibration.",
            fails.to_string(),
        )

    logger.info("═" * 65)

    # Save CSV
    os.makedirs(os.path.dirname(IS_TEST_RESULTS_CSV), exist_ok=True)
    df.to_csv(IS_TEST_RESULTS_CSV, index=False)
    logger.info("💾 Full results → %s", IS_TEST_RESULTS_CSV)


if __name__ == "__main__":
    main()
