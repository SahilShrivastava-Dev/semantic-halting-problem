"""
test_information_score.py

Validates the IS formula across multiple answer quality tiers.
Reads optimized_weights.json and checks:
    IS(good) > IS(convergent) > IS(poor)

Usage
-----
    python test_information_score.py
    python test_information_score.py --provider openai --eval-model gpt-4o-mini
"""

import argparse
import json
import logging
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness,
)

from config import (
    DEFAULT_IS_WEIGHTS,
    DEFAULT_EVAL_MODELS,
    DEFAULT_PROVIDER,
    EMBEDDING_MODEL_NAME,
    IS_TEST_RESULTS_CSV,
    METRIC_COLS,
    RAGAS_MAX_WORKERS,
    RAGAS_EVAL_TIMEOUT,
    WEIGHTS_FILE,
)
from providers import get_ragas_llm

try:
    from ragas.run_config import RunConfig as _RagasRunConfig
    _HAS_RUN_CONFIG = True
except ImportError:
    _HAS_RUN_CONFIG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_weights() -> tuple[dict[str, float], str, object]:
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data["weights"], data.get("data_source", "unknown"), data.get("r2_score", "N/A")
    logger.warning("%s not found — using equal default weights.", WEIGHTS_FILE)
    return dict(DEFAULT_IS_WEIGHTS), "default", "N/A"


IS_WEIGHTS, _data_source, _r2 = _load_weights()

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
                "groundwater salinity."
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
        ],
        "ground_truth": (
            "IV labetalol was administered for acute BP reduction (target <25% in hour 1), "
            "followed by oral amlodipine 5mg and ramipril 5mg. Discharged in stable "
            "condition with 48-72 hour cardiology follow-up and weekly BP monitoring."
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
                "(2) weekly home blood pressure monitoring for the first month, and "
                "(3) lifestyle modifications including sodium restriction to <2g/day."
            ),
            "convergent": (
                "The patient was treated with intravenous labetalol to lower blood pressure "
                "acutely, with a target reduction under 25% in the first hour. "
                "Oral amlodipine and ramipril were started for ongoing control. The patient "
                "is discharged stable with cardiology follow-up within 48-72 hours and renal labs within one week."
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


def compute_information_score(scores: dict[str, float]) -> float:
    return round(sum(IS_WEIGHTS[m] * scores.get(m, 0.0) for m in METRIC_COLS), 4)


def _evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    ragas_llm,
    embeddings,
    metrics,
    run_config=None,
) -> dict[str, float]:
    ds = Dataset.from_dict({
        "question":     [question],
        "answer":       [answer],
        "contexts":     [contexts],
        "ground_truth": [ground_truth],
    })
    eval_kwargs: dict = dict(dataset=ds, metrics=metrics, llm=ragas_llm, embeddings=embeddings)
    if run_config is not None:
        eval_kwargs["run_config"] = run_config
    result = evaluate(**eval_kwargs)
    return {
        m.name: round(
            float(result[m.name][0] if isinstance(result[m.name], list) else result[m.name]),
            4,
        )
        for m in metrics
    }


def main(provider: str = DEFAULT_PROVIDER, eval_model: str | None = None) -> None:
    eval_model = eval_model or DEFAULT_EVAL_MODELS[provider]
    ragas_llm = get_ragas_llm(provider, eval_model)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    _n_workers = RAGAS_MAX_WORKERS if provider == "groq" else min(2, RAGAS_MAX_WORKERS + 1)
    _run_config = (
        _RagasRunConfig(max_workers=_n_workers, timeout=RAGAS_EVAL_TIMEOUT)
        if _HAS_RUN_CONFIG else None
    )

    logger.info("═" * 65)
    logger.info("IS VALIDATION  weights=%s", IS_WEIGHTS)
    logger.info("═" * 65)

    all_rows: list[dict] = []

    for scenario_id, data in TEST_CASES.items():
        logger.info("─" * 65)
        logger.info("Scenario: %s", scenario_id.upper().replace("_", " "))

        for quality, answer in data["answers"].items():
            scores     = _evaluate_single(
                data["question"], answer, data["contexts"],
                data["ground_truth"], ragas_llm, embeddings, metrics,
                run_config=_run_config,
            )
            info_score = compute_information_score(scores)
            logger.info("  [%s] IS=%.4f", quality.upper(), info_score)
            all_rows.append({
                "Scenario":          scenario_id,
                "Quality":           quality,
                **scores,
                "Information Score": info_score,
            })

    df = pd.DataFrame(all_rows)
    pivot = df.pivot_table(index="Scenario", columns="Quality", values="Information Score")
    available = [c for c in ["good", "convergent", "poor"] if c in pivot.columns]
    pivot = pivot[available]

    if "good" in pivot.columns and "convergent" in pivot.columns:
        pivot["good>convergent"] = pivot["good"] > pivot["convergent"]
    if "convergent" in pivot.columns and "poor" in pivot.columns:
        pivot["convergent>poor"] = pivot["convergent"] > pivot["poor"]

    logger.info("═" * 65)
    logger.info("VALIDATION SUMMARY\n%s", pivot.to_string())

    all_pass = (
        "good>convergent" in pivot.columns
        and "convergent>poor" in pivot.columns
        and bool(pivot["good>convergent"].all())
        and bool(pivot["convergent>poor"].all())
    )
    if all_pass:
        logger.info("✅ VALIDATION PASSED")
    else:
        logger.warning("⚠️  PARTIAL PASS")

    os.makedirs(os.path.dirname(IS_TEST_RESULTS_CSV), exist_ok=True)
    df.to_csv(IS_TEST_RESULTS_CSV, index=False)
    logger.info("💾 Results → %s", IS_TEST_RESULTS_CSV)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER)
    parser.add_argument("--eval-model", type=str, default=None)
    args = parser.parse_args()
    main(provider=args.provider, eval_model=args.eval_model)
