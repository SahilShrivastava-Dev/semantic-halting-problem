"""
main.py

Central pipeline orchestrator for the Semantic Halting Problem (SHP) project.
Runs all four stages sequentially, with data flowing between them via JSON files:

    Stage 1 → agent_workflow.py      → produces agent_results.json
    Stage 2 → ragas_eval.py          → reads agent_results.json  → produces ragas_scores.json
    Stage 3 → optimize_score.py      → reads ragas_scores.json   → produces optimized_weights.json
    Stage 4 → test_information_score.py → reads optimized_weights.json → validates formula

Run with:
    python3 main.py
"""
import subprocess
import sys


def run_script(script_name: str, step_description: str):
    """
    Executes a Python script as a subprocess, using the current virtual environment.

    Args:
        script_name (str):       Filename of the script to run.
        step_description (str):  Human-readable label shown in the pipeline header.

    Raises:
        SystemExit: If the script exits with a non-zero return code.
    """
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_description}")
    print(f"   Executing: {script_name}")
    print(f"{'='*60}\n")

    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed during '{script_name}'. Exit code: {e.returncode}")
        sys.exit(1)


def main():
    print("🌟 INITIALIZING SHP PIPELINE 🌟")
    print("""
Data flow:
  agent_workflow.py  →  agent_results.json
                               ↓
  ragas_eval.py      →  ragas_scores.json
                               ↓
  optimize_score.py  →  optimized_weights.json
                               ↓
  test_information_score.py  →  validation report
""")

    # Stage 1: Run the multi-agent loop; saves agent_results.json
    run_script("agent_workflow.py",
               "Semantic Entropy Multi-Agent Exoskeleton")

    # Stage 2: Evaluate REAL agent outputs; saves ragas_scores.json
    run_script("ragas_eval.py",
               "Ragas LLM Evaluation (reads agent_results.json)")

    # Stage 3: Derive optimal weights from real scores; saves optimized_weights.json
    run_script("optimize_score.py",
               "Information Score Weight Optimisation (reads ragas_scores.json)")

    # Stage 4: Validate the formula is working correctly
    run_script("test_information_score.py",
               "Information Score Validation (reads optimized_weights.json)")

    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("""
Artefacts produced:
  📄 agent_results.json       — final drafts from every scenario
  📄 ragas_scores.json        — Ragas metric scores per scenario
  📄 optimized_weights.json   — learned IS formula weights
  📄 doc/information_score_test_results.csv  — full validation table
""")


if __name__ == "__main__":
    main()
