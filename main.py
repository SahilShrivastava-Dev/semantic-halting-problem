"""
main.py

This is the central pipeline orchestrator. It triggers the entire Semantic Halting Problem (SHP)
workflow sequentially:
1. Agent Simulation (agent_workflow.py)
2. Ragas Evaluation (ragas_eval.py)
3. Score Optimization (optimize_score.py)
"""
import subprocess
import sys

def run_script(script_name: str, step_description: str):
    print(f"\n{'='*50}")
    print(f"🚀 STEP: {step_description}")
    print(f"Executing: {script_name}")
    print(f"{'='*50}\n")
    
    # Use the python executable from the current virtual environment
    python_exec = sys.executable
    
    try:
        subprocess.run([python_exec, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed during {script_name}. Exit code: {e.returncode}")
        sys.exit(1)

def main():
    print("🌟 INITIALIZING SHP PIPELINE 🌟")
    
    # Step 1: Run the multi-agent workflow
    run_script("agent_workflow.py", "Semantic Entropy Multi-Agent Exoskeleton")
    
    # Step 2: Evaluate the output
    run_script("ragas_eval.py", "Ragas LLM Evaluation (HuggingFace)")
    
    # Step 3: Optimize the Information Score weights
    run_script("optimize_score.py", "Machine Learning Weight Optimization")
    
    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
