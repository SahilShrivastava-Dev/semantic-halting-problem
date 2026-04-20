# Beginner's Guide to the Semantic Halting Problem (SHP) Codebase

Welcome to the SHP project! If you're new to this codebase, don't worry—it's broken down into four easy-to-understand parts.

## What is this project?
When multiple AI agents (like a "Writer" and a "Critic") talk to each other, they can sometimes get stuck in an endless loop, arguing over minor details forever. This project is a proof-of-concept for how to mathematically detect when they are stuck and force them to stop.

## How the Code is Organized

### 1. The Core Simulation
- **`agent_workflow.py`**: This is the heart of the simulation. It sets up a graph (like a flowchart) where a Writer writes a draft, and a Critic reviews it.
- **`agents.py`**: Contains the actual instructions for the Writer and Critic. Right now, they are "mock" agents (they just spit out pre-written text) so we can test the loop safely without paying for API keys.
- **`semantic_entropy.py`**: The "Math Engine". It takes the drafts, converts them into numbers (embeddings), and calculates how different they are. If the difference is too small, it tells `agent_workflow.py` to stop the loop!

### 2. The Grader
- **`ragas_eval.py`**: Once the Writer and Critic finish their report, we need to know if the report is actually good. This file uses free AI models from Hugging Face to grade the report on things like "Faithfulness" (did it hallucinate?) and "Relevancy" (did it answer the question?).

### 3. The Optimizer
- **`optimize_score.py`**: We have all these grades (Faithfulness, Relevancy, etc.), but how do we combine them into one final "Information Score"? This file uses basic Machine Learning (Linear Regression) to calculate the perfect weights for our final formula.

### 4. The Orchestrator
- **`main.py`**: Think of this as the master switch. When you run `main.py`, it automatically runs the simulation, evaluates the output, and calculates the optimal score in chronological order.
