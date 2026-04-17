# Semantic Halting Problem (SHP) Exoskeleton

This repository provides an exoskeleton (boilerplate) for solving the **Semantic Halting Problem** in multi-agent workflows (like those built on LangGraph) using **Semantic Entropy Convergence**.

## The Problem
In multi-agent systems, agents often fall into Circular Reasoning Loops. The standard engineering hack is to hardcode a `max_steps` limit. This is mathematically unsound and forces the system to stop prematurely or burn unnecessary API tokens on trivial arguments.

## The Solution
By treating the agentic workflow as a dynamical system, we apply the Banach Fixed-Point Theorem to the embedding space:
1. **Embed** the state output of each loop into a high-dimensional vector.
2. **Calculate** the cosine distance between the current state and the previous state.
3. **Halt** when the distance approaches zero ($d(S_t, S_{t-1}) \to 0$), indicating that Semantic Entropy has collapsed and no new value is being generated.

## Running the Exoskeleton

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the LangGraph simulation:
```bash
python main.py
```

Check the Wiki for practical use cases and examples!
