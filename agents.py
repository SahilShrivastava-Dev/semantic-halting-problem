from langchain_core.messages import HumanMessage, AIMessage

def writer_node(state):
    """
    The Writer agent. In a real system, this calls an LLM.
    Here we simulate the writer making iterative updates to a report.
    """
    history = state.get("history", [])
    loop_count = state.get("loop_count", 0)
    
    # Mock responses for the Dubai Real Estate use case
    if loop_count == 0:
        draft = "Draft 1: This is a 5-page report on the new Dubai property."
    elif loop_count == 1:
        draft = "Draft 2: This is a 5-page report on the new Dubai property. Added a section on structural integrity with steel beams."
    elif loop_count == 2:
        draft = "Draft 3: This is a 5-page report on the new Dubai property. Added a section on structural integrity with steel beams. The foundation is a cement base."
    else:
        # Trivial changes that cause entropy to collapse (deadlock)
        words = ["cement base", "concrete foundation"]
        choice = words[loop_count % 2]
        draft = f"Draft {loop_count + 1}: This is a 5-page report on the new Dubai property. Added a section on structural integrity with steel beams. The foundation is a {choice}."

    print(f"\n[Writer] Generated Draft {loop_count + 1}")
    return {"current_draft": draft}

def critic_node(state):
    """
    The Critic agent. Simulates critique that drives the writer to update.
    """
    draft = state.get("current_draft", "")
    loop_count = state.get("loop_count", 0)
    
    if loop_count == 0:
        feedback = "The report is missing a section on structural integrity. Please add it."
    elif loop_count == 1:
        feedback = "Good addition. But what about the foundation material? Please specify."
    else:
        # Endless pedantic arguing
        words = ["concrete foundation", "cement base"]
        choice = words[loop_count % 2]
        feedback = f"Change the foundation wording to '{choice}' instead."
        
    print(f"[Critic] Feedback: {feedback}")
    
    history = state.get("history", [])
    history.append({"draft": draft, "feedback": feedback})
    
    return {
        "history": history,
        "loop_count": loop_count + 1
    }
