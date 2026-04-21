"""
agents.py

Defines real LLM-powered Writer and Critic agent nodes for the LangGraph workflow.

Both agents use Qwen/Qwen2.5-7B-Instruct via the Hugging Face Inference API
(no local GPU required). The Writer generates and improves technical drafts;
the Critic provides specific, actionable editorial feedback until the report
is either approved or the semantic entropy converges.

This is the core of the Semantic Halting Problem demonstration: unlike mock
agents with scripted responses, real LLMs organically exhaust their critique
vocabulary, causing semantic convergence to emerge naturally.
"""
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

MAX_RETRIES = 3

def _llm_call_with_retry(messages: list, context: str = "") -> object:
    """
    Calls the LLM with exponential-backoff retry on transient API errors.

    Args:
        messages (list): List of SystemMessage / HumanMessage objects.
        context (str):   Label for logging (e.g., 'writer' or 'critic').

    Returns:
        The LangChain response object.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"[{context}] API error (attempt {attempt}/{MAX_RETRIES}). "
                  f"Retrying in {wait}s... ({type(e).__name__}: {str(e)[:80]}")
            time.sleep(wait)

# ─────────────────────────────────────────────────────────────
# LLM Initialisation (shared by both agents, loaded once)
# Using Groq's free tier: fast inference, no per-credit quota.
# ─────────────────────────────────────────────────────────────
print("[Agents] Initialising Groq llama-3.1-8b-instant for Writer and Critic...")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=700,
)

# ─────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────
WRITER_SYSTEM = """\
You are a professional technical report writer. Your sole job is to write and
iteratively improve a detailed, factual report.

Rules:
- Write 3-4 focused, information-dense paragraphs.
- Be specific, factual, and professional. Use domain terminology correctly.
- When feedback is provided, address it directly and thoroughly in the new draft.
- Output ONLY the report text. Do not include preambles like "Here is the report:".
"""

CRITIC_SYSTEM = """\
You are a strict, detail-oriented technical editor. Your job is to review a
report and provide ONE specific, actionable critique.

Rules:
- Give exactly ONE critique in 1-3 sentences. Be precise about what is missing
  or technically insufficient.
- Do NOT nitpick synonyms, stylistic preferences, or trivial wording choices.
- Only critique substantive content gaps (missing data, incomplete reasoning,
  incorrect terminology, missing safety standards, etc.).
- If the report is genuinely complete, technically accurate, and professionally
  written with no substantive gaps remaining, respond with exactly the word:
  APPROVED
"""


# ─────────────────────────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────────────────────────
def writer_node(state: dict) -> dict:
    """
    The Writer agent. Calls Qwen2.5-7B-Instruct to generate or improve a
    technical draft based on the scenario brief and the critic's last feedback.

    On loop 0: generates an initial draft from the scenario brief.
    On subsequent loops: rewrites the draft to address the critic's feedback.

    Args:
        state (dict): LangGraph workflow state containing:
            - scenario (dict): The active test scenario with topic and brief.
            - history (list):  Previous (draft, feedback) pairs.
            - loop_count (int): Current iteration number.

    Returns:
        dict: State update with 'current_draft' set to the new draft text.
    """
    scenario   = state.get("scenario", {})
    loop_count = state.get("loop_count", 0)
    history    = state.get("history", [])
    topic      = scenario.get("topic", "")
    brief      = scenario.get("initial_brief", topic)

    if loop_count == 0:
        user_content = (
            f"Topic: {topic}\n\n"
            f"Brief: {brief}\n\n"
            "Write the first draft of this report."
        )
    else:
        last = history[-1]
        user_content = (
            f"Topic: {topic}\n\n"
            f"Current Draft:\n{last['draft']}\n\n"
            f"Editor Feedback:\n{last['feedback']}\n\n"
            "Rewrite and improve the report, fully addressing the feedback above."
        )

    response = _llm_call_with_retry([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=user_content)
    ], context="Writer")
    draft = response.content.strip()

    word_count = len(draft.split())
    print(f"\n[Writer] Draft {loop_count + 1} generated ({word_count} words)")
    print(f"         Preview: ...{draft[:120]}...")
    return {"current_draft": draft}


def critic_node(state: dict) -> dict:
    """
    The Critic agent. Calls Qwen2.5-7B-Instruct to review the current draft
    and provide ONE specific, substantive critique — or to approve the report
    if no meaningful improvements remain.

    Returning "APPROVED" signals that the LLM itself has detected convergence,
    independent of the semantic entropy measurement. Both signals are used to
    halt the loop in agent_workflow.py.

    Args:
        state (dict): LangGraph workflow state containing:
            - scenario (dict):     The active test scenario.
            - current_draft (str): The latest draft from the Writer.
            - loop_count (int):    Current iteration number.
            - history (list):      Previous (draft, feedback) pairs.

    Returns:
        dict: State update with 'history' appended and 'loop_count' incremented.
    """
    scenario   = state.get("scenario", {})
    draft      = state.get("current_draft", "")
    loop_count = state.get("loop_count", 0)
    history    = list(state.get("history", []))
    topic      = scenario.get("topic", "")

    user_content = (
        f"Topic: {topic}\n\n"
        f"Report Draft:\n{draft}\n\n"
        "Provide your editorial review."
    )

    response = _llm_call_with_retry([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=user_content)
    ], context="Critic")
    feedback = response.content.strip()

    is_approved = feedback.strip().upper().startswith("APPROVED")
    status = "✅ APPROVED by Critic LLM" if is_approved else f"Feedback: {feedback[:100]}..."
    print(f"[Critic] {status}")

    history.append({"draft": draft, "feedback": feedback})
    return {
        "history":    history,
        "loop_count": loop_count + 1,
    }
