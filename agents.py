"""
agents.py

Defines the Writer and Critic LLM-powered agent nodes for the LangGraph workflow.

Both agents share a single ChatGroq LLM instance (llama-3.1-8b-instant) loaded
once at module import time.  The Writer generates and iteratively improves
technical drafts; the Critic provides ONE specific, actionable critique per
round — or returns ``APPROVED`` when no substantive gaps remain.

Design rationale
----------------
Real LLMs are used intentionally: unlike scripted mock agents, a real model
organically exhausts its critique vocabulary across rounds, causing semantic
convergence to emerge naturally.  This is the core demonstration of the
Semantic Halting Problem — the system detects and halts the deadlock
*before* it is artificially forced to stop.

Retry policy
------------
Every LLM call goes through ``_llm_call_with_retry`` which applies
exponential-backoff on transient API errors (rate-limits, 5xx, etc.).
The backoff schedule is: 2^attempt seconds (i.e., 2s, 4s, 8s, …).
"""

import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from config import (
    AGENT_LLM_MODEL,
    AGENT_LLM_TEMPERATURE,
    AGENT_LLM_MAX_TOKENS,
    MAX_LLM_RETRIES,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LLM — loaded once at module import, shared by both agent nodes.
# ─────────────────────────────────────────────────────────────
logger.info("Initialising Groq %s for Writer and Critic agents...", AGENT_LLM_MODEL)
_llm = ChatGroq(
    model=AGENT_LLM_MODEL,
    temperature=AGENT_LLM_TEMPERATURE,
    max_tokens=AGENT_LLM_MAX_TOKENS,
)

# ─────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────
_WRITER_SYSTEM = """\
You are a professional technical report writer. Your sole job is to write and
iteratively improve a detailed, factual report.

Rules:
- Write 3-4 focused, information-dense paragraphs.
- Be specific, factual, and professional. Use domain terminology correctly.
- When feedback is provided, address it directly and thoroughly in the new draft.
- Output ONLY the report text. Do not include preambles like "Here is the report:".
"""

_CRITIC_SYSTEM = """\
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
# Internal helper
# ─────────────────────────────────────────────────────────────
def _llm_call_with_retry(messages: list, context: str = "") -> object:
    """
    Invoke the shared LLM with exponential-backoff retry on transient errors.

    Args:
        messages (list): Ordered list of ``SystemMessage`` / ``HumanMessage``
            objects to send to the model.
        context (str): Human-readable label (e.g. ``"Writer"`` or ``"Critic"``)
            used in log messages to aid debugging.

    Returns:
        The Langchain response object (has a ``.content`` attribute).

    Raises:
        Exception: Re-raises the last exception after ``MAX_LLM_RETRIES``
            consecutive failures.
    """
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            return _llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            if attempt == MAX_LLM_RETRIES:
                logger.error(
                    "[%s] LLM call failed after %d attempts. Last error: %s",
                    context, MAX_LLM_RETRIES, exc,
                )
                raise
            wait: int = 2 ** attempt
            logger.warning(
                "[%s] API error on attempt %d/%d (%s: %s). Retrying in %ds...",
                context, attempt, MAX_LLM_RETRIES,
                type(exc).__name__, str(exc)[:120], wait,
            )
            time.sleep(wait)


# ─────────────────────────────────────────────────────────────
# LangGraph agent nodes
# ─────────────────────────────────────────────────────────────
def writer_node(state: dict) -> dict:
    """
    Writer agent — generates or revises a technical report draft.

    On round 0 (``loop_count == 0``): produces the initial draft from the
    scenario topic and brief.

    On subsequent rounds: incorporates the Critic's last feedback, producing
    a revised draft that directly addresses the critique.

    Args:
        state (dict): LangGraph workflow state containing:
            - ``scenario`` (dict): Active scenario with ``topic`` and
              ``initial_brief`` keys.
            - ``history`` (list[dict]): All previous ``{draft, feedback}``
              pairs; last entry is the Critic's most recent feedback.
            - ``loop_count`` (int): Zero-indexed current iteration count.

    Returns:
        dict: Partial state update ``{"current_draft": <new_draft_text>}``.
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

    response = _llm_call_with_retry(
        [SystemMessage(content=_WRITER_SYSTEM), HumanMessage(content=user_content)],
        context="Writer",
    )
    draft: str = response.content.strip()

    word_count = len(draft.split())
    logger.info("[Writer] Draft %d generated (%d words)", loop_count + 1, word_count)
    logger.debug("[Writer] Preview: ...%s...", draft[:120])

    return {"current_draft": draft}


def critic_node(state: dict) -> dict:
    """
    Critic agent — reviews the current draft and returns one actionable critique,
    or ``APPROVED`` when the report is genuinely complete.

    Returning ``APPROVED`` triggers the Critic-Approval halt signal in
    ``agent_workflow.py``, independently of the semantic-entropy halt.

    Args:
        state (dict): LangGraph workflow state containing:
            - ``scenario`` (dict): Active scenario (used for ``topic``).
            - ``current_draft`` (str): Latest draft from the Writer.
            - ``loop_count`` (int): Current iteration number (pre-increment).
            - ``history`` (list[dict]): All previous ``{draft, feedback}`` pairs.

    Returns:
        dict: Partial state update with:
            - ``history``: Appended with ``{draft, feedback}`` for this round.
            - ``loop_count``: Incremented by 1.
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

    response = _llm_call_with_retry(
        [SystemMessage(content=_CRITIC_SYSTEM), HumanMessage(content=user_content)],
        context="Critic",
    )
    feedback: str = response.content.strip()

    is_approved = feedback.strip().upper().startswith("APPROVED")
    if is_approved:
        logger.info("[Critic] APPROVED by Critic LLM after %d round(s).", loop_count + 1)
    else:
        logger.info("[Critic] Feedback: %s...", feedback[:100])

    history.append({"draft": draft, "feedback": feedback})
    return {
        "history":    history,
        "loop_count": loop_count + 1,
    }
