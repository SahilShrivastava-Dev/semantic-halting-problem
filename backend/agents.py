"""
agents.py

Writer and Critic LLM agent nodes for the SHP LangGraph workflow.

Provider-agnostic: both agents accept a pre-built LLM instance so the
caller (agent_workflow.py) can pass any Groq or OpenAI model without
changing the agent logic.

Design rationale
----------------
Real LLMs are used intentionally. Unlike mock agents, a real model
organically exhausts critique vocabulary across rounds, causing semantic
convergence to emerge naturally — demonstrating the Semantic Halting
Problem without artificial forcing.

Retry policy
------------
Every LLM call goes through _llm_call_with_retry with exponential backoff.
Schedule: 2^attempt seconds (2s, 4s, 8s, ...).
"""

import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from config import MAX_LLM_RETRIES

logger = logging.getLogger(__name__)

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
def _llm_call_with_retry(llm: BaseChatModel, messages: list, context: str = "") -> object:
    """
    Invoke an LLM with exponential-backoff retry on transient errors.

    Args:
        llm:      Configured Langchain chat model (Groq or OpenAI).
        messages: Ordered list of SystemMessage / HumanMessage objects.
        context:  Human-readable label for log messages (e.g. "Writer").

    Returns:
        Langchain response object with a .content attribute.

    Raises:
        Exception: Re-raises the last exception after MAX_LLM_RETRIES failures.
    """
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if attempt == MAX_LLM_RETRIES:
                logger.error(
                    "[%s] LLM call failed after %d attempts. Last error: %s",
                    context, MAX_LLM_RETRIES, exc,
                )
                raise
            wait = 2 ** attempt
            logger.warning(
                "[%s] API error on attempt %d/%d (%s: %s). Retrying in %ds...",
                context, attempt, MAX_LLM_RETRIES,
                type(exc).__name__, str(exc)[:120], wait,
            )
            time.sleep(wait)


# ─────────────────────────────────────────────────────────────
# Agent node factories
# ─────────────────────────────────────────────────────────────
def make_writer_node(llm: BaseChatModel, emit=None):
    """
    Return a LangGraph writer node bound to the given LLM.

    Args:
        llm:  Pre-built chat model from providers.get_llm().
        emit: Optional event callback (dict) -> None for streaming.

    Returns:
        Callable (state: dict) -> dict  compatible with LangGraph add_node.
    """
    _emit = emit or (lambda e: None)

    def writer_node(state: dict) -> dict:
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
            llm,
            [SystemMessage(content=_WRITER_SYSTEM), HumanMessage(content=user_content)],
            context="Writer",
        )
        draft: str = response.content.strip()
        word_count = len(draft.split())

        logger.info("[Writer] Draft %d generated (%d words)", loop_count + 1, word_count)

        _emit({
            "type":       "draft_generated",
            "round":      loop_count + 1,
            "word_count": word_count,
            "preview":    draft[:300],
            "full_draft": draft,
        })

        return {"current_draft": draft}

    return writer_node


def make_critic_node(llm: BaseChatModel, emit=None):
    """
    Return a LangGraph critic node bound to the given LLM.

    Returning APPROVED triggers the Critic-Approval halt signal in
    check_convergence, independently of the semantic-entropy halt.

    Args:
        llm:  Pre-built chat model from providers.get_llm().
        emit: Optional event callback (dict) -> None for streaming.

    Returns:
        Callable (state: dict) -> dict  compatible with LangGraph add_node.
    """
    _emit = emit or (lambda e: None)

    def critic_node(state: dict) -> dict:
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
            llm,
            [SystemMessage(content=_CRITIC_SYSTEM), HumanMessage(content=user_content)],
            context="Critic",
        )
        feedback: str = response.content.strip()
        is_approved = feedback.strip().upper().startswith("APPROVED")

        if is_approved:
            logger.info(
                "[Critic] APPROVED after %d round(s).", loop_count + 1
            )
        else:
            logger.info("[Critic] Feedback: %s...", feedback[:100])

        _emit({
            "type":        "critic_feedback",
            "round":       loop_count + 1,
            "feedback":    feedback,
            "is_approved": is_approved,
        })

        history.append({"draft": draft, "feedback": feedback})
        return {
            "history":    history,
            "loop_count": loop_count + 1,
        }

    return critic_node
