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

from shp.config import MAX_LLM_RETRIES

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────
_WRITER_SYSTEM = """\
You are a careful research assistant answering a question using ONLY the provided
context passages. Your job is to write and iteratively improve a grounded answer.

Rules:
- Answer the question directly and completely. For multi-hop questions, make
  EVERY reasoning step explicit and name the entities/facts that connect them.
- Ground every claim in the provided context. Do NOT add facts that are not
  supported by the context. If the context is insufficient, say so explicitly.
- When editor feedback is provided, address it directly and thoroughly.
- Be concise but complete: a focused answer with its supporting reasoning, not a
  padded essay. Output ONLY the answer text, with no preamble.
"""

_CRITIC_SYSTEM = """\
You are a strict, detail-oriented editor reviewing a grounded answer against the
source context. Provide ONE specific, actionable critique.

Rules:
- Give exactly ONE critique in 1-3 sentences. Be precise about what is missing,
  unsupported, or factually incomplete relative to the context and question.
- Prioritise substantive gaps: a missing reasoning hop, a claim not supported by
  the context, an unanswered part of the question, or a factual error.
- Do NOT nitpick synonyms, style, or wording.
- If the answer fully and correctly addresses the question, is entirely grounded
  in the context, and has no substantive gaps, respond with exactly: APPROVED
"""


def _format_contexts(contexts) -> str:
    """Render retrieved context passages as a numbered block for the prompt."""
    if not contexts:
        return ""
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))


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
        question   = scenario.get("question", "")
        contexts   = scenario.get("contexts", [])
        brief      = scenario.get("initial_brief", question or topic)
        ctx_block  = _format_contexts(contexts)

        # RAG mode when a question + contexts are present; else fall back to the
        # legacy topic/brief report mode so existing topic scenarios still run.
        if question and ctx_block:
            task_header = f"Question: {question}\n\nContext passages:\n{ctx_block}\n\n"
        else:
            task_header = f"Topic: {topic}\n\nBrief: {brief}\n\n"

        if loop_count == 0:
            user_content = task_header + "Write your first grounded answer."
        else:
            last = history[-1]
            user_content = (
                task_header
                + f"Current Draft:\n{last['draft']}\n\n"
                + f"Editor Feedback:\n{last['feedback']}\n\n"
                + "Rewrite and improve the answer, fully addressing the feedback above."
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
        question   = scenario.get("question", "")
        contexts   = scenario.get("contexts", [])
        ctx_block  = _format_contexts(contexts)

        if question and ctx_block:
            user_content = (
                f"Question: {question}\n\n"
                f"Context passages:\n{ctx_block}\n\n"
                f"Answer Draft:\n{draft}\n\n"
                "Provide your editorial review."
            )
        else:
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
