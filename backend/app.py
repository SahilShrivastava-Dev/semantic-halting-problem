"""
app.py

FastAPI server for the Semantic Halting Problem (SHP) dashboard.

Endpoints
---------
    GET  /api/models            — Available providers and model lists.
    GET  /api/scenarios         — All scenarios from test_scenarios.json.
    GET  /api/weights           — Currently loaded IS weights + R².
    POST /api/run               — Run a single scenario synchronously (returns JSON).
    WS   /ws                    — WebSocket: stream real-time events during a run.

WebSocket protocol
------------------
    Client → Server:  JSON run config
        {
          "provider":    "groq" | "openai",
          "agent_model": "llama-3.1-8b-instant",
          "eval_model":  "llama-3.1-8b-instant",
          "scenario_id": "dubai_real_estate"   // or null for custom
          "topic":       "...",                // used when scenario_id is null
          "question":    "...",
          "contexts":    ["...", "..."],
          "ground_truth": "..."
        }

    Server → Client:  JSON events (see agent_workflow.py for full schemas)
        { "type": "scenario_start", ... }
        { "type": "draft_generated", ... }
        { "type": "is_score", ... }
        { "type": "convergence_metrics", ... }
        { "type": "critic_feedback", ... }
        { "type": "halt_signal", ... }
        { "type": "scenario_complete", ... }
        { "type": "error", "message": "..." }

Running
-------
    cd backend
    uvicorn app:app --reload --port 8000
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    DEFAULT_PROVIDER,
    DEFAULT_AGENT_MODELS,
    DEFAULT_EVAL_MODELS,
    RAGAS_EVAL_TIMEOUT,
    RAGAS_MAX_WORKERS,
    SCENARIOS_FILE,
    WEIGHTS_FILE,
)
from providers import list_providers
from agent_workflow import build_graph, run_scenario
from optimize_score import optimize_information_score_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Semantic Halting Problem API",
    description="Real-time multi-agent convergence analysis with ML-style visualisation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)


# ─────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/api/models")
async def get_models():
    """Return available providers and their model lists."""
    return {
        "providers": list_providers(),
        "defaults": {
            "provider":    DEFAULT_PROVIDER,
            "agent_model": DEFAULT_AGENT_MODELS,
            "eval_model":  DEFAULT_EVAL_MODELS,
        },
    }


@app.get("/api/scenarios")
async def get_scenarios():
    """Return all scenarios from test_scenarios.json."""
    if not os.path.exists(SCENARIOS_FILE):
        return JSONResponse(status_code=404, content={"error": "Scenarios file not found."})
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as fh:
        scenarios = json.load(fh)
    return {"scenarios": scenarios}


@app.get("/api/topics")
async def get_topics():
    """Return distinct topics from test_scenarios.json grouped by topic_id."""
    if not os.path.exists(SCENARIOS_FILE):
        return JSONResponse(status_code=404, content={"error": "Scenarios file not found."})
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as fh:
        scenarios = json.load(fh)
    seen: dict[str, dict] = {}
    for s in scenarios:
        tid = s.get("topic_id")
        if tid and tid not in seen:
            seen[tid] = {
                "topic_id":    tid,
                "topic":       s.get("topic", tid),
                "train_count": 0,
                "val_count":   0,
                "test_count":  0,
            }
        if tid:
            seen[tid][f"{s['split']}_count"] += 1
    return {"topics": list(seen.values())}


@app.get("/api/weights")
async def get_weights():
    """Return currently optimised IS weights (or defaults if not yet calibrated)."""
    if not os.path.exists(WEIGHTS_FILE):
        return {
            "weights": {
                "faithfulness":      0.25,
                "answer_relevancy":  0.25,
                "context_precision": 0.25,
                "context_recall":    0.25,
            },
            "r2_score":    None,
            "data_source": "default",
        }
    with open(WEIGHTS_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────
# WebSocket — real-time streaming run
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Stream agent events in real time over WebSocket.

    The client sends a run config JSON, then receives a stream of typed
    events until scenario_complete or error is received.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    try:
        raw = await websocket.receive_text()
        config: dict[str, Any] = json.loads(raw)
    except Exception as exc:
        await _send(websocket, {"type": "error", "message": f"Invalid config JSON: {exc}"})
        await websocket.close()
        return

    provider    = config.get("provider", DEFAULT_PROVIDER)
    agent_model = config.get("agent_model", DEFAULT_AGENT_MODELS.get(provider))
    eval_model  = config.get("eval_model",  DEFAULT_EVAL_MODELS.get(provider))
    scenario_id = config.get("scenario_id")

    # Resolve scenario
    scenario = _resolve_scenario(config, scenario_id)
    if scenario is None:
        await _send(websocket, {
            "type":    "error",
            "message": f"Scenario '{scenario_id}' not found in {SCENARIOS_FILE}.",
        })
        await websocket.close()
        return

    await _send(websocket, {
        "type":        "run_config",
        "provider":    provider,
        "agent_model": agent_model,
        "eval_model":  eval_model,
        "scenario_id": scenario["id"],
        "topic":       scenario["topic"],
    })

    # Run blocking workflow in thread; stream events via asyncio Queue
    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: dict):
        asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    def _run():
        try:
            graph = build_graph(
                provider=provider,
                agent_model=agent_model,
                eval_model=eval_model,
                emit=emit,
            )
            result = run_scenario(graph, scenario, emit=emit)
            emit({"type": "_done", "result": result})
        except Exception as exc:
            logger.exception("Workflow error")
            emit({"type": "error", "message": str(exc)})
            emit({"type": "_done", "result": None})

    future = loop.run_in_executor(_executor, _run)

    # Per-event silence timeout: how long to wait for the NEXT event before giving up.
    # With RAGAS_MAX_WORKERS=1 on Groq, a single metric can block for up to
    # RAGAS_EVAL_TIMEOUT seconds. We add a buffer for retries/rate-limit backoff.
    _event_timeout = RAGAS_EVAL_TIMEOUT * RAGAS_MAX_WORKERS + 120

    # Forward queue events to WebSocket until "_done"
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=float(_event_timeout))
        except asyncio.TimeoutError:
            await _send(websocket, {
                "type":    "error",
                "message": f"Workflow timed out — no event in {_event_timeout}s. "
                           "Try a faster provider (OpenAI) or reduce max_rounds.",
            })
            break

        if event.get("type") == "_done":
            break

        await _send(websocket, event)

    await future  # ensure thread cleanup
    logger.info("WebSocket run complete.")

    try:
        await websocket.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
async def _send(ws: WebSocket, data: dict) -> None:
    try:
        await ws.send_text(json.dumps(data))
    except Exception:
        pass


def _resolve_scenario(config: dict, scenario_id: str | None) -> dict | None:
    """
    Return a scenario dict either from the JSON file or built from custom fields.
    """
    if scenario_id and scenario_id != "custom":
        if os.path.exists(SCENARIOS_FILE):
            with open(SCENARIOS_FILE, "r", encoding="utf-8") as fh:
                all_scenarios = json.load(fh)
            for s in all_scenarios:
                if s["id"] == scenario_id:
                    return s
        return None

    # Custom scenario from config fields
    topic      = config.get("topic", "Custom Topic")
    question   = config.get("question", "")
    contexts   = config.get("contexts", [])
    ground_truth = config.get("ground_truth", "")

    if not question or not contexts:
        return None

    return {
        "id":          "custom",
        "split":       "custom",
        "topic":       topic,
        "question":    question,
        "contexts":    contexts,
        "ground_truth": ground_truth,
    }


def _load_topic_questions(topic_id: str) -> dict[str, list[dict]]:
    """Load all scenarios for a topic_id, partitioned by split."""
    if not os.path.exists(SCENARIOS_FILE):
        return {"train": [], "val": [], "test": []}
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as fh:
        all_scenarios = json.load(fh)
    result: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in all_scenarios:
        if s.get("topic_id") == topic_id and s.get("split") in result:
            result[s["split"]].append(s)
    return result


# ─────────────────────────────────────────────────────────────
# Pipeline WebSocket — full train→calibrate→val→test run
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws/pipeline")
async def pipeline_websocket_endpoint(websocket: WebSocket):
    """
    Stream a complete 4-phase pipeline for a given topic.

    Client sends:
        { "topic_id": "arjun_life", "provider": "groq",
          "agent_model": "...", "eval_model": "..." }

    Server emits (in order):
        pipeline_start
        pipeline_q_start  / per-round events / pipeline_q_complete  (repeated per question)
        weights_learned
        pipeline_complete
        error (on failure)
    """
    await websocket.accept()
    logger.info("Pipeline WebSocket accepted.")

    try:
        raw = await websocket.receive_text()
        cfg: dict[str, Any] = json.loads(raw)
    except Exception as exc:
        await _send(websocket, {"type": "error", "message": f"Invalid config: {exc}"})
        await websocket.close()
        return

    topic_id    = cfg.get("topic_id", "")
    provider    = cfg.get("provider", DEFAULT_PROVIDER)
    agent_model = cfg.get("agent_model", DEFAULT_AGENT_MODELS.get(provider))
    eval_model  = cfg.get("eval_model",  DEFAULT_EVAL_MODELS.get(provider))

    splits = _load_topic_questions(topic_id)
    if not any(splits.values()):
        await _send(websocket, {"type": "error", "message": f"No scenarios found for topic_id='{topic_id}'."})
        await websocket.close()
        return

    train_qs = splits["train"]
    val_qs   = splits["val"]
    test_qs  = splits["test"]

    await _send(websocket, {
        "type":        "pipeline_start",
        "topic_id":    topic_id,
        "provider":    provider,
        "agent_model": agent_model,
        "eval_model":  eval_model,
        "train_count": len(train_qs),
        "val_count":   len(val_qs),
        "test_count":  len(test_qs),
    })

    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(event: dict):
        asyncio.run_coroutine_threadsafe(queue.put(event), loop)

    _event_timeout = RAGAS_EVAL_TIMEOUT * RAGAS_MAX_WORKERS + 120

    async def _drain_until_done() -> tuple[bool, dict | None]:
        """Forward queue events to client until '_done' sentinel; return (ok, result)."""
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=float(_event_timeout))
            except asyncio.TimeoutError:
                await _send(websocket, {
                    "type":    "error",
                    "message": f"Pipeline timed out — no event in {_event_timeout}s.",
                })
                return False, None
            if event.get("type") == "_done":
                return True, event.get("result")
            await _send(websocket, event)

    # Build the graph once and reuse it across all questions in this topic.
    graph_future = loop.run_in_executor(
        _executor,
        lambda: build_graph(provider=provider, agent_model=agent_model, eval_model=eval_model, emit=emit),
    )
    graph = await graph_future

    train_results: list[dict] = []
    val_results:   list[dict] = []
    test_results:  list[dict] = []

    async def _run_phase(questions: list[dict], phase: str, results_bucket: list[dict]) -> bool:
        for q_idx, scenario in enumerate(questions):
            await _send(websocket, {
                "type":        "pipeline_q_start",
                "phase":       phase,
                "q_idx":       q_idx,
                "total":       len(questions),
                "question":    scenario["question"],
                "scenario_id": scenario["id"],
            })

            def _worker(s=scenario):
                try:
                    result = run_scenario(graph, s, emit=emit)
                    emit({"type": "_done", "result": result})
                except Exception as exc:
                    logger.exception("Pipeline question error")
                    emit({"type": "error", "message": str(exc)})
                    emit({"type": "_done", "result": None})

            loop.run_in_executor(_executor, _worker)
            ok, result = await _drain_until_done()
            if not ok:
                return False

            # Extract final IS from scenario result returned by run_scenario
            final_is = 0.0
            if result and isinstance(result, dict):
                history = result.get("is_score_history", [])
                if history:
                    final_is = history[-1]

            q_complete: dict = {
                "type":        "pipeline_q_complete",
                "phase":       phase,
                "q_idx":       q_idx,
                "total":       len(questions),
                "question":    scenario["question"],
                "scenario_id": scenario["id"],
                "final_is":    round(float(final_is), 4),
                "halt_reason": result.get("halt_reason", "unknown") if result else "error",
            }
            results_bucket.append(q_complete)
            await _send(websocket, q_complete)

        return True

    # ── Phase 1: Train ────────────────────────────────────────────────────────
    ok = await _run_phase(train_qs, "train", train_results)
    if not ok:
        await websocket.close()
        return

    # ── Calibrate: learn IS weights from training data ────────────────────────
    await _send(websocket, {"type": "calibrating", "message": "Learning IS weights from training data…"})

    def _calibrate():
        try:
            return optimize_information_score_weights()
        except Exception as exc:
            return {"error": str(exc)}

    weights_result = await loop.run_in_executor(_executor, _calibrate)
    if "error" in weights_result:
        await _send(websocket, {"type": "error", "message": f"Weight calibration failed: {weights_result['error']}"})
    else:
        # Reload full output from file (includes r2_score)
        weights_output = weights_result
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as fh:
                weights_output = json.load(fh)
        await _send(websocket, {
            "type":        "weights_learned",
            "weights":     weights_output.get("weights", weights_result),
            "r2_score":    weights_output.get("r2_score"),
            "data_source": weights_output.get("data_source", "real"),
        })

    # ── Phase 2: Validate ─────────────────────────────────────────────────────
    ok = await _run_phase(val_qs, "val", val_results)
    if not ok:
        await websocket.close()
        return

    # ── Phase 3: Test ─────────────────────────────────────────────────────────
    ok = await _run_phase(test_qs, "test", test_results)
    if not ok:
        await websocket.close()
        return

    await _send(websocket, {
        "type":          "pipeline_complete",
        "topic_id":      topic_id,
        "train_results": train_results,
        "val_results":   val_results,
        "test_results":  test_results,
    })

    logger.info("Pipeline WebSocket complete.")
    try:
        await websocket.close()
    except Exception:
        pass
