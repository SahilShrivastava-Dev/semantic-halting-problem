import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Play, Square, RefreshCw, Activity, FlaskConical, MessageSquare } from 'lucide-react'

import ProviderSelector from './components/ProviderSelector'
import ScenarioPanel from './components/ScenarioPanel'
import ConvergenceChart from './components/ConvergenceChart'
import DistanceChart from './components/DistanceChart'
import ISGainChart from './components/ISGainChart'
import WeightsChart from './components/WeightsChart'
import MetricsBreakdown from './components/MetricsBreakdown'
import LogStream from './components/LogStream'
import HaltBadge from './components/HaltBadge'
import ConversationView from './components/ConversationView'
import PipelineView from './components/PipelineView'
import { useWebSocket } from './hooks/useWebSocket'
import { usePipelineWebSocket } from './hooks/usePipelineWebSocket'

import type {
  Scenario, ModelsResponse, WeightsResponse, RunState, RoundDataPoint,
  RoundConversation, LogEntry, SHPEvent, RunConfig,
  PipelineState, PipelineEvent, TopicInfo, PipelineConfig, PipelineQuestionResult,
} from './types'

const API = 'http://localhost:8000'

// ── App mode ────────────────────────────────────────────────────────────────
type AppMode = 'single' | 'pipeline'

// ── Single-run state defaults ────────────────────────────────────────────────
const DEFAULT_RUN_STATE: RunState = {
  status:              'idle',
  provider:            'groq',
  agentModel:          'llama-3.1-8b-instant',
  evalModel:           'llama-3.1-8b-instant',
  scenarioId:          '',
  topic:               '',
  question:            '',
  rounds:              0,
  haltReason:          '',
  finalISScore:        0,
  roundData:           [],
  conversations:       [],
  isWeights:           { faithfulness: 0.25, answer_relevancy: 0.25, context_precision: 0.25, context_recall: 0.25 },
  logs:                [],
  currentDraftPreview: '',
  lastCriticFeedback:  '',
  error:               null,
}

// ── Pipeline state defaults ───────────────────────────────────────────────────
const DEFAULT_PIPELINE_STATE: PipelineState = {
  status:            'idle',
  phase:             '',
  topicId:           '',
  trainResults:      [],
  valResults:        [],
  testResults:       [],
  weights:           null,
  r2:                null,
  activeConversation: [],
  activeQuestion:    '',
  logs:              [],
  error:             null,
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function emptyConv(round: number): RoundConversation {
  return {
    round,
    draft:          '',
    wordCount:      0,
    isScore:        null,
    isGain:         null,
    evaluating:     false,
    metrics:        { faithfulness: null, answer_relevancy: null, context_precision: null, context_recall: null },
    criticFeedback: null,
    isApproved:     false,
  }
}

function genId() {
  return Math.random().toString(36).slice(2)
}

// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  // ── Remote data ─────────────────────────────────────────────────────────────
  const [providers, setProviders]           = useState<Record<string, string[]>>({})
  const [scenarios, setScenarios]           = useState<Scenario[]>([])
  const [topics, setTopics]                 = useState<TopicInfo[]>([])
  const [calibratedWeights, setCalibratedWeights] = useState<WeightsResponse | null>(null)
  const [apiError, setApiError]             = useState<string | null>(null)

  // ── Mode toggle ──────────────────────────────────────────────────────────────
  const [appMode, setAppMode] = useState<AppMode>('single')

  // ── Shared provider config ────────────────────────────────────────────────────
  const [provider, setProvider]       = useState('groq')
  const [agentModel, setAgentModel]   = useState('llama-3.1-8b-instant')
  const [evalModel, setEvalModel]     = useState('llama-3.1-8b-instant')

  // ── Single-run config ─────────────────────────────────────────────────────────
  const [selectedId, setSelectedId]   = useState<string>('')
  const [customTopic, setCustomTopic] = useState('')
  const [customQ, setCustomQ]         = useState('')
  const [customCtx, setCustomCtx]     = useState('')
  const [customGT, setCustomGT]       = useState('')

  // ── Single-run state ──────────────────────────────────────────────────────────
  const [run, setRun]       = useState<RunState>(DEFAULT_RUN_STATE)
  const roundMapRef         = useRef<Map<number, RoundDataPoint>>(new Map())
  const convMapRef          = useRef<Map<number, RoundConversation>>(new Map())

  const { status: wsStatus, connect, disconnect, error: wsError } = useWebSocket()

  // ── Pipeline config ────────────────────────────────────────────────────────────
  const [selectedTopic, setSelectedTopic] = useState<string>('')

  // ── Pipeline state ─────────────────────────────────────────────────────────────
  const [pipeline, setPipeline]   = useState<PipelineState>(DEFAULT_PIPELINE_STATE)
  const pipelineConvMapRef        = useRef<Map<number, RoundConversation>>(new Map())

  const {
    status: pipelineWsStatus,
    connect: pipelineConnect,
    disconnect: pipelineDisconnect,
    error: pipelineWsError,
  } = usePipelineWebSocket()

  // ── Fetch initial data ─────────────────────────────────────────────────────────
  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/models`).then(r => r.json()) as Promise<ModelsResponse>,
      fetch(`${API}/api/scenarios`).then(r => r.json()),
      fetch(`${API}/api/weights`).then(r => r.json()) as Promise<WeightsResponse>,
      fetch(`${API}/api/topics`).then(r => r.json()),
    ])
      .then(([modelsRes, scenariosRes, weightsRes, topicsRes]) => {
        setProviders(modelsRes.providers)
        setScenarios(scenariosRes.scenarios ?? [])
        setCalibratedWeights(weightsRes)
        setTopics(topicsRes.topics ?? [])
        setSelectedId(scenariosRes.scenarios?.[0]?.id ?? '')
        setSelectedTopic(topicsRes.topics?.[0]?.topic_id ?? '')
      })
      .catch(() => {
        setApiError('Cannot reach backend at port 8000. Start the server first.')
      })
  }, [])

  // Sync agent/eval model when provider changes
  useEffect(() => {
    const models = providers[provider]
    if (models?.length) {
      setAgentModel(models[0])
      setEvalModel(models[0])
    }
  }, [provider, providers])

  // ── Single-run event handler ───────────────────────────────────────────────────
  const handleEvent = useCallback((event: SHPEvent) => {
    setRun((prev) => {
      let next = { ...prev }
      const now = new Date().toISOString()

      const addLog = (msg: string, level: LogEntry['level'] = 'info', round?: number): void => {
        const entry: LogEntry = { id: genId(), timestamp: now, level, message: msg, round }
        next = { ...next, logs: [...next.logs, entry] }
      }

      switch (event.type) {
        case 'run_config':
          addLog(`Provider: ${event.provider} | Agent: ${event.agent_model} | Judge: ${event.eval_model}`)
          addLog(`Scenario: ${event.topic}`)
          break

        case 'scenario_start':
          next.question = event.question
          addLog(`▶ Starting: ${event.scenario_id}`, 'info')
          addLog(`Q: ${event.question}`, 'info')
          break

        case 'draft_generated': {
          const e = event
          next.currentDraftPreview = e.preview
          next.rounds = e.round

          const rp = roundMapRef.current.get(e.round) ?? {
            round: e.round, is_score: 0, distance: null, is_gain: null,
            faithfulness: null, answer_relevancy: null, context_precision: null, context_recall: null,
          }
          roundMapRef.current.set(e.round, rp)

          const cv = convMapRef.current.get(e.round) ?? emptyConv(e.round)
          convMapRef.current.set(e.round, { ...cv, draft: e.full_draft, wordCount: e.word_count })
          next.conversations = Array.from(convMapRef.current.values()).sort((a, b) => a.round - b.round)

          addLog(`[Writer] Round ${e.round} — ${e.word_count} words`, 'info', e.round)
          break
        }

        case 'evaluating': {
          const cv = convMapRef.current.get(event.round) ?? emptyConv(event.round)
          convMapRef.current.set(event.round, { ...cv, evaluating: true })
          next.conversations = Array.from(convMapRef.current.values()).sort((a, b) => a.round - b.round)
          addLog(`[Ragas] Scoring round ${event.round}…`, 'info', event.round)
          break
        }

        case 'is_score': {
          const e = event

          const rp = roundMapRef.current.get(e.round) ?? {
            round: e.round, is_score: 0, distance: null, is_gain: null,
            faithfulness: null, answer_relevancy: null, context_precision: null, context_recall: null,
          }
          const prevRp = roundMapRef.current.get(e.round - 1)
          const updatedRp: RoundDataPoint = {
            ...rp,
            is_score:          e.score,
            faithfulness:      e.metrics.faithfulness      ?? rp.faithfulness,
            answer_relevancy:  e.metrics.answer_relevancy  ?? rp.answer_relevancy,
            context_precision: e.metrics.context_precision ?? rp.context_precision,
            context_recall:    e.metrics.context_recall    ?? rp.context_recall,
            is_gain:           prevRp ? e.score - prevRp.is_score : null,
          }
          roundMapRef.current.set(e.round, updatedRp)
          next.roundData = Array.from(roundMapRef.current.values()).sort((a, b) => a.round - b.round)

          const prevCv = convMapRef.current.get(e.round - 1)
          const cv = convMapRef.current.get(e.round) ?? emptyConv(e.round)
          convMapRef.current.set(e.round, {
            ...cv,
            isScore:    e.score,
            isGain:     prevCv?.isScore != null ? e.score - prevCv.isScore : null,
            evaluating: false,
            metrics: {
              faithfulness:      e.metrics.faithfulness      ?? null,
              answer_relevancy:  e.metrics.answer_relevancy  ?? null,
              context_precision: e.metrics.context_precision ?? null,
              context_recall:    e.metrics.context_recall    ?? null,
            },
          })
          next.conversations = Array.from(convMapRef.current.values()).sort((a, b) => a.round - b.round)
          if (e.weights) next.isWeights = e.weights

          addLog(
            `[IS] Round ${e.round}: ${e.score.toFixed(3)} | F:${(e.metrics.faithfulness ?? 0).toFixed(2)} Rel:${(e.metrics.answer_relevancy ?? 0).toFixed(2)} Prec:${(e.metrics.context_precision ?? 0).toFixed(2)} Rec:${(e.metrics.context_recall ?? 0).toFixed(2)}`,
            e.error ? 'warn' : 'info', e.round,
          )
          break
        }

        case 'convergence_metrics': {
          const e = event
          const rp = roundMapRef.current.get(e.round) ?? {
            round: e.round, is_score: 0, distance: null, is_gain: null,
            faithfulness: null, answer_relevancy: null, context_precision: null, context_recall: null,
          }
          roundMapRef.current.set(e.round, { ...rp, distance: e.distance, is_gain: e.is_gain })
          next.roundData = Array.from(roundMapRef.current.values()).sort((a, b) => a.round - b.round)

          const gainStr = e.is_gain >= 0 ? `+${e.is_gain.toFixed(3)}` : e.is_gain.toFixed(3)
          addLog(
            `[Distance] ${e.distance.toFixed(4)} (ε=${e.threshold}) | IS Δ: ${gainStr}`,
            e.distance < e.threshold ? 'success' : 'info', e.round,
          )
          break
        }

        case 'critic_feedback': {
          const e = event
          next.lastCriticFeedback = e.feedback
          const cv = convMapRef.current.get(e.round) ?? emptyConv(e.round)
          convMapRef.current.set(e.round, { ...cv, criticFeedback: e.feedback, isApproved: e.is_approved })
          next.conversations = Array.from(convMapRef.current.values()).sort((a, b) => a.round - b.round)
          addLog(
            e.is_approved ? '[Critic] APPROVED ✓' : `[Critic] ${e.feedback.slice(0, 100)}…`,
            e.is_approved ? 'success' : 'info', e.round,
          )
          break
        }

        case 'halt_signal': {
          const e = event
          next.haltReason   = e.reason
          next.finalISScore = e.final_is_score
          addLog(
            `🛑 HALT — ${e.reason.replace(/_/g, ' ')} | round ${e.round} | IS: ${e.final_is_score.toFixed(4)}`,
            'success', e.round,
          )
          break
        }

        case 'scenario_complete': {
          const e = event
          next.status       = 'complete'
          next.rounds       = e.rounds
          next.haltReason   = e.halt_reason
          next.finalISScore = e.final_is_score
          addLog(`✅ Done — ${e.rounds} rounds | ${e.halt_reason} | IS=${e.final_is_score.toFixed(4)}`, 'success')
          break
        }

        case 'error': {
          next.status = 'error'
          next.error  = event.message
          addLog(`✖ Error: ${event.message}`, 'error')
          break
        }
      }

      return next
    })
  }, [])

  // ── Pipeline event handler ─────────────────────────────────────────────────────
  const handlePipelineEvent = useCallback((event: PipelineEvent) => {
    const now = new Date().toISOString()
    const addLog = (msg: string, level: LogEntry['level'] = 'info'): LogEntry => ({
      id: genId(), timestamp: now, level, message: msg,
    })

    setPipeline((prev) => {
      const next = { ...prev }

      switch (event.type) {
        case 'pipeline_start':
          next.status = 'running'
          next.phase  = 'train'
          next.topicId = event.topic_id
          next.logs = [...next.logs, addLog(`Pipeline started: ${event.topic_id} | train=${event.train_count} val=${event.val_count} test=${event.test_count}`)]
          break

        case 'pipeline_q_start':
          next.phase           = event.phase
          next.activeQuestion  = event.question
          next.activeConversation = []
          pipelineConvMapRef.current.clear()
          next.logs = [...next.logs, addLog(`[${event.phase.toUpperCase()}] Q${event.q_idx + 1}/${event.total}: ${event.question}`)]
          break

        case 'draft_generated': {
          const e = event as import('./types').DraftGeneratedEvent
          const cv = pipelineConvMapRef.current.get(e.round) ?? emptyConv(e.round)
          pipelineConvMapRef.current.set(e.round, { ...cv, draft: e.full_draft, wordCount: e.word_count })
          next.activeConversation = Array.from(pipelineConvMapRef.current.values()).sort((a, b) => a.round - b.round)
          next.logs = [...next.logs, addLog(`  [Writer] Round ${e.round} — ${e.word_count} words`)]
          break
        }

        case 'evaluating': {
          const e = event as import('./types').EvaluatingEvent
          const cv = pipelineConvMapRef.current.get(e.round) ?? emptyConv(e.round)
          pipelineConvMapRef.current.set(e.round, { ...cv, evaluating: true })
          next.activeConversation = Array.from(pipelineConvMapRef.current.values()).sort((a, b) => a.round - b.round)
          break
        }

        case 'is_score': {
          const e = event as import('./types').ISScoreEvent
          const prevCv = pipelineConvMapRef.current.get(e.round - 1)
          const cv = pipelineConvMapRef.current.get(e.round) ?? emptyConv(e.round)
          pipelineConvMapRef.current.set(e.round, {
            ...cv,
            isScore:    e.score,
            isGain:     prevCv?.isScore != null ? e.score - prevCv.isScore : null,
            evaluating: false,
            metrics: {
              faithfulness:      e.metrics.faithfulness      ?? null,
              answer_relevancy:  e.metrics.answer_relevancy  ?? null,
              context_precision: e.metrics.context_precision ?? null,
              context_recall:    e.metrics.context_recall    ?? null,
            },
          })
          next.activeConversation = Array.from(pipelineConvMapRef.current.values()).sort((a, b) => a.round - b.round)
          next.logs = [...next.logs, addLog(`  [IS] Round ${e.round}: ${e.score.toFixed(3)}`)]
          break
        }

        case 'critic_feedback': {
          const e = event as import('./types').CriticFeedbackEvent
          const cv = pipelineConvMapRef.current.get(e.round) ?? emptyConv(e.round)
          pipelineConvMapRef.current.set(e.round, { ...cv, criticFeedback: e.feedback, isApproved: e.is_approved })
          next.activeConversation = Array.from(pipelineConvMapRef.current.values()).sort((a, b) => a.round - b.round)
          next.logs = [...next.logs, addLog(
            e.is_approved ? '  [Critic] APPROVED ✓' : `  [Critic] ${e.feedback.slice(0, 80)}…`,
            e.is_approved ? 'success' : 'info',
          )]
          break
        }

        case 'pipeline_q_complete': {
          const result: PipelineQuestionResult = {
            phase:       event.phase,
            q_idx:       event.q_idx,
            question:    event.question,
            scenario_id: event.scenario_id,
            final_is:    event.final_is,
            halt_reason: event.halt_reason,
          }
          if (event.phase === 'train') {
            next.trainResults = [...next.trainResults, result]
          } else if (event.phase === 'val') {
            next.valResults = [...next.valResults, result]
          } else {
            next.testResults = [...next.testResults, result]
          }
          next.logs = [...next.logs, addLog(
            `  ✓ ${event.phase} Q${event.q_idx + 1} — IS=${event.final_is.toFixed(4)} (${event.halt_reason})`,
            'success',
          )]
          break
        }

        case 'calibrating':
          next.phase = 'calibrating'
          next.logs  = [...next.logs, addLog('⚙ Calibrating IS weights from training data…')]
          break

        case 'weights_learned':
          next.weights = event.weights
          next.r2      = event.r2_score ?? null
          next.logs    = [...next.logs, addLog(
            `✓ Weights learned — R²=${event.r2_score != null ? (event.r2_score * 100).toFixed(1) : 'n/a'}%`,
            'success',
          )]
          break

        case 'pipeline_complete':
          next.status = 'complete'
          next.logs   = [...next.logs, addLog(`🏁 Pipeline complete — train:${event.train_results.length} val:${event.val_results.length} test:${event.test_results.length}`, 'success')]
          break

        case 'error':
          next.status = 'error'
          next.error  = (event as import('./types').ErrorEvent).message
          next.logs   = [...next.logs, addLog(`✖ ${(event as import('./types').ErrorEvent).message}`, 'error')]
          break
      }

      return next
    })
  }, [])

  // ── Single-run controls ───────────────────────────────────────────────────────
  const handleRun = () => {
    if (wsStatus === 'running') {
      disconnect()
      setRun(p => ({ ...p, status: 'idle' }))
      return
    }

    roundMapRef.current.clear()
    convMapRef.current.clear()

    const isCustom = selectedId === 'custom'
    const config: RunConfig = {
      provider,
      agent_model: agentModel,
      eval_model:  evalModel,
      scenario_id: isCustom ? null : selectedId,
      ...(isCustom && {
        topic:        customTopic || 'Custom Topic',
        question:     customQ,
        contexts:     customCtx.split('\n').map(s => s.trim()).filter(Boolean),
        ground_truth: customGT,
      }),
    }

    setRun({
      ...DEFAULT_RUN_STATE,
      status:     'connecting',
      provider,
      agentModel,
      evalModel,
      scenarioId: selectedId,
      topic:      scenarios.find(s => s.id === selectedId)?.topic ?? customTopic ?? '',
      question:   scenarios.find(s => s.id === selectedId)?.question ?? customQ ?? '',
      isWeights:  calibratedWeights?.weights ?? DEFAULT_RUN_STATE.isWeights,
    })

    connect(config, handleEvent)
  }

  const handleReset = () => {
    disconnect()
    roundMapRef.current.clear()
    convMapRef.current.clear()
    setRun(DEFAULT_RUN_STATE)
  }

  // ── Pipeline controls ─────────────────────────────────────────────────────────
  const handlePipelineRun = () => {
    if (pipelineWsStatus === 'running') {
      pipelineDisconnect()
      setPipeline(p => ({ ...p, status: 'idle' }))
      return
    }

    pipelineConvMapRef.current.clear()
    setPipeline({ ...DEFAULT_PIPELINE_STATE, status: 'connecting', topicId: selectedTopic })

    const config: PipelineConfig = {
      topic_id:    selectedTopic,
      provider,
      agent_model: agentModel,
      eval_model:  evalModel,
    }

    pipelineConnect(config, handlePipelineEvent)
  }

  const handlePipelineReset = () => {
    pipelineDisconnect()
    pipelineConvMapRef.current.clear()
    setPipeline(DEFAULT_PIPELINE_STATE)
  }

  // ── Derived values ────────────────────────────────────────────────────────────
  const isRunning         = wsStatus === 'running' || wsStatus === 'connecting'
  const isPipelineRunning = pipelineWsStatus === 'running' || pipelineWsStatus === 'connecting'
  const haltKey           = isRunning ? 'running' : run.haltReason || 'idle'
  const haltRound         = run.haltReason ? run.rounds : undefined

  const displayWeights = run.isWeights
  const displayR2      = calibratedWeights?.r2_score ?? null
  const displaySource  = calibratedWeights?.data_source ?? 'default'

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-[1600px] mx-auto px-4 py-3 flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center">
              <Activity size={14} className="text-white" />
            </div>
            <div>
              <span className="font-bold text-slate-100 text-sm">SHP</span>
              <span className="text-slate-500 text-sm ml-2">Semantic Halting Problem</span>
            </div>
          </div>

          {/* Mode toggle */}
          <div className="ml-6 flex items-center bg-slate-800 rounded-lg p-1 border border-slate-700">
            <button
              onClick={() => setAppMode('single')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all
                ${appMode === 'single'
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'}`}
            >
              <MessageSquare size={12} /> Single Question
            </button>
            <button
              onClick={() => setAppMode('pipeline')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all
                ${appMode === 'pipeline'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'}`}
            >
              <FlaskConical size={12} /> Full Pipeline
            </button>
          </div>

          <div className="ml-auto flex items-center gap-2 text-xs text-slate-600">
            <span>LangGraph + Ragas + Banach Fixed-Point</span>
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-4 py-5 space-y-5">
        {/* ── Backend error ─────────────────────────────────────────────────── */}
        {apiError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-sm text-red-400">
            {apiError}
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            SINGLE-QUESTION MODE
        ══════════════════════════════════════════════════════════════════ */}
        {appMode === 'single' && (
          <>
            {/* Control panel */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 space-y-4">
              <div className="flex flex-wrap items-center gap-4">
                <ProviderSelector
                  provider={provider}
                  agentModel={agentModel}
                  evalModel={evalModel}
                  availableProviders={providers}
                  onProviderChange={(p) => { setProvider(p); setSelectedId('') }}
                  onAgentModelChange={setAgentModel}
                  onEvalModelChange={setEvalModel}
                  disabled={isRunning}
                />

                <div className="ml-auto flex items-center gap-2">
                  {run.status !== 'idle' && (
                    <button
                      onClick={handleReset}
                      disabled={isRunning}
                      className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm
                                 text-slate-400 hover:text-slate-200 hover:bg-slate-700
                                 disabled:opacity-40 transition-all"
                    >
                      <RefreshCw size={14} /> Reset
                    </button>
                  )}
                  <button
                    onClick={handleRun}
                    disabled={!selectedId && !(customQ && customCtx)}
                    className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold
                                transition-all shadow disabled:opacity-40 disabled:cursor-not-allowed
                                ${isRunning
                                  ? 'bg-red-600 hover:bg-red-500 text-white'
                                  : 'bg-blue-600 hover:bg-blue-500 text-white'}`}
                  >
                    {isRunning ? <><Square size={14} /> Stop</> : <><Play size={14} /> Run</>}
                  </button>
                </div>
              </div>

              <ScenarioPanel
                scenarios={scenarios}
                selectedId={selectedId}
                onSelectScenario={setSelectedId}
                customTopic={customTopic}
                customQuestion={customQ}
                customContexts={customCtx}
                customGroundTruth={customGT}
                onCustomChange={(f, v) => {
                  if (f === 'topic') setCustomTopic(v)
                  else if (f === 'question') setCustomQ(v)
                  else if (f === 'contexts') setCustomCtx(v)
                  else if (f === 'groundTruth') setCustomGT(v)
                }}
                disabled={isRunning}
              />
            </div>

            {/* Halt status bar */}
            <div className="flex flex-wrap items-center gap-3">
              <HaltBadge
                reason={haltKey as any}
                rounds={run.rounds > 0 ? run.rounds : undefined}
                finalScore={run.finalISScore > 0 ? run.finalISScore : undefined}
              />
              {run.error && (
                <div className="text-xs text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg px-3 py-2">
                  {run.error}
                </div>
              )}
              {isRunning && (
                <div className="flex items-center gap-2 text-xs text-blue-400 font-mono">
                  <div className="w-2 h-2 rounded-full bg-blue-400 pulse-ring" />
                  Round {run.rounds}…
                </div>
              )}
            </div>

            {/* Conversation view */}
            {(run.conversations.length > 0 || run.status === 'running' || run.status === 'connecting') && (
              <ConversationView
                conversations={run.conversations}
                question={run.question}
                isWeights={run.isWeights}
                isRunning={isRunning}
              />
            )}

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ConvergenceChart data={run.roundData} haltRound={haltRound} />
              <DistanceChart    data={run.roundData} haltRound={haltRound} />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ISGainChart  data={run.roundData} />
              <WeightsChart weights={displayWeights} r2Score={displayR2} dataSource={displaySource} />
            </div>
            <MetricsBreakdown data={run.roundData} />
            <LogStream logs={run.logs} maxHeight="220px" />
          </>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            FULL PIPELINE MODE
        ══════════════════════════════════════════════════════════════════ */}
        {appMode === 'pipeline' && (
          <>
            {/* Pipeline control panel */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-4 space-y-4">
              <div className="flex flex-wrap items-center gap-4">
                <ProviderSelector
                  provider={provider}
                  agentModel={agentModel}
                  evalModel={evalModel}
                  availableProviders={providers}
                  onProviderChange={setProvider}
                  onAgentModelChange={setAgentModel}
                  onEvalModelChange={setEvalModel}
                  disabled={isPipelineRunning}
                />

                {/* Topic picker */}
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-slate-500 font-medium">Topic</label>
                  <select
                    value={selectedTopic}
                    onChange={e => setSelectedTopic(e.target.value)}
                    disabled={isPipelineRunning}
                    className="bg-slate-700 border border-slate-600 text-slate-200 text-sm rounded-lg
                               px-3 py-1.5 focus:ring-1 focus:ring-blue-500 focus:outline-none
                               disabled:opacity-50 min-w-[220px]"
                  >
                    {topics.length === 0 && (
                      <option value="">Loading topics…</option>
                    )}
                    {topics.map(t => (
                      <option key={t.topic_id} value={t.topic_id}>
                        {t.topic} ({t.train_count}T / {t.val_count}V / {t.test_count}Te)
                      </option>
                    ))}
                  </select>
                </div>

                <div className="ml-auto flex items-center gap-2">
                  {pipeline.status !== 'idle' && (
                    <button
                      onClick={handlePipelineReset}
                      disabled={isPipelineRunning}
                      className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm
                                 text-slate-400 hover:text-slate-200 hover:bg-slate-700
                                 disabled:opacity-40 transition-all"
                    >
                      <RefreshCw size={14} /> Reset
                    </button>
                  )}
                  <button
                    onClick={handlePipelineRun}
                    disabled={!selectedTopic}
                    className={`flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold
                                transition-all shadow disabled:opacity-40 disabled:cursor-not-allowed
                                ${isPipelineRunning
                                  ? 'bg-red-600 hover:bg-red-500 text-white'
                                  : 'bg-indigo-600 hover:bg-indigo-500 text-white'}`}
                  >
                    {isPipelineRunning
                      ? <><Square size={14} /> Stop</>
                      : <><FlaskConical size={14} /> Run Pipeline</>}
                  </button>
                </div>
              </div>

              {/* Pipeline description */}
              <div className="text-xs text-slate-500 border-t border-slate-700 pt-3">
                Runs all training questions → learns IS weights → validates → tests.
                Watch the IS score curve grow like an ML training graph.
              </div>
            </div>

            {/* Pipeline error */}
            {(pipeline.error || pipelineWsError) && (
              <div className="text-xs text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg px-3 py-2">
                {pipeline.error ?? pipelineWsError}
              </div>
            )}

            {/* Pipeline view */}
            <PipelineView
              pipeline={pipeline}
              activeConversation={pipeline.activeConversation}
              activeQuestion={pipeline.activeQuestion}
            />

            {/* Pipeline logs */}
            {pipeline.logs.length > 0 && (
              <LogStream logs={pipeline.logs} maxHeight="200px" />
            )}
          </>
        )}
      </main>

      {/* ── Footer ──────────────────────────────────────────────────────────── */}
      <footer className="border-t border-slate-800 mt-8 py-4 text-center text-xs text-slate-600">
        Semantic Halting Problem · Banach Fixed-Point Theorem in embedding space · Ragas + LangGraph
      </footer>
    </div>
  )
}
