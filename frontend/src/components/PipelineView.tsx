/**
 * PipelineView
 *
 * Full ML-style training pipeline visualisation.
 *
 * Layout (top → bottom):
 *   A. IS Curve — three coloured lines (train / val / test) revealing point-by-point
 *   B. Weight Bar Chart — shown after calibration, with plain-English labels
 *   C. Active Conversation — reuses ConversationView for the currently running question
 */

import React from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { Brain, FlaskConical, TestTube2, Loader2, CheckCircle2, AlertCircle } from 'lucide-react'
import type { PipelineState, PipelineQuestionResult, RoundConversation } from '../types'
import ConversationView from './ConversationView'

// ── Helpers ───────────────────────────────────────────────────────────────────

const PHASE_COLOURS: Record<string, string> = {
  train: '#3b82f6', // blue-500
  val:   '#eab308', // yellow-500
  test:  '#22c55e', // green-500
}

const WEIGHT_COLOURS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981']

const WEIGHT_LABELS: Record<string, string> = {
  faithfulness:      'Stuck to facts?',
  answer_relevancy:  'Answered the question?',
  context_precision: 'Used relevant context?',
  context_recall:    'Covered all context?',
}

interface ISCurvePoint {
  x: number
  train?: number
  val?: number
  test?: number
  label: string
}

function buildCurveData(
  trainResults: PipelineQuestionResult[],
  valResults: PipelineQuestionResult[],
  testResults: PipelineQuestionResult[],
): ISCurvePoint[] {
  const maxLen = Math.max(
    trainResults.length,
    valResults.length,
    testResults.length,
    1,
  )
  return Array.from({ length: maxLen }, (_, i) => ({
    x:     i + 1,
    label: `Q${i + 1}`,
    train: trainResults[i]  !== undefined ? trainResults[i].final_is  : undefined,
    val:   valResults[i]    !== undefined ? valResults[i].final_is    : undefined,
    test:  testResults[i]   !== undefined ? testResults[i].final_is   : undefined,
  }))
}

// ── Phase status badge ─────────────────────────────────────────────────────────

function PhaseBadge({
  label,
  icon: Icon,
  active,
  done,
  count,
}: {
  label: string
  icon: React.ElementType
  active: boolean
  done: boolean
  count: number
}) {
  const base = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all'
  const style = done
    ? 'bg-green-900/40 text-green-300 border border-green-700'
    : active
    ? 'bg-blue-900/40 text-blue-300 border border-blue-500 animate-pulse'
    : 'bg-gray-800 text-gray-500 border border-gray-700'
  return (
    <div className={`${base} ${style}`}>
      <Icon className="w-4 h-4" />
      {label} ({count})
      {done && <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />}
      {active && !done && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
    </div>
  )
}

// ── IS Curve chart ────────────────────────────────────────────────────────────

function ISCurveChart({
  trainResults,
  valResults,
  testResults,
  weightsLearnedAt,
}: {
  trainResults: PipelineQuestionResult[]
  valResults: PipelineQuestionResult[]
  testResults: PipelineQuestionResult[]
  weightsLearnedAt: number | null
}) {
  const data = buildCurveData(trainResults, valResults, testResults)
  const hasData = data.some(d => d.train !== undefined)

  if (!hasData) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-500 text-sm">
        Waiting for first question to complete…
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="label"
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af', fontSize: 12 }}
          label={{ value: 'Question #', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }}
        />
        <YAxis
          domain={[0, 1]}
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af', fontSize: 12 }}
          tickFormatter={v => v.toFixed(1)}
          label={{ value: 'IS Score', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
          labelStyle={{ color: '#f9fafb', fontWeight: 600 }}
          itemStyle={{ color: '#d1d5db' }}
          formatter={(v: number, name: string) => [v.toFixed(4), name.charAt(0).toUpperCase() + name.slice(1)]}
        />
        <Legend
          wrapperStyle={{ color: '#9ca3af', fontSize: 12 }}
          formatter={v => v.charAt(0).toUpperCase() + v.slice(1)}
        />
        {weightsLearnedAt !== null && (
          <ReferenceLine
            x={`Q${weightsLearnedAt}`}
            stroke="#6366f1"
            strokeDasharray="6 3"
            label={{ value: 'Weights learned', fill: '#818cf8', fontSize: 11, position: 'top' }}
          />
        )}
        <Line
          type="monotone"
          dataKey="train"
          stroke={PHASE_COLOURS.train}
          strokeWidth={2}
          dot={{ r: 4, fill: PHASE_COLOURS.train }}
          connectNulls={false}
          activeDot={{ r: 6 }}
        />
        <Line
          type="monotone"
          dataKey="val"
          stroke={PHASE_COLOURS.val}
          strokeWidth={2}
          dot={{ r: 4, fill: PHASE_COLOURS.val }}
          connectNulls={false}
          activeDot={{ r: 6 }}
        />
        <Line
          type="monotone"
          dataKey="test"
          stroke={PHASE_COLOURS.test}
          strokeWidth={2}
          dot={{ r: 4, fill: PHASE_COLOURS.test }}
          connectNulls={false}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ── Weight bar chart ──────────────────────────────────────────────────────────

function WeightsChart({
  weights,
  r2,
}: {
  weights: Record<string, number>
  r2: number | null
}) {
  const data = Object.entries(weights).map(([key, value], i) => ({
    key,
    label: WEIGHT_LABELS[key] ?? key,
    value: Math.round(value * 100),
    fill:  WEIGHT_COLOURS[i % WEIGHT_COLOURS.length],
  }))

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm font-semibold text-white">Learned IS Weights</p>
        {r2 !== null && (
          <span className="text-xs text-indigo-300 bg-indigo-900/40 border border-indigo-700 px-2 py-0.5 rounded-full">
            Fit quality (R²): {(r2 * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="label"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            interval={0}
          />
          <YAxis
            domain={[0, 100]}
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            tickFormatter={v => `${v}%`}
          />
          <Tooltip
            contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
            formatter={(v: number) => [`${v}%`, 'Weight']}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-500 mt-2">
        These weights tell us which quality dimensions matter most.
        A higher weight means that dimension has more influence on the final IS score.
      </p>
    </div>
  )
}

// ── Phase result table ────────────────────────────────────────────────────────

function PhaseResultTable({
  results,
  phase,
}: {
  results: PipelineQuestionResult[]
  phase: 'train' | 'val' | 'test'
}) {
  if (results.length === 0) return null
  const colour = PHASE_COLOURS[phase]
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: colour }}>
        {phase} results
      </p>
      <div className="space-y-1">
        {results.map((r, i) => (
          <div key={i} className="flex items-center gap-3 text-xs bg-gray-800/50 rounded px-3 py-1.5">
            <span className="text-gray-400 w-4">{i + 1}.</span>
            <span className="text-gray-300 flex-1 truncate" title={r.question}>{r.question}</span>
            <span
              className="font-mono font-semibold"
              style={{ color: colour }}
            >
              IS {r.final_is.toFixed(4)}
            </span>
            <span className="text-gray-500 text-[10px]">{r.halt_reason}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  pipeline: PipelineState
  activeConversation: RoundConversation[]
  activeQuestion: string
}

export default function PipelineView({ pipeline, activeConversation, activeQuestion }: Props) {
  const trainDone = pipeline.trainResults.length > 0
  const weightsLearned = pipeline.weights !== null
  const valDone = pipeline.valResults.length > 0
  const testDone = pipeline.testResults.length > 0

  // Index at which the vertical "weights learned" line should appear
  const weightsLearnedAt = weightsLearned ? pipeline.trainResults.length : null

  return (
    <div className="space-y-6">

      {/* ── Phase status strip ─────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">
        <PhaseBadge
          label="Train"
          icon={Brain}
          active={pipeline.phase === 'train'}
          done={trainDone}
          count={pipeline.trainResults.length}
        />
        <div className="text-gray-600 text-xs">→ calibrate →</div>
        <PhaseBadge
          label="Validate"
          icon={FlaskConical}
          active={pipeline.phase === 'val'}
          done={valDone}
          count={pipeline.valResults.length}
        />
        <div className="text-gray-600 text-xs">→</div>
        <PhaseBadge
          label="Test"
          icon={TestTube2}
          active={pipeline.phase === 'test'}
          done={testDone}
          count={pipeline.testResults.length}
        />
        {pipeline.status === 'complete' && (
          <span className="ml-auto flex items-center gap-1.5 text-green-400 text-sm font-medium">
            <CheckCircle2 className="w-4 h-4" /> Pipeline complete
          </span>
        )}
        {pipeline.status === 'error' && (
          <span className="ml-auto flex items-center gap-1.5 text-red-400 text-sm font-medium">
            <AlertCircle className="w-4 h-4" /> {pipeline.error}
          </span>
        )}
        {pipeline.phase === 'calibrating' && (
          <span className="ml-auto flex items-center gap-1.5 text-indigo-400 text-sm font-medium animate-pulse">
            <Loader2 className="w-4 h-4 animate-spin" /> Learning IS weights…
          </span>
        )}
      </div>

      {/* ── A. IS Curve ────────────────────────────────────────────────── */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white mb-1">Information Score Curve</h3>
        <p className="text-xs text-gray-500 mb-4">
          Like training / validation / test loss in ML — watch how IS improves (or plateaus) across questions in each phase.
        </p>
        <ISCurveChart
          trainResults={pipeline.trainResults}
          valResults={pipeline.valResults}
          testResults={pipeline.testResults}
          weightsLearnedAt={weightsLearnedAt}
        />
      </div>

      {/* ── B. Weights chart (post-calibration) ───────────────────────── */}
      {weightsLearned && pipeline.weights && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <WeightsChart weights={pipeline.weights} r2={pipeline.r2} />
        </div>
      )}

      {/* ── Phase result tables ────────────────────────────────────────── */}
      {(trainDone || valDone || testDone) && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
          <PhaseResultTable results={pipeline.trainResults} phase="train" />
          <PhaseResultTable results={pipeline.valResults}   phase="val"   />
          <PhaseResultTable results={pipeline.testResults}  phase="test"  />
        </div>
      )}

      {/* ── C. Active conversation ─────────────────────────────────────── */}
      {(pipeline.status === 'running' || pipeline.status === 'complete') && activeConversation.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800">
            <p className="text-sm font-semibold text-white">Active Question</p>
            {pipeline.status === 'running' && (
              <span className="flex items-center gap-1.5 text-xs text-blue-400">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                {pipeline.phase} phase
              </span>
            )}
          </div>
          <div className="p-4">
            <ConversationView
              conversations={activeConversation}
              question={activeQuestion}
              isWeights={{}}
              isRunning={pipeline.status === 'running'}
            />
          </div>
        </div>
      )}
    </div>
  )
}
