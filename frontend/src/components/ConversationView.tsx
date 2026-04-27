/**
 * ConversationView
 *
 * Shows the live Writer → Critic dialogue round by round.
 * Each round card contains:
 *   - Writer's full draft
 *   - Metric bars with plain-English labels and explanations
 *   - IS score with trend arrow
 *   - Critic's feedback
 *
 * Designed so someone with no technical background can follow what
 * is happening and WHY the Information Score changes.
 */

import React, { useEffect, useRef } from 'react'
import { PenLine, Scale, ArrowUp, ArrowDown, Minus, Loader2, CheckCircle } from 'lucide-react'
import type { RoundConversation } from '../types'

// ── Metric meta ─────────────────────────────────────────────────────────────
const METRIC_META: Record<string, { label: string; explain: string; color: string }> = {
  faithfulness: {
    label:   'Stuck to the facts?',
    explain: 'Did the writer only state things that are actually in the provided context? (1 = perfectly faithful, 0 = made things up)',
    color:   'bg-blue-500',
  },
  answer_relevancy: {
    label:   'Answered the question?',
    explain: 'Does the response directly address what was asked? A vague or off-topic reply scores low here.',
    color:   'bg-violet-500',
  },
  context_precision: {
    label:   'Used the right context parts?',
    explain: 'Were the most relevant pieces of context placed prominently, or did the writer focus on less important details?',
    color:   'bg-amber-500',
  },
  context_recall: {
    label:   'Covered all key facts?',
    explain: 'Did the writer include all the important information from the context, or leave out key facts?',
    color:   'bg-emerald-500',
  },
}

// ── Sub-components ───────────────────────────────────────────────────────────

function MetricBar({ name, value }: { name: string; value: number | null }) {
  const meta = METRIC_META[name]
  if (!meta) return null
  const pct = value != null ? Math.round(value * 100) : null

  return (
    <div className="group relative">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs text-slate-400 w-44 shrink-0">{meta.label}</span>
        <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
          {pct != null ? (
            <div
              className={`h-full rounded-full transition-all duration-700 ${meta.color}`}
              style={{ width: `${pct}%` }}
            />
          ) : (
            <div className="h-full bg-slate-600 rounded-full animate-pulse w-1/3" />
          )}
        </div>
        <span className="text-xs font-mono text-slate-300 w-10 text-right shrink-0">
          {pct != null ? `${pct}%` : '…'}
        </span>
      </div>
      {/* Tooltip on hover */}
      <div className="absolute left-0 top-6 z-20 hidden group-hover:block
                      bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                      text-xs text-slate-300 max-w-xs shadow-xl pointer-events-none">
        {meta.explain}
      </div>
    </div>
  )
}

function ISScoreBadge({
  score, gain, isBaseline,
}: { score: number | null; gain: number | null; isBaseline: boolean }) {
  if (score == null) {
    return (
      <div className="flex items-center gap-1.5 text-slate-500 text-sm">
        <Loader2 size={13} className="animate-spin" />
        Scoring…
      </div>
    )
  }

  const pct = Math.round(score * 100)

  let gainEl = (
    <span className="flex items-center gap-0.5 text-slate-400 text-xs">
      <Minus size={10} /> baseline
    </span>
  )
  if (!isBaseline && gain != null) {
    const improved = gain > 0.005
    const worsened = gain < -0.005
    gainEl = improved ? (
      <span className="flex items-center gap-0.5 text-emerald-400 text-xs font-medium">
        <ArrowUp size={11} /> +{(gain * 100).toFixed(1)}% improved
      </span>
    ) : worsened ? (
      <span className="flex items-center gap-0.5 text-red-400 text-xs font-medium">
        <ArrowDown size={11} /> {(gain * 100).toFixed(1)}% dropped
      </span>
    ) : (
      <span className="flex items-center gap-0.5 text-slate-400 text-xs">
        <Minus size={10} /> no change
      </span>
    )
  }

  const scoreColor =
    pct >= 70 ? 'text-emerald-400' :
    pct >= 45 ? 'text-amber-400'  :
    'text-red-400'

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-baseline gap-1">
        <span className={`text-2xl font-bold tabular-nums ${scoreColor}`}>{pct}</span>
        <span className="text-slate-500 text-sm">/100</span>
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-xs text-slate-500">Information Score</span>
        {gainEl}
      </div>
    </div>
  )
}

// ── Round card ───────────────────────────────────────────────────────────────

function RoundCard({
  conv, prevScore, isLast, isRunning,
}: {
  conv: RoundConversation
  prevScore: number | null
  isLast: boolean
  isRunning: boolean
}) {
  const hasDraft   = conv.draft.length > 0
  const hasMetrics = conv.isScore != null
  const hasCritic  = conv.criticFeedback != null

  return (
    <div className={`relative flex flex-col gap-0 ${isLast ? 'pb-2' : 'pb-6'}`}>
      {/* ── Timeline connector ─────────────────── */}
      {!isLast && (
        <div className="absolute left-[19px] top-10 bottom-0 w-0.5 bg-slate-700" />
      )}

      {/* ── Round header ───────────────────────── */}
      <div className="flex items-center gap-3 mb-3">
        <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 z-10
                         border-2 font-bold text-sm
                         ${conv.isApproved
                           ? 'bg-emerald-900/60 border-emerald-500 text-emerald-300'
                           : 'bg-slate-800 border-slate-600 text-slate-300'
                         }`}>
          {conv.round + 1}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-400 text-sm">Round {conv.round + 1}</span>
          {conv.isApproved && (
            <span className="flex items-center gap-1 text-xs text-emerald-400 font-medium">
              <CheckCircle size={12} /> Approved
            </span>
          )}
          {isLast && isRunning && !conv.isApproved && (
            <span className="flex items-center gap-1 text-xs text-blue-400">
              <Loader2 size={11} className="animate-spin" /> in progress…
            </span>
          )}
        </div>
      </div>

      <div className="ml-[52px] space-y-3">

        {/* ── Writer block ───────────────────────── */}
        {hasDraft && (
          <div className="bg-slate-800/70 border border-slate-700 rounded-xl overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-700 bg-slate-800">
              <PenLine size={13} className="text-blue-400" />
              <span className="text-xs font-semibold text-blue-300 uppercase tracking-wide">Writer</span>
              <span className="ml-auto text-xs text-slate-500">{conv.wordCount} words</span>
            </div>
            <div className="px-4 py-3 text-sm text-slate-300 leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto">
              {conv.draft}
            </div>
          </div>
        )}

        {/* ── Evaluation block ───────────────────── */}
        {(hasDraft && (conv.evaluating || hasMetrics)) && (
          <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-700/60 bg-slate-800/80">
              <Scale size={13} className="text-amber-400" />
              <span className="text-xs font-semibold text-amber-300 uppercase tracking-wide">Evaluation</span>
              {conv.evaluating && !hasMetrics && (
                <span className="ml-auto flex items-center gap-1 text-xs text-slate-500">
                  <Loader2 size={10} className="animate-spin" /> Ragas running…
                </span>
              )}
            </div>

            <div className="px-4 py-3 space-y-3">
              {/* Score */}
              <ISScoreBadge
                score={conv.isScore}
                gain={conv.isGain}
                isBaseline={prevScore == null}
              />

              {/* Metric bars */}
              {hasMetrics && (
                <div className="space-y-2 pt-1 border-t border-slate-700/40">
                  <p className="text-xs text-slate-500 mb-2">
                    Hover any bar for an explanation of what the metric measures.
                  </p>
                  {Object.keys(METRIC_META).map((k) => (
                    <MetricBar key={k} name={k} value={conv.metrics[k as keyof typeof conv.metrics]} />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Critic block ───────────────────────── */}
        {hasCritic && (
          <div className={`border rounded-xl overflow-hidden
                           ${conv.isApproved
                             ? 'bg-emerald-900/20 border-emerald-700/50'
                             : 'bg-slate-800/50 border-slate-700/60'
                           }`}>
            <div className={`flex items-center gap-2 px-4 py-2.5 border-b
                             ${conv.isApproved
                               ? 'border-emerald-700/40 bg-emerald-900/30'
                               : 'border-slate-700/60 bg-slate-800/80'
                             }`}>
              <span className="text-sm">⚖️</span>
              <span className={`text-xs font-semibold uppercase tracking-wide
                               ${conv.isApproved ? 'text-emerald-300' : 'text-slate-300'}`}>
                Critic
              </span>
              {conv.isApproved && (
                <span className="ml-auto text-xs text-emerald-400 font-medium">✓ Approved — stopping here</span>
              )}
            </div>
            <div className="px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap text-slate-300">
              {conv.criticFeedback}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

interface Props {
  conversations: RoundConversation[]
  question: string
  isWeights: Record<string, number>
  isRunning: boolean
}

export default function ConversationView({ conversations, question, isRunning }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom as new rounds arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [conversations.length, conversations[conversations.length - 1]?.criticFeedback])

  if (conversations.length === 0) return null

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-slate-700 bg-slate-800/80">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Live Conversation</h2>
            {question && (
              <p className="text-xs text-slate-400 mt-0.5">
                <span className="text-slate-500">Q: </span>{question}
              </p>
            )}
          </div>
          <div className="text-xs text-slate-500 shrink-0 mt-0.5">
            {conversations.length} round{conversations.length !== 1 ? 's' : ''}
          </div>
        </div>
      </div>

      {/* Explanation banner — shown only once */}
      <div className="px-5 py-2.5 bg-blue-950/30 border-b border-blue-900/30">
        <p className="text-xs text-blue-300/80 leading-relaxed">
          <strong className="text-blue-200">How to read this:</strong> The Writer tries to answer the question using the provided facts.
          Ragas scores how well it did across 4 dimensions. The Critic then tells the Writer what to improve.
          The loop stops when the answer stops getting better (Information Score converges).
        </p>
      </div>

      {/* Conversation timeline */}
      <div className="px-5 py-5 max-h-[700px] overflow-y-auto space-y-0">
        {conversations.map((conv, i) => (
          <RoundCard
            key={conv.round}
            conv={conv}
            prevScore={i > 0 ? (conversations[i - 1].isScore ?? null) : null}
            isLast={i === conversations.length - 1}
            isRunning={isRunning}
          />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
