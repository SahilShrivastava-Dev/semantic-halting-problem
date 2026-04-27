import React, { useState } from 'react'
import { Edit3, Sparkles, FlaskConical, Plus, X } from 'lucide-react'
import type { Scenario } from '../types'

const PROFILE_COLORS: Record<string, string> = {
  fast_critic:        'text-emerald-400 bg-emerald-400/10 border-emerald-400/30',
  entropy_convergence:'text-blue-400   bg-blue-400/10   border-blue-400/30',
  is_plateau:         'text-amber-400  bg-amber-400/10  border-amber-400/30',
  failsafe:           'text-red-400    bg-red-400/10    border-red-400/30',
}
const PROFILE_LABELS: Record<string, string> = {
  fast_critic:        '✓ quick',
  entropy_convergence:'≈ converges',
  is_plateau:         '~ plateaus',
  failsafe:           '⚠ hard',
}

interface Props {
  scenarios: Scenario[]
  selectedId: string | null
  onSelectScenario: (id: string) => void
  customTopic: string
  customQuestion: string
  customContexts: string
  customGroundTruth: string
  onCustomChange: (field: string, value: string) => void
  disabled: boolean
}

export default function ScenarioPanel({
  scenarios,
  selectedId,
  onSelectScenario,
  customTopic,
  customQuestion,
  customContexts,
  customGroundTruth,
  onCustomChange,
  disabled,
}: Props) {
  const [mode, setMode] = useState<'demo' | 'research' | 'custom'>('demo')
  const [contexts, setContexts] = useState<string[]>(['', ''])

  const demos    = scenarios.filter((s) => s.split === 'demo')
  const research = scenarios.filter((s) => s.split !== 'demo')

  const selectedScenario = scenarios.find((s) => s.id === selectedId)

  function addContext() {
    const updated = [...contexts, '']
    setContexts(updated)
    onCustomChange('contexts', updated.join('\n'))
  }
  function removeContext(i: number) {
    const updated = contexts.filter((_, idx) => idx !== i)
    setContexts(updated)
    onCustomChange('contexts', updated.join('\n'))
  }
  function updateContext(i: number, val: string) {
    const updated = contexts.map((c, idx) => idx === i ? val : c)
    setContexts(updated)
    onCustomChange('contexts', updated.join('\n'))
  }

  const tabClass = (t: typeof mode) =>
    `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-all disabled:opacity-50
     ${mode === t ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-200'}`

  return (
    <div className="space-y-3">
      {/* ── Mode tabs ─────────────────────────────────────────── */}
      <div className="flex gap-1 bg-slate-900 rounded-lg p-1 w-fit">
        <button onClick={() => { setMode('demo');     onSelectScenario('') }}
                disabled={disabled} className={tabClass('demo')}>
          <Sparkles size={13} /> Try a Demo
        </button>
        <button onClick={() => { setMode('research'); onSelectScenario('') }}
                disabled={disabled} className={tabClass('research')}>
          <FlaskConical size={13} /> Research
        </button>
        <button onClick={() => { setMode('custom');  onSelectScenario('custom') }}
                disabled={disabled} className={tabClass('custom')}>
          <Edit3 size={13} /> Your Own
        </button>
      </div>

      {/* ── Demo scenarios ────────────────────────────────────── */}
      {mode === 'demo' && (
        <div className="space-y-2">
          <p className="text-xs text-slate-500">
            Simple, everyday questions — easy to follow and understand what the agents are doing.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            {demos.map((s) => (
              <button
                key={s.id}
                onClick={() => onSelectScenario(s.id)}
                disabled={disabled}
                className={`text-left p-3.5 rounded-xl border transition-all
                  ${selectedId === s.id
                    ? 'border-blue-500 bg-blue-500/10 ring-1 ring-blue-500/30'
                    : 'border-slate-700 bg-slate-800 hover:border-blue-700/50 hover:bg-slate-700/50'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <div className="text-base mb-1">{s.topic.split(':').pop()?.trim().split(' ').slice(0, 2).join(' ')}</div>
                <p className="text-sm font-medium text-slate-200 leading-snug mb-2">
                  {s.topic.replace(/^.*Demo:\s*/i, '').replace(/^[^\w]*/, '')}
                </p>
                <p className="text-xs text-slate-500 line-clamp-2">{s.question}</p>
                {s.convergence_profile && (
                  <span className={`inline-block mt-2 text-xs px-2 py-0.5 rounded border font-mono ${PROFILE_COLORS[s.convergence_profile] ?? ''}`}>
                    {PROFILE_LABELS[s.convergence_profile] ?? s.convergence_profile}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Research scenarios ────────────────────────────────── */}
      {mode === 'research' && (
        <div className="space-y-2">
          <p className="text-xs text-slate-500">
            Technical scenarios spanning different convergence behaviours — used for the IS weight regression.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {research.map((s) => (
              <button
                key={s.id}
                onClick={() => onSelectScenario(s.id)}
                disabled={disabled}
                className={`text-left p-3 rounded-lg border transition-all
                  ${selectedId === s.id
                    ? 'border-blue-500 bg-blue-500/10'
                    : 'border-slate-700 bg-slate-800 hover:border-slate-600'
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <div className="flex items-start justify-between gap-2 mb-1">
                  <span className="text-sm font-medium text-slate-200 leading-snug">
                    {s.topic.split(':').pop()?.trim() ?? s.topic}
                  </span>
                  {s.convergence_profile && (
                    <span className={`shrink-0 text-xs px-1.5 py-0.5 rounded border font-mono
                                     ${PROFILE_COLORS[s.convergence_profile] ?? 'text-slate-400 bg-slate-400/10 border-slate-400/30'}`}>
                      {PROFILE_LABELS[s.convergence_profile] ?? s.convergence_profile}
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-500 line-clamp-2">{s.question}</p>
                <span className="text-xs text-slate-600 mt-1 inline-block">{s.split}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Custom scenario ───────────────────────────────────── */}
      {mode === 'custom' && (
        <div className="space-y-3 bg-slate-800 rounded-xl p-4 border border-slate-700">
          <div className="text-xs text-slate-400 leading-relaxed bg-slate-900/50 rounded-lg px-3 py-2">
            <strong className="text-slate-300">How this works:</strong> Write 2–4 short facts as your "context database",
            then ask a question about them. The Writer will answer using only those facts,
            and you can watch how it improves with each round of Critic feedback.
          </div>

          {/* Context facts */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-2">
              Context facts <span className="text-slate-500 font-normal">(your "database")</span>
            </label>
            <div className="space-y-2">
              {contexts.map((ctx, i) => (
                <div key={i} className="flex gap-2 items-start">
                  <span className="mt-2 text-xs text-slate-600 w-5 shrink-0">#{i + 1}</span>
                  <input
                    type="text"
                    value={ctx}
                    onChange={(e) => updateContext(i, e.target.value)}
                    disabled={disabled}
                    placeholder={
                      i === 0 ? 'e.g. Sahil lives in Mumbai, India.'
                    : i === 1 ? 'e.g. Mumbai is on the west coast of India.'
                    : 'Another fact…'
                    }
                    className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                               text-sm text-slate-200 placeholder-slate-600
                               focus:outline-none focus:ring-1 focus:ring-blue-500
                               disabled:opacity-50"
                  />
                  {contexts.length > 2 && (
                    <button
                      onClick={() => removeContext(i)}
                      disabled={disabled}
                      className="mt-2 text-slate-600 hover:text-red-400 transition-colors"
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>
            <button
              onClick={addContext}
              disabled={disabled || contexts.length >= 6}
              className="mt-2 flex items-center gap-1 text-xs text-slate-500 hover:text-blue-400
                         transition-colors disabled:opacity-40"
            >
              <Plus size={12} /> Add another fact
            </button>
          </div>

          {/* Question */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Your question <span className="text-slate-500 font-normal">(what should the Writer answer?)</span>
            </label>
            <input
              type="text"
              value={customQuestion}
              onChange={(e) => onCustomChange('question', e.target.value)}
              disabled={disabled}
              placeholder="e.g. Where exactly does Sahil live and what city is it in?"
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                         text-sm text-slate-200 placeholder-slate-600
                         focus:outline-none focus:ring-1 focus:ring-blue-500
                         disabled:opacity-50"
            />
          </div>

          {/* Ground truth */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Ideal answer <span className="text-slate-500 font-normal">(used as reference to score the Writer)</span>
            </label>
            <input
              type="text"
              value={customGroundTruth}
              onChange={(e) => onCustomChange('groundTruth', e.target.value)}
              disabled={disabled}
              placeholder="e.g. Sahil lives in Mumbai, India."
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                         text-sm text-slate-200 placeholder-slate-600
                         focus:outline-none focus:ring-1 focus:ring-blue-500
                         disabled:opacity-50"
            />
          </div>

          {/* Topic (optional, collapsed) */}
          <div>
            <label className="block text-xs text-slate-500 mb-1">
              Title <span className="text-slate-600">(optional)</span>
            </label>
            <input
              type="text"
              value={customTopic}
              onChange={(e) => onCustomChange('topic', e.target.value)}
              disabled={disabled}
              placeholder="e.g. Where does Sahil live?"
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                         text-sm text-slate-200 placeholder-slate-600
                         focus:outline-none focus:ring-1 focus:ring-blue-500
                         disabled:opacity-50"
            />
          </div>
        </div>
      )}

      {/* ── Selected scenario preview ─────────────────────────── */}
      {selectedScenario && mode !== 'custom' && (
        <div className="bg-slate-800/40 rounded-lg px-3 py-2 border border-slate-700/40">
          <p className="text-xs text-slate-400 line-clamp-2">
            <span className="text-slate-600">Selected Q: </span>
            {selectedScenario.question}
          </p>
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {selectedScenario.contexts.slice(0, 2).map((ctx, i) => (
              <span key={i} className="text-xs text-slate-600 bg-slate-900/60 rounded px-2 py-0.5 truncate max-w-xs">
                "{ctx.slice(0, 60)}{ctx.length > 60 ? '…' : ''}"
              </span>
            ))}
            {selectedScenario.contexts.length > 2 && (
              <span className="text-xs text-slate-600">+{selectedScenario.contexts.length - 2} more facts</span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
