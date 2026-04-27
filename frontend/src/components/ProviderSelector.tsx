import React from 'react'
import { Cpu, Key } from 'lucide-react'

interface Props {
  provider: string
  agentModel: string
  evalModel: string
  availableProviders: Record<string, string[]>
  onProviderChange: (provider: string) => void
  onAgentModelChange: (model: string) => void
  onEvalModelChange:  (model: string) => void
  disabled: boolean
}

const PROVIDER_LABELS: Record<string, string> = {
  groq:   'Groq (Free)',
  openai: 'OpenAI',
}

const PROVIDER_COLORS: Record<string, string> = {
  groq:   'text-orange-400',
  openai: 'text-green-400',
}

export default function ProviderSelector({
  provider,
  agentModel,
  evalModel,
  availableProviders,
  onProviderChange,
  onAgentModelChange,
  onEvalModelChange,
  disabled,
}: Props) {
  const models = availableProviders[provider] ?? []

  return (
    <div className="flex flex-wrap items-center gap-3">
      {/* Provider pill buttons */}
      <div className="flex items-center gap-1 bg-slate-800 rounded-lg p-1">
        {Object.keys(availableProviders).map((p) => (
          <button
            key={p}
            onClick={() => onProviderChange(p)}
            disabled={disabled}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              provider === p
                ? 'bg-blue-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <span className={provider === p ? 'text-white' : PROVIDER_COLORS[p]}>
              {PROVIDER_LABELS[p] ?? p}
            </span>
          </button>
        ))}
      </div>

      {/* Agent model */}
      <div className="flex items-center gap-2">
        <Cpu size={14} className="text-slate-500" />
        <label className="text-xs text-slate-400">Agent</label>
        <select
          value={agentModel}
          onChange={(e) => onAgentModelChange(e.target.value)}
          disabled={disabled}
          className="bg-slate-800 border border-slate-700 rounded-md text-sm text-slate-200
                     px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* Eval model */}
      <div className="flex items-center gap-2">
        <Key size={14} className="text-slate-500" />
        <label className="text-xs text-slate-400">Judge</label>
        <select
          value={evalModel}
          onChange={(e) => onEvalModelChange(e.target.value)}
          disabled={disabled}
          className="bg-slate-800 border border-slate-700 rounded-md text-sm text-slate-200
                     px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>
    </div>
  )
}
