import React from 'react'
import { CheckCircle, Cpu, TrendingDown, AlertTriangle } from 'lucide-react'
import type { HaltReason } from '../types'

const HALT_CONFIG: Record<HaltReason | 'running' | 'idle', {
  label:   string
  color:   string
  bg:      string
  border:  string
  Icon:    React.FC<any>
}> = {
  critic_approved: {
    label:  'Critic Approved',
    color:  'text-green-400',
    bg:     'bg-green-400/10',
    border: 'border-green-400/30',
    Icon:   CheckCircle,
  },
  entropy_convergence: {
    label:  'Entropy Convergence',
    color:  'text-orange-400',
    bg:     'bg-orange-400/10',
    border: 'border-orange-400/30',
    Icon:   TrendingDown,
  },
  no_information_gain: {
    label:  'No IS Gain',
    color:  'text-yellow-400',
    bg:     'bg-yellow-400/10',
    border: 'border-yellow-400/30',
    Icon:   TrendingDown,
  },
  failsafe: {
    label:  'Failsafe (max rounds)',
    color:  'text-red-400',
    bg:     'bg-red-400/10',
    border: 'border-red-400/30',
    Icon:   AlertTriangle,
  },
  running: {
    label:  'Running…',
    color:  'text-blue-400',
    bg:     'bg-blue-400/10',
    border: 'border-blue-400/30',
    Icon:   Cpu,
  },
  idle: {
    label:  'Ready',
    color:  'text-slate-500',
    bg:     'bg-slate-700/30',
    border: 'border-slate-700',
    Icon:   Cpu,
  },
  '': {
    label:  'Ready',
    color:  'text-slate-500',
    bg:     'bg-slate-700/30',
    border: 'border-slate-700',
    Icon:   Cpu,
  },
}

interface Props {
  reason: HaltReason | 'running' | 'idle'
  rounds?: number
  finalScore?: number
}

export default function HaltBadge({ reason, rounds, finalScore }: Props) {
  const cfg = HALT_CONFIG[reason] ?? HALT_CONFIG['idle']
  const { Icon } = cfg

  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${cfg.bg} ${cfg.border}`}>
      <div className={`${cfg.color} ${reason === 'running' ? 'pulse-ring' : ''}`}>
        <Icon size={18} />
      </div>
      <div>
        <div className={`text-sm font-semibold ${cfg.color}`}>
          {cfg.label}
        </div>
        {(rounds != null || finalScore != null) && (
          <div className="text-xs text-slate-500 font-mono mt-0.5">
            {rounds != null && <span>Rounds: {rounds}</span>}
            {rounds != null && finalScore != null && <span className="mx-2">·</span>}
            {finalScore != null && <span>Final IS: {finalScore.toFixed(4)}</span>}
          </div>
        )}
      </div>
    </div>
  )
}
