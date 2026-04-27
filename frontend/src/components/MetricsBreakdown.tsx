/**
 * MetricsBreakdown — Per-metric Ragas scores across rounds.
 *
 * ML analogy: Per-layer loss curves / task-specific metrics.
 * Shows how each Ragas metric (Faithfulness, Relevancy, Precision, Recall)
 * evolves independently across the writer-critic loop.
 */

import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import type { RoundDataPoint } from '../types'

const METRIC_CONFIG = [
  { key: 'faithfulness',      color: '#a78bfa', label: 'Faithfulness' },
  { key: 'answer_relevancy',  color: '#60a5fa', label: 'Relevancy' },
  { key: 'context_precision', color: '#34d399', label: 'Precision' },
  { key: 'context_recall',    color: '#f59e0b', label: 'Recall' },
]

interface Props {
  data: RoundDataPoint[]
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-xl space-y-1">
      <p className="text-slate-400 mb-1">Round {label}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span style={{ color: p.color }}>{p.name}:</span>
          <span className="font-mono text-slate-200">{p.value?.toFixed(4) ?? 'N/A'}</span>
        </div>
      ))}
    </div>
  )
}

export default function MetricsBreakdown({ data }: Props) {
  const hasMetrics = data.some((d) => d.faithfulness != null)

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-slate-200">
          Per-Metric Scores
          <span className="ml-2 text-xs font-normal text-slate-500">Ragas breakdown</span>
        </h3>
        <p className="text-xs text-slate-500 mt-0.5">
          Like per-task metrics — each Ragas dimension tracked independently
        </p>
      </div>

      {!hasMetrics ? (
        <div className="flex items-center justify-center h-40">
          <p className="text-slate-600 text-sm">Waiting for Ragas evaluation...</p>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="round"
              tick={{ fill: '#64748b', fontSize: 11 }}
              label={{ value: 'Round', position: 'insideBottom', offset: -2, fill: '#475569', fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickFormatter={(v) => v.toFixed(1)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
              formatter={(value) => <span style={{ color: '#94a3b8' }}>{value}</span>}
            />
            {METRIC_CONFIG.map((m) => (
              <Line
                key={m.key}
                type="monotone"
                dataKey={m.key}
                name={m.label}
                stroke={m.color}
                strokeWidth={1.5}
                dot={{ r: 3, fill: m.color }}
                activeDot={{ r: 5 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
