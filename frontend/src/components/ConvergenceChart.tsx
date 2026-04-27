/**
 * ConvergenceChart — IS Score over rounds.
 *
 * ML analogy: Training accuracy curve. Rises as the model (Writer) improves.
 * When it plateaus (Δ ≈ 0), we've reached the semantic fixed-point.
 */

import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Dot,
} from 'recharts'
import type { RoundDataPoint } from '../types'

interface Props {
  data: RoundDataPoint[]
  haltRound?: number
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">Round {label}</p>
      <p className="text-blue-400 font-mono font-semibold">
        IS Score: {payload[0]?.value?.toFixed(4)}
      </p>
    </div>
  )
}

export default function ConvergenceChart({ data, haltRound }: Props) {
  const isEmpty = data.length === 0

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">
            Information Score
            <span className="ml-2 text-xs font-normal text-slate-500">↑ higher is better</span>
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Like training accuracy — rises toward convergence
          </p>
        </div>
        {data.length > 0 && (
          <div className="text-right">
            <div className="text-lg font-bold font-mono text-blue-400">
              {data[data.length - 1].is_score.toFixed(4)}
            </div>
            <div className="text-xs text-slate-500">latest</div>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={180}>
        {isEmpty ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-600 text-sm">Waiting for first round...</p>
          </div>
        ) : (
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
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip content={<CustomTooltip />} />
            {haltRound != null && (
              <ReferenceLine
                x={haltRound}
                stroke="#f97316"
                strokeDasharray="4 2"
                label={{ value: 'HALT', position: 'top', fill: '#f97316', fontSize: 10 }}
              />
            )}
            <Line
              type="monotone"
              dataKey="is_score"
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={(props) => {
                const isLast = props.index === data.length - 1
                return (
                  <Dot
                    {...props}
                    r={isLast ? 5 : 3}
                    fill={isLast ? '#3b82f6' : '#1d4ed8'}
                    stroke={isLast ? '#93c5fd' : 'none'}
                    strokeWidth={isLast ? 2 : 0}
                  />
                )
              }}
              activeDot={{ r: 6, fill: '#60a5fa' }}
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  )
}
