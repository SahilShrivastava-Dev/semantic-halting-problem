/**
 * ISGainChart — IS gain (delta) per round.
 *
 * ML analogy: Gradient magnitude per epoch.
 * Positive = still learning. Zero/negative = gradient vanished = halt.
 */

import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts'
import type { RoundDataPoint } from '../types'

interface Props {
  data: RoundDataPoint[]
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  const val = payload[0]?.value
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">Round {label}</p>
      <p className={`font-mono font-semibold ${val >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        IS Gain: {val >= 0 ? '+' : ''}{val?.toFixed(4)}
      </p>
      <p className="text-xs text-slate-500 mt-0.5">
        {val > 0 ? 'Information added' : val === 0 ? 'No new information' : 'Information lost'}
      </p>
    </div>
  )
}

export default function ISGainChart({ data }: Props) {
  const gainData = data
    .filter((d) => d.is_gain != null)
    .map((d) => ({ ...d, is_gain: d.is_gain! }))

  const isEmpty = gainData.length === 0

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">
            IS Gain per Round
            <span className="ml-2 text-xs font-normal text-slate-500">Δ IS</span>
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Like gradient magnitude — halts when it hits zero
          </p>
        </div>
        {gainData.length > 0 && (
          <div className="text-right">
            <div className={`text-lg font-bold font-mono ${
              gainData[gainData.length - 1].is_gain >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {gainData[gainData.length - 1].is_gain >= 0 ? '+' : ''}
              {gainData[gainData.length - 1].is_gain.toFixed(4)}
            </div>
            <div className="text-xs text-slate-500">latest</div>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={180}>
        {isEmpty ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-600 text-sm">Needs 2 rounds to compute...</p>
          </div>
        ) : (
          <BarChart data={gainData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="round"
              tick={{ fill: '#64748b', fontSize: 11 }}
              label={{ value: 'Round', position: 'insideBottom', offset: -2, fill: '#475569', fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickFormatter={(v) => v.toFixed(3)}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0} stroke="#64748b" strokeWidth={1.5} />
            <Bar dataKey="is_gain" radius={[3, 3, 0, 0]}>
              {gainData.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.is_gain >= 0 ? '#22c55e' : '#ef4444'}
                  fillOpacity={0.8}
                />
              ))}
            </Bar>
          </BarChart>
        )}
      </ResponsiveContainer>
    </div>
  )
}
