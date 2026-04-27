/**
 * DistanceChart — Cosine distance over rounds.
 *
 * ML analogy: Training loss curve. Decays toward 0.
 * The convergence threshold line (ε = 0.06) is the "early stopping boundary".
 * When distance crosses below this line for k=2 consecutive rounds, halt fires.
 */

import React from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Dot, ReferenceArea,
} from 'recharts'
import type { RoundDataPoint } from '../types'

const CONVERGENCE_THRESHOLD = 0.06

interface Props {
  data: RoundDataPoint[]
  haltRound?: number
  threshold?: number
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  const val = payload[0]?.value
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">Round {label}</p>
      {val != null ? (
        <>
          <p className="text-orange-400 font-mono font-semibold">
            Distance: {val.toFixed(6)}
          </p>
          <p className={`text-xs mt-0.5 ${val < CONVERGENCE_THRESHOLD ? 'text-green-400' : 'text-slate-500'}`}>
            {val < CONVERGENCE_THRESHOLD ? '⬇ Below threshold' : 'Above threshold'}
          </p>
        </>
      ) : (
        <p className="text-slate-500">No embedding yet</p>
      )}
    </div>
  )
}

export default function DistanceChart({ data, haltRound, threshold = CONVERGENCE_THRESHOLD }: Props) {
  const distData = data.filter((d) => d.distance != null)
  const isEmpty = distData.length === 0
  const latest = distData[distData.length - 1]?.distance

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">
            Cosine Distance
            <span className="ml-2 text-xs font-normal text-slate-500">↓ decays to 0</span>
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Like training loss — measures semantic change between rounds
          </p>
        </div>
        {latest != null && (
          <div className="text-right">
            <div className={`text-lg font-bold font-mono ${latest < threshold ? 'text-green-400' : 'text-orange-400'}`}>
              {latest.toFixed(4)}
            </div>
            <div className="text-xs text-slate-500">
              {latest < threshold ? 'converged' : `ε = ${threshold}`}
            </div>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={180}>
        {isEmpty ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-600 text-sm">Needs 2 rounds to compute...</p>
          </div>
        ) : (
          <LineChart data={distData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            {/* Green zone below threshold */}
            <ReferenceArea y2={threshold} fill="#22c55e" fillOpacity={0.05} />
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
            <ReferenceLine
              y={threshold}
              stroke="#22c55e"
              strokeDasharray="4 2"
              label={{ value: `ε=${threshold}`, position: 'right', fill: '#22c55e', fontSize: 10 }}
            />
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
              dataKey="distance"
              stroke="#f97316"
              strokeWidth={2.5}
              dot={(props) => {
                const val = distData[props.index]?.distance ?? 1
                const isLast = props.index === distData.length - 1
                const color = val < threshold ? '#22c55e' : '#f97316'
                return (
                  <Dot
                    {...props}
                    r={isLast ? 5 : 3}
                    fill={color}
                    stroke={isLast ? '#fed7aa' : 'none'}
                    strokeWidth={isLast ? 2 : 0}
                  />
                )
              }}
              activeDot={{ r: 6, fill: '#fb923c' }}
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  )
}
