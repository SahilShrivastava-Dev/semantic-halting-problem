/**
 * WeightsChart — Learned IS metric weights.
 *
 * ML analogy: Visualising learned model parameters (weight magnitudes).
 * Shows which Ragas metrics matter most after linear regression calibration.
 */

import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LabelList,
} from 'recharts'

const METRIC_COLORS: Record<string, string> = {
  faithfulness:      '#a78bfa',
  answer_relevancy:  '#60a5fa',
  context_precision: '#34d399',
  context_recall:    '#f59e0b',
}

const METRIC_SHORT: Record<string, string> = {
  faithfulness:      'Faith.',
  answer_relevancy:  'Relevancy',
  context_precision: 'Precision',
  context_recall:    'Recall',
}

interface Props {
  weights: Record<string, number>
  r2Score: number | null
  dataSource?: string
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-300 font-medium">{payload[0]?.payload?.metric}</p>
      <p className="font-mono text-purple-400 font-semibold">
        Weight: {(payload[0]?.value * 100).toFixed(1)}%
      </p>
    </div>
  )
}

export default function WeightsChart({ weights, r2Score, dataSource }: Props) {
  const chartData = Object.entries(weights).map(([metric, weight]) => ({
    metric: METRIC_SHORT[metric] ?? metric,
    full_metric: metric,
    weight,
  }))

  return (
    <div className="bg-slate-800 rounded-xl p-4 border border-slate-700">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-200">
            Learned IS Weights
            <span className="ml-2 text-xs font-normal text-slate-500">w₁…w₄</span>
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Like model parameters — learned via linear regression
          </p>
        </div>
        <div className="text-right">
          {r2Score != null ? (
            <>
              <div className={`text-sm font-bold font-mono ${r2Score >= 0.75 ? 'text-green-400' : r2Score >= 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
                R² = {r2Score.toFixed(3)}
              </div>
              <div className="text-xs text-slate-500">{dataSource ?? 'calibrated'}</div>
            </>
          ) : (
            <div className="text-xs text-slate-500">equal default</div>
          )}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 0, right: 40, left: 10, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, 1]}
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <YAxis
            type="category"
            dataKey="metric"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            width={65}
          />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="weight" radius={[0, 4, 4, 0]} barSize={22}>
            <LabelList
              dataKey="weight"
              position="right"
              formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              style={{ fill: '#94a3b8', fontSize: 11 }}
            />
            {chartData.map((entry, index) => (
              <Cell
                key={index}
                fill={METRIC_COLORS[entry.full_metric] ?? '#6366f1'}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
