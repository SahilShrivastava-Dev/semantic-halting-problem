import React, { useEffect, useRef } from 'react'
import { Terminal } from 'lucide-react'
import type { LogEntry } from '../types'

const LEVEL_STYLES: Record<string, string> = {
  info:    'text-slate-300',
  warn:    'text-yellow-400',
  error:   'text-red-400',
  success: 'text-green-400',
}

const LEVEL_PREFIX: Record<string, string> = {
  info:    '·',
  warn:    '⚠',
  error:   '✖',
  success: '✓',
}

interface Props {
  logs: LogEntry[]
  maxHeight?: string
}

export default function LogStream({ logs, maxHeight = '220px' }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  return (
    <div className="bg-slate-900 rounded-xl border border-slate-700 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-slate-700 bg-slate-800/50">
        <Terminal size={14} className="text-slate-500" />
        <span className="text-xs font-medium text-slate-400">Agent Log</span>
        <span className="ml-auto text-xs text-slate-600 font-mono">{logs.length} entries</span>
      </div>

      <div
        className="log-stream overflow-y-auto p-3 font-mono text-xs space-y-0.5"
        style={{ maxHeight }}
      >
        {logs.length === 0 ? (
          <p className="text-slate-600 py-2">Waiting for run to start...</p>
        ) : (
          logs.map((entry) => (
            <div key={entry.id} className={`flex gap-2 leading-5 ${LEVEL_STYLES[entry.level]}`}>
              <span className="shrink-0 text-slate-600">
                {entry.timestamp.slice(11, 19)}
              </span>
              {entry.round != null && (
                <span className="shrink-0 text-blue-500/70">
                  [R{entry.round}]
                </span>
              )}
              <span className="shrink-0">{LEVEL_PREFIX[entry.level]}</span>
              <span className="break-all">{entry.message}</span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
