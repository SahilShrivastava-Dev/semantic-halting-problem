import React, { useState } from 'react'
import { FileText, MessageSquare, ChevronDown, ChevronUp } from 'lucide-react'

interface Props {
  draftPreview: string
  criticFeedback: string
  round: number
}

export default function DraftViewer({ draftPreview, criticFeedback, round }: Props) {
  const [showDraft, setShowDraft] = useState(false)

  if (!draftPreview && !criticFeedback) return null

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-slate-700 bg-slate-800/80">
        <span className="text-xs font-semibold text-slate-400">
          Latest Agent Output — Round {round}
        </span>
      </div>

      <div className="divide-y divide-slate-700">
        {/* Draft preview */}
        {draftPreview && (
          <div className="p-4">
            <div
              className="flex items-center gap-2 cursor-pointer"
              onClick={() => setShowDraft((v) => !v)}
            >
              <FileText size={14} className="text-blue-400" />
              <span className="text-xs font-medium text-blue-400">Writer Draft</span>
              <div className="ml-auto text-slate-600">
                {showDraft ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </div>
            </div>
            {showDraft && (
              <p className="mt-3 text-xs text-slate-300 leading-relaxed whitespace-pre-wrap font-mono bg-slate-900/50 rounded-lg p-3">
                {draftPreview}
                {draftPreview.length >= 300 && (
                  <span className="text-slate-600"> …[truncated]</span>
                )}
              </p>
            )}
            {!showDraft && (
              <p className="mt-2 text-xs text-slate-400 line-clamp-2 leading-relaxed">
                {draftPreview}
              </p>
            )}
          </div>
        )}

        {/* Critic feedback */}
        {criticFeedback && (
          <div className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <MessageSquare size={14} className={
                criticFeedback.toUpperCase().startsWith('APPROVED')
                  ? 'text-green-400'
                  : 'text-yellow-400'
              } />
              <span className={`text-xs font-medium ${
                criticFeedback.toUpperCase().startsWith('APPROVED')
                  ? 'text-green-400'
                  : 'text-yellow-400'
              }`}>
                {criticFeedback.toUpperCase().startsWith('APPROVED') ? 'Critic: APPROVED' : 'Critic Feedback'}
              </span>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">{criticFeedback}</p>
          </div>
        )}
      </div>
    </div>
  )
}
