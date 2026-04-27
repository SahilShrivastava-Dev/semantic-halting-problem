import { useCallback, useRef, useState } from 'react'
import type { PipelineEvent, PipelineConfig } from '../types'

type Status = 'idle' | 'connecting' | 'running' | 'complete' | 'error'

interface UsePipelineWebSocketReturn {
  status: Status
  connect: (config: PipelineConfig, onEvent: (event: PipelineEvent) => void) => void
  disconnect: () => void
  error: string | null
}

const WS_URL = 'ws://localhost:8000/ws/pipeline'

export function usePipelineWebSocket(): UsePipelineWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setStatus('idle')
  }, [])

  const connect = useCallback(
    (config: PipelineConfig, onEvent: (event: PipelineEvent) => void) => {
      if (wsRef.current) wsRef.current.close()
      setStatus('connecting')
      setError(null)

      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setStatus('running')
        ws.send(JSON.stringify(config))
      }

      ws.onmessage = (msgEvent) => {
        try {
          const event: PipelineEvent = JSON.parse(msgEvent.data)
          if (event.type === 'pipeline_complete') {
            setStatus('complete')
          } else if (event.type === 'error') {
            setError((event as { type: 'error'; message: string }).message)
            setStatus('error')
          }
          onEvent(event)
        } catch (e) {
          console.error('Failed to parse pipeline WebSocket message:', e)
        }
      }

      ws.onerror = () => {
        setError('Pipeline WebSocket connection failed. Make sure the backend is running on port 8000.')
        setStatus('error')
      }

      ws.onclose = () => {
        wsRef.current = null
      }
    },
    []
  )

  return { status, connect, disconnect, error }
}
