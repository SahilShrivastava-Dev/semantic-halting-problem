import { useCallback, useRef, useState } from 'react'
import type { SHPEvent, RunConfig } from '../types'

type Status = 'idle' | 'connecting' | 'running' | 'complete' | 'error'

interface UseWebSocketReturn {
  status: Status
  connect: (config: RunConfig, onEvent: (event: SHPEvent) => void) => void
  disconnect: () => void
  error: string | null
}

const WS_URL = 'ws://localhost:8000/ws'

export function useWebSocket(): UseWebSocketReturn {
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
    (config: RunConfig, onEvent: (event: SHPEvent) => void) => {
      // Clean up any existing connection
      if (wsRef.current) {
        wsRef.current.close()
      }

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
          const event: SHPEvent = JSON.parse(msgEvent.data)

          if (event.type === 'scenario_complete') {
            setStatus('complete')
          } else if (event.type === 'error') {
            setError((event as { type: 'error'; message: string }).message)
            setStatus('error')
          }

          onEvent(event)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.onerror = () => {
        setError('WebSocket connection failed. Make sure the backend is running on port 8000.')
        setStatus('error')
      }

      ws.onclose = (evt) => {
        wsRef.current = null
        if (evt.code !== 1000 && status === 'running') {
          setStatus('error')
          setError('Connection closed unexpectedly.')
        }
      }
    },
    [status]
  )

  return { status, connect, disconnect, error }
}
