import { useEffect, useRef, useCallback } from 'react';
import { useRunStore } from '../store/runStore';
import type { AttemptUpdate, RunComplete } from '../types/autored';

export function useWebSocket(runId: string | undefined) {
  const wsRef = useRef<WebSocket | null>(null);
  const { addAttempt, setSelectedRun } = useRunStore();

  const connect = useCallback(() => {
    if (!runId) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/run/${runId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'attempt_update') {
          const msg = data as AttemptUpdate;
          addAttempt(msg.attempt);
        } else if (data.type === 'run_complete') {
          const msg = data as RunComplete;
          setSelectedRun(msg.run);
          ws.close();
          wsRef.current = null;
        }
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };

    ws.onclose = () => { wsRef.current = null; };
    ws.onerror = (error) => { console.error('WebSocket error:', error); };
  }, [runId, addAttempt, setSelectedRun]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => { disconnect(); };
  }, [disconnect]);

  return { connect, disconnect, isConnected: wsRef.current?.readyState === WebSocket.OPEN };
}
