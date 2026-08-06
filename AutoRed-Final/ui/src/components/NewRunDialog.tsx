import { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import type { AutoRedRun } from '../types/autored';

interface NewRunDialogProps {
  onClose: () => void;
  onSuccess: () => void;
}

export default function NewRunDialog({ onClose, onSuccess }: NewRunDialogProps) {
  const navigate = useNavigate();
  const { setSelectedRun } = useRunStore();

  const [maxAttempts, setMaxAttempts] = useState(20);
  const [scenarioId, setScenarioId] = useState('');
  const [running, setRunning] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [currentAttempt, setCurrentAttempt] = useState(0);
  const [totalAttempts, setTotalAttempts] = useState(0);
  const [success, setSuccess] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const wsConnected = useRef(false);
  const runCompleted = useRef(false);

  const connectWebSocket = useCallback((rid: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/run/${rid}`;
    console.log(`[NewRun WS] Opening WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    wsConnected.current = false;
    runCompleted.current = false;

    ws.onopen = () => {
      wsConnected.current = true;
      console.log(`[NewRun WS] ✓ WebSocket opened for run_id=${rid}`);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log(`[NewRun WS] Received message type=${data.type}, run_id=${data.run_id}`);

        if (data.type === 'attempt_update') {
          const attemptNum = data.attempt.attempt_number;
          console.log(`[NewRun WS] Attempt #${attemptNum} received, ground_truth_found=${data.attempt.ground_truth_found}`);
          setCurrentAttempt(attemptNum);
          setTotalAttempts(attemptNum);
          if (data.attempt.ground_truth_found) {
            console.log('[NewRun WS] ✓ Ground truth found in this attempt!');
            setSuccess(true);
          }
        } else if (data.type === 'run_complete') {
          runCompleted.current = true;
          console.log('[NewRun WS] ✓ run_complete received');
          const rawRun = (data as { run: unknown }).run;
          if (rawRun && typeof rawRun === 'object' && 'error' in rawRun) {
            console.error('[NewRun WS] Run completed with error:', (rawRun as { error: string }).error);
            setError((rawRun as { error: string }).error);
          } else {
            const run = rawRun as AutoRedRun;
            console.log('[NewRun WS] Run complete - success:', run.result?.ground_truth_success, 'attempts:', run.result?.total_attempts);
            console.log('[NewRun WS] Run attempts array length:', run.attempts?.length);
            setSuccess(Boolean(
              run.result?.ground_truth_success
              || run.result?.extractor_success
              || run.result?.verified_success
            ));
            setTotalAttempts(run.result?.total_attempts ?? 0);
            setSelectedRun(run);
          }
          ws.close();
          wsRef.current = null;
          setRunning(false);
        }
      } catch (e) {
        console.error('[NewRun WS] Message parse error:', e);
      }
    };

    ws.onclose = (event) => {
      console.log(`[NewRun WS] WebSocket closed: code=${event.code}, reason=${event.reason}`);
      const wasConnected = wsConnected.current;
      wsRef.current = null;
      wsConnected.current = false;
      if (running && wasConnected && !runCompleted.current && !error) {
        setError('Live connection closed before the run completed.');
        setRunning(false);
      }
    };

    ws.onerror = (err) => {
      console.error('[NewRun WS] WebSocket error:', err);
      // Only surface error if we never connected (server unreachable)
      if (!wsConnected.current) {
        setError('Cannot connect to server. Is the backend running on port 8001?');
        setRunning(false);
      }
    };
  }, [setSelectedRun, running, error]);

  const handleStart = async () => {
    setRunning(true);
    setError(null);
    setSuccess(null);
    setCurrentAttempt(0);

    try {
      console.log('[NewRun] Starting run flow...');

      // Health check first
      console.log('[NewRun] Checking server health...');
      const healthRes = await fetch('/api/models/status');
      if (!healthRes.ok) {
        console.error('[NewRun] Health check failed:', healthRes.status);
        setError('Cannot reach server. Is the backend running on port 8001?');
        setRunning(false);
        return;
      }

      const health = await healthRes.json();
      console.log('[NewRun] Health check passed:', health);
      if (!health.victim?.loaded) {
        setError('Server is starting up — models are still loading. Wait a moment and try again.');
        setRunning(false);
        return;
      }

      // Generate run_id client-side — we'll use THIS same ID for both WebSocket and POST
      const rid = `run_${Date.now()}`;
      setRunId(rid);
      console.log(`[NewRun] Generated run_id: ${rid}`);

      // Step 1: Connect WebSocket FIRST (before experiment starts)
      console.log(`[NewRun] Connecting WebSocket to /ws/run/${rid}...`);
      connectWebSocket(rid);

      // Wait for WebSocket to open before starting experiment
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('WebSocket timeout')), 5000);
        const checkWs = setInterval(() => {
          if (wsConnected.current) {
            clearTimeout(timeout);
            clearInterval(checkWs);
            console.log('[NewRun] ✓ WebSocket connected');
            resolve();
          }
        }, 50);
      });

      // Step 2: Start experiment, passing the SAME run_id for WebSocket routing
      const params = new URLSearchParams({
        run_id: rid,  // CRITICAL: pass client run_id so server routes WS messages correctly
        max_attempts: String(maxAttempts),
        ...(scenarioId ? { scenario_id: scenarioId } : {}),
      });

      console.log(`[NewRun] Starting experiment via POST /api/run?${params}`);
      const res = await fetch(`/api/run?${params}`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        console.error('[NewRun] POST failed:', res.status, err);
        setError(err.detail || `Server error (${res.status})`);
        setRunning(false);
        wsRef.current?.close();
        return;
      }

      const data = await res.json();
      console.log('[NewRun] POST response:', data);

      // Verify server used our run_id
      if (data.run_id && data.run_id !== rid) {
        console.warn(`[NewRun] Server returned different run_id: ${data.run_id} (expected ${rid}). Reconnecting WebSocket...`);
        setRunId(data.run_id);
        if (wsRef.current) wsRef.current.close();
        connectWebSocket(data.run_id);
      } else {
        console.log(`[NewRun] ✓ Server confirmed run_id: ${data.run_id}`);
      }
    } catch (e) {
      console.error('[NewRun] Exception during start:', e);
      setError('Cannot connect to server. Check that the backend is running and SSH tunnel is active.');
      setRunning(false);
    }
  };

  const handleDone = () => {
    if (wsRef.current) wsRef.current.close();
    if (runId && !error) {
      navigate(`/run/${runId}`);
    } else {
      onSuccess();
    }
  };

  const handleCancel = () => {
    if (wsRef.current && running) {
      if (!confirm('Run is in progress. Cancel?')) return;
      wsRef.current.close();
    }
    onClose();
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const progress = maxAttempts > 0 ? (currentAttempt / maxAttempts) * 100 : 0;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-200">
          <h2 className="text-lg font-bold text-slate-900">
            {running ? 'Running Experiment...' : 'New Experiment Run'}
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            {running
              ? 'Models are pre-loaded — run will start immediately'
              : 'Start a new AutoRed attack scenario with pre-loaded models'}
          </p>
        </div>

        {/* Body */}
        {!running ? (
          <div className="px-6 py-5 space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Max Attempts
              </label>
              <input
                type="number"
                min={1}
                max={100}
                value={maxAttempts}
                onChange={(e) => setMaxAttempts(Number(e.target.value))}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Scenario ID <span className="text-slate-400">(optional, random if empty)</span>
              </label>
              <input
                type="text"
                value={scenarioId}
                onChange={(e) => setScenarioId(e.target.value)}
                placeholder="e.g., 89021"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        ) : (
          <div className="px-6 py-8">
            {/* Progress bar */}
            <div className="mb-4">
              <div className="flex items-center justify-between text-sm text-slate-600 mb-1">
                <span>Attempt {currentAttempt} / {maxAttempts}</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(progress, 100)}%` }}
                />
              </div>
            </div>

            {runId && (
              <p className="text-xs text-slate-500 text-center font-mono mb-4">
                Run ID: {runId}
              </p>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            {success !== null && !error && (
              <div className={`rounded-lg p-4 text-center ${success ? 'bg-green-50' : 'bg-slate-50'}`}>
                <p className={`text-lg font-bold ${success ? 'text-green-700' : 'text-slate-700'}`}>
                  {success ? '✓ Success — Access Code Extracted' : '✗ Failed — Max Attempts Reached'}
                </p>
                <p className="text-sm text-slate-500 mt-1">
                  Total attempts: {totalAttempts}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-200 flex items-center justify-end gap-3">
          {!running ? (
            <>
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleStart}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
              >
                Start Run
              </button>
            </>
          ) : (
            <>
              <button
                onClick={handleCancel}
                className="px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                Cancel
              </button>
              {(success !== null || error) && (
                <button
                  onClick={handleDone}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
                >
                  {success ? 'View Results' : 'Done'}
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
