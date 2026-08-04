import { useCallback, useEffect, useRef, useState } from 'react';
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

  const connectWebSocket = useCallback(
    (rid: string) => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/run/${rid}`;
      console.log(`[NewRun WS] Opening WebSocket: ${wsUrl}`);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      wsConnected.current = false;
      runCompleted.current = false;

      ws.onopen = () => {
        wsConnected.current = true;
        console.log(`[NewRun WS] WebSocket opened for run_id=${rid}`);
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
              console.log('[NewRun WS] Ground truth found in this attempt!');
              setSuccess(true);
            }
          } else if (data.type === 'run_complete') {
            runCompleted.current = true;
            console.log('[NewRun WS] run_complete received');
            const rawRun = (data as { run: unknown }).run;
            if (rawRun && typeof rawRun === 'object' && 'error' in rawRun) {
              console.error('[NewRun WS] Run completed with error:', (rawRun as { error: string }).error);
              setError((rawRun as { error: string }).error);
            } else {
              const run = rawRun as AutoRedRun;
              console.log('[NewRun WS] Run complete - success:', run.result?.ground_truth_success, 'attempts:', run.result?.total_attempts);
              setSuccess(
                Boolean(run.result?.ground_truth_success || run.result?.extractor_success || run.result?.verified_success),
              );
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
        if (!wsConnected.current) {
          setError('Cannot connect to server. Is the backend running on port 8001?');
          setRunning(false);
        }
      };
    },
    [setSelectedRun, running, error],
  );

  const handleStart = async () => {
    setRunning(true);
    setError(null);
    setSuccess(null);
    setCurrentAttempt(0);

    try {
      console.log('[NewRun] Starting run flow...');
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

      const rid = `run_${Date.now()}`;
      setRunId(rid);
      console.log(`[NewRun] Generated run_id: ${rid}`);

      console.log(`[NewRun] Connecting WebSocket to /ws/run/${rid}...`);
      connectWebSocket(rid);

      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('WebSocket timeout')), 5000);
        const checkWs = setInterval(() => {
          if (wsConnected.current) {
            clearTimeout(timeout);
            clearInterval(checkWs);
            console.log('[NewRun] WebSocket connected');
            resolve();
          }
        }, 50);
      });

      const params = new URLSearchParams({
        run_id: rid,
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

      if (data.run_id && data.run_id !== rid) {
        console.warn(`[NewRun] Server returned different run_id: ${data.run_id} (expected ${rid}). Reconnecting WebSocket...`);
        setRunId(data.run_id);
        if (wsRef.current) wsRef.current.close();
        connectWebSocket(data.run_id);
      } else {
        console.log(`[NewRun] Server confirmed run_id: ${data.run_id}`);
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
      <div className="w-full max-w-lg overflow-hidden rounded-2xl border border-stone-200 bg-white shadow-2xl dark:border-stone-800 dark:bg-stone-900">
        <div className="border-b border-stone-200 px-6 py-4 dark:border-stone-800">
          <h2 className="font-display text-lg font-semibold text-stone-900 dark:text-stone-100">
            {running ? 'Running Experiment…' : 'New Experiment Run'}
          </h2>
          <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
            {running
              ? 'Models are pre-loaded — run will start immediately.'
              : 'Start a new AutoRed attack scenario with pre-loaded models.'}
          </p>
        </div>

        {!running ? (
          <div className="space-y-4 px-6 py-5">
            <div>
              <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
                Max Attempts
              </label>
              <input
                type="number"
                min={1}
                max={100}
                value={maxAttempts}
                onChange={(e) => setMaxAttempts(Number(e.target.value))}
                className="input w-full"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium text-stone-700 dark:text-stone-300">
                Scenario ID <span className="text-stone-400">(optional, random if empty)</span>
              </label>
              <input
                type="text"
                value={scenarioId}
                onChange={(e) => setScenarioId(e.target.value)}
                placeholder="e.g., 89021"
                className="input w-full"
              />
            </div>
          </div>
        ) : (
          <div className="px-6 py-8">
            <div className="mb-4">
              <div className="mb-1 flex items-center justify-between text-sm text-stone-600 dark:text-stone-400">
                <span>
                  Attempt {currentAttempt} / {maxAttempts}
                </span>
                <span>{Math.round(progress)}%</span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-stone-200 dark:bg-stone-800">
                <div
                  className="h-2.5 rounded-full bg-teal-600 transition-all duration-300 dark:bg-teal-500"
                  style={{ width: `${Math.min(progress, 100)}%` }}
                />
              </div>
            </div>

            {runId && <p className="mb-4 text-center font-mono text-xs text-stone-500 dark:text-stone-400">Run ID: {runId}</p>}

            {error && (
              <div className="rounded-lg border border-rose-200 bg-rose-50 p-3 dark:border-rose-900/50 dark:bg-rose-950/20">
                <p className="text-sm text-rose-700 dark:text-rose-300">{error}</p>
              </div>
            )}

            {success !== null && !error && (
              <div className={`rounded-lg p-4 text-center ${success ? 'border border-emerald-200 bg-emerald-50 dark:border-emerald-900/50 dark:bg-emerald-950/20' : 'border border-stone-200 bg-stone-50 dark:border-stone-800 dark:bg-stone-800/50'}`}>
                <p className={`text-lg font-bold ${success ? 'text-emerald-700 dark:text-emerald-400' : 'text-stone-700 dark:text-stone-300'}`}>
                  {success ? 'Success — Access Code Extracted' : 'Failed — Max Attempts Reached'}
                </p>
                <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">Total attempts: {totalAttempts}</p>
              </div>
            )}
          </div>
        )}

        <div className="flex items-center justify-end gap-3 border-t border-stone-200 px-6 py-4 dark:border-stone-800">
          {!running ? (
            <>
              <button onClick={onClose} className="btn-default">Cancel</button>
              <button onClick={handleStart} className="btn-primary">Start Run</button>
            </>
          ) : (
            <>
              <button onClick={handleCancel} className="btn-danger">Cancel</button>
              {(success !== null || error) && (
                <button onClick={handleDone} className="btn-primary">{success ? 'View Results' : 'Done'}</button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
