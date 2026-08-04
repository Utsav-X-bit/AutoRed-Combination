import {
  useCallback,
  useEffect,
  useState,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import TimelineSidebar from '../components/TimelineSidebar';
import GeneratorCard from '../components/GeneratorCard';
import VictimCard from '../components/VictimCard';
import ExtractorCard from '../components/ExtractorCard';
import VerifierCard from '../components/VerifierCard';
import AnalyticsPanel from '../components/AnalyticsPanel';
import InvestigationTabs from '../components/InvestigationTabs';
import PlannerInsightsPanel from '../components/PlannerInsightsPanel';
import ResizeHandle from '../components/ResizeHandle';
import { isRunSuccessful } from '../utils/success';

const DEFAULT_LEFT_WIDTH = 280;
const DEFAULT_RIGHT_WIDTH = 320;
const DEFAULT_BOTTOM_HEIGHT = 360;
const MIN_SIDE_WIDTH = 200;
const MIN_CENTER_WIDTH = 420;
const MIN_BOTTOM_HEIGHT = 160;
const MIN_MAIN_HEIGHT = 220;

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), Math.max(min, max));

const readStoredSize = (key: string, fallback: number) => {
  if (typeof window === 'undefined') return fallback;
  const value = Number(window.localStorage.getItem(key));
  return Number.isFinite(value) ? value : fallback;
};

export default function InvestigationPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { selectedRun, selectedAttemptIndex, setSelectedRun, clearSelectedRun } = useRunStore();
  const [loadError, setLoadError] = useState<string | null>(null);
  const [leftWidth, setLeftWidth] = useState(() =>
    readStoredSize('autored.layout.leftWidth', DEFAULT_LEFT_WIDTH),
  );
  const [rightWidth, setRightWidth] = useState(() =>
    readStoredSize('autored.layout.rightWidth', DEFAULT_RIGHT_WIDTH),
  );
  const [bottomHeight, setBottomHeight] = useState(() =>
    readStoredSize('autored.layout.bottomHeight', DEFAULT_BOTTOM_HEIGHT),
  );
  const [showTimeline, setShowTimeline] = useState(true);
  const [showPlanner, setShowPlanner] = useState(true);
  const [showAnalytics, setShowAnalytics] = useState(true);
  const [showTabs, setShowTabs] = useState(true);

  useEffect(() => {
    window.localStorage.setItem('autored.layout.leftWidth', String(leftWidth));
    window.localStorage.setItem('autored.layout.rightWidth', String(rightWidth));
    window.localStorage.setItem('autored.layout.bottomHeight', String(bottomHeight));
  }, [leftWidth, rightWidth, bottomHeight]);

  const resizeFromPointer = useCallback((
    event: ReactPointerEvent<HTMLDivElement>,
    axis: 'x' | 'y',
    currentSize: number,
    applyDelta: (startSize: number, delta: number) => void,
  ) => {
    event.preventDefault();
    const startPosition = axis === 'x' ? event.clientX : event.clientY;
    const cursor = axis === 'x' ? 'col-resize' : 'row-resize';
    document.body.classList.add('is-resizing');
    document.body.style.cursor = cursor;

    const handleMove = (pointerEvent: PointerEvent) => {
      const position = axis === 'x' ? pointerEvent.clientX : pointerEvent.clientY;
      applyDelta(currentSize, position - startPosition);
    };
    const handleEnd = () => {
      document.body.classList.remove('is-resizing');
      document.body.style.cursor = '';
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleEnd);
      window.removeEventListener('pointercancel', handleEnd);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleEnd);
    window.addEventListener('pointercancel', handleEnd);
  }, []);

  const updateLeftWidth = useCallback((value: number) => {
    const max = window.innerWidth - rightWidth - MIN_CENTER_WIDTH;
    setLeftWidth(clamp(value, MIN_SIDE_WIDTH, max));
  }, [rightWidth]);

  const updateRightWidth = useCallback((value: number) => {
    const max = window.innerWidth - leftWidth - MIN_CENTER_WIDTH;
    setRightWidth(clamp(value, MIN_SIDE_WIDTH, max));
  }, [leftWidth]);

  const updateBottomHeight = useCallback((value: number) => {
    const max = window.innerHeight - MIN_MAIN_HEIGHT;
    setBottomHeight(clamp(value, MIN_BOTTOM_HEIGHT, max));
  }, []);

  const resetLayout = useCallback(() => {
    setLeftWidth(DEFAULT_LEFT_WIDTH);
    setRightWidth(DEFAULT_RIGHT_WIDTH);
    setBottomHeight(DEFAULT_BOTTOM_HEIGHT);
  }, []);

  useEffect(() => {
    if (!runId) return;
    clearSelectedRun();
    setLoadError(null);

    fetch(`/api/run/${encodeURIComponent(runId)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.json();
      })
      .then((data) => {
        if (!data.attempts || !Array.isArray(data.attempts)) {
          setLoadError('Invalid run data: missing attempts. The run file may be corrupted.');
          return;
        }
        setSelectedRun(data);
      })
      .catch((err) => {
        setLoadError(`Failed to load run: ${err.message}`);
      });
  }, [runId, clearSelectedRun, setSelectedRun]);

  if (loadError) {
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center bg-stone-50 dark:bg-stone-950">
        <div className="text-center">
          <p className="font-display font-semibold text-rose-600 dark:text-rose-400">Error Loading Run</p>
          <p className="mt-2 text-sm text-stone-500 dark:text-stone-400">{loadError}</p>
          <button
            onClick={() => navigate('/runs')}
            className="mt-4 rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-800 dark:bg-teal-600 dark:hover:bg-teal-500"
          >
            ← Back to Runs
          </button>
        </div>
      </div>
    );
  }

  if (!selectedRun) {
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center bg-stone-50 dark:bg-stone-950">
        <div className="flex items-center gap-3 text-stone-500 dark:text-stone-400">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-stone-300 border-t-teal-600 dark:border-stone-700 dark:border-t-teal-500" />
          Loading run…
        </div>
      </div>
    );
  }

  const attempt = selectedRun.attempts?.[selectedAttemptIndex];
  const runSucceeded = isRunSuccessful(selectedRun);
  const hasPlannerData = selectedRun.attempts.some((item) => Boolean(item.generator.plan_raw));
  if (!attempt) {
    return (
      <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center bg-stone-50 dark:bg-stone-950">
        <div className="text-center">
          <p className="font-display font-semibold text-amber-600 dark:text-amber-400">No Attempt Data</p>
          <p className="mt-2 text-sm text-stone-500 dark:text-stone-400">
            Attempt {selectedAttemptIndex + 1} not found (total: {selectedRun.attempts?.length || 0})
          </p>
          <button
            onClick={() => navigate('/runs')}
            className="mt-4 rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-800 dark:bg-teal-600 dark:hover:bg-teal-500"
          >
            ← Back to Runs
          </button>
        </div>
      </div>
    );
  }

  const ToggleButton = ({
    active,
    onClick,
    label,
    title,
  }: {
    active: boolean;
    onClick: () => void;
    label: string;
    title?: string;
  }) => (
    <button
      onClick={onClick}
      title={title}
      className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${
        active
          ? 'bg-stone-900 text-white dark:bg-teal-600 dark:text-white'
          : 'bg-white text-stone-600 ring-1 ring-stone-200 hover:bg-stone-50 dark:bg-stone-900 dark:text-stone-300 dark:ring-stone-800 dark:hover:bg-stone-800'
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="h-[calc(100vh-3.5rem)] flex flex-col bg-stone-50 dark:bg-stone-950">
      {/* Top Bar */}
      <header className="flex flex-shrink-0 items-center justify-between gap-3 border-b border-stone-200 bg-white px-4 py-2.5 dark:border-stone-800 dark:bg-stone-900">
        <div className="flex min-w-0 items-center gap-3">
          <button
            onClick={() => navigate('/runs')}
            className="flex items-center gap-1 rounded-lg px-2 py-1 text-sm font-medium text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-900 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-100"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 18l-6-6 6-6" /></svg>
            Runs
          </button>
          <span className="text-stone-300 dark:text-stone-700">|</span>
          <h1 className="font-mono text-sm font-semibold text-stone-900 dark:text-stone-100 truncate">
            {selectedRun.experiment.run_id}
          </h1>
          <span className="hidden rounded-full bg-stone-100 px-2 py-0.5 font-mono text-[10px] text-stone-600 dark:bg-stone-800 dark:text-stone-400 sm:inline">
            {selectedRun.experiment.scenario_id}
          </span>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
              runSucceeded
                ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400'
                : 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400'
            }`}
          >
            {runSucceeded ? 'SUCCESS' : 'FAILED'}
          </span>
        </div>

        <div className="flex items-center gap-2 text-sm text-stone-500 dark:text-stone-400">
          <span className="hidden sm:inline">Attempt</span>
          <span className="font-mono font-medium text-stone-900 dark:text-stone-100">
            {attempt.attempt_number}/{selectedRun.result.total_attempts}
          </span>
        </div>

        <div className="flex flex-shrink-0 flex-wrap items-center gap-2">
          <ToggleButton active={showTimeline} onClick={() => setShowTimeline((v) => !v)} label="Timeline" title="Show or hide the attempt timeline" />
          <ToggleButton active={showPlanner} onClick={() => setShowPlanner((v) => !v)} label="Planner" title={hasPlannerData ? 'Show or hide planner analysis' : 'Legacy run: planner output is inferred'} />
          <ToggleButton active={showAnalytics} onClick={() => setShowAnalytics((v) => !v)} label="Analytics" title="Show or hide analytics panel" />
          <ToggleButton active={showTabs} onClick={() => setShowTabs((v) => !v)} label="Tabs" title="Show or hide deep analysis tabs" />
          <button
            onClick={resetLayout}
            className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-900 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-100"
            title="Reset all panel sizes"
          >
            Reset
          </button>
          <div className="h-5 w-px bg-stone-200 dark:bg-stone-800 mx-1" />
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/json`}
            download={`${selectedRun.experiment.run_id}.json`}
            className="rounded-lg bg-white px-2.5 py-1.5 text-xs font-medium text-stone-700 ring-1 ring-stone-200 transition-colors hover:bg-stone-50 dark:bg-stone-900 dark:text-stone-300 dark:ring-stone-800 dark:hover:bg-stone-800"
          >
            JSON
          </a>
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/csv`}
            download={`${selectedRun.experiment.run_id}.csv`}
            className="rounded-lg bg-white px-2.5 py-1.5 text-xs font-medium text-stone-700 ring-1 ring-stone-200 transition-colors hover:bg-stone-50 dark:bg-stone-900 dark:text-stone-300 dark:ring-stone-800 dark:hover:bg-stone-800"
          >
            CSV
          </a>
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/html`}
            download={`${selectedRun.experiment.run_id}.html`}
            className="rounded-lg bg-white px-2.5 py-1.5 text-xs font-medium text-stone-700 ring-1 ring-stone-200 transition-colors hover:bg-stone-50 dark:bg-stone-900 dark:text-stone-300 dark:ring-stone-800 dark:hover:bg-stone-800"
          >
            HTML
          </a>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex min-h-0 flex-1 flex-row overflow-hidden">
          {showTimeline && (
            <>
              <aside
                className="flex-shrink-0 min-w-0 overflow-hidden"
                style={{ width: leftWidth }}
              >
                <TimelineSidebar />
              </aside>
              <ResizeHandle
                direction="vertical"
                label="Resize attempt timeline"
                onPointerDown={(event) => resizeFromPointer(
                  event,
                  'x',
                  leftWidth,
                  (start, delta) => updateLeftWidth(start + delta),
                )}
                onKeyboardResize={(delta) => updateLeftWidth(leftWidth + delta)}
                onReset={() => setLeftWidth(DEFAULT_LEFT_WIDTH)}
              />
            </>
          )}

          <main className="min-w-0 flex-1 overflow-y-auto bg-stone-50 p-4 dark:bg-stone-950 lg:p-5">
            <div className="mx-auto w-full max-w-5xl space-y-4">
              {showPlanner && (
                <PlannerInsightsPanel run={selectedRun} selectedAttemptIndex={selectedAttemptIndex} />
              )}

              {/* Attempt Header */}
              <div className="flex items-center justify-between rounded-xl border border-stone-200 bg-white px-4 py-3 shadow-sm dark:border-stone-800 dark:bg-stone-900">
                <div>
                  <h2 className="font-display text-lg font-semibold text-stone-900 dark:text-stone-100">
                    Attempt {attempt.attempt_number}
                  </h2>
                  <p className="text-xs text-stone-500 dark:text-stone-400">{formatDateTime(attempt.timestamp)}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-purple-100 px-2.5 py-1 text-xs font-semibold text-purple-700 dark:bg-purple-950/50 dark:text-purple-300">
                    {attempt.generator.strategy}
                  </span>
                  <span
                    className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                      attempt.judge.decision === 'ATTACK'
                        ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400'
                        : 'bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-400'
                    }`}
                  >
                    {attempt.judge.decision} ({attempt.judge.confidence.toFixed(2)})
                  </span>
                </div>
              </div>

              {/* Pipeline */}
              <GeneratorCard attempt={attempt} />
              <div className="flex justify-center py-1 text-stone-300 dark:text-stone-700">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M19 12l-7 7-7-7" /></svg>
              </div>
              <VictimCard attempt={attempt} accessCode={selectedRun.scenario.access_code} />
              <div className="flex justify-center py-1 text-stone-300 dark:text-stone-700">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M19 12l-7 7-7-7" /></svg>
              </div>
              <ExtractorCard attempt={attempt} />
              <div className="flex justify-center py-1 text-stone-300 dark:text-stone-700">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M19 12l-7 7-7-7" /></svg>
              </div>
              <VerifierCard attempt={attempt} />
            </div>
          </main>

          {showAnalytics && (
            <>
              <ResizeHandle
                direction="vertical"
                label="Resize analytics panel"
                onPointerDown={(event) => resizeFromPointer(
                  event,
                  'x',
                  rightWidth,
                  (start, delta) => updateRightWidth(start - delta),
                )}
                onKeyboardResize={(delta) => updateRightWidth(rightWidth - delta)}
                onReset={() => setRightWidth(DEFAULT_RIGHT_WIDTH)}
              />
              <aside
                className="flex-shrink-0 min-w-0 overflow-hidden"
                style={{ width: rightWidth }}
              >
                <AnalyticsPanel />
              </aside>
            </>
          )}
        </div>

        {showTabs && (
          <>
            <ResizeHandle
              direction="horizontal"
              label="Resize investigation tabs"
              onPointerDown={(event) => resizeFromPointer(
                event,
                'y',
                bottomHeight,
                (start, delta) => updateBottomHeight(start - delta),
              )}
              onKeyboardResize={(delta) => updateBottomHeight(bottomHeight - delta)}
              onReset={() => setBottomHeight(DEFAULT_BOTTOM_HEIGHT)}
            />
            <section
              className="flex-shrink-0 min-h-0 overflow-hidden border-t border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900"
              style={{ height: bottomHeight }}
            >
              <InvestigationTabs />
            </section>
          </>
        )}
      </div>
    </div>
  );
}

function formatDateTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? 'n/a'
    : date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
