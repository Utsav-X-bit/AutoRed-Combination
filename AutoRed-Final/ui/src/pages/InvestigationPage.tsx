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

const DEFAULT_LEFT_WIDTH = 256;
const DEFAULT_RIGHT_WIDTH = 288;
const DEFAULT_BOTTOM_HEIGHT = 360;
const MIN_SIDE_WIDTH = 180;
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
    console.log('[InvestigationPage] Loading run:', runId);

    fetch(`/api/run/${encodeURIComponent(runId)}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.json();
      })
      .then((data) => {
        console.log('[InvestigationPage] Run loaded:', {
          run_id: data.experiment?.run_id,
          attempts_count: data.attempts?.length,
          result: data.result,
        });
        if (!data.attempts || !Array.isArray(data.attempts)) {
          console.error('[InvestigationPage] Run data missing attempts array:', data);
          setLoadError('Invalid run data: missing attempts. The run file may be corrupted.');
          return;
        }
        setSelectedRun(data);
      })
      .catch((err) => {
        console.error('[InvestigationPage] Failed to load run:', err);
        setLoadError(`Failed to load run: ${err.message}`);
      });
  }, [runId, clearSelectedRun, setSelectedRun]);

  if (loadError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <p className="text-red-600 font-medium mb-2">Error Loading Run</p>
          <p className="text-sm text-slate-500 mb-4">{loadError}</p>
          <button
            onClick={() => navigate('/runs')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
          >
            ← Back to Runs
          </button>
        </div>
      </div>
    );
  }

  if (!selectedRun) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <p className="text-slate-500">Loading run...</p>
      </div>
    );
  }

  const attempt = selectedRun.attempts?.[selectedAttemptIndex];
  const runSucceeded = isRunSuccessful(selectedRun);
  const hasPlannerData = selectedRun.attempts.some((item) => Boolean(item.generator.plan_raw));
  if (!attempt) {
    console.error('[InvestigationPage] Attempt not found:', {
      index: selectedAttemptIndex,
      total_attempts: selectedRun.attempts?.length,
    });
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <p className="text-yellow-600 font-medium mb-2">No Attempt Data</p>
          <p className="text-sm text-slate-500 mb-4">
            Attempt {selectedAttemptIndex + 1} not found (total: {selectedRun.attempts?.length || 0})
          </p>
          <button
            onClick={() => navigate('/runs')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
          >
            ← Back to Runs
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Top Bar */}
      <header className="bg-white border-b border-slate-200 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/runs')} className="text-sm text-slate-500 hover:text-slate-900 transition-colors">
            ← Runs
          </button>
          <span className="text-slate-300">|</span>
          <h1 className="font-mono text-sm font-bold text-slate-900">{selectedRun.experiment.run_id}</h1>
          <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-slate-100 text-slate-700">
            scenario {selectedRun.experiment.scenario_id}
          </span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${runSucceeded ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {runSucceeded ? 'SUCCESS' : 'FAILED'}
          </span>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span>Attempt {attempt.attempt_number}/{selectedRun.result.total_attempts}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowTimeline((value) => !value)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${showTimeline ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-700'}`}
            title="Show or hide the attempt timeline"
          >
            Timeline
          </button>
          <button
            onClick={() => setShowPlanner((value) => !value)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${showPlanner ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-700'}`}
            title={hasPlannerData ? 'Show or hide the planner analysis' : 'Legacy run: planner output is inferred from history'}
          >
            Planner
          </button>
          <button
            onClick={() => setShowAnalytics((value) => !value)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${showAnalytics ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-700'}`}
            title="Show or hide the analytics panel"
          >
            Analytics
          </button>
          <button
            onClick={() => setShowTabs((value) => !value)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${showTabs ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-700'}`}
            title="Show or hide the deep analysis tabs"
          >
            Tabs
          </button>
          <button
            onClick={resetLayout}
            className="px-3 py-1.5 text-slate-500 hover:text-slate-900 text-xs font-medium transition-colors"
            title="Reset all panel sizes"
          >
            Reset Layout
          </button>
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/json`}
            download={`${selectedRun.experiment.run_id}.json`}
            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-medium transition-colors"
          >
            Export JSON
          </a>
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/csv`}
            download={`${selectedRun.experiment.run_id}.csv`}
            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-medium transition-colors"
          >
            Export CSV
          </a>
          <a
            href={`/api/export/${selectedRun.experiment.run_id}/html`}
            download={`${selectedRun.experiment.run_id}.html`}
            className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-medium transition-colors"
          >
            Export HTML
          </a>
        </div>
      </header>

      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <div className="flex-1 min-h-0 flex overflow-hidden">
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

          <main className="flex-1 min-w-0 overflow-y-auto bg-slate-50 p-6">
            <div className="w-full max-w-6xl mx-auto space-y-4">
            {showPlanner && (
              <PlannerInsightsPanel run={selectedRun} selectedAttemptIndex={selectedAttemptIndex} />
            )}
            {/* Attempt Header */}
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-slate-900">Attempt {attempt.attempt_number}</h2>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
                  {attempt.generator.strategy}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${attempt.judge.decision === 'ATTACK' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {attempt.judge.decision} ({attempt.judge.confidence.toFixed(2)})
                </span>
              </div>
            </div>

            {/* Pipeline */}
            <GeneratorCard attempt={attempt} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
            <VictimCard attempt={attempt} accessCode={selectedRun.scenario.access_code} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
            <ExtractorCard attempt={attempt} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
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
              className="flex-shrink-0 min-h-0 overflow-hidden"
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
