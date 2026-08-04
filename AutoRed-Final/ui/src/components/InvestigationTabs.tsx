import { useState } from 'react';
import ScenarioTab from './ScenarioTab';
import AttackEvolutionTab from './AttackEvolutionTab';
import StrategyHeatmapTab from './StrategyHeatmapTab';
import ModelHeatmapTab from './ModelHeatmapTab';
import ExtractorDebuggerTab from './ExtractorDebuggerTab';
import VerificationTraceTab from './VerificationTraceTab';
import TokenAnalyticsTab from './TokenAnalyticsTab';

const tabs = [
  { id: 'scenario', label: 'Scenario' },
  { id: 'evolution', label: 'Attack Evolution' },
  { id: 'heatmap', label: 'Strategy Heatmap' },
  { id: 'model', label: 'Model Perf' },
  { id: 'extractor', label: 'Extractor Debugger' },
  { id: 'verification', label: 'Verification' },
  { id: 'tokens', label: 'Token Analytics' },
];

export default function InvestigationTabs() {
  const [activeTab, setActiveTab] = useState('scenario');

  return (
    <div className="flex h-full min-h-0 flex-col bg-white dark:bg-stone-900">
      <div className="flex flex-shrink-0 items-center gap-1 overflow-x-auto border-b border-stone-200 px-4 dark:border-stone-800">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`whitespace-nowrap border-b-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'border-teal-500 text-teal-600 dark:border-teal-500 dark:text-teal-400'
                : 'border-transparent text-stone-500 hover:text-stone-700 dark:text-stone-400 dark:hover:text-stone-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto bg-stone-50 p-4 dark:bg-stone-950">
        {activeTab === 'scenario' && <ScenarioTab />}
        {activeTab === 'evolution' && <AttackEvolutionTab />}
        {activeTab === 'heatmap' && <StrategyHeatmapTab />}
        {activeTab === 'model' && <ModelHeatmapTab />}
        {activeTab === 'extractor' && <ExtractorDebuggerTab />}
        {activeTab === 'verification' && <VerificationTraceTab />}
        {activeTab === 'tokens' && <TokenAnalyticsTab />}
      </div>
    </div>
  );
}
