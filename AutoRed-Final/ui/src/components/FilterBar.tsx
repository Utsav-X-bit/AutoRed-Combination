import { useEffect, useState } from 'react';
import type { RunListItem } from '../types/autored';

interface FilterBarProps {
  runs: RunListItem[];
  onFilter: (filtered: RunListItem[]) => void;
}

export default function FilterBar({ runs, onFilter }: FilterBarProps) {
  const [showOnly, setShowOnly] = useState<'all' | 'success' | 'failure'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    let filtered = [...runs];

    if (showOnly === 'success') {
      filtered = filtered.filter((r) => r.success);
    } else if (showOnly === 'failure') {
      filtered = filtered.filter((r) => !r.success);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (r) =>
          r.run_id.toLowerCase().includes(q) ||
          r.scenario_id.toLowerCase().includes(q) ||
          r.access_code.toLowerCase().includes(q) ||
          r.generator.toLowerCase().includes(q) ||
          r.victim.toLowerCase().includes(q),
      );
    }

    onFilter(filtered);
  }, [runs, searchQuery, showOnly, onFilter]);

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex items-center gap-1 rounded-lg border border-stone-200 bg-white p-1 dark:border-stone-800 dark:bg-stone-900">
        {(['all', 'success', 'failure'] as const).map((opt) => (
          <button
            key={opt}
            onClick={() => setShowOnly(opt)}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              showOnly === opt
                ? 'bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900'
                : 'text-stone-600 hover:bg-stone-100 dark:text-stone-400 dark:hover:bg-stone-800'
            }`}
          >
            {opt.charAt(0).toUpperCase() + opt.slice(1)}
          </button>
        ))}
      </div>

      <div className="min-w-[200px] flex-1">
        <input
          type="text"
          placeholder="Search runs (ID, access code, model...)"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="input w-full"
        />
      </div>

      <span className="text-xs text-stone-500 dark:text-stone-400">Filter run history</span>
    </div>
  );
}
