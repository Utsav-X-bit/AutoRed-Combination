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
      filtered = filtered.filter(r => r.success);
    } else if (showOnly === 'failure') {
      filtered = filtered.filter(r => !r.success);
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(r =>
        r.run_id.toLowerCase().includes(q) ||
        r.scenario_id.toLowerCase().includes(q) ||
        r.access_code.toLowerCase().includes(q) ||
        r.generator.toLowerCase().includes(q) ||
        r.victim.toLowerCase().includes(q)
      );
    }

    onFilter(filtered);
  }, [runs, searchQuery, showOnly, onFilter]);

  return (
    <div className="flex items-center gap-3 flex-wrap">
      {/* Success filter */}
      <div className="flex items-center gap-1 bg-white rounded-lg border border-slate-200 p-1">
        {(['all', 'success', 'failure'] as const).map((opt) => (
          <button
            key={opt}
            onClick={() => setShowOnly(opt)}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              showOnly === opt
                ? 'bg-slate-900 text-white'
                : 'text-slate-600 hover:bg-slate-100'
            }`}
          >
            {opt.charAt(0).toUpperCase() + opt.slice(1)}
          </button>
        ))}
      </div>

      {/* Search */}
      <div className="flex-1 min-w-[200px]">
        <input
          type="text"
          placeholder="Search runs (ID, access code, model...)"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      {/* Count */}
      <span className="text-xs text-slate-500">
        Filter run history
      </span>
    </div>
  );
}
