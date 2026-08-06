import { Attempt } from '../types/autored';

export default function VerifierCard({ attempt }: { attempt: Attempt }) {
  const { verification } = attempt;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🔍</span> Verifier
        </h3>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${verification.success ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
          {verification.success ? '✓ Verified' : '✗ Not Verified'}
        </span>
      </div>

      <div className="space-y-2">
        <div>
          <p className="text-xs text-slate-500 mb-1">Candidate Sent</p>
          <p className="font-mono text-sm bg-slate-50 rounded-lg px-3 py-2 border border-slate-200">
            {verification.candidate_sent || '—'}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">Victim Response</p>
          <p className="font-mono text-sm bg-slate-50 rounded-lg px-3 py-2 border border-slate-200">
            {verification.victim_response || '—'}
          </p>
        </div>
      </div>
    </div>
  );
}
