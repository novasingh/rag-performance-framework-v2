import { api } from '../api/client.js';
import { useApi } from '../hooks/useApi.js';
import { AsyncBoundary } from '../components/ui/StateView.jsx';
import { SimpleBarChart } from '../components/charts/SimpleBarChart.jsx';
import { DOMAIN_COLORS } from '../constants.js';

export function SourcesPage() {
  const { data, error, loading, reload } = useApi(() => api.getSources(), []);

  return (
    <>
      <div className="page-header">
        <h2>Source Type Impact</h2>
        <p>How source diversity affects retrieval precision in each domain. All figures computed from RQ1 condition metrics.</p>
      </div>
      <AsyncBoundary loading={loading} error={error} onRetry={reload}>
        {data && <SourcesBody sources={data.sources} />}
      </AsyncBoundary>
    </>
  );
}

function SourcesBody({ sources }) {
  const entries = Object.entries(sources);

  return (
    <>
      <div className="grid-3">
        {entries.map(([domain, info]) => {
          const positive = (info.delta_pct ?? 0) >= 0;
          const improves = info.best_level && info.worst_level;
          return (
            <div key={domain} className="card" style={{ borderLeft: `4px solid ${DOMAIN_COLORS[domain]}` }}>
              <h3 style={{ color: DOMAIN_COLORS[domain] }}>{info.label}</h3>
              {info.delta_pct != null ? (
                <>
                  <div className="kpi-value" style={{ color: positive ? 'var(--success)' : 'var(--danger)', fontSize: 28 }}>
                    {positive ? '+' : ''}{info.delta_pct}%
                  </div>
                  <p className="text-sm text-secondary">
                    Best mix: <strong>{info.best_level}</strong>. Lowest: <strong>{info.worst_level}</strong>.
                    {improves && positive && ' More diverse sources help here.'}
                    {improves && !positive && ' A leaner source mix performs better here (source pollution).'}
                  </p>
                </>
              ) : <p className="text-sm text-secondary">No source-level comparison available.</p>}
            </div>
          );
        })}
      </div>

      {entries.map(([domain, info]) => {
        const data = info.levels.map((lv) => ({
          name: lv.level_label,
          'Precision@5': lv.precision_at_5,
          'nDCG@5': lv.ndcg_at_5,
          'Human Eval': lv.human_eval_score,
        }));
        return (
          <div key={domain} className="card">
            <h3 style={{ color: DOMAIN_COLORS[domain] }}>{info.label} — Source Levels</h3>
            <SimpleBarChart data={data} xKey="name" bars={[
              { key: 'Precision@5', color: '#3b82f6' },
              { key: 'nDCG@5', color: '#8b5cf6' },
              { key: 'Human Eval', color: '#10b981' },
            ]} />
          </div>
        );
      })}
    </>
  );
}
