import { api } from '../api/client.js';
import { useApi } from '../hooks/useApi.js';
import { useMeta } from '../context/MetaContext.jsx';
import { AsyncBoundary } from '../components/ui/StateView.jsx';
import { SimpleBarChart } from '../components/charts/SimpleBarChart.jsx';
import { DOMAIN_COLORS } from '../constants.js';

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const num = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d));

export function DashboardPage() {
  const { data, error, loading, reload } = useApi(() => api.getConditions(), []);
  const { metrics } = useMeta();

  return (
    <>
      <div className="page-header">
        <h2>Domain Comparison Dashboard</h2>
        <p>Measured performance across Technology, Healthcare, and History domains (RQ1).</p>
      </div>
      <AsyncBoundary loading={loading} error={error} onRetry={reload}>
        {data && <DashboardBody conditions={data.conditions} metrics={metrics} />}
      </AsyncBoundary>
    </>
  );
}

function DashboardBody({ conditions, metrics }) {
  const rows = Object.entries(conditions).map(([id, c]) => ({ id, ...c }));

  // Average key metrics per domain (computed from condition data, not hardcoded).
  const domains = [...new Set(rows.map((r) => r.domain))];
  const avg = (list, key) => {
    const vals = list.map((r) => r[key]).filter((v) => v != null);
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0;
  };
  const domainData = domains.map((d) => {
    const list = rows.filter((r) => r.domain === d);
    return {
      name: d[0].toUpperCase() + d.slice(1),
      [metrics?.precision_at_5?.label || 'Precision@5']: avg(list, 'precision_at_5'),
      [metrics?.ndcg_at_5?.label || 'nDCG@5']: avg(list, 'ndcg_at_5'),
      [metrics?.hallucination_rate?.label || 'Hallucination']: avg(list, 'hallucination_rate'),
    };
  });
  const bars = [
    { key: metrics?.precision_at_5?.label || 'Precision@5', color: '#3b82f6' },
    { key: metrics?.ndcg_at_5?.label || 'nDCG@5', color: '#8b5cf6' },
    { key: metrics?.hallucination_rate?.label || 'Hallucination', color: '#ef4444' },
  ];

  return (
    <>
      <div className="card">
        <h3>Domain-Level Performance</h3>
        <SimpleBarChart data={domainData} xKey="name" bars={bars} />
      </div>

      <div className="card">
        <h3>All Experimental Conditions</h3>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Condition</th><th>Domain</th><th>Freshness</th><th>Source Config</th>
                <th>N</th><th>P@5</th><th>nDCG@5</th><th>BERTScore</th><th>Halluc.</th><th>Human</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.id}>
                  <td className="fw-600">{c.id}</td>
                  <td><span className="domain-badge" style={{ background: DOMAIN_COLORS[c.domain] || '#94a3b8' }}>{c.domain}</span></td>
                  <td className="text-xs">{c.freshness}</td>
                  <td className="text-xs">{c.source_config}</td>
                  <td>{c.n_records}</td>
                  <td>{pct(c.precision_at_5)}</td>
                  <td>{num(c.ndcg_at_5)}</td>
                  <td>{num(c.bertscore_f1)}</td>
                  <td>{pct(c.hallucination_rate)}</td>
                  <td>{num(c.human_eval_score, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
