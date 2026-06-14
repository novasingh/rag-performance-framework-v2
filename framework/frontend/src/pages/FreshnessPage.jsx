import { api } from '../api/client.js';
import { useApi } from '../hooks/useApi.js';
import { AsyncBoundary } from '../components/ui/StateView.jsx';
import { DecayLineChart } from '../components/charts/DecayLineChart.jsx';
import { DOMAINS, DOMAIN_COLORS } from '../constants.js';

export function FreshnessPage() {
  const { data, error, loading, reload } = useApi(() => api.getFreshness(), []);

  return (
    <>
      <div className="page-header">
        <h2>Freshness Impact</h2>
        <p>Precision@5 swept across document age using the trained RQ3 model. Curves show each domain's sensitivity to staleness.</p>
      </div>
      <AsyncBoundary loading={loading} error={error} onRetry={reload}>
        {data && <FreshnessBody data={data} />}
      </AsyncBoundary>
    </>
  );
}

function FreshnessBody({ data }) {
  const chartData = data.curves.map((row) => ({ name: `${row.age_days}d`, ...row }));

  return (
    <>
      <div className="card">
        <h3>Model-Predicted Precision@5 vs Document Age</h3>
        <DecayLineChart data={chartData} xKey="name" domains={DOMAINS} />
      </div>

      <div className="card">
        <h3>Freshness Sensitivity (best-fit decay)</h3>
        <p className="text-sm text-secondary mb-4">
          Best-fitting decay curve and variance explained (R²) per domain, from the RQ3 model.
        </p>
        <table>
          <thead>
            <tr><th>Domain</th><th>Best-Fit Curve</th><th>R²</th></tr>
          </thead>
          <tbody>
            {Object.entries(data.decay_fit || {}).map(([domain, fit]) => (
              <tr key={domain}>
                <td><span className="domain-badge" style={{ background: DOMAIN_COLORS[domain] || '#94a3b8' }}>{domain}</span></td>
                <td style={{ textTransform: 'capitalize' }}>{fit.best_fit || '—'}</td>
                <td>{fit.r_squared ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
