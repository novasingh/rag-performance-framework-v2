import { useState } from 'react';
import { api } from '../api/client.js';
import { useApi } from '../hooks/useApi.js';
import { useMeta } from '../context/MetaContext.jsx';
import { AsyncBoundary } from '../components/ui/StateView.jsx';
import { KPICard } from '../components/cards/KPICard.jsx';
import { LocoScatter } from '../components/charts/LocoScatter.jsx';
import { DOMAIN_COLORS } from '../constants.js';

export function ValidationPage() {
  const { data, error, loading, reload } = useApi(() => api.getValidation(), []);
  const { metrics } = useMeta();

  return (
    <>
      <div className="page-header">
        <h2>Framework Validation (RQ4)</h2>
        <p>Leave-one-condition-out validation: predicted vs actual performance, with boundary conditions flagged.</p>
      </div>
      <AsyncBoundary loading={loading} error={error} onRetry={reload}>
        {data && <ValidationBody data={data} metrics={metrics} />}
      </AsyncBoundary>
    </>
  );
}

function ValidationBody({ data, metrics }) {
  const metricIds = Object.keys(data.loco_data || {});
  const [active, setActive] = useState(metricIds[0] || 'precision_at_5');

  return (
    <>
      <div className="grid-4">
        <KPICard value={data.n_samples} label="Observations" color="#3b82f6" />
        <KPICard value={data.boundary_conditions.length} label="Boundary Conditions" sub="across all metrics"
          color={data.boundary_conditions.length ? '#f59e0b' : '#10b981'} />
        {Object.entries(data.boundary_counts_by_domain || {}).slice(0, 2).map(([d, c]) => (
          <KPICard key={d} value={c} label={`${d} boundaries`} color={DOMAIN_COLORS[d]} />
        ))}
      </div>

      <div className="card">
        <h3>Actual vs Predicted (LOCO)</h3>
        <div className="tabs">
          {metricIds.map((m) => (
            <button key={m} className={`tab ${active === m ? 'active' : ''}`} onClick={() => setActive(m)}>
              {metrics?.[m]?.label || m}
            </button>
          ))}
        </div>
        <LocoScatter data={data.loco_data[active] || []} />
        <p className="text-xs text-secondary mt-2">
          Points on the dashed diagonal indicate perfect prediction. Red-outlined points are boundary conditions.
        </p>
      </div>

      {data.boundary_conditions.length > 0 && (
        <div className="card">
          <h3>Boundary Conditions</h3>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr><th>Condition</th><th>Domain</th><th>Metric</th><th>Actual</th><th>Predicted</th><th>95% CI</th></tr>
              </thead>
              <tbody>
                {data.boundary_conditions.map((b, i) => (
                  <tr key={i}>
                    <td className="fw-600">{b.condition}</td>
                    <td><span className="domain-badge" style={{ background: DOMAIN_COLORS[b.domain] || '#94a3b8' }}>{b.domain}</span></td>
                    <td>{metrics?.[b.metric]?.label || b.metric}</td>
                    <td>{b.actual_mean?.toFixed(4)}</td>
                    <td>{b.predicted_mean?.toFixed(4)}</td>
                    <td>[{b.ci_lower?.toFixed(4)}, {b.ci_upper?.toFixed(4)}]</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
