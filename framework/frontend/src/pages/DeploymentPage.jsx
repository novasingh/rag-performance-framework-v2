import { api } from '../api/client.js';
import { useApi } from '../hooks/useApi.js';
import { AsyncBoundary } from '../components/ui/StateView.jsx';
import { Callout } from '../components/Callout.jsx';
import { DOMAIN_COLORS } from '../constants.js';

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

const STEPS = [
  'Pick your target domain and the expected average age of your knowledge base.',
  'Set your intended source-type mix (single-source, two-source, or full diversity).',
  'Read the predicted hallucination rate and precision in the Prediction Tool.',
  'If predicted hallucination exceeds your risk tolerance, increase update frequency or reduce source diversity before spending compute on embedding and deployment.',
];

export function DeploymentPage({ onNavigate }) {
  const { data, error, loading, reload } = useApi(() => api.getDeploymentGuidelines(), []);

  return (
    <>
      <div className="page-header">
        <div className="eyebrow">RO4 Deliverable</div>
        <h2>Deployment Guidelines</h2>
        <p>
          Evidence-based rules for configuring and maintaining a RAG knowledge base, derived from the
          experiments (RQ1–RQ4) and the trained predictive model. Use these alongside the Prediction Tool.
        </p>
      </div>

      <AsyncBoundary loading={loading} error={error} onRetry={reload}>
        {data && <DeploymentBody data={data} onNavigate={onNavigate} />}
      </AsyncBoundary>
    </>
  );
}

function DeploymentBody({ data, onNavigate }) {
  const domains = Object.entries(data.domains);

  return (
    <>
      {/* Per-domain guidance cards (data-driven) */}
      <div className="grid-3">
        {domains.map(([domain, g]) => {
          const color = DOMAIN_COLORS[domain] || '#6366f1';
          return (
            <div key={domain} className="card" style={{ borderTop: `3px solid ${color}` }}>
              <div className="flex-between mb-2">
                <h3 style={{ margin: 0, color }}>{data.domain_labels?.[domain] || domain}</h3>
                {g.reliable === true && <span className="tag tag-green">Reliable</span>}
                {g.reliable === false && <span className="tag tag-yellow">{g.boundary_conditions} edge cases</span>}
              </div>

              <table>
                <tbody>
                  <tr><td>Update cadence</td><td className="fw-700">{g.update_cadence?.cadence}</td></tr>
                  <tr><td>Best source mix</td><td className="fw-700">{g.best_source_level}</td></tr>
                  <tr><td>Predicted hallucination</td><td className="fw-700">{pct(g.predicted?.hallucination_rate)}</td></tr>
                  <tr><td>Predicted Precision@5</td><td className="fw-700">{pct(g.predicted?.precision_at_5)}</td></tr>
                </tbody>
              </table>

              {g.source_pollution ? (
                <Callout type="danger" title="Source pollution risk">
                  Adding the most diverse sources lowered measured precision here — keep the source mix lean.
                </Callout>
              ) : (
                <Callout type="tip" title="Diversity helps">
                  More source diversity improved measured precision in this domain.
                </Callout>
              )}
            </div>
          );
        })}
      </div>

      {/* The three core rules (research findings) */}
      <div className="card">
        <h3>Rule 1 — Domain volatility dictates update strategy</h3>
        <p className="text-sm text-secondary mb-4">
          The rate of staleness is not universal; it depends on domain volatility. Cadences below are
          derived from each domain's model-measured freshness sensitivity.
        </p>
        <div className="grid-3">
          {domains.map(([domain, g]) => (
            <div key={domain} className="recommendation" style={{ borderColor: DOMAIN_COLORS[domain] }}>
              <h4 style={{ textTransform: 'capitalize' }}>{domain}</h4>
              <p><strong>{g.update_cadence?.cadence}.</strong> {g.update_cadence?.rationale}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Rule 2 — Source diversity is a double-edged sword</h3>
        <p className="text-sm text-secondary mb-2">
          "More data is always better" does not hold. The right mix is domain-dependent.
        </p>
        <Callout type="warning" title="Check before you add sources">
          In medium-volatility domains, adding broad or technical sources can hurt retrieval precision and
          raise hallucinations. Prefer curated, single-source data unless the evidence shows diversity helps.
        </Callout>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr><th>Domain</th><th>Recommended Mix</th><th>Diversity Effect</th></tr>
            </thead>
            <tbody>
              {domains.map(([domain, g]) => (
                <tr key={domain}>
                  <td><span className="domain-badge" style={{ background: DOMAIN_COLORS[domain] }}>{domain}</span></td>
                  <td>{(g.best_source_configs || []).join(', ') || g.best_source_level}</td>
                  <td>{g.source_pollution
                    ? <span className="tag tag-red">Restrict diversity</span>
                    : <span className="tag tag-green">Increase diversity</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3>Rule 3 — Trust the framework within its boundaries</h3>
        <Callout type="info" title="Leave-one-condition-out validation (RQ4)">
          The framework was validated against held-out conditions. Domains marked “Reliable” above had no
          boundary conditions; where edge cases exist, treat predictions as directional rather than exact.
        </Callout>
      </div>

      {/* How-to */}
      <div className="card">
        <h3>Using the framework before deployment</h3>
        <div className="timeline">
          {STEPS.map((s, i) => (
            <div key={i} className="tl-item">
              <div className="tl-dot">{i + 1}</div>
              <div className="tl-body"><p style={{ marginTop: 8 }}>{s}</p></div>
            </div>
          ))}
        </div>
        <button className="btn btn-primary mt-2" onClick={() => onNavigate?.('predict')}>
          Go to the Prediction Tool →
        </button>
      </div>
    </>
  );
}
