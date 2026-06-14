import { useEffect, useState } from 'react';
import { api } from '../api/client.js';
import { useMeta } from '../context/MetaContext.jsx';
import { ErrorView } from '../components/ui/StateView.jsx';
import { MetricCard } from '../components/cards/MetricCard.jsx';
import { PerformanceRadar } from '../components/charts/PerformanceRadar.jsx';
import { DOMAINS, FRESHNESS_PRESETS, SOURCE_OPTIONS, METRIC_COLORS } from '../constants.js';

function Selector({ title, options, value, onChange, describe }) {
  return (
    <div className="mb-4">
      <div className="text-sm fw-600 mb-2">{title}</div>
      <div className="selector-group">
        {options.map((opt) => (
          <div
            key={opt.id}
            className={`selector-option ${value === opt.id ? 'selected' : ''}`}
            onClick={() => onChange(opt.id)}
          >
            <div className="indicator" />
            <div>
              <div className="label">{opt.label}</div>
              {describe && <div className="desc">{describe(opt)}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function PredictPage() {
  const { metrics } = useMeta();
  const [domain, setDomain] = useState('technology');
  const [freshness, setFreshness] = useState('<1week');
  const [source, setSource] = useState('academic');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    const days = FRESHNESS_PRESETS.find((f) => f.id === freshness)?.days ?? 30;
    setLoading(true);
    setError(null);
    api.predict({ domain, avg_age_days: days, source_config: source })
      .then((res) => active && setResult(res))
      .catch((err) => active && setError(err))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, [domain, freshness, source]);

  const preds = result?.predictions || {};
  const inputs = result?.inputs || {};
  const recs = result?.recommendations || {};
  const readiness = result?.deployment_readiness_score ?? 0;

  const radarPoints = Object.entries(preds).map(([id, v]) => {
    const meta = metrics?.[id];
    const [lo, hi] = meta?.scale ?? [0, 1];
    let norm = ((v.expected - lo) / (hi - lo)) * 100;
    if (meta?.direction === 'lower') norm = 100 - norm;
    return { metric: meta?.label || id, value: Math.max(0, Math.min(100, norm)) };
  });

  const readinessColor = readiness > 80 ? '#10b981' : readiness > 60 ? '#f59e0b' : '#ef4444';

  return (
    <>
      <div className="page-header">
        <h2>Interactive Prediction Tool</h2>
        <p>Select dataset factors to predict RAG performance before deployment.</p>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>Input Factors</h3>
          <Selector title="A. Domain Volatility" options={DOMAINS} value={domain} onChange={setDomain}
            describe={(o) => o.volatility} />
          <Selector title="B. Dataset Freshness" options={FRESHNESS_PRESETS} value={freshness} onChange={setFreshness}
            describe={(o) => `~${o.days} days average age`} />
          <Selector title="C. Source Type Configuration" options={SOURCE_OPTIONS} value={source} onChange={setSource} />
        </div>

        <div>
          <div className="card">
            <h3>Resolved Inputs</h3>
            <table>
              <tbody>
                <tr><td>Domain</td><td className="fw-600">{inputs.domain_label || domain}</td></tr>
                <tr><td>Average Age</td><td className="fw-600">{inputs.avg_age_days != null ? `${inputs.avg_age_days} days` : '—'}</td></tr>
                <tr><td>Freshness Score</td><td className="fw-600">{inputs.freshness_score ?? '—'}</td></tr>
                <tr><td>Source Level</td><td className="fw-600">{inputs.source_level || '—'}</td></tr>
                <tr><td>Source Diversity Index</td><td className="fw-600">{inputs.source_diversity_index ?? '—'}</td></tr>
              </tbody>
            </table>
            {loading && <p className="text-sm text-secondary mt-2">Updating prediction…</p>}
          </div>

          {error && <ErrorView error={error} />}

          {result && (
            <div className="card">
              <h3>Predicted Performance</h3>
              <div className="grid-2">
                {Object.entries(preds).map(([id, v]) => (
                  <MetricCard
                    key={id}
                    name={id}
                    label={metrics?.[id]?.label}
                    meta={metrics?.[id]}
                    value={v.expected}
                    lower={v.lower_bound}
                    upper={v.upper_bound}
                    mae={v.mae_margin}
                    color={METRIC_COLORS[id] || '#3b82f6'}
                  />
                ))}
              </div>
            </div>
          )}

          {result && (
            <div className="card">
              <h3>Performance Profile</h3>
              <PerformanceRadar points={radarPoints} />
            </div>
          )}
        </div>
      </div>

      {result && (
        <div className="card">
          <h3>Decision Support</h3>
          <div className="grid-3">
            <div>
              <div className="recommendation" style={{ borderColor: '#3b82f6' }}>
                <h4>Recommended Update Frequency</h4>
                <p>{recs.update_frequency}</p>
              </div>
              <div className="recommendation" style={{ borderColor: '#8b5cf6' }}>
                <h4>Recommended Source Mix</h4>
                <p>{recs.source_mix}</p>
              </div>
            </div>
            <div>
              <div className="recommendation" style={{ borderColor: '#10b981' }}>
                <h4>Expected Performance Range</h4>
                <p>{recs.performance_range}</p>
              </div>
              <div className="recommendation" style={{ borderColor: '#f59e0b' }}>
                <h4>Key Finding</h4>
                <p>{recs.key_finding || 'No source-level difference measured for this domain.'}</p>
              </div>
            </div>
            <div>
              <div className="recommendation" style={{ borderColor: readinessColor }}>
                <h4>Deployment Readiness</h4>
                <div style={{ fontSize: 36, fontWeight: 700, color: readinessColor }}>{readiness}%</div>
                <p>{readiness > 80 ? 'Ready for deployment' : readiness > 60 ? 'Conditionally ready' : 'Needs configuration review'}</p>
              </div>
              <div className="card" style={{ margin: 0 }}>
                <h4>Risk Assessment</h4>
                {(recs.risks || []).map((r, i) => (
                  <p key={i} className="text-sm text-secondary mb-2">⚠ {r}</p>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
