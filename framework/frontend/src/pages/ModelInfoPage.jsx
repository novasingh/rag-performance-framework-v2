import { api } from '../api/client.js';
import { useApi } from '../hooks/useApi.js';
import { useMeta } from '../context/MetaContext.jsx';
import { AsyncBoundary } from '../components/ui/StateView.jsx';
import { KPICard } from '../components/cards/KPICard.jsx';
import { SimpleBarChart } from '../components/charts/SimpleBarChart.jsx';

const num = (v) => (v == null ? '—' : Number(v).toFixed(4));

export function ModelInfoPage() {
  const { data, error, loading, reload } = useApi(() => api.getModelInfo(), []);
  const { metrics } = useMeta();

  return (
    <>
      <div className="page-header">
        <h2>Model Information (RQ3)</h2>
        <p>Cross-validated comparison of the OLS, Random Forest, and XGBoost models, plus feature importances.</p>
      </div>
      <AsyncBoundary loading={loading} error={error} onRetry={reload}>
        {data && <ModelInfoBody data={data} metrics={metrics} />}
      </AsyncBoundary>
    </>
  );
}

function ModelInfoBody({ data, metrics }) {
  const label = (id) => metrics?.[id]?.label || id;
  const maeChart = data.comparison.map((c) => ({
    metric: label(c.metric),
    OLS: c.ols_mae,
    'Random Forest': c.rf_mae,
    XGBoost: c.xgb_mae,
  }));

  return (
    <>
      <div className="grid-4">
        <KPICard value={data.n_samples} label="Training Samples" color="#3b82f6" />
        <KPICard value={data.n_features} label="Features" color="#8b5cf6" />
        <KPICard value={data.comparison.length} label="Predicted Metrics" color="#06b6d4" />
        <KPICard value="10-fold" label="Cross-Validation" color="#10b981" />
      </div>

      <div className="card">
        <h3>Model Comparison — MAE by Metric (lower is better)</h3>
        <SimpleBarChart data={maeChart} xKey="metric" bars={[
          { key: 'OLS', color: '#3b82f6' },
          { key: 'Random Forest', color: '#10b981' },
          { key: 'XGBoost', color: '#f59e0b' },
        ]} />
      </div>

      <div className="card">
        <h3>Cross-Validated Errors</h3>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr><th>Metric</th><th>OLS MAE</th><th>RF MAE</th><th>XGB MAE</th><th>OLS R²</th><th>Best</th></tr>
            </thead>
            <tbody>
              {data.comparison.map((c) => {
                const best = Math.min(c.ols_mae, c.rf_mae, c.xgb_mae);
                const winner = best === c.rf_mae ? 'Random Forest' : best === c.xgb_mae ? 'XGBoost' : 'OLS';
                return (
                  <tr key={c.metric}>
                    <td className="fw-600">{label(c.metric)}</td>
                    <td>{num(c.ols_mae)}</td>
                    <td>{num(c.rf_mae)}</td>
                    <td>{num(c.xgb_mae)}</td>
                    <td>{num(c.ols_r2)}</td>
                    <td><span className="tag tag-green">{winner}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {Object.keys(data.feature_importance || {}).length > 0 && (
        <div className="card">
          <h3>Random Forest Feature Importance</h3>
          {Object.entries(data.feature_importance).map(([metric, byModel]) => {
            const fi = byModel.random_forest;
            if (!fi) return null;
            const chart = Object.entries(fi)
              .map(([feature, imp]) => ({ feature, Importance: Number(imp) }))
              .sort((a, b) => b.Importance - a.Importance)
              .slice(0, 8);
            return (
              <div key={metric} style={{ marginBottom: 24 }}>
                <h4 className="text-sm fw-600 mb-2">{label(metric)}</h4>
                <SimpleBarChart data={chart} xKey="feature" bars={[{ key: 'Importance', color: '#8b5cf6' }]} height={220} />
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
