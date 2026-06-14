// Renders a single predicted metric with its CI band and MAE.
// Formatting is driven by the metric metadata from the API (scale/direction),
// so nothing about the metric is hardcoded here.

function isPercentScale(meta) {
  const [lo, hi] = meta?.scale ?? [0, 1];
  return lo === 0 && hi === 1;
}

export function MetricCard({ name, label, value, lower, upper, mae, color, meta }) {
  const percent = isPercentScale(meta);
  const [lo, hi] = meta?.scale ?? [0, 1];

  const display = percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(2);
  const fmt = (v) => (percent ? `${(v * 100).toFixed(1)}%` : v.toFixed(2));
  const barPct = ((value - lo) / (hi - lo)) * 100;

  return (
    <div className="metric-card">
      <div className="name">{label || name}</div>
      <div className="value" style={{ color }}>{display}</div>
      {Number.isFinite(lower) && Number.isFinite(upper) && (
        <div className="range">CI: [{fmt(lower)}, {fmt(upper)}]</div>
      )}
      {Number.isFinite(mae) && <div className="mae">MAE: ±{mae.toFixed(4)}</div>}
      <div className="bar" style={{ marginTop: 8 }}>
        <div className="bar-fill" style={{ width: `${Math.min(100, Math.max(0, barPct))}%`, background: color }} />
      </div>
    </div>
  );
}
