export function KPICard({ value, label, sub, color }) {
  return (
    <div className="kpi card">
      <div className="kpi-value" style={{ color: color || 'var(--text)' }}>{value}</div>
      <div className="kpi-label">{label}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}
