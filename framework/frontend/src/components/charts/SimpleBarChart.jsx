import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

const tooltipStyle = { background: 'var(--bg-card)', border: '1px solid var(--border)' };

export function SimpleBarChart({ data, xKey, bars, height = 280 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey={xKey} tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
        <YAxis tick={{ fontSize: 11, fill: 'var(--text-light)' }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend />
        {bars.map((b) => (
          <Bar key={b.key} dataKey={b.key} name={b.name || b.key} fill={b.color || '#3b82f6'} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
