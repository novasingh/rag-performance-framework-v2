import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { DOMAIN_COLORS } from '../../constants.js';

const tooltipStyle = { background: 'var(--bg-card)', border: '1px solid var(--border)' };

// `data` rows contain an x key plus one numeric series per domain id.
export function DecayLineChart({ data, xKey, domains, height = 300 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey={xKey} tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
        <YAxis tick={{ fontSize: 11, fill: 'var(--text-light)' }} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => (v == null ? '-' : Number(v).toFixed(3))} />
        <Legend />
        {domains.map((d) => (
          <Line
            key={d.id}
            type="monotone"
            dataKey={d.id}
            name={d.label}
            stroke={DOMAIN_COLORS[d.id] || '#3b82f6'}
            strokeWidth={2}
            dot={{ r: 3 }}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
