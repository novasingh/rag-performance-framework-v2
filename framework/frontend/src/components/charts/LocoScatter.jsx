import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Cell,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { DOMAIN_COLORS } from '../../constants.js';

const tooltipStyle = { background: 'var(--bg-card)', border: '1px solid var(--border)' };

// Actual-vs-predicted scatter for RQ4 LOCO validation. The diagonal reference
// line marks a perfect prediction.
export function LocoScatter({ data, height = 300 }) {
  if (!data?.length) return null;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis type="number" dataKey="actual" name="Actual" domain={['auto', 'auto']}
          tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
        <YAxis type="number" dataKey="predicted" name="Predicted" domain={['auto', 'auto']}
          tick={{ fontSize: 11, fill: 'var(--text-light)' }} />
        <ZAxis range={[70, 70]} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: '3 3' }}
          formatter={(v) => (typeof v === 'number' ? v.toFixed(4) : v)} />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />
        <Scatter data={data}>
          {data.map((entry, i) => (
            <Cell key={i} fill={DOMAIN_COLORS[entry.domain] || '#94a3b8'}
              stroke={entry.is_boundary ? '#ef4444' : 'none'} strokeWidth={entry.is_boundary ? 2 : 0} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
