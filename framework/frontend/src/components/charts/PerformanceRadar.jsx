import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer,
} from 'recharts';

// `points` = [{ metric: label, value: 0..100 }]. Normalization happens upstream
// using metric metadata so this component stays presentational.
export function PerformanceRadar({ points, height = 280 }) {
  if (!points?.length) return null;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={points}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-light)' }} />
        <Radar name="Performance" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
