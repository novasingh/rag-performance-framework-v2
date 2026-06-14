// ─── UI configuration ────────────────────────────────────────────────────────
// Presentation-only constants (colors, labels, selectable options).
// No performance numbers live here — those always come from the API.

export const DOMAIN_COLORS = {
  technology: '#2196F3',
  healthcare: '#4CAF50',
  history: '#FF9800',
};

export const DOMAINS = [
  { id: 'technology', label: 'Technology', volatility: 'High Volatility' },
  { id: 'healthcare', label: 'Healthcare', volatility: 'Medium Volatility' },
  { id: 'history', label: 'History', volatility: 'Low Volatility' },
];

// Document-age presets used by the prediction tool (average age in days).
export const FRESHNESS_PRESETS = [
  { id: '<1week', label: '< 1 Week', days: 3.5 },
  { id: '1wk-1mo', label: '1 Week - 1 Month', days: 18.5 },
  { id: '1-6mo', label: '1 - 6 Months', days: 105 },
  { id: '>6mo', label: '> 6 Months', days: 365 },
];

// Generic source configurations understood by the backend.
export const SOURCE_OPTIONS = [
  { id: 'academic', label: 'Academic Only' },
  { id: 'academic+news', label: 'Academic + News' },
  { id: 'full', label: 'Academic + News + Technical' },
];

// Per-metric chart colors (keyed by the metric ids the API returns).
export const METRIC_COLORS = {
  precision_at_5: '#3b82f6',
  ndcg_at_5: '#8b5cf6',
  bertscore_f1: '#06b6d4',
  hallucination_rate: '#ef4444',
  human_eval_score: '#10b981',
};

export const NAV_ITEMS = [
  { id: 'info', label: 'Research Overview', icon: 'Info', section: 'About' },
  { id: 'predict', label: 'Prediction Tool', icon: 'Predict', section: 'Use the Framework' },
  { id: 'deployment', label: 'Deployment Guide', icon: 'Rocket', section: 'Use the Framework' },
  { id: 'dashboard', label: 'Domain Dashboard', icon: 'Dashboard', section: 'Evidence' },
  { id: 'sources', label: 'Source Impact', icon: 'Sources', section: 'Evidence' },
  { id: 'freshness', label: 'Freshness Impact', icon: 'Freshness', section: 'Evidence' },
  { id: 'validation', label: 'Validation (RQ4)', icon: 'Validate', section: 'Evidence' },
  { id: 'model', label: 'Model Info (RQ3)', icon: 'Findings', section: 'Evidence' },
];
