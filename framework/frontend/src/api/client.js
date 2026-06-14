// ─── API client ────────────────────────────────────────────────────────────
// Thin wrapper around fetch. All endpoints are proxied to the FastAPI backend
// by Vite (see vite.config.js), so we use same-origin relative paths here.

const API_BASE = import.meta.env.VITE_API_BASE ?? '';

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch (cause) {
    throw new ApiError(
      'Cannot reach the backend. Is the API running on port 8000?',
      0,
    );
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

// ─── Endpoint functions ──────────────────────────────────────────────────────
export const api = {
  getSummary: () => request('/summary'),
  getConditions: () => request('/conditions'),
  getSources: () => request('/sources'),
  getFreshness: () => request('/freshness'),
  getValidation: () => request('/validation'),
  getModelInfo: () => request('/model-info'),
  getMetricsMeta: () => request('/metrics-meta'),
  getDeploymentGuidelines: () => request('/deployment-guidelines'),
  getHealth: () => request('/health'),
  predict: (payload) =>
    request('/predict', { method: 'POST', body: JSON.stringify(payload) }),
};

export { ApiError };
