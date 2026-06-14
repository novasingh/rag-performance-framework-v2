// Shared loading / error / empty states so pages don't repeat the boilerplate.

export function Loading({ label = 'Loading...' }) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: 40 }}>
      <div className="shimmer" style={{ height: 12, width: '40%', margin: '0 auto 12px' }} />
      <p className="text-sm text-secondary">{label}</p>
    </div>
  );
}

export function ErrorView({ error, onRetry }) {
  return (
    <div className="card" style={{ borderLeft: '4px solid var(--danger)' }}>
      <h3 style={{ color: 'var(--danger)' }}>Something went wrong</h3>
      <p className="text-sm text-secondary mb-4">{error?.message || 'Unknown error.'}</p>
      {onRetry && (
        <button className="btn btn-outline" onClick={onRetry}>Retry</button>
      )}
    </div>
  );
}

// Wraps a page body: shows loading/error states, otherwise renders children.
export function AsyncBoundary({ loading, error, onRetry, children }) {
  if (loading) return <Loading />;
  if (error) return <ErrorView error={error} onRetry={onRetry} />;
  return children;
}
