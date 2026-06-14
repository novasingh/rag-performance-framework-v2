import { useCallback, useEffect, useState } from 'react';

// ─── useApi ──────────────────────────────────────────────────────────────────
// Runs an async API function on mount (and when deps change), exposing
// { data, error, loading, reload }. Keeps every page free of fetch boilerplate.
export function useApi(fn, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const run = useCallback(() => {
    let active = true;
    setLoading(true);
    setError(null);
    Promise.resolve(fn())
      .then((res) => active && setData(res))
      .catch((err) => active && setError(err))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => run(), [run]);

  return { data, error, loading, reload: run };
}
