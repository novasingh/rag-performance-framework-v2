import { createContext, useContext } from 'react';

// Holds /metrics-meta (metric labels, scales, directions + domain labels) so
// components can format values without hardcoding metric definitions.
const MetaContext = createContext({ metrics: {}, domains: {}, domain_volatility: {} });

export function MetaProvider({ value, children }) {
  return <MetaContext.Provider value={value}>{children}</MetaContext.Provider>;
}

export function useMeta() {
  return useContext(MetaContext);
}

// Convenience: ordered [id, meta] pairs for iterating metrics consistently.
export function useMetricList() {
  const { metrics } = useMeta();
  return Object.entries(metrics || {});
}
