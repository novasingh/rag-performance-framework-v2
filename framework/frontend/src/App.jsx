import { useEffect, useState } from 'react';
import { api } from './api/client.js';
import { useApi } from './hooks/useApi.js';
import { MetaProvider } from './context/MetaContext.jsx';
import { Sidebar } from './components/Sidebar.jsx';
import { AsyncBoundary } from './components/ui/StateView.jsx';

import { InfoPage } from './pages/InfoPage.jsx';
import { PredictPage } from './pages/PredictPage.jsx';
import { DeploymentPage } from './pages/DeploymentPage.jsx';
import { DashboardPage } from './pages/DashboardPage.jsx';
import { SourcesPage } from './pages/SourcesPage.jsx';
import { FreshnessPage } from './pages/FreshnessPage.jsx';
import { ValidationPage } from './pages/ValidationPage.jsx';
import { ModelInfoPage } from './pages/ModelInfoPage.jsx';

const PAGES = {
  info: InfoPage,
  predict: PredictPage,
  deployment: DeploymentPage,
  dashboard: DashboardPage,
  sources: SourcesPage,
  freshness: FreshnessPage,
  validation: ValidationPage,
  model: ModelInfoPage,
};

export default function App() {
  // The Research Overview (Info) page is the default landing view.
  const [page, setPage] = useState('info');
  const [dark, setDark] = useState(false);

  const { data: meta, error, loading, reload } = useApi(() => api.getMetricsMeta(), []);

  useEffect(() => {
    document.body.classList.toggle('dark', dark);
  }, [dark]);

  useEffect(() => {
    // Scroll to top whenever the page changes.
    document.querySelector('.main')?.scrollTo({ top: 0, behavior: 'smooth' });
  }, [page]);

  const Page = PAGES[page] || InfoPage;

  return (
    <MetaProvider value={meta || { metrics: {}, domains: {}, domain_volatility: {} }}>
      <Sidebar active={page} onNavigate={setPage} dark={dark} onToggleTheme={() => setDark((d) => !d)} />
      <main className="main">
        <AsyncBoundary loading={loading} error={error} onRetry={reload}>
          <Page onNavigate={setPage} />
        </AsyncBoundary>
      </main>
    </MetaProvider>
  );
}
