import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// All backend endpoints are proxied to the FastAPI server on port 8000.
const BACKEND = 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: Object.fromEntries(
      ['/predict', '/batch-predict', '/model-info', '/validation', '/conditions',
       '/summary', '/sources', '/freshness', '/metrics-meta', '/deployment-guidelines', '/health']
        .map((p) => [p, BACKEND])
    ),
  },
});
