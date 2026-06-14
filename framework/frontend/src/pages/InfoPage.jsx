import { api } from '../api/client.js';
import { useApi } from '../hooks/useApi.js';

// ─── Static research narrative (sourced from the dissertation) ────────────────
const RESEARCH = {
  title: 'A Framework for Predicting Retrieval-Augmented Generation System Performance Based on Relevant Factors',
  author: 'Akashdeep Singh',
  matric: '24072095',
  supervisor: 'AP Dr. Mumtaz Begum Binti Peer Mustafa',
  abstract:
    'RAG systems ground language-model responses in externally retrieved content, but organizations lack a way to predict how knowledge-base properties affect performance before deployment. This study quantifies how dataset freshness and source-type diversity relate to RAG performance across domains with different information volatility, and packages the result as a validated predictive framework that translates dataset characteristics into expected performance ranges with documented boundary conditions.',
};

const QUESTIONS = [
  { ro: 'RO1', rq: 'RQ1', text: 'How do dataset freshness and source type relate to RAG performance across high-, medium-, and low-volatility domains?' },
  { ro: 'RO2', rq: 'RQ2', text: 'What is the individual contribution of source-type configurations across multiple metrics, and does it vary by domain?' },
  { ro: 'RO3', rq: 'RQ3', text: 'Can a predictive framework estimate RAG performance from dataset-level factors across domains with different volatility?' },
  { ro: 'RO4', rq: 'RQ4', text: 'How accurately does the framework estimate performance across configurations, and what are its reliable operating boundaries?' },
];

const PHASES = [
  { n: '1', tag: 'RO1 · RQ1', title: 'Factor-Performance Analysis', desc: 'Build three domain datasets across four freshness levels and three source configurations; quantify freshness–performance relationships with two-way ANOVA and effect sizes.' },
  { n: '2', tag: 'RO2 · RQ2', title: 'Experimental Design & Execution', desc: 'Run 12 controlled conditions over a 2,400-query bank; collect automated metrics + human expert ratings (Cohen\'s κ ≥ 0.70) and the Source Diversity Index.' },
  { n: '3', tag: 'RO3 · RQ3', title: 'Framework Development', desc: 'Fit OLS regression, Random Forest, and XGBoost with 10-fold cross-validation, plus decay-curve modelling per domain — the predictive model that powers this tool.' },
  { n: '4', tag: 'RO4 · RQ4', title: 'Validation & Benchmarking', desc: 'Leave-one-condition-out validation establishes benchmark ranges (95% CI) and documents boundary conditions where estimates are reliable.' },
];

const DELIVERABLES = [
  { state: 'done', title: 'Factor–performance relationships (RO1)', desc: 'Per-condition metrics across 3 domains × 4 freshness levels × source configs — see Domain Dashboard & Freshness Impact.' },
  { state: 'done', title: 'Source-type contribution analysis (RO2)', desc: 'Source Diversity Index linked to outcomes, with human-eval reliability — see Source Impact.' },
  { state: 'done', title: 'Predictive framework (RO3)', desc: 'Cross-validated OLS/RF/XGBoost models with error bounds, served live by the Prediction Tool — see Model Info.' },
  { state: 'done', title: 'Validation + benchmark ranges (RO4)', desc: 'LOCO validation with 95% CI ranges and boundary conditions — see Validation.' },
  { state: 'done', title: 'Deployment guidelines', desc: 'Evidence-based update-cadence and source-mix rules per domain — see Deployment Guide.' },
];

export function InfoPage({ onNavigate }) {
  const { data: summary } = useApi(() => api.getSummary(), []);
  const he = summary?.human_evaluation || {};

  return (
    <>
      <section className="hero">
        <div className="eyebrow">Master's Dissertation · Predictive Framework</div>
        <h1>{RESEARCH.title}</h1>
        <p className="lead">{RESEARCH.abstract}</p>
        <div className="meta">
          <span><b>Author:</b> {RESEARCH.author} ({RESEARCH.matric})</span>
          <span><b>Supervisor:</b> {RESEARCH.supervisor}</span>
        </div>
        <div style={{ display: 'flex', gap: 10, marginTop: 24, flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={() => onNavigate?.('predict')}>Open the Prediction Tool →</button>
          <button className="btn btn-outline" style={{ color: '#fff', borderColor: 'rgba(255,255,255,.25)' }}
            onClick={() => onNavigate?.('deployment')}>Deployment Guide</button>
        </div>
      </section>

      <div className="stat-strip animate-fade">
        <div className="s"><div className="v">{summary?.total_observations ?? '—'}</div><div className="l">Query-level observations</div></div>
        <div className="s"><div className="v">{summary?.conditions ?? 12}</div><div className="l">Experimental conditions</div></div>
        <div className="s"><div className="v">{summary?.domains ?? 3}</div><div className="l">Domains (volatility levels)</div></div>
        <div className="s"><div className="v">{he.kappa != null ? `κ ${he.kappa.toFixed(2)}` : '—'}</div><div className="l">Inter-rater agreement</div></div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3>The problem</h3>
          <div className="prose">
            <p>RAG adoption has outpaced understanding of how <strong>controllable, dataset-level factors</strong> influence performance. Teams guess at update frequency and source mix instead of deciding from evidence.</p>
            <p>This research focuses on two factors that are <strong>stable across models, directly controllable, and demonstrably influential</strong>: dataset <strong>freshness</strong> and <strong>source-type</strong> composition.</p>
          </div>
        </div>
        <div className="card">
          <h3>The contribution</h3>
          <div className="prose">
            <p>A <strong>prospective</strong> predictive framework: it estimates expected performance ranges for an untested configuration <strong>before deployment</strong>, instead of requiring historical data for every new setup.</p>
            <p>It integrates both factors (and their interaction), generalizes across three volatility levels, and documents where its predictions can be trusted.</p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Research questions & objectives</h3>
        <div className="grid-2">
          {QUESTIONS.map((q) => (
            <div key={q.ro} className="recommendation">
              <h4><span className="tag tag-blue" style={{ marginRight: 8 }}>{q.ro} · {q.rq}</span></h4>
              <p>{q.text}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Methodology — four phases</h3>
        <div className="timeline">
          {PHASES.map((p) => (
            <div key={p.n} className="tl-item">
              <div className="tl-dot">{p.n}</div>
              <div className="tl-body">
                <div className="tl-tag">{p.tag}</div>
                <h4>{p.title}</h4>
                <p>{p.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="flex-between mb-4">
          <h3 style={{ margin: 0 }}>Deliverables — and whether this app fulfils them</h3>
          <span className="pill">5 / 5 delivered</span>
        </div>
        <ul className="checklist">
          {DELIVERABLES.map((d) => (
            <li key={d.title}>
              <span className={`check-icon ${d.state === 'done' ? 'check-done' : 'check-partial'}`}>
                {d.state === 'done' ? '✓' : '!'}
              </span>
              <div>
                <div className="c-title">{d.title}</div>
                <div className="c-desc">{d.desc}</div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}
