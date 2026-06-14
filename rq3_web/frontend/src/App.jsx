import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Database, FileText, CheckCircle, BarChart2, TrendingDown } from 'lucide-react';
import { 
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, Legend, CartesianGrid, LineChart, Line
} from 'recharts';
import { AlertTriangle, Download, Plus } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [models, setModels] = useState(null);
  
  // Predictor State
  const [domain, setDomain] = useState('technology');
  const [ageDays, setAgeDays] = useState(14);
  const [diversity, setDiversity] = useState(0.35);
  const [predictions, setPredictions] = useState(null);
  const [scenarios, setScenarios] = useState([]);

  // Boundary condition check (from RQ4 findings)
  const isBoundaryCondition = domain === 'history' && diversity > 0.15;

  // Fetch initial models data and run first prediction
  useEffect(() => {
    const init = async () => {
      try {
        const modelRes = await axios.get(`${API_BASE}/models`);
        setModels(modelRes.data);
        runPrediction(domain, ageDays, diversity);
      } catch (err) {
        console.error("Failed to load models. Is the FastAPI backend running?", err);
      }
    };
    init();
  }, []);

  // Debounced prediction fetcher
  useEffect(() => {
    const timer = setTimeout(() => {
      runPrediction(domain, ageDays, diversity);
    }, 300);
    return () => clearTimeout(timer);
  }, [domain, ageDays, diversity]);

  const runPrediction = async (d, a, div) => {
    try {
      const res = await axios.post(`${API_BASE}/predict`, {
        domain: d,
        avg_age_days: Number(a),
        source_diversity_index: Number(div)
      });
      setPredictions(res.data.predictions);
    } catch (err) {
      console.error(err);
    }
  };

  if (!models || !predictions) {
    return (
      <div className="dashboard-container" style={{ justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center' }}>
          <Activity className="animate-spin" size={48} color="var(--accent-cyan)" />
          <h2 style={{ marginTop: '1rem' }}>Loading Predictor Models...</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Ensure FastAPI backend is running on port 8000</p>
        </div>
      </div>
    );
  }

  const handleSaveScenario = () => {
    const newScenario = {
      id: Date.now(),
      domain,
      ageDays,
      diversity,
      ndcg: predictions.ndcg_at_5.expected.toFixed(3),
      precision: predictions.precision_at_5.expected.toFixed(3),
      hallucination: predictions.hallucination_rate.expected.toFixed(3),
      bertscore: predictions.bertscore_f1.expected.toFixed(3)
    };
    setScenarios([...scenarios, newScenario]);
  };

  const exportCSV = () => {
    if (scenarios.length === 0) return;
    const headers = ['Domain', 'Age (Days)', 'Diversity', 'nDCG@5', 'Precision@5', 'Hallucination', 'BERTScore'];
    const rows = scenarios.map(s => [s.domain, s.ageDays, s.diversity, s.ndcg, s.precision, s.hallucination, s.bertscore]);
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "rq4_scenarios.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Format data for Radar Chart
  const radarData = [
    { metric: 'BERTScore', value: predictions.bertscore_f1.expected * 100, fullMark: 100 },
    { metric: 'Precision@5', value: predictions.precision_at_5.expected * 100, fullMark: 100 },
    { metric: 'nDCG@5', value: predictions.ndcg_at_5.expected * 100, fullMark: 100 },
    { metric: 'Accuracy (Inv Hallucination)', value: (1 - predictions.hallucination_rate.expected) * 100, fullMark: 100 },
  ];

  // Format data for Feature Importance Chart
  const featImportances = models.random_forest.bertscore_f1.feature_importances;
  const barData = Object.entries(featImportances).map(([key, val]) => ({
    name: key.replace(/_/g, ' '),
    importance: val
  })).sort((a, b) => b.importance - a.importance);

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Predictive Framework (RQ3)</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Dynamically estimate RAG effectiveness based on dataset factors</p>
      </header>

      <div className="tabs">
        <button 
          className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          Dashboard
        </button>
        <button 
          className={`tab-btn ${activeTab === 'models' ? 'active' : ''}`}
          onClick={() => setActiveTab('models')}
        >
          Raw JSON Explorer
        </button>
      </div>

      {activeTab === 'dashboard' ? (
        <div className="dashboard-grid">
          {/* LEFT: Controls Panel */}
          <div className="glass-panel controls-panel">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <Database size={20} color="var(--accent-purple)"/> Dataset Parameters
            </h3>
            
            <div className="control-group">
              <div className="control-label">
                <span>Domain</span>
                <span className="control-value" style={{textTransform: 'capitalize'}}>{domain}</span>
              </div>
              <select value={domain} onChange={(e) => setDomain(e.target.value)}>
                <option value="technology">Technology (High Volatility)</option>
                <option value="healthcare">Healthcare (Medium Volatility)</option>
                <option value="history">History (Low Volatility)</option>
              </select>
            </div>

            <div className="control-group" style={{marginTop: '1rem'}}>
              <div className="control-label">
                <span>Average Data Age (Days)</span>
                <span className="control-value">{ageDays}</span>
              </div>
              <input 
                type="range" 
                min="0" max="730" step="1"
                value={ageDays} 
                onChange={(e) => setAgeDays(e.target.value)} 
              />
              <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)'}}>
                <span>0 (Fresh)</span>
                <span>730 (2 Years)</span>
              </div>
            </div>

            <div className="control-group" style={{marginTop: '1rem'}}>
              <div className="control-label">
                <span>Source Diversity Index (SDI)</span>
                <span className="control-value">{diversity}</span>
              </div>
              <input 
                type="range" 
                min="0" max="0.45" step="0.01"
                value={diversity} 
                onChange={(e) => setDiversity(e.target.value)} 
              />
              <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)'}}>
                <span>0.0 (Single)</span>
                <span>0.45 (Max Mix)</span>
              </div>
            </div>
            
            <button className="action-btn" style={{marginTop: '1rem'}} onClick={handleSaveScenario}>
              <Plus size={18} /> Save to Scenarios Table
            </button>
          </div>

          {/* RIGHT: Results Area */}
          <div className="results-area">
            
            {isBoundaryCondition && (
              <div className="boundary-warning">
                <AlertTriangle className="warning-icon" size={24} />
                <div className="warning-text">
                  <h4>RQ4 Boundary Condition Exceeded</h4>
                  <p>The predictive framework is unreliable for Historical domains with high source diversity. OLS regression extrapolates poorly here (MAE &gt; 1.0) because historical facts do not decay, causing non-linear interactions with source diversity.</p>
                </div>
              </div>
            )}

            {/* Metrics Grid */}
            <div className="metrics-grid">
              <div className="glass-panel metric-card">
                <div className="metric-header">
                  <CheckCircle size={16} color="var(--accent-cyan)" /> nDCG@5
                </div>
                <div className="metric-value">{predictions.ndcg_at_5.expected.toFixed(3)}</div>
                <div className="metric-bounds">± {predictions.ndcg_at_5.mae_margin.toFixed(3)} MAE</div>
              </div>
              
              <div className="glass-panel metric-card emerald">
                <div className="metric-header">
                  <CheckCircle size={16} color="var(--accent-emerald)" /> Precision@5
                </div>
                <div className="metric-value">{predictions.precision_at_5.expected.toFixed(3)}</div>
                <div className="metric-bounds">± {predictions.precision_at_5.mae_margin.toFixed(3)} MAE</div>
              </div>

              <div className="glass-panel metric-card rose">
                <div className="metric-header">
                  <Activity size={16} color="var(--accent-rose)" /> Hallucination Rate
                </div>
                <div className="metric-value">{predictions.hallucination_rate.expected.toFixed(3)}</div>
                <div className="metric-bounds">± {predictions.hallucination_rate.mae_margin.toFixed(3)} MAE</div>
              </div>

              <div className="glass-panel metric-card purple">
                <div className="metric-header">
                  <FileText size={16} color="var(--accent-purple)" /> BERTScore
                </div>
                <div className="metric-value">{predictions.bertscore_f1.expected.toFixed(3)}</div>
                <div className="metric-bounds">± {predictions.bertscore_f1.mae_margin.toFixed(3)} MAE</div>
              </div>
            </div>

            {/* Charts Grid */}
            <div className="charts-grid">
              <div className="glass-panel chart-container">
                <div className="chart-title" style={{display: 'flex', alignItems: 'center', gap:'0.5rem'}}>
                  <TrendingDown size={18} color="var(--accent-cyan)"/> Performance Profile
                </div>
                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                      <PolarGrid stroke="rgba(255,255,255,0.1)" />
                      <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                      <Radar name="Performance" dataKey="value" stroke="var(--accent-cyan)" fill="var(--accent-cyan)" fillOpacity={0.4} />
                      <RechartsTooltip contentStyle={{backgroundColor: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.1)'}} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="glass-panel chart-container">
                <div className="chart-title" style={{display: 'flex', alignItems: 'center', gap:'0.5rem'}}>
                  <BarChart2 size={18} color="var(--accent-purple)"/> RF Feature Importance
                </div>
                <div className="chart-wrapper">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={barData} margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(255,255,255,0.05)" />
                      <XAxis type="number" tick={{ fill: 'var(--text-secondary)' }} />
                      <YAxis dataKey="name" type="category" width={100} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <RechartsTooltip contentStyle={{backgroundColor: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.1)'}} />
                      <Bar dataKey="importance" fill="var(--accent-purple)" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="glass-panel json-viewer">
          <h3>Raw Models Payload (rq3_models.json)</h3>
          <p style={{color: 'var(--text-secondary)', marginBottom: '1rem', fontSize: '0.875rem'}}>
            Shows all calculated OLS equations, RF Feature Importances, and cross-validation metrics.
          </p>
          <pre>{JSON.stringify(models, null, 2)}</pre>
        </div>
      )}

      {/* Scenarios Table (Rendered only on Dashboard tab if there are scenarios) */}
      {activeTab === 'dashboard' && scenarios.length > 0 && (
        <div className="glass-panel scenarios-container">
          <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
            <h3>Saved Research Scenarios</h3>
            <button className="action-btn" style={{width: 'auto', padding: '0.5rem 1rem'}} onClick={exportCSV}>
              <Download size={16} /> Export CSV
            </button>
          </div>
          <table className="scenarios-table">
            <thead>
              <tr>
                <th>Domain</th>
                <th>Age (Days)</th>
                <th>SDI (Diversity)</th>
                <th>nDCG@5</th>
                <th>Precision@5</th>
                <th>Hallucination</th>
                <th>BERTScore</th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map(s => (
                <tr key={s.id}>
                  <td style={{textTransform: 'capitalize'}}>{s.domain}</td>
                  <td>{s.ageDays}</td>
                  <td>{s.diversity}</td>
                  <td>{s.ndcg}</td>
                  <td>{s.precision}</td>
                  <td>{s.hallucination}</td>
                  <td>{s.bertscore}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

    </div>
  );
}

export default App;
