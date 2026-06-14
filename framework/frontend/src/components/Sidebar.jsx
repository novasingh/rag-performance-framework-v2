import { Icons } from './Icons.jsx';
import { NAV_ITEMS } from '../constants.js';

export function Sidebar({ active, onNavigate, dark, onToggleTheme }) {
  // Group nav items by their section, preserving order.
  const sections = [];
  for (const item of NAV_ITEMS) {
    let group = sections.find((s) => s.name === item.section);
    if (!group) {
      group = { name: item.section, items: [] };
      sections.push(group);
    }
    group.items.push(item);
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-badge">🧭</div>
        <h1>RAG Performance Prediction Framework</h1>
        <p>Predicting RAG effectiveness from dataset-level factors</p>
      </div>

      <nav className="sidebar-nav">
        {sections.map((section) => (
          <div key={section.name}>
            <div className="nav-section">{section.name}</div>
            {section.items.map((item) => {
              const Icon = Icons[item.icon] || Icons.Framework;
              return (
                <button
                  key={item.id}
                  className={active === item.id ? 'active' : ''}
                  onClick={() => onNavigate(item.id)}
                >
                  <Icon />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button className="theme-toggle" onClick={onToggleTheme}>
          {dark ? <Icons.Light /> : <Icons.Dark />}
          <span>{dark ? 'Light mode' : 'Dark mode'}</span>
        </button>
      </div>
    </aside>
  );
}
