import { Icons } from './Icons.jsx';

const ICON = { info: Icons.Info, warning: Icons.Alert, danger: Icons.Alert, tip: Icons.Bulb };

// Markdown-style admonition box (info / warning / danger / tip).
export function Callout({ type = 'info', title, children }) {
  const Icon = ICON[type] || Icons.Info;
  return (
    <div className={`callout callout-${type}`}>
      <span className="ic"><Icon /></span>
      <div>
        {title && <h4>{title}</h4>}
        <p>{children}</p>
      </div>
    </div>
  );
}
