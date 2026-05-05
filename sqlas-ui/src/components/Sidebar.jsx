import {
  LayoutDashboard, FlaskConical, Trophy, BarChart3,
  History, Settings, ChevronRight, Zap,
} from 'lucide-react'

const NAV = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { id: 'evaluate',  icon: FlaskConical,    label: 'Evaluate' },
  { id: 'benchmark', icon: Trophy,          label: 'Benchmark' },
  { id: 'results',   icon: BarChart3,       label: 'Results' },
  { id: 'history',   icon: History,         label: 'History' },
]

const S = {
  sidebar: {
    width: 220,
    minWidth: 220,
    background: '#ffffff',
    borderRight: '1px solid #e5e7eb',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    userSelect: 'none',
  },
  logo: {
    padding: '20px 18px 16px',
    borderBottom: '1px solid #f3f4f6',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  logoIcon: {
    width: 32, height: 32,
    background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
    borderRadius: 8,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: '#fff', flexShrink: 0,
  },
  logoText: { fontWeight: 700, fontSize: 15, color: '#111827', letterSpacing: '-0.3px' },
  logoSub:  { fontSize: 11, color: '#9ca3af', fontWeight: 400, marginTop: 1 },
  nav: { flex: 1, padding: '12px 10px', display: 'flex', flexDirection: 'column', gap: 2 },
  sectionLabel: {
    fontSize: 10, fontWeight: 700, color: '#9ca3af',
    letterSpacing: '0.1em', textTransform: 'uppercase',
    padding: '8px 8px 4px',
  },
  item: (active) => ({
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '8px 10px',
    borderRadius: 7,
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: active ? 600 : 400,
    color: active ? '#2563eb' : '#374151',
    background: active ? '#eff6ff' : 'transparent',
    transition: 'all 0.12s',
  }),
  footer: {
    padding: '12px 10px',
    borderTop: '1px solid #f3f4f6',
  },
  badge: {
    marginLeft: 'auto',
    background: 'linear-gradient(135deg, #eff6ff, #f5f3ff)',
    border: '1px solid #ddd6fe',
    color: '#7c3aed',
    fontSize: 10, fontWeight: 700,
    padding: '1px 7px',
    borderRadius: 10,
  },
}

export default function Sidebar({ active, onNav }) {
  return (
    <nav style={S.sidebar}>
      {/* Logo */}
      <div style={S.logo}>
        <div style={S.logoIcon}>
          <Zap size={16} strokeWidth={2.5} />
        </div>
        <div>
          <div style={S.logoText}>SQLAS</div>
          <div style={S.logoSub}>SQL Agent Evaluator</div>
        </div>
      </div>

      {/* Navigation */}
      <div style={S.nav}>
        <div style={S.sectionLabel}>Navigation</div>

        {NAV.map(({ id, icon: Icon, label }) => (
          <div
            key={id}
            style={S.item(active === id)}
            onClick={() => onNav(id)}
            onMouseEnter={e => { if (active !== id) e.currentTarget.style.background = '#f1f4f9' }}
            onMouseLeave={e => { if (active !== id) e.currentTarget.style.background = 'transparent' }}
          >
            <Icon size={16} strokeWidth={active === id ? 2.2 : 1.8} />
            <span>{label}</span>
            {id === 'benchmark' && <span style={S.badge}>NEW</span>}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div style={S.footer}>
        <div
          style={{ ...S.item(active === 'settings'), gap: 10 }}
          onClick={() => onNav('settings')}
          onMouseEnter={e => { if (active !== 'settings') e.currentTarget.style.background = '#f1f4f9' }}
          onMouseLeave={e => { if (active !== 'settings') e.currentTarget.style.background = 'transparent' }}
        >
          <Settings size={16} strokeWidth={1.8} />
          <span>Settings</span>
        </div>
        <div style={{ padding: '8px 10px 2px', fontSize: 11, color: '#9ca3af' }}>
          v2.6.0 · MIT License
        </div>
      </div>
    </nav>
  )
}
