/* Shared UI primitives */

export function PageHeader({ title, subtitle, action }) {
  return (
    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom: 22 }}>
      <div>
        <h1 style={{ fontSize: 20, fontWeight: 700, color: '#111827', marginBottom: 3 }}>{title}</h1>
        {subtitle && <p style={{ fontSize: 13, color: '#6b7280' }}>{subtitle}</p>}
      </div>
      {action && (
        <Btn onClick={action.onClick} icon={action.icon} variant="primary" size="sm">
          {action.label}
        </Btn>
      )}
    </div>
  )
}

export function Card({ title, subtitle, action, icon, children }) {
  return (
    <div style={{
      background: '#ffffff',
      border: '1px solid #e5e7eb',
      borderRadius: 10,
      overflow: 'hidden',
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    }}>
      {(title || action) && (
        <div style={{
          padding: '14px 18px 12px',
          borderBottom: '1px solid #f3f4f6',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {icon && <span style={{ color: '#6b7280' }}>{icon}</span>}
            <div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#111827' }}>{title}</div>
              {subtitle && <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 1 }}>{subtitle}</div>}
            </div>
          </div>
          {action && (
            <button onClick={action.onClick} style={{
              display: 'flex', alignItems: 'center', gap: 5,
              fontSize: 12, color: '#2563eb', fontWeight: 500,
              background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'var(--font)',
            }}>
              {action.icon}{action.label}
            </button>
          )}
        </div>
      )}
      <div style={{ padding: '16px 18px' }}>{children}</div>
    </div>
  )
}

export function StatCard({ icon, label, value, delta, color }) {
  const colors = {
    blue:   { bg:'#eff6ff', text:'#1d4ed8', border:'#2563eb' },
    purple: { bg:'#f5f3ff', text:'#6d28d9', border:'#7c3aed' },
    green:  { bg:'#ecfdf5', text:'#065f46', border:'#059669' },
  }
  const c = colors[color] || colors.blue
  return (
    <div style={{
      background: '#fff', border: '1px solid #e5e7eb', borderRadius: 10,
      padding: '16px 18px', borderTop: `3px solid ${c.border}`,
      boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <div style={{ background: c.bg, color: c.text, borderRadius: 7, padding: 6, display: 'flex' }}>
          {icon}
        </div>
        <span style={{ fontSize: 12, color: '#6b7280', fontWeight: 500 }}>{label}</span>
      </div>
      <div style={{ fontSize: 24, fontWeight: 700, color: '#111827', marginBottom: 4 }}>{value}</div>
      {delta && <div style={{ fontSize: 11, color: '#6b7280' }}>{delta}</div>}
    </div>
  )
}

export function Badge({ type = 'gray', children }) {
  const styles = {
    green:  { bg:'#ecfdf5', color:'#065f46', border:'#a7f3d0' },
    yellow: { bg:'#fffbeb', color:'#92400e', border:'#fde68a' },
    red:    { bg:'#fef2f2', color:'#991b1b', border:'#fca5a5' },
    blue:   { bg:'#eff6ff', color:'#1e40af', border:'#bfdbfe' },
    purple: { bg:'#f5f3ff', color:'#5b21b6', border:'#ddd6fe' },
    gray:   { bg:'#f9fafb', color:'#374151', border:'#e5e7eb' },
  }
  const s = styles[type] || styles.gray
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 8px', borderRadius: 20,
      fontSize: 11, fontWeight: 600,
      background: s.bg, color: s.color, border: `1px solid ${s.border}`,
      whiteSpace: 'nowrap',
    }}>
      {children}
    </span>
  )
}

export function ScoreBar({ value }) {
  const v = parseFloat(value) || 0
  const color = v >= 0.8 ? '#059669' : v >= 0.6 ? '#2563eb' : v >= 0.4 ? '#d97706' : '#dc2626'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 100 }}>
      <div style={{ flex: 1, background: '#f3f4f6', borderRadius: 3, height: 6, overflow: 'hidden' }}>
        <div style={{ width: `${v * 100}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 600, color, minWidth: 38, textAlign: 'right' }}>
        {v.toFixed(3)}
      </span>
    </div>
  )
}

export function Table({ cols, rows }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {cols.map(c => (
              <th key={c} style={{
                padding: '8px 12px', textAlign: 'left',
                fontSize: 11, fontWeight: 600, color: '#6b7280',
                letterSpacing: '0.06em', textTransform: 'uppercase',
                borderBottom: '1px solid #e5e7eb', whiteSpace: 'nowrap',
              }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: i < rows.length - 1 ? '1px solid #f3f4f6' : 'none' }}>
              {row.map((cell, j) => (
                <td key={j} style={{ padding: '10px 12px', verticalAlign: 'middle' }}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Input({ label, value, onChange, placeholder, type = 'text', mono }) {
  return (
    <div>
      {label && <label style={{ fontSize: 12, fontWeight: 500, color: '#374151', display:'block', marginBottom: 5 }}>{label}</label>}
      <input
        type={type}
        value={value || ''}
        onChange={e => onChange && onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width: '100%', padding: '8px 10px',
          border: '1px solid #d1d5db', borderRadius: 7,
          fontSize: 13, color: '#111827', outline: 'none',
          fontFamily: mono ? 'var(--mono)' : 'var(--font)',
          transition: 'border-color 0.15s',
          background: '#fff',
        }}
        onFocus={e => e.target.style.borderColor = '#2563eb'}
        onBlur={e  => e.target.style.borderColor = '#d1d5db'}
      />
    </div>
  )
}

export function Select({ label, value, onChange, options, small }) {
  return (
    <div>
      {label && <label style={{ fontSize: 12, fontWeight: 500, color: '#374151', display:'block', marginBottom: 5 }}>{label}</label>}
      <select
        value={value || ''}
        onChange={e => onChange && onChange(e.target.value)}
        style={{
          padding: small ? '4px 8px' : '8px 10px',
          border: '1px solid #d1d5db', borderRadius: 7,
          fontSize: small ? 12 : 13, color: '#111827', outline: 'none',
          fontFamily: 'var(--font)', background: '#fff', cursor: 'pointer',
          width: small ? 'auto' : '100%',
        }}
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  )
}

export function Btn({ children, onClick, icon, loading, disabled, full, variant = 'primary', size }) {
  const variants = {
    primary: { bg: '#2563eb', hover: '#1d4ed8', color: '#fff', border: '#2563eb' },
    purple:  { bg: '#7c3aed', hover: '#6d28d9', color: '#fff', border: '#7c3aed' },
    success: { bg: '#059669', hover: '#047857', color: '#fff', border: '#059669' },
    outline: { bg: '#fff',    hover: '#f8f9fb', color: '#374151', border: '#d1d5db' },
  }
  const v = variants[variant] || variants.primary
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
        padding: size === 'sm' ? '7px 14px' : '9px 18px',
        background: disabled ? '#e5e7eb' : v.bg,
        color: disabled ? '#9ca3af' : v.color,
        border: `1px solid ${disabled ? '#e5e7eb' : v.border}`,
        borderRadius: 8, fontSize: 13, fontWeight: 600,
        cursor: disabled ? 'not-allowed' : 'pointer',
        width: full ? '100%' : 'auto',
        fontFamily: 'var(--font)',
        transition: 'all 0.15s',
        opacity: loading ? 0.75 : 1,
      }}
      onMouseEnter={e => { if (!disabled && !loading) e.currentTarget.style.background = v.hover }}
      onMouseLeave={e => { if (!disabled && !loading) e.currentTarget.style.background = v.bg }}
    >
      {icon && !loading && icon}
      {loading && <span style={{ width: 14, height: 14, border: '2px solid rgba(255,255,255,0.3)', borderTop: '2px solid #fff', borderRadius: '50%', animation: 'spin 0.8s linear infinite', display: 'inline-block' }} />}
      {children}
    </button>
  )
}
