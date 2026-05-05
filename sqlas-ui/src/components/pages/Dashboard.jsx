import { TrendingUp, ShieldCheck, FlaskConical, CheckCircle2, AlertTriangle, ArrowUpRight } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'
import { PageHeader, Card, StatCard, Badge, ScoreBar, Table } from '../ui'

const trend = [
  { date: 'Apr 28', score: 0.71 }, { date: 'Apr 29', score: 0.74 },
  { date: 'Apr 30', score: 0.76 }, { date: 'May 1',  score: 0.78 },
  { date: 'May 2',  score: 0.80 }, { date: 'May 3',  score: 0.82 },
  { date: 'May 4',  score: 0.84 },
]

const catData = [
  { cat: 'Correctness', score: 0.82 },
  { cat: 'Quality',     score: 0.78 },
  { cat: 'Safety',      score: 0.97 },
  { cat: 'Agentic',     score: 0.74 },
]

const recent = [
  { id: 1, run: 'eval-2025-05-04-a', model: 'GPT-4o',    tests: 25, score: 0.84, verdict: 'PASS',  time: '2m ago' },
  { id: 2, run: 'eval-2025-05-04-b', model: 'Claude 3.5',tests: 25, score: 0.81, verdict: 'PASS',  time: '1h ago' },
  { id: 3, run: 'spider-50-gpt4',    model: 'GPT-4o',    tests: 50, score: 0.78, verdict: 'PASS',  time: '3h ago' },
  { id: 4, run: 'eval-2025-05-03',   model: 'GPT-4o-mini',tests:25, score: 0.61, verdict: 'WARN',  time: '1d ago' },
]

export default function Dashboard({ onNav }) {
  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Dashboard"
        subtitle="Evaluation overview and recent activity"
        action={{ label: 'New Evaluation', onClick: () => onNav('evaluate') }}
      />

      {/* Stat cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 22 }}>
        <StatCard icon={<FlaskConical size={18} />} label="Total Runs" value="48" delta="+6 this week" color="blue" />
        <StatCard icon={<TrendingUp size={18} />}   label="Avg Score"  value="0.817" delta="+0.03 vs last week" color="purple" />
        <StatCard icon={<CheckCircle2 size={18} />} label="Pass Rate"  value="87%" delta="21 of 24 passed" color="green" />
        <StatCard icon={<ShieldCheck size={18} />}  label="Safety Score" value="0.971" delta="All runs safe" color="blue" />
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 14, marginBottom: 22 }}>
        <Card title="Score Trend" subtitle="Last 7 days">
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={trend} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
              <defs>
                <linearGradient id="blueGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#2563eb" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
              <YAxis domain={[0.65, 0.90]} tick={{ fontSize: 11, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, fontSize: 12 }}
                formatter={v => [v.toFixed(4), 'Score']}
              />
              <Area type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={2} fill="url(#blueGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        <Card title="By Category">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={catData} margin={{ top: 4, right: 4, bottom: 0, left: -24 }} barSize={18}>
              <XAxis dataKey="cat" tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: '#9ca3af' }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, fontSize: 12 }}
                formatter={v => [v.toFixed(3), 'Score']}
              />
              <Bar dataKey="score" fill="#7c3aed" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Recent runs */}
      <Card title="Recent Runs" action={{ label: 'View all', onClick: () => onNav('history') }}>
        <Table
          cols={['Run', 'Model', 'Tests', 'Score', 'Verdict', 'Time']}
          rows={recent.map(r => [
            <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: '#374151' }}>{r.run}</span>,
            <span style={{ color: '#6b7280' }}>{r.model}</span>,
            r.tests,
            <ScoreBar value={r.score} />,
            <Badge type={r.verdict === 'PASS' ? 'green' : 'yellow'}>{r.verdict}</Badge>,
            <span style={{ color: '#9ca3af', fontSize: 12 }}>{r.time}</span>,
          ])}
        />
      </Card>
    </div>
  )
}
