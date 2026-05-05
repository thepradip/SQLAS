import { PageHeader, Card, Badge, ScoreBar, Table } from '../ui'
import { Filter } from 'lucide-react'

const runs = [
  { id:'eval-050a', model:'GPT-4o',      n:25, overall:0.843, correct:0.820, quality:0.781, safety:0.971, verdict:'PASS',  date:'2025-05-04 14:22' },
  { id:'eval-050b', model:'Claude 3.5',  n:25, overall:0.812, correct:0.795, quality:0.754, safety:0.962, verdict:'PASS',  date:'2025-05-04 11:05' },
  { id:'spider-50', model:'GPT-4o',      n:50, overall:0.783, correct:0.760, quality:0.741, safety:0.998, verdict:'PASS',  date:'2025-05-04 09:14' },
  { id:'eval-049',  model:'GPT-4o-mini', n:25, overall:0.614, correct:0.590, quality:0.580, safety:0.871, verdict:'WARN',  date:'2025-05-03 16:40' },
  { id:'eval-048',  model:'GPT-4o',      n:25, overall:0.798, correct:0.780, quality:0.760, safety:0.955, verdict:'PASS',  date:'2025-05-03 10:20' },
]

const VT = { PASS: 'green', WARN: 'yellow', FAIL: 'red' }

export default function History() {
  return (
    <div style={{ padding: 28 }}>
      <PageHeader title="History" subtitle="Past evaluation runs" action={{ label: 'Filter', icon: <Filter size={13} /> }} />

      <Card title={`${runs.length} evaluation runs`}>
        <Table
          cols={['Run ID','Model','Tests','Overall','Correctness','Quality','Safety','Verdict','Date']}
          rows={runs.map(r => [
            <span style={{ fontFamily:'var(--mono)',fontSize:12,color:'#374151' }}>{r.id}</span>,
            <span style={{color:'#6b7280'}}>{r.model}</span>,
            r.n,
            <ScoreBar value={r.overall} />,
            <ScoreBar value={r.correct} />,
            <ScoreBar value={r.quality} />,
            <ScoreBar value={r.safety} />,
            <Badge type={VT[r.verdict]||'gray'}>{r.verdict}</Badge>,
            <span style={{color:'#9ca3af',fontSize:12}}>{r.date}</span>,
          ])}
        />
      </Card>
    </div>
  )
}
