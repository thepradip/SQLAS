import { useState } from 'react'
import { Plus, Trash2, Play, Upload } from 'lucide-react'
import { PageHeader, Card, Btn, Input, Select } from '../ui'

const defaultCases = [
  { question: 'How many active users are there?', gold: 'SELECT COUNT(*) FROM users WHERE status = \'active\'', category: 'easy' },
  { question: 'Show top 5 products by revenue',   gold: '',                                                    category: 'medium' },
]

export default function Evaluate() {
  const [cases, setCases]     = useState(defaultCases)
  const [agentUrl, setAgentUrl] = useState('http://localhost:8000')
  const [judgeModel, setJudge]  = useState('gpt-4o')
  const [weights, setWeights]   = useState('WEIGHTS_V4')
  const [running, setRunning]   = useState(false)
  const [done, setDone]         = useState(false)

  const addCase = () => setCases(p => [...p, { question: '', gold: '', category: 'medium' }])
  const delCase = i => setCases(p => p.filter((_,j) => j !== i))
  const updCase = (i, k, v) => setCases(p => p.map((c,j) => j === i ? { ...c, [k]: v } : c))

  const run = () => {
    setRunning(true)
    setTimeout(() => { setRunning(false); setDone(true) }, 2200)
  }

  return (
    <div style={{ padding: 28 }}>
      <PageHeader title="Evaluate" subtitle="Run SQLAS evaluation against your SQL agent" />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 18, alignItems: 'start' }}>

        {/* Left: test cases */}
        <div>
          <Card title="Test Cases" action={{ label: 'Import JSON', icon: <Upload size={13} /> }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {cases.map((c, i) => (
                <div key={i} style={{ background: '#f8f9fb', border: '1px solid #e5e7eb', borderRadius: 8, padding: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: '#374151' }}>Test {i + 1}</span>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <Select value={c.category} onChange={v => updCase(i, 'category', v)}
                        options={['easy','medium','hard','extra hard']} small />
                      <button onClick={() => delCase(i)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#9ca3af', display: 'flex', alignItems: 'center' }}>
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                  <Input
                    label="Question"
                    value={c.question}
                    onChange={v => updCase(i, 'question', v)}
                    placeholder="How many active users are there?"
                  />
                  <div style={{ marginTop: 8 }}>
                    <Input
                      label="Gold SQL (optional)"
                      value={c.gold}
                      onChange={v => updCase(i, 'gold', v)}
                      placeholder="SELECT COUNT(*) FROM users WHERE status = 'active'"
                      mono
                    />
                  </div>
                </div>
              ))}

              <button onClick={addCase} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                padding: '10px', border: '1.5px dashed #d1d5db', borderRadius: 8,
                color: '#6b7280', fontSize: 13, cursor: 'pointer', background: 'none',
                fontFamily: 'var(--font)',
              }}>
                <Plus size={15} /> Add test case
              </button>
            </div>
          </Card>
        </div>

        {/* Right: config */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <Card title="Agent Configuration">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <Input label="Agent endpoint" value={agentUrl} onChange={setAgentUrl} placeholder="http://localhost:8000" />
              <Select label="LLM Judge" value={judgeModel} onChange={setJudge}
                options={['gpt-4o','gpt-4o-mini','claude-opus-4-7','claude-sonnet-4-6']} />
              <Select label="Weight profile" value={weights} onChange={setWeights}
                options={['WEIGHTS_V4','WEIGHTS_V3','WEIGHTS_V2','WEIGHTS']} />
              <Input label="Database path" placeholder="./my_database.db" />
            </div>
          </Card>

          <Card title="Cost Estimate">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[
                ['Test cases',  cases.length],
                ['LLM calls',   `~${cases.length * 6}`],
                ['Est. cost',   `~$${(cases.length * 0.002).toFixed(3)}`],
                ['Est. time',   `~${Math.max(1, Math.round(cases.length * 0.4))} min`],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: '#6b7280' }}>{k}</span>
                  <span style={{ fontWeight: 500, color: '#111827' }}>{v}</span>
                </div>
              ))}
            </div>
          </Card>

          <Btn
            onClick={run}
            loading={running}
            disabled={cases.filter(c=>c.question).length === 0}
            icon={<Play size={15} />}
            full
          >
            {running ? 'Running evaluation…' : 'Run Evaluation'}
          </Btn>

          {done && (
            <div style={{ background: '#ecfdf5', border: '1px solid #a7f3d0', borderRadius: 8, padding: 14, textAlign: 'center' }}>
              <div style={{ color: '#065f46', fontWeight: 600, fontSize: 13 }}>✓ Evaluation complete</div>
              <div style={{ color: '#047857', fontSize: 12, marginTop: 4 }}>View results in the Results tab</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
