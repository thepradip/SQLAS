import { useState } from 'react'
import { Trophy, Info, Play, DollarSign } from 'lucide-react'
import { PageHeader, Card, Btn, Select, Input, Badge } from '../ui'

const DIFFICULTIES = ['easy','medium','hard','extra hard']
const TYPES = ['simple','aggregation','join','nested']

export default function Benchmark() {
  const [dataset,    setDataset]    = useState('Spider')
  const [dataDir,    setDataDir]    = useState('./spider')
  const [nSamples,   setNSamples]   = useState(50)
  const [difficulty, setDifficulty] = useState([])
  const [queryTypes, setQueryTypes] = useState([])
  const [seed,       setSeed]       = useState(42)
  const [running,    setRunning]    = useState(false)
  const [progress,   setProgress]   = useState(0)

  const estCost = (nSamples * 0.005).toFixed(2)
  const estTime = Math.max(1, Math.round(nSamples * 0.04))

  const toggle = (arr, setArr, val) =>
    setArr(arr.includes(val) ? arr.filter(v => v !== val) : [...arr, val])

  const run = () => {
    setRunning(true); setProgress(0)
    const iv = setInterval(() => {
      setProgress(p => { if (p >= 100) { clearInterval(iv); setRunning(false); return 100; } return p + 2; })
    }, estTime * 20)
  }

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Benchmark"
        subtitle="Evaluate against Spider and BIRD academic benchmarks"
      />

      {/* Dataset selector */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
        {['Spider','BIRD'].map(d => (
          <div key={d}
            onClick={() => { setDataset(d); setDataDir('./' + d.toLowerCase()) }}
            style={{
              flex: 1, padding: '16px 20px', borderRadius: 10, cursor: 'pointer',
              border: `2px solid ${dataset === d ? '#2563eb' : '#e5e7eb'}`,
              background: dataset === d ? '#eff6ff' : '#fff',
              transition: 'all 0.15s',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ fontWeight: 600, color: dataset === d ? '#1d4ed8' : '#111827', marginBottom: 4 }}>
                  {d}
                </div>
                <div style={{ fontSize: 12, color: '#6b7280' }}>
                  {d === 'Spider' ? '10,181 questions · 200 databases · 4 difficulty levels' : '12,751 questions · real noisy DBs · VES metric'}
                </div>
              </div>
              {d === 'Spider' && <Badge type="blue">Popular</Badge>}
              {d === 'BIRD'   && <Badge type="purple">Harder</Badge>}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 18, alignItems: 'start' }}>
        {/* Left */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <Card title="Sampling Configuration"
            subtitle={`Smart stratified sampling saves 60× on tokens vs running all questions`}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <Input label={`${dataset} directory`} value={dataDir} onChange={setDataDir}
                placeholder={`./spider`} />

              <div>
                <label style={{ fontSize: 12, fontWeight: 500, color: '#374151', display: 'block', marginBottom: 6 }}>
                  Sample size: <strong style={{ color: '#2563eb' }}>{nSamples}</strong> questions
                </label>
                <input type="range" min={10} max={200} value={nSamples}
                  onChange={e => setNSamples(+e.target.value)}
                  style={{ width: '100%', accentColor: '#2563eb' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#9ca3af', marginTop: 4 }}>
                  <span>10 (fast)</span><span>50 (balanced)</span><span>200 (thorough)</span>
                </div>
              </div>

              <div>
                <label style={{ fontSize: 12, fontWeight: 500, color: '#374151', display: 'block', marginBottom: 8 }}>
                  Difficulty filter <span style={{ color: '#9ca3af', fontWeight: 400 }}>(empty = all)</span>
                </label>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {DIFFICULTIES.map(d => (
                    <FilterChip key={d} label={d} active={difficulty.includes(d)}
                      onClick={() => toggle(difficulty, setDifficulty, d)} />
                  ))}
                </div>
              </div>

              {dataset === 'Spider' && (
                <div>
                  <label style={{ fontSize: 12, fontWeight: 500, color: '#374151', display: 'block', marginBottom: 8 }}>
                    Query type filter <span style={{ color: '#9ca3af', fontWeight: 400 }}>(empty = all)</span>
                  </label>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {TYPES.map(t => (
                      <FilterChip key={t} label={t} active={queryTypes.includes(t)}
                        onClick={() => toggle(queryTypes, setQueryTypes, t)} color="purple" />
                    ))}
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', gap: 12 }}>
                <Input label="Random seed" value={seed} onChange={v => setSeed(+v)} type="number" />
              </div>
            </div>
          </Card>

          {/* Progress */}
          {running && (
            <Card title="Running benchmark…">
              <div style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
                  <span>Progress</span><span>{progress}%</span>
                </div>
                <div style={{ background: '#e5e7eb', borderRadius: 4, height: 8, overflow: 'hidden' }}>
                  <div style={{ background: 'linear-gradient(90deg,#2563eb,#7c3aed)', height: '100%', width: `${progress}%`, borderRadius: 4, transition: 'width 0.3s' }} />
                </div>
              </div>
              <div style={{ fontSize: 12, color: '#6b7280' }}>
                Evaluating question {Math.round(progress * nSamples / 100)} of {nSamples}…
              </div>
            </Card>
          )}
        </div>

        {/* Right */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <Card title="Cost Estimate">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                ['Questions',  nSamples],
                ['LLM calls',  `~${nSamples * 3}`],
                ['Safety checks', `${nSamples} (free)`],
                ['Est. cost',  `~$${estCost}`],
                ['Est. time',  `~${estTime} min`],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: '#6b7280' }}>{k}</span>
                  <span style={{ fontWeight: 500, color: '#111827' }}>{v}</span>
                </div>
              ))}
              <div style={{ height: 1, background: '#f3f4f6', margin: '4px 0' }} />
              <div style={{ fontSize: 11, color: '#9ca3af', lineHeight: 1.6 }}>
                Full {dataset} ({dataset === 'Spider' ? '1,034' : '1,534'} questions) would cost ~${dataset === 'Spider' ? '15–30' : '20–40'}.
                Smart sampling gives representative results at 60× lower cost.
              </div>
            </div>
          </Card>

          <Card title="Stratified Distribution">
            {[
              ['Easy',       '20%', Math.round(nSamples * 0.20)],
              ['Medium',     '30%', Math.round(nSamples * 0.30)],
              ['Hard',       '30%', Math.round(nSamples * 0.30)],
              ['Extra hard', '20%', Math.round(nSamples * 0.20)],
            ].map(([label, pct, n]) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <span style={{ fontSize: 12, color: '#374151', width: 70 }}>{label}</span>
                <div style={{ flex: 1, background: '#f3f4f6', borderRadius: 4, height: 6, overflow: 'hidden' }}>
                  <div style={{ background: '#2563eb', height: '100%', width: pct, borderRadius: 4, opacity: 0.7 }} />
                </div>
                <span style={{ fontSize: 11, color: '#6b7280', width: 50, textAlign: 'right' }}>{pct} ({n})</span>
              </div>
            ))}
          </Card>

          <Btn onClick={run} loading={running} icon={<Play size={15} />} full>
            {running ? `Running ${dataset}…` : `Run ${dataset} Benchmark`}
          </Btn>
        </div>
      </div>
    </div>
  )
}

function FilterChip({ label, active, onClick, color = 'blue' }) {
  const colors = {
    blue:   { active: { bg: '#eff6ff', color: '#2563eb', border: '#bfdbfe' }, inactive: { bg: '#fff', color: '#6b7280', border: '#e5e7eb' } },
    purple: { active: { bg: '#f5f3ff', color: '#7c3aed', border: '#ddd6fe' }, inactive: { bg: '#fff', color: '#6b7280', border: '#e5e7eb' } },
  }
  const c = active ? colors[color].active : colors[color].inactive
  return (
    <button onClick={onClick} style={{
      padding: '4px 12px', borderRadius: 20, fontSize: 12, fontWeight: active ? 600 : 400,
      cursor: 'pointer', border: `1.5px solid ${c.border}`,
      background: c.bg, color: c.color, fontFamily: 'var(--font)', transition: 'all 0.12s',
    }}>
      {label}
    </button>
  )
}
