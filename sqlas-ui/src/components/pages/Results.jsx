import { useState } from 'react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { PageHeader, Card, Badge, ScoreBar, Table } from '../ui'
import { Download, ChevronDown, ChevronRight, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'

/* ── Mock data representing a full SQLASScores object ── */
const SCORES = {
  // Correctness
  execution_accuracy: 0.820, syntax_valid: 1.0, semantic_equivalence: 0.840,
  result_set_similarity: 0.790, result_coverage: 0.950,
  // Context
  context_precision: 0.810, context_recall: 0.780, entity_recall: 0.760, noise_robustness: 0.820,
  // SQL Quality
  sql_quality: 0.810, schema_compliance: 0.960, complexity_match: 0.830,
  // Production
  efficiency_score: 0.880, data_scan_efficiency: 0.920, execution_success: 0.960,
  empty_result_penalty: 0.960, row_explosion_detected: false, execution_time_ms: 42, result_row_count: 128,
  // Response
  faithfulness: 0.950, answer_relevance: 0.900, answer_completeness: 0.840, fluency: 0.920,
  // Safety
  read_only_compliance: 1.0, sql_injection_score: 1.0, prompt_injection_score: 0.980,
  pii_access_score: 0.880, pii_leakage_score: 1.0, guardrail_score: 0.972, safety_score: 0.971,
  // Visualization
  chart_spec_validity: 0.920, chart_data_alignment: 0.940, chart_llm_validation: 0.850, visualization_score: 0.903,
  // Agentic
  agent_mode: 'react', steps_taken: 4, steps_efficiency: 0.800, schema_grounding: 1.0,
  planning_quality: 0.870, tool_use_accuracy: 0.840, agentic_score: 0.860,
  plan_compliance_score: 1.0, first_attempt_score: 1.0,
  // Cache
  cache_hit: true, cache_type: 'semantic', tokens_saved: 8600, few_shot_count: 2,
  // Composite
  correctness_score: 0.820, quality_score: 0.781, safety_composite_score: 0.971,
  verdict: 'PASS', overall_score: 0.843,
}

const CATEGORIES = [
  {
    id: 'correctness', label: 'Correctness', weight: '50% of composite',
    color: '#2563eb', bg: '#eff6ff',
    metrics: [
      { key: 'execution_accuracy',     label: 'Execution Accuracy',     weight: '50%', desc: 'Row-by-row numeric match vs gold SQL' },
      { key: 'semantic_equivalence',   label: 'Semantic Equivalence',   weight: '25%', desc: 'LLM judge: does SQL answer the intent?' },
      { key: 'result_coverage',        label: 'Result Coverage',        weight: '15%', desc: 'Truncation penalty for GROUP BY queries' },
      { key: 'result_set_similarity',  label: 'Result Set Similarity',  weight: '10%', desc: 'Jaccard similarity on result sets' },
      { key: 'syntax_valid',           label: 'Syntax Valid',           weight: '—',   desc: 'sqlglot AST parse check' },
    ],
  },
  {
    id: 'context', label: 'Context Quality', weight: 'RAGAS-mapped',
    color: '#7c3aed', bg: '#f5f3ff',
    metrics: [
      { key: 'context_precision',  label: 'Context Precision',  weight: '3%',  desc: 'Only relevant schema elements used' },
      { key: 'context_recall',     label: 'Context Recall',     weight: '3%',  desc: 'All required schema elements present' },
      { key: 'entity_recall',      label: 'Entity Recall',      weight: '2%',  desc: 'Strict entity match (tables, columns, literals)' },
      { key: 'noise_robustness',   label: 'Noise Robustness',   weight: '2%',  desc: 'Irrelevant schema elements ignored' },
    ],
  },
  {
    id: 'quality', label: 'SQL Quality', weight: '20% of quality score',
    color: '#0891b2', bg: '#ecfeff',
    metrics: [
      { key: 'sql_quality',       label: 'SQL Quality',        weight: '20%', desc: 'LLM: join/aggregation/filter correctness' },
      { key: 'schema_compliance', label: 'Schema Compliance',  weight: '10%', desc: 'All referenced tables/columns exist in schema' },
      { key: 'complexity_match',  label: 'Complexity Match',   weight: '10%', desc: 'Query complexity appropriate for the question' },
    ],
  },
  {
    id: 'production', label: 'Production', weight: '7% of composite',
    color: '#0369a1', bg: '#f0f9ff',
    metrics: [
      { key: 'efficiency_score',      label: 'VES Efficiency',         weight: '3%',  desc: 'Valid Efficiency Score: correct AND fast' },
      { key: 'data_scan_efficiency',  label: 'Data Scan Efficiency',   weight: '3%',  desc: 'No full scans, no row explosion' },
      { key: 'execution_success',     label: 'Execution Success',      weight: '3%',  desc: 'SQL executed without runtime error' },
      { key: 'empty_result_penalty',  label: 'Empty Result Penalty',   weight: '2%',  desc: 'Penalises empty results when non-empty expected' },
      { key: 'execution_time_ms',     label: 'Execution Time',         weight: 'ms',  desc: 'Raw SQL execution latency in milliseconds' },
      { key: 'result_row_count',      label: 'Result Row Count',       weight: 'count',desc: 'Number of rows returned' },
      { key: 'row_explosion_detected',label: 'Row Explosion Detected', weight: 'bool',desc: 'Detected suspiciously large JOIN result' },
    ],
  },
  {
    id: 'response', label: 'Response Quality', weight: '8% of composite',
    color: '#7c3aed', bg: '#faf5ff',
    metrics: [
      { key: 'faithfulness',        label: 'Faithfulness',         weight: '20%', desc: 'Claims grounded in SQL result data' },
      { key: 'answer_relevance',    label: 'Answer Relevance',     weight: '15%', desc: 'Response directly answers the question' },
      { key: 'answer_completeness', label: 'Answer Completeness',  weight: '10%', desc: 'All key data points surfaced' },
      { key: 'fluency',             label: 'Fluency',              weight: '5%',  desc: 'Readability score (1–5, normalised)' },
    ],
  },
  {
    id: 'safety', label: 'Safety & Guardrails', weight: '15–20% of composite',
    color: '#059669', bg: '#ecfdf5',
    metrics: [
      { key: 'read_only_compliance',    label: 'Read-Only Compliance',     weight: '25%', desc: 'AST-validated: no DDL/DML anywhere in SQL' },
      { key: 'sql_injection_score',     label: 'SQL Injection Score',      weight: '15%', desc: 'Stacked queries, UNION SELECT, tautologies' },
      { key: 'prompt_injection_score',  label: 'Prompt Injection Score',   weight: '10%', desc: 'Jailbreak patterns in question/response' },
      { key: 'pii_access_score',        label: 'PII Access Score',         weight: '10%', desc: 'PII columns accessed in SQL' },
      { key: 'pii_leakage_score',       label: 'PII Leakage Score',        weight: '5%',  desc: 'PII patterns in response text' },
      { key: 'guardrail_score',         label: 'Guardrail Score',          weight: '35%', desc: 'Composite of all five safety dimensions' },
    ],
  },
  {
    id: 'visualization', label: 'Visualization', weight: '7% of composite',
    color: '#d97706', bg: '#fffbeb',
    metrics: [
      { key: 'chart_spec_validity',    label: 'Chart Spec Validity',    weight: '—', desc: 'Renderable chart payload structure' },
      { key: 'chart_data_alignment',   label: 'Chart Data Alignment',   weight: '—', desc: 'Chart keys match SQL result columns' },
      { key: 'chart_llm_validation',   label: 'Chart LLM Validation',   weight: '—', desc: 'Chart type appropriate for data and question' },
      { key: 'visualization_score',    label: 'Visualization Score',    weight: '—', desc: 'Composite of all three chart metrics' },
    ],
  },
  {
    id: 'agentic', label: 'Agentic Quality', weight: '10% of V4',
    color: '#dc2626', bg: '#fef2f2',
    metrics: [
      { key: 'plan_compliance_score', label: 'Plan Compliance',       weight: '—',   desc: 'create_plan called before execute_sql?' },
      { key: 'first_attempt_score',   label: 'First Attempt Success', weight: '—',   desc: 'SQL succeeded without retries?' },
      { key: 'steps_efficiency',      label: 'Steps Efficiency',      weight: '30%', desc: 'Step count vs optimal (default 3)' },
      { key: 'schema_grounding',      label: 'Schema Grounding',      weight: '30%', desc: 'Schema inspected before querying?' },
      { key: 'planning_quality',      label: 'Planning Quality',      weight: '40%', desc: 'LLM judge on reasoning sequence' },
      { key: 'tool_use_accuracy',     label: 'Tool Use Accuracy',     weight: '—',   desc: 'LLM judge on tool selection' },
      { key: 'agentic_score',         label: 'Agentic Score',         weight: '10%', desc: 'Composite of efficiency + grounding + planning' },
      { key: 'agent_mode',            label: 'Agent Mode',            weight: 'str', desc: '"react" or "pipeline"' },
      { key: 'steps_taken',           label: 'Steps Taken',           weight: 'int', desc: 'Number of tool calls in ReAct loop' },
    ],
  },
  {
    id: 'cache', label: 'Cache Performance', weight: 'Informational',
    color: '#6b7280', bg: '#f9fafb',
    metrics: [
      { key: 'cache_hit',      label: 'Cache Hit',       weight: '—',     desc: 'Was this query served from cache?' },
      { key: 'cache_type',     label: 'Cache Type',      weight: 'str',   desc: '"exact", "semantic", or ""' },
      { key: 'tokens_saved',   label: 'Tokens Saved',    weight: 'int',   desc: 'Tokens saved vs full LLM pipeline' },
      { key: 'few_shot_count', label: 'Few-Shot Count',  weight: 'int',   desc: 'Verified examples injected into prompt' },
    ],
  },
]

const radarData = [
  { dim: 'Correctness', score: 0.820 },
  { dim: 'SQL Quality', score: 0.870 },
  { dim: 'Response',    score: 0.903 },
  { dim: 'Safety',      score: 0.971 },
  { dim: 'Context',     score: 0.793 },
  { dim: 'Agentic',     score: 0.860 },
]

function MetricValue({ k, v }) {
  if (typeof v === 'boolean') {
    return v
      ? <span style={{ color: '#dc2626', fontWeight: 600, fontSize: 12 }}>⚠ Yes</span>
      : <span style={{ color: '#059669', fontWeight: 600, fontSize: 12 }}>✓ No</span>
  }
  if (typeof v === 'string') {
    return <Badge type={v === 'PASS' ? 'green' : v === 'react' ? 'purple' : v === 'semantic' ? 'blue' : 'gray'}>{v}</Badge>
  }
  if (typeof v === 'number') {
    if (k.endsWith('_ms'))    return <span style={{ fontSize: 12, color: '#6b7280' }}>{v}ms</span>
    if (k.endsWith('_count') || k === 'steps_taken' || k === 'result_row_count' || k === 'tokens_saved')
      return <span style={{ fontSize: 12, fontWeight: 600, color: '#374151' }}>{v.toLocaleString()}</span>
    return <ScoreBar value={v} />
  }
  return <span style={{ fontSize: 12, color: '#6b7280' }}>{String(v)}</span>
}

function CategoryBlock({ cat, scores }) {
  const [open, setOpen] = useState(true)
  const catScore = scores[`${cat.id}_score`] || scores[`${cat.id}_composite_score`]

  return (
    <div style={{ border: '1px solid #e5e7eb', borderRadius: 10, overflow: 'hidden', marginBottom: 12 }}>
      {/* Header */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 16px', background: '#fff', border: 'none', cursor: 'pointer',
          borderBottom: open ? '1px solid #f3f4f6' : 'none', fontFamily: 'var(--font)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 10, height: 10, borderRadius: '50%', background: cat.color, flexShrink: 0 }} />
          <span style={{ fontWeight: 600, fontSize: 14, color: '#111827' }}>{cat.label}</span>
          <span style={{ fontSize: 11, color: '#9ca3af' }}>{cat.metrics.length} metrics · {cat.weight}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {catScore !== undefined && typeof catScore === 'number' && (
            <span style={{ fontSize: 13, fontWeight: 700, color: cat.color }}>{catScore.toFixed(3)}</span>
          )}
          {open ? <ChevronDown size={15} color="#9ca3af" /> : <ChevronRight size={15} color="#9ca3af" />}
        </div>
      </button>

      {/* Metric rows */}
      {open && (
        <div>
          {cat.metrics.map((m, i) => (
            <div key={m.key} style={{
              display: 'flex', alignItems: 'center',
              padding: '10px 16px',
              background: i % 2 === 1 ? '#fafafa' : '#fff',
              borderBottom: i < cat.metrics.length - 1 ? '1px solid #f3f4f6' : 'none',
            }}>
              {/* Label + description */}
              <div style={{ width: 220, flexShrink: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: '#374151' }}>{m.label}</div>
                <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 1 }}>{m.desc}</div>
              </div>
              {/* Weight */}
              <div style={{ width: 60, flexShrink: 0, fontSize: 11, color: '#d1d5db', fontWeight: 500 }}>
                {m.weight}
              </div>
              {/* Value */}
              <div style={{ flex: 1 }}>
                <MetricValue k={m.key} v={scores[m.key] ?? 0} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Results() {
  const [view, setView] = useState('summary') // 'summary' | 'all' | 'safety'

  return (
    <div style={{ padding: 28 }}>
      <PageHeader
        title="Results"
        subtitle="All 45 SQLAS metrics across 9 categories"
        action={{ label: 'Export JSON', icon: <Download size={14} /> }}
      />

      {/* View toggle */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, background: '#f3f4f6', padding: 4, borderRadius: 8, width: 'fit-content' }}>
        {[['summary','Summary'], ['all','All 45 Metrics'], ['safety','Safety Detail']].map(([id,label]) => (
          <button key={id} onClick={() => setView(id)} style={{
            padding: '6px 16px', borderRadius: 6, fontSize: 13, fontWeight: view===id ? 600 : 400,
            background: view===id ? '#fff' : 'transparent',
            color: view===id ? '#111827' : '#6b7280',
            border: view===id ? '1px solid #e5e7eb' : '1px solid transparent',
            cursor: 'pointer', fontFamily: 'var(--font)',
            boxShadow: view===id ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
          }}>{label}</button>
        ))}
      </div>

      {view === 'summary' && <SummaryView scores={SCORES} />}
      {view === 'all'     && <AllMetricsView scores={SCORES} />}
      {view === 'safety'  && <SafetyView scores={SCORES} />}
    </div>
  )
}

function SummaryView({ scores }) {
  return (
    <>
      {/* Top 5 metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 12, marginBottom: 18 }}>
        {[
          ['Overall Score',    scores.overall_score,          '#2563eb'],
          ['Correctness',      scores.correctness_score,      '#2563eb'],
          ['Quality',          scores.quality_score,          '#7c3aed'],
          ['Safety',           scores.safety_composite_score, '#059669'],
          ['Pass Rate',        0.88,                          '#7c3aed'],
        ].map(([label, val, color]) => (
          <div key={label} style={{
            background:'#fff', border:'1px solid #e5e7eb', borderRadius:10,
            padding:'14px 16px', borderTop:`3px solid ${color}`,
          }}>
            <div style={{ fontSize:11, color:'#6b7280', marginBottom:5, fontWeight:500 }}>{label}</div>
            <div style={{ fontSize:22, fontWeight:700, color:'#111827' }}>
              {label === 'Pass Rate' ? '88%' : val.toFixed(4)}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14, marginBottom:18 }}>
        {/* 3-dimension verdict */}
        <Card title="Three-Dimension Verdict">
          {[
            { label:'Correctness', score:scores.correctness_score,      threshold:0.5, color:'#2563eb' },
            { label:'Quality',     score:scores.quality_score,          threshold:0.6, color:'#7c3aed' },
            { label:'Safety',      score:scores.safety_composite_score, threshold:0.9, color:'#059669' },
          ].map(({ label, score, threshold, color }) => (
            <div key={label} style={{ marginBottom:14 }}>
              <div style={{ display:'flex', justifyContent:'space-between', marginBottom:5 }}>
                <span style={{ fontSize:13, fontWeight:500 }}>{label}</span>
                <div style={{ display:'flex', gap:8, alignItems:'center' }}>
                  <span style={{ fontSize:11, color:'#9ca3af' }}>≥ {threshold}</span>
                  <span style={{ fontSize:13, fontWeight:700, color }}>{score.toFixed(3)}</span>
                  <span style={{ fontSize:11, fontWeight:600, color: score >= threshold ? '#059669' : '#dc2626' }}>
                    {score >= threshold ? 'PASS' : 'FAIL'}
                  </span>
                </div>
              </div>
              <div style={{ background:'#f3f4f6', borderRadius:4, height:8, overflow:'hidden', position:'relative' }}>
                <div style={{ background:color, height:'100%', width:`${score*100}%`, borderRadius:4 }} />
                <div style={{ position:'absolute', top:0, left:`${threshold*100}%`, width:2, height:'100%', background:'#6b7280', opacity:0.4 }} />
              </div>
            </div>
          ))}
          <div style={{ background:'#eff6ff', border:'1px solid #bfdbfe', borderRadius:8, padding:'10px 14px', textAlign:'center' }}>
            <span style={{ fontSize:13, fontWeight:700, color:'#1d4ed8' }}>Verdict: PASS</span>
          </div>
        </Card>

        {/* Radar */}
        <Card title="Category Radar">
          <ResponsiveContainer width="100%" height={230}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="dim" tick={{ fontSize: 11, fill: '#6b7280' }} />
              <Tooltip contentStyle={{ background:'#fff', border:'1px solid #e5e7eb', borderRadius:8, fontSize:12 }}
                formatter={v => [v.toFixed(3),'Score']} />
              <Radar dataKey="score" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.15} strokeWidth={2} dot />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Category summary bar */}
      <Card title="All 9 Categories at a Glance">
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:10 }}>
          {CATEGORIES.map(cat => {
            const s = SCORES[cat.id + '_score'] || SCORES[cat.id + '_composite_score'] || SCORES[cat.id + 'ness_score']
            const disp = typeof s === 'number' ? s : (SCORES[cat.metrics[0]?.key] || 0)
            return (
              <div key={cat.id} style={{
                display:'flex', alignItems:'center', gap:10, padding:'10px 12px',
                border:'1px solid #e5e7eb', borderRadius:8, background:'#fafafa',
              }}>
                <div style={{ width:8, height:8, borderRadius:'50%', background:cat.color, flexShrink:0 }} />
                <div style={{ flex:1, minWidth:0 }}>
                  <div style={{ fontSize:12, fontWeight:500, color:'#374151' }}>{cat.label}</div>
                  <div style={{ fontSize:10, color:'#9ca3af' }}>{cat.metrics.length} metrics</div>
                </div>
                <span style={{ fontSize:13, fontWeight:700, color:cat.color }}>
                  {typeof disp === 'number' ? disp.toFixed(3) : '—'}
                </span>
              </div>
            )
          })}
        </div>
      </Card>
    </>
  )
}

function AllMetricsView({ scores }) {
  return (
    <div>
      <div style={{ fontSize:13, color:'#6b7280', marginBottom:16 }}>
        All 45 SQLAS metrics grouped by category. Click a category header to expand/collapse.
      </div>
      {CATEGORIES.map(cat => (
        <CategoryBlock key={cat.id} cat={cat} scores={scores} />
      ))}
    </div>
  )
}

function SafetyView({ scores }) {
  const safetyMetrics = CATEGORIES.find(c => c.id === 'safety').metrics
  const issues = safetyMetrics.filter(m => typeof scores[m.key] === 'number' && scores[m.key] < 1.0)

  return (
    <div>
      {/* Safety verdict card */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:12, marginBottom:18 }}>
        {[
          ['Guardrail Score',     scores.guardrail_score,    '#059669'],
          ['Read-Only',          scores.read_only_compliance, '#2563eb'],
          ['SQL Injection',      scores.sql_injection_score,  '#2563eb'],
          ['PII Protection',     Math.min(scores.pii_access_score, scores.pii_leakage_score), '#7c3aed'],
        ].map(([label, val, color]) => (
          <div key={label} style={{
            background:'#fff', border:'1px solid #e5e7eb', borderRadius:10,
            padding:'14px 16px', borderTop:`3px solid ${color}`,
          }}>
            <div style={{ fontSize:11, color:'#6b7280', marginBottom:5, fontWeight:500 }}>{label}</div>
            <div style={{ fontSize:22, fontWeight:700, color:'#111827' }}>{val.toFixed(3)}</div>
            {val === 1.0
              ? <div style={{ fontSize:11, color:'#059669', marginTop:4, fontWeight:500 }}>✓ Clean</div>
              : <div style={{ fontSize:11, color:'#dc2626', marginTop:4, fontWeight:500 }}>⚠ Issue detected</div>}
          </div>
        ))}
      </div>

      {/* All safety metrics */}
      <Card title="Safety Metrics Detail">
        {safetyMetrics.map((m, i) => (
          <div key={m.key} style={{
            display:'flex', alignItems:'center', padding:'12px 0',
            borderBottom: i < safetyMetrics.length-1 ? '1px solid #f3f4f6' : 'none',
          }}>
            <div style={{ width:28, flexShrink:0 }}>
              {scores[m.key] === 1.0
                ? <CheckCircle2 size={16} color="#059669" />
                : scores[m.key] < 0.7
                  ? <XCircle size={16} color="#dc2626" />
                  : <AlertTriangle size={16} color="#d97706" />}
            </div>
            <div style={{ width:220, flexShrink:0 }}>
              <div style={{ fontSize:13, fontWeight:500, color:'#374151' }}>{m.label}</div>
              <div style={{ fontSize:11, color:'#9ca3af' }}>{m.desc}</div>
            </div>
            <div style={{ flex:1 }}>
              <ScoreBar value={typeof scores[m.key] === 'number' ? scores[m.key] : 0} />
            </div>
          </div>
        ))}
      </Card>

      {issues.length > 0 && (
        <div style={{ marginTop:14, background:'#fef2f2', border:'1px solid #fca5a5', borderRadius:10, padding:'16px 18px' }}>
          <div style={{ fontSize:13, fontWeight:600, color:'#991b1b', marginBottom:10 }}>Issues Detected</div>
          {issues.map(m => (
            <div key={m.key} style={{ display:'flex', gap:8, fontSize:13, color:'#b91c1c', marginBottom:6 }}>
              <span>⚠</span>
              <span><strong>{m.label}</strong> scored {(SCORES[m.key]).toFixed(3)} — {m.desc}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
