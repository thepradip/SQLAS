import { useState } from 'react'
import { PageHeader, Card, Input, Select, Btn } from '../ui'
import { Save, Key, Database, Plug } from 'lucide-react'

export default function Settings() {
  const [saved, setSaved] = useState(false)
  const save = () => { setSaved(true); setTimeout(() => setSaved(false), 2000) }

  return (
    <div style={{ padding: 28 }}>
      <PageHeader title="Settings" subtitle="Configure API keys, database, and integrations" />

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <Card title="LLM Judge" icon={<Key size={15} />}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <Select label="Default judge model"
                options={['gpt-4o','gpt-4o-mini','claude-opus-4-7','claude-sonnet-4-6']} />
              <Input label="OpenAI API key" type="password" placeholder="sk-..." />
              <Input label="Anthropic API key" type="password" placeholder="sk-ant-..." />
              <Input label="Azure OpenAI endpoint" placeholder="https://your-resource.openai.azure.com/" />
              <Input label="Azure deployment name" placeholder="gpt-4o" />
            </div>
          </Card>

          <Card title="Database" icon={<Database size={15} />}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <Input label="Default database path" placeholder="./my_database.db" />
              <Select label="Default dialect" options={['SQLite','PostgreSQL','MySQL','Snowflake','BigQuery']} />
            </div>
          </Card>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <Card title="Integrations" icon={<Plug size={15} />}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {[
                { label: 'MLflow experiment', placeholder: 'sqlas-evaluation', desc: 'Log evaluation runs to MLflow' },
                { label: 'W&B project',        placeholder: 'sql-evals',        desc: 'Log to Weights & Biases' },
                { label: 'LangSmith project',  placeholder: 'my-sql-agent',     desc: 'Log to LangSmith' },
              ].map(({ label, placeholder, desc }) => (
                <div key={label}>
                  <Input label={label} placeholder={placeholder} />
                  <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 3 }}>{desc}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Evaluation Defaults">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <Select label="Default weight profile" options={['WEIGHTS_V4','WEIGHTS_V3','WEIGHTS_V2','WEIGHTS']} />
              <Input label="Pass threshold" type="number" placeholder="0.6" />
              <div style={{ display: 'flex', gap: 10 }}>
                <Input label="Spider directory" placeholder="./spider" />
                <Input label="BIRD directory"   placeholder="./bird" />
              </div>
            </div>
          </Card>

          <Btn onClick={save} icon={<Save size={14} />} full variant={saved ? 'success' : 'primary'}>
            {saved ? '✓ Saved' : 'Save Settings'}
          </Btn>
        </div>
      </div>
    </div>
  )
}
