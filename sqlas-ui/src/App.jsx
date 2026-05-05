import { useState } from 'react'
import Sidebar from './components/Sidebar'
import Dashboard from './components/pages/Dashboard'
import Evaluate  from './components/pages/Evaluate'
import Benchmark from './components/pages/Benchmark'
import Results   from './components/pages/Results'
import History   from './components/pages/History'
import Settings  from './components/pages/Settings'

const PAGES = {
  dashboard: Dashboard,
  evaluate:  Evaluate,
  benchmark: Benchmark,
  results:   Results,
  history:   History,
  settings:  Settings,
}

export default function App() {
  const [page, setPage] = useState('dashboard')
  const Page = PAGES[page] || Dashboard

  return (
    <div className="app">
      <Sidebar active={page} onNav={setPage} />
      <main style={{ flex: 1, overflow: 'auto', background: '#f8f9fb' }}>
        <Page onNav={setPage} />
      </main>
    </div>
  )
}
