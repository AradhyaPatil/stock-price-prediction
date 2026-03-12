import { useState } from 'react'
import { TrendingUp, Settings, ChevronLeft, ChevronRight, Search, Cpu } from 'lucide-react'

const POPULAR_STOCKS = [
  'AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSM','BRK-B','TSLA','LLY',
  'JPM','V','UNH','WMT','XOM','MA','JNJ','NFLX','AMD','PLTR','COIN',
  'RELIANCE.NS','TCS.NS','INFY.NS','HDFCBANK.NS','ICICIBANK.NS',
  'SBIN.NS','BAJFINANCE.NS','TITAN.NS','WIPRO.NS','HCLTECH.NS',
]

function SliderField({ label, value, min, max, step, onChange }) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between gap-3">
        <label className="text-[11px] uppercase tracking-[0.2em] text-text-muted">{label}</label>
        <span className="rounded-full border border-line-subtle px-2.5 py-0.5 text-[11px] font-semibold text-accent-gold">{value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full cursor-pointer"
      />
      <div className="flex justify-between text-[10px] text-text-muted">
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  )
}

export default function Sidebar({ params, onFetch, onTrain, loading, training }) {
  const [local, setLocal] = useState({ ...params })
  const [collapsed, setCollapsed] = useState(false)

  const set = (k, v) => setLocal(p => ({ ...p, [k]: v }))

  if (collapsed) {
    return (
      <div className="w-14 flex-shrink-0 bg-bg-secondary border-r border-line-subtle flex flex-col items-center pt-5">
        <button
          onClick={() => setCollapsed(false)}
          className="rounded-full border border-line-subtle bg-white/[0.03] p-2 text-text-muted hover:text-text-primary transition"
          title="Open sidebar"
        >
          <ChevronRight size={18} />
        </button>
      </div>
    )
  }

  return (
    <aside className="w-80 flex-shrink-0 bg-bg-secondary border-r border-line-subtle flex flex-col overflow-hidden">
      <div className="px-5 py-5 border-b border-line-subtle">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-text-muted">
              <TrendingUp size={14} className="text-accent-gold" />
              Market Desk
            </div>
            <div className="mt-2 font-display text-2xl text-white">Control Panel</div>
            <div className="mt-1 text-sm text-text-soft">Configure symbol, history, and model settings.</div>
          </div>
          <button
            onClick={() => setCollapsed(true)}
            className="rounded-full border border-line-subtle bg-white/[0.03] p-2 text-text-muted hover:text-text-primary transition"
            title="Collapse sidebar"
          >
            <ChevronLeft size={16} />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-5 space-y-6">

        <div>
          <label className="block text-[11px] uppercase tracking-[0.2em] text-text-muted mb-2">
            Popular Stocks
          </label>
          <select
            className="w-full bg-bg-panel text-text-primary rounded-xl px-4 py-3 text-sm border border-line-subtle focus:outline-none focus:border-accent-gold"
            defaultValue=""
            onChange={e => { if (e.target.value) set('ticker', e.target.value) }}
          >
            <option value="">— Select or type below —</option>
            {POPULAR_STOCKS.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-[11px] uppercase tracking-[0.2em] text-text-muted mb-2">
            Stock Ticker
          </label>
          <div className="flex items-center bg-bg-panel border border-line-subtle rounded-xl px-3">
            <Search size={15} className="text-text-muted" />
            <input
              type="text"
              className="w-full bg-transparent text-white rounded-xl px-3 py-3 text-sm font-mono font-semibold tracking-[0.15em] focus:outline-none"
              value={local.ticker}
              onChange={e => set('ticker', e.target.value.toUpperCase().trim())}
              placeholder="e.g. AAPL"
            />
          </div>
        </div>

        <div>
          <label className="block text-[11px] uppercase tracking-[0.2em] text-text-muted mb-2">
            Historical Period
          </label>
          <select
            className="w-full bg-bg-panel text-text-primary rounded-xl px-4 py-3 text-sm border border-line-subtle focus:outline-none focus:border-accent-gold"
            value={local.period}
            onChange={e => set('period', e.target.value)}
          >
            {['1y','2y','5y','10y','max'].map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <div className="rounded-2xl border border-line-subtle bg-white/[0.02] p-4">
          <div className="flex items-center gap-2 mb-4">
            <Settings size={14} className="text-text-muted" />
            <span className="text-sm font-semibold text-text-primary">Model Parameters</span>
          </div>
          <div className="space-y-5">
            <SliderField label="Lookback Window" value={local.lookback}
            min={20} max={120} step={10} onChange={v => set('lookback', v)} />
            <SliderField label="Training Epochs" value={local.epochs}
            min={10} max={500} step={10} onChange={v => set('epochs', v)} />
            <SliderField label="Forecast Horizon" value={local.forecast_days}
            min={5} max={90} step={5} onChange={v => set('forecast_days', v)} />
          </div>
        </div>

        <div className="rounded-2xl border border-line-subtle bg-[rgba(202,167,106,0.06)] p-4">
          <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.2em] text-text-muted">
            <Cpu size={14} className="text-accent-gold" />
            Model Note
          </div>
          <p className="mt-3 text-sm text-text-soft leading-6">
            Higher epochs improve fit stability but increase runtime. Keep lookback and forecast balanced for better practical performance.
          </p>
        </div>
      </div>

      <div className="px-5 py-5 border-t border-line-subtle space-y-3">
        <button
          onClick={() => onFetch(local)}
          disabled={loading || !local.ticker}
          className="w-full py-3 rounded-xl font-semibold text-sm transition
            bg-white/[0.04] hover:bg-white/[0.08] text-text-primary border border-line-subtle
            disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? 'Loading Market Data...' : 'Load Market Data'}
        </button>
        <button
          onClick={() => onTrain(local)}
          disabled={training || !local.ticker}
          className="w-full py-3 rounded-xl font-bold text-sm transition
            bg-[linear-gradient(135deg,#d0b07a,#b48a49)] hover:brightness-105 text-[#0d1721]
            disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {training ? 'Training Model...' : 'Train Model & Forecast'}
        </button>
        
      </div>
    </aside>
  )
}
