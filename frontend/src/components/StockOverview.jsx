import { Building2, Landmark, TrendingUp, Trophy, Wallet } from 'lucide-react'

function MetricCard({ label, value, color = 'default', icon: Icon }) {
  const colors = {
    green:   'text-accent-green',
    red:     'text-accent-red',
    blue:    'text-accent-blue',
    gold:    'text-accent-gold',
    default: 'text-text-primary',
  }
  return (
    <div className="market-card rounded-2xl p-4 text-center transition-transform hover:-translate-y-0.5 min-h-[170px] flex flex-col justify-center">
      <div className="flex items-center justify-center gap-2">
        {Icon ? <Icon size={15} className="text-text-muted" /> : null}
        <div className="text-[11px] uppercase tracking-[0.2em] text-text-muted">{label}</div>
      </div>
      <div className={`mt-4 text-[24px] font-semibold leading-tight ${colors[color] ?? colors.default} break-words`}>{value}</div>
    </div>
  )
}

function fmt(n) {
  if (n == null) return 'N/A'
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`
  if (n >= 1e9)  return `$${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6)  return `$${(n / 1e6).toFixed(2)}M`
  return `$${Number(n).toLocaleString()}`
}

export default function StockOverview({ info, chart }) {
  const price    = info?.currentPrice ?? info?.regularMarketPrice ?? chart?.close?.at(-1)
  const prev     = info?.previousClose ?? chart?.close?.at(-2)
  const change   = price != null && prev != null ? price - prev : 0
  const pct      = prev ? (change / prev) * 100 : 0
  const sign     = change >= 0 ? '+' : ''
  const chColor  = change >= 0 ? 'green' : 'red'
  const high52   = info?.fiftyTwoWeekHigh ?? (chart?.high ? Math.max(...chart.high.filter(Boolean)) : null)

  return (
    <section className="space-y-4">
      <div className="flex items-end justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.26em] text-text-muted">Market Snapshot</div>
          <h2 className="mt-2 font-display text-3xl text-white">{info?.longName || info?.shortName || 'Selected Company'}</h2>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricCard label="Company" value={info?.longName || info?.shortName || 'N/A'} icon={Building2} />
        <MetricCard label="Current Price" value={price ? `$${Number(price).toFixed(2)}` : 'N/A'} color="blue" icon={Wallet} />
        <MetricCard label="Daily Change" value={prev ? `${sign}${pct.toFixed(2)}%` : 'N/A'} color={chColor} icon={TrendingUp} />
        <MetricCard label="52W High" value={high52 ? `$${Number(high52).toFixed(2)}` : 'N/A'} color="green" icon={Trophy} />
        <MetricCard label="Market Cap" value={fmt(info?.marketCap)} color="gold" icon={Landmark} />
      </div>
    </section>
  )
}
