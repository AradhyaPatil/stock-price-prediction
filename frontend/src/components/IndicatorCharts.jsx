import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip, ReferenceLine, Legend,
  ComposedChart, Bar, Cell,
} from 'recharts'

const ttStyle = {
  background: '#102539',
  border: '1px solid rgba(222,234,245,0.12)',
  borderRadius: 12,
  color: '#edf3f8',
  fontSize: 12,
}

function ChartCard({ title, children }) {
  return (
    <div className="market-card rounded-[24px] p-5 md:p-6">
      <div className="section-divider">
        <h3 className="font-display text-2xl text-white">{title}</h3>
      </div>
      {children}
    </div>
  )
}

export default function IndicatorCharts({ chart }) {
  if (!chart) return null

  const SAMPLE = 200

  const rsiData = chart.dates
    .map((d, i) => ({ date: d.slice(5), rsi: chart.rsi?.[i] }))
    .filter(d => d.rsi != null)
    .slice(-SAMPLE)

  const macdData = chart.dates
    .map((d, i) => {
      const m = chart.macd?.[i]
      const s = chart.macdSignal?.[i]
      return {
        date: d.slice(5),
        macd: m,
        signal: s,
        hist: m != null && s != null ? m - s : null,
      }
    })
    .filter(d => d.macd != null)
    .slice(-SAMPLE)

  return (
    <section className="space-y-4">
      <div>
        <div className="text-[11px] uppercase tracking-[0.28em] text-text-muted">Momentum Studies</div>
        <h2 className="mt-2 font-display text-3xl text-white">RSI and MACD diagnostics</h2>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard title="RSI (14) - Relative Strength Index">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rsiData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(222,234,245,0.06)" />
              <XAxis dataKey="date" tick={{ fill: '#8ea2b5', fontSize: 10 }} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fill: '#8ea2b5', fontSize: 10 }} width={46} label={{ value: 'RSI', angle: -90, position: 'insideLeft', fill: '#8ea2b5', fontSize: 11 }} />
              <Tooltip contentStyle={ttStyle} formatter={v => [v?.toFixed(2), 'RSI']} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#9db0c0' }} />
              <ReferenceLine y={70} stroke="#d96a5d" strokeDasharray="4 2" opacity={0.7} label={{ value: 'Overbought', fill: '#d96a5d', fontSize: 10 }} />
              <ReferenceLine y={30} stroke="#4fbf8f" strokeDasharray="4 2" opacity={0.7} label={{ value: 'Oversold', fill: '#4fbf8f', fontSize: 10 }} />
              <Line type="monotone" dataKey="rsi" stroke="#6fa8dc" dot={false} strokeWidth={2.6} isAnimationActive={false} name="RSI" />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="MACD - Trend & Momentum Divergence">
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={macdData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(222,234,245,0.06)" />
              <XAxis dataKey="date" tick={{ fill: '#8ea2b5', fontSize: 10 }} interval="preserveStartEnd" />
              <YAxis tick={{ fill: '#8ea2b5', fontSize: 10 }} width={48} label={{ value: 'MACD', angle: -90, position: 'insideLeft', fill: '#8ea2b5', fontSize: 11 }} />
              <Tooltip contentStyle={ttStyle} formatter={v => [v?.toFixed(4), '']} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#9db0c0' }} />
              <Bar dataKey="hist" isAnimationActive={false}>
                {macdData.map((e, i) => (
                  <Cell key={i} fill={(e.hist ?? 0) >= 0 ? 'rgba(79,191,143,0.45)' : 'rgba(217,106,93,0.45)'} />
                ))}
              </Bar>
              <Line type="monotone" dataKey="macd" stroke="#6fa8dc" dot={false} strokeWidth={2.8} isAnimationActive={false} name="MACD" />
              <Line type="monotone" dataKey="signal" stroke="#caa76a" dot={false} strokeWidth={2.2} strokeDasharray="6 4" isAnimationActive={false} name="Signal" />
            </ComposedChart>
          </ResponsiveContainer>
          <div className="mt-2 flex gap-4 text-xs text-text-muted">
            <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#6fa8dc]" />MACD</span>
            <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#caa76a]" />Signal</span>
          </div>
        </ChartCard>
      </div>
    </section>
  )
}
