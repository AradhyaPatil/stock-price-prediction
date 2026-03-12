import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip, Legend,
} from 'recharts'

const ttStyle = {
  background: '#102539',
  border: '1px solid rgba(222,234,245,0.12)',
  borderRadius: 12,
  color: '#edf3f8',
  fontSize: 12,
}

function SectionHeader({ title }) {
  return (
    <div className="section-divider mt-6 first:mt-0">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.28em] text-text-muted">{title}</h2>
    </div>
  )
}

function MetricCard({ label, value, color = 'default' }) {
  const colors = {
    green: 'text-accent-green',
    red: 'text-accent-red',
    blue: 'text-accent-blue',
    gold: 'text-accent-gold',
    default: 'text-text-primary',
  }

  return (
    <div className="market-card flex flex-1 items-center rounded-xl px-4 py-3">
      <div className="flex w-full items-center justify-between gap-4">
        <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-text-muted shrink-0">{label}</div>
        <div className={`text-base font-semibold text-right ${colors[color] ?? colors.default}`}>{value}</div>
      </div>
    </div>
  )
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

export default function TrainingPanel({ result, ticker, forecastDays }) {
  if (!result) return null

  const { metrics, prediction, training, forecast, current_price } = result

  const lossData = training.loss.map((l, i) => ({
    epoch: i + 1,
    loss: l,
    val_loss: training.val_loss[i] ?? null,
  }))

  const stride = Math.max(1, Math.floor(prediction.dates.length / 300))
  const predData = prediction.dates
    .filter((_, i) => i % stride === 0)
    .map((d, i) => ({
      date: d.slice(5),
      actual: prediction.actual[i * stride],
      predicted: prediction.predicted[i * stride],
    }))

  const recentMap = {}
  forecast.recent_dates.forEach((d, i) => { recentMap[d] = forecast.recent_prices[i] })
  const forecastMap = {}
  forecast.dates.forEach((d, i) => {
    forecastMap[d] = { price: forecast.prices[i], upper: forecast.upper[i], lower: forecast.lower[i] }
  })

  const allDates = [...new Set([...forecast.recent_dates, ...forecast.dates])].sort()
  const combinedData = allDates.map(d => ({
    date: d.slice(5),
    actual: recentMap[d] ?? null,
    forecast: forecastMap[d]?.price ?? null,
    upper: forecastMap[d]?.upper ?? null,
    lower: forecastMap[d]?.lower ?? null,
  }))

  const lastForecast = forecast.prices.at(-1)
  const forecastChange = lastForecast - current_price
  const forecastPct = (forecastChange / current_price) * 100
  const fcSign = forecastChange >= 0 ? '+' : ''
  const fcColor = forecastChange >= 0 ? 'green' : 'red'
  const signal = forecastChange >= 0 ? 'Bullish Bias' : 'Bearish Bias'
  const r2Color = metrics.r2 > 0.8 ? 'green' : metrics.r2 > 0.5 ? 'gold' : 'red'

  return (
    <div className="space-y-6">
      <SectionHeader title="LSTM Model Training & Forecast" />

      <div className="market-card rounded-[24px] p-5 text-sm text-text-soft md:p-6">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-[0.24em] text-text-muted">Training Completion</div>
            <div className="mt-2 text-xl font-semibold text-white">Forecast engine finished for {ticker}</div>
          </div>
          <div className="grid gap-2 text-sm md:text-right">
            <div>Epochs executed: <strong className="text-accent-blue">{training.epochs}</strong></div>
            <div>Final loss: <strong className="text-accent-blue">{training.loss.at(-1)?.toFixed(5)}</strong></div>
            <div>Final val loss: <strong className="text-accent-gold">{training.val_loss.at(-1)?.toFixed(5) ?? 'N/A'}</strong></div>
          </div>
        </div>
      </div>

      <div className="grid items-stretch gap-4 xl:grid-cols-[2fr_1fr]">
        <div>
          <ChartCard title="Training Loss">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(222,234,245,0.06)" />
                <XAxis dataKey="epoch" tick={{ fill: '#8ea2b5', fontSize: 10 }} label={{ value: '', position: 'insideBottom', dy: 12, fill: '#8ea2b5', fontSize: 11 }} height={40} />
                <YAxis tick={{ fill: '#8ea2b5', fontSize: 10 }} width={70} tickFormatter={v => Number(v).toFixed(4)} label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#8ea2b5', fontSize: 11 }} />
                <Tooltip contentStyle={ttStyle} formatter={v => [v?.toFixed(6), '']} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#9db0c0' }} />
                <Line type="monotone" dataKey="loss" stroke="#6fa8dc" dot={false} strokeWidth={2.8} isAnimationActive={false} name="Train Loss" />
                {lossData[0]?.val_loss != null && (
                  <Line type="monotone" dataKey="val_loss" stroke="#caa76a" dot={false} strokeWidth={2.4} strokeDasharray="6 3" isAnimationActive={false} name="Val Loss" />
                )}
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-2 flex gap-4 text-xs text-text-muted">
              <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#6fa8dc]" />Train Loss</span>
              <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#caa76a]" />Val Loss</span>
            </div>
          </ChartCard>
        </div>

        <div className="market-card flex h-full flex-col rounded-[24px] p-4 md:p-5">
          <div className="section-divider">
            <h3 className="font-display text-xl text-white">Performance Metrics</h3>
          </div>
          <div className="flex flex-1 flex-col gap-2">
            <MetricCard label="RMSE" value={`$${metrics.rmse.toFixed(2)}`} color="red" />
            <MetricCard label="MAE" value={`$${metrics.mae.toFixed(2)}`} color="blue" />
            <MetricCard label="MAPE" value={`${metrics.mape.toFixed(2)}%`} color="gold" />
            <MetricCard label="R² Score" value={metrics.r2.toFixed(4)} color={r2Color} />
          </div>
        </div>
      </div>

      <ChartCard title="Actual vs Predicted (Test Set)">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={predData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(222,234,245,0.06)" />
            <XAxis dataKey="date" tick={{ fill: '#8ea2b5', fontSize: 10 }} interval={Math.max(1, Math.floor(predData.length / 8))} />
            <YAxis tick={{ fill: '#8ea2b5', fontSize: 10 }} width={70} tickFormatter={v => `$${Number(v).toFixed(0)}`} label={{ value: 'Price', angle: -90, position: 'insideLeft', fill: '#8ea2b5', fontSize: 11 }} />
            <Tooltip contentStyle={ttStyle} formatter={v => (v != null ? [`$${v.toFixed(2)}`, ''] : [null, ''])} />
            <Legend wrapperStyle={{ fontSize: 11, color: '#9db0c0' }} />
            <Line type="monotone" dataKey="actual" stroke="#4fbf8f" dot={false} strokeWidth={2.8} isAnimationActive={false} name="Actual" />
            <Line type="monotone" dataKey="predicted" stroke="#d96a5d" dot={false} strokeWidth={2.2} strokeDasharray="6 3" isAnimationActive={false} name="Predicted" />
          </LineChart>
        </ResponsiveContainer>
        <div className="mt-2 flex gap-4 text-xs text-text-muted">
          <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#4fbf8f]" />Actual</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#d96a5d]" style={{ borderTop: '2px dashed #d96a5d', background: 'transparent' }} />Predicted</span>
        </div>
      </ChartCard>

      <SectionHeader title="Future Price Forecast" />
      <ChartCard title={`${ticker} - ${forecastDays}-Day Price Forecast (Monte Carlo)`}>
        <ResponsiveContainer width="100%" height={420}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(222,234,245,0.06)" />
            <XAxis dataKey="date" tick={{ fill: '#8ea2b5', fontSize: 10 }} interval={Math.max(1, Math.floor(combinedData.length / 10))} />
            <YAxis tick={{ fill: '#8ea2b5', fontSize: 10 }} width={70} tickFormatter={v => `$${Number(v).toFixed(0)}`} label={{ value: 'Price', angle: -90, position: 'insideLeft', fill: '#8ea2b5', fontSize: 11 }} />
            <Tooltip contentStyle={ttStyle} formatter={v => (v != null ? [`$${v.toFixed(2)}`, ''] : [null, ''])} />
            <Legend wrapperStyle={{ fontSize: 11, color: '#9db0c0' }} />
            <Line type="monotone" dataKey="actual" stroke="#6fa8dc" dot={false} strokeWidth={2.8} connectNulls={false} isAnimationActive={false} name="Recent Actual" />
            <Line type="monotone" dataKey="upper" stroke="rgba(224,179,90,0.42)" dot={false} strokeWidth={1.6} strokeDasharray="3 3" connectNulls={false} isAnimationActive={false} name="Upper Bound" />
            <Line type="monotone" dataKey="lower" stroke="rgba(224,179,90,0.42)" dot={false} strokeWidth={1.6} strokeDasharray="3 3" connectNulls={false} isAnimationActive={false} name="Lower Bound" />
            <Line type="monotone" dataKey="forecast" stroke="#caa76a" dot={false} strokeWidth={2.5} connectNulls={false} isAnimationActive={false} name={`Forecast (${forecastDays}d)`} />
          </LineChart>
        </ResponsiveContainer>
        <div className="mt-2 flex gap-4 text-xs text-text-muted">
          <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#6fa8dc]" />Recent Actual</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 bg-[#caa76a]" />Forecast</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-0.5 w-4" style={{ background: 'rgba(224,179,90,0.4)' }} />Confidence Band</span>
        </div>
      </ChartCard>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Current Price" value={`$${current_price.toFixed(2)}`} color="blue" />
        <MetricCard label={`Predicted (${forecastDays}d)`} value={`$${lastForecast.toFixed(2)}`} color={fcColor} />
        <MetricCard label="Expected Change" value={`${fcSign}${forecastPct.toFixed(2)}%`} color={fcColor} />
        <MetricCard label="Signal" value={signal} color={fcColor} />
      </div>

      <div className="market-card rounded-[24px] border-l-4 border-l-accent-red p-4 text-sm leading-7 text-text-soft">
        <strong className="text-white">Disclaimer:</strong> Predictions are generated by a machine learning model for educational purposes only. Stock markets are inherently unpredictable. <strong className="text-white">Do not use this for actual trading or investment decisions.</strong>
      </div>
    </div>
  )
}
