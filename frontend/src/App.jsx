import { useState, useCallback, useEffect, useRef } from 'react'
import axios from 'axios'
import Sidebar from './components/Sidebar'
import StockOverview from './components/StockOverview'
import PriceChart from './components/PriceChart'
import IndicatorCharts from './components/IndicatorCharts'
import TrainingPanel from './components/TrainingPanel'

export default function App() {
  const [stockData, setStockData] = useState(null)
  const [trainingResult, setTrainingResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [training, setTraining] = useState(false)
  const [error, setError] = useState(null)
  const [trainError, setTrainError] = useState(null)
  const [params, setParams] = useState({
    ticker: 'AAPL', period: '5y', lookback: 60, epochs: 100, forecast_days: 30,
  })

  // Training timer
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef(null)

  useEffect(() => {
    if (training) {
      setElapsed(0)
      timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000)
    } else {
      clearInterval(timerRef.current)
    }
    return () => clearInterval(timerRef.current)
  }, [training])

  const fetchStock = useCallback(async (ticker, period) => {
    setLoading(true)
    setError(null)
    setTrainingResult(null)
    try {
      const res = await axios.get(`/api/stock/${encodeURIComponent(ticker)}`, {
        params: { period },
      })
      setStockData(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message)
      setStockData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const trainModel = useCallback(async p => {
    setTraining(true)
    setTrainError(null)
    setTrainingResult(null)
    try {
      const res = await axios.post('/api/train', p, { timeout: 7200000 })
      setTrainingResult(res.data)
    } catch (e) {
      setTrainError(e.response?.data?.detail || e.message)
    } finally {
      setTraining(false)
    }
  }, [])

  const handleFetch = (newParams) => {
    setParams(newParams)
    fetchStock(newParams.ticker, newParams.period)
  }

  const handleTrain = (newParams) => {
    setParams(newParams)
    trainModel(newParams)
  }

  const mins = String(Math.floor(elapsed / 60)).padStart(2, '0')
  const secs = String(elapsed % 60).padStart(2, '0')

  return (
    <div className="min-h-screen bg-bg-primary text-text-primary">
      <div className="flex h-screen overflow-hidden">
      <Sidebar
        params={params}
        onFetch={handleFetch}
        onTrain={handleTrain}
        loading={loading}
        training={training}
      />

      <main className="flex-1 overflow-y-auto p-4 md:p-6 xl:p-8">
        <div className="mx-auto max-w-[1600px] space-y-6">
          <section className="market-card rounded-[20px] px-5 py-4 md:px-6 md:py-5">
            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
              <div>
                <h1 className="mt-2 font-display text-2xl text-white md:text-3xl">Stock Price Prediction Dashboard</h1>
              </div>
              <div className="text-sm text-text-soft">
                Active Symbol: <span className="font-semibold text-text-primary">{params.ticker}</span>
              </div>
            </div>
          </section>

          {error && (
            <div className="market-card rounded-2xl border-l-4 border-l-accent-red p-4 text-sm text-[#f6c2bb]">
              {error}
            </div>
          )}

          {loading && (
            <div className="market-card rounded-2xl p-8 text-center text-sm text-text-muted animate-pulse">
              Loading market data and refreshing analytics...
            </div>
          )}

          {stockData && !loading && (
            <>
              <StockOverview info={stockData.info} chart={stockData.chart} />
              <PriceChart chart={stockData.chart} ticker={params.ticker} />
              <IndicatorCharts chart={stockData.chart} />
            </>
          )}

          {training && (
            <section className="market-card rounded-2xl p-6 md:p-8">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="text-[11px] uppercase tracking-[0.24em] text-text-muted">Model Execution</div>
                  <div className="mt-2 text-2xl font-semibold text-white">Running forecast pipeline for {params.ticker}</div>
                  <div className="mt-2 text-sm text-text-soft">{params.epochs} epochs · lookback {params.lookback}d · forecast {params.forecast_days}d</div>
                </div>
                <div className="rounded-2xl border border-line-subtle bg-white/[0.03] px-6 py-4 text-center">
                  <div className="text-[11px] uppercase tracking-[0.24em] text-text-muted">Elapsed</div>
                  <div className="mt-2 font-mono text-3xl font-semibold text-accent-gold">{mins}:{secs}</div>
                </div>
              </div>
            </section>
          )}

          {trainError && (
            <div className="market-card rounded-2xl border-l-4 border-l-accent-red p-4 text-sm text-[#f6c2bb]">
              Training failed: {trainError}
            </div>
          )}

          {trainingResult && !training && (
            <TrainingPanel
              result={trainingResult}
              ticker={params.ticker}
              forecastDays={params.forecast_days}
            />
          )}

          {!stockData && !loading && (
            <section className="market-card rounded-2xl p-6 md:p-8">
              <div className="max-w-3xl space-y-4">
                <div className="text-[11px] uppercase tracking-[0.26em] text-text-muted">Start Coverage</div>
                <h2 className="font-display text-3xl text-white">Load a symbol to open the market workspace.</h2>
                <p className="text-sm leading-7 text-text-soft">Use the left control panel to select a ticker, refresh historical data, and launch model training.</p>
                <div className="flex flex-wrap gap-3 text-sm text-text-soft">
                  <span className="market-chip rounded-full px-3 py-2">AAPL</span>
                  <span className="market-chip rounded-full px-3 py-2">NVDA</span>
                  <span className="market-chip rounded-full px-3 py-2">TSLA</span>
                  <span className="market-chip rounded-full px-3 py-2">RELIANCE.NS</span>
                  <span className="market-chip rounded-full px-3 py-2">TCS.NS</span>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>
      </div>
    </div>
  )
}
