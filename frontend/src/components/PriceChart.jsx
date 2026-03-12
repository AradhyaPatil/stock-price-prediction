import { useEffect, useRef } from 'react'
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
} from 'lightweight-charts'

export default function PriceChart({ chart, ticker }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current || !chart?.dates?.length) return

    const el = containerRef.current
    const c = createChart(el, {
      width: el.clientWidth,
      height: 500,
      localization: {
        priceFormatter: p => `$${Number(p).toFixed(2)}`,
      },
      layout: {
        background: { type: ColorType.Solid, color: '#102539' },
        textColor: '#9db0c0',
      },
      grid: {
        vertLines: { color: 'rgba(222,234,245,0.05)' },
        horzLines: { color: 'rgba(222,234,245,0.05)' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: 'rgba(222,234,245,0.12)' },
      timeScale: { borderColor: 'rgba(222,234,245,0.12)', timeVisible: true },
    })

    const candles = c.addCandlestickSeries({
      upColor: '#4fbf8f',
      downColor: '#d96a5d',
      borderUpColor: '#4fbf8f',
      borderDownColor: '#d96a5d',
      wickUpColor: '#4fbf8f',
      wickDownColor: '#d96a5d',
    })

    candles.setData(
      chart.dates
        .map((d, i) => ({
          time: d,
          open: chart.open[i],
          high: chart.high[i],
          low: chart.low[i],
          close: chart.close[i],
        }))
        .filter(d => d.open != null && d.high != null && d.low != null && d.close != null)
    )

    const volumeSeries = c.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      color: '#6fa8dc',
    })
    c.priceScale('volume').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } })

    volumeSeries.setData(
      chart.dates
        .map((d, i) => ({
          time: d,
          value: chart.volume[i],
          color: chart.close[i] >= chart.open[i]
            ? 'rgba(79,191,143,0.45)'
            : 'rgba(217,106,93,0.45)',
        }))
        .filter(d => d.value != null)
    )

    if (chart.sma20?.some(v => v != null)) {
      const s = c.addLineSeries({ color: '#6fa8dc', lineWidth: 1.5, title: 'SMA 20' })
      s.setData(chart.dates.map((d, i) => ({ time: d, value: chart.sma20[i] })).filter(d => d.value != null))
    }
    if (chart.sma50?.some(v => v != null)) {
      const s = c.addLineSeries({ color: '#caa76a', lineWidth: 1.5, title: 'SMA 50' })
      s.setData(chart.dates.map((d, i) => ({ time: d, value: chart.sma50[i] })).filter(d => d.value != null))
    }
    if (chart.sma200?.some(v => v != null)) {
      const s = c.addLineSeries({ color: '#f0d9a3', lineWidth: 1.5, lineStyle: LineStyle.Dashed, title: 'SMA 200' })
      s.setData(chart.dates.map((d, i) => ({ time: d, value: chart.sma200[i] })).filter(d => d.value != null))
    }
    if (chart.bbUpper?.some(v => v != null)) {
      const s = c.addLineSeries({
        color: 'rgba(224,179,90,0.65)',
        lineWidth: 1.4,
        lineStyle: LineStyle.Dotted,
        title: 'BB Upper',
      })
      s.setData(chart.dates.map((d, i) => ({ time: d, value: chart.bbUpper[i] })).filter(d => d.value != null))
    }
    if (chart.bbLower?.some(v => v != null)) {
      const s = c.addLineSeries({
        color: 'rgba(111,168,220,0.65)',
        lineWidth: 1.4,
        lineStyle: LineStyle.Dashed,
        title: 'BB Lower',
      })
      s.setData(chart.dates.map((d, i) => ({ time: d, value: chart.bbLower[i] })).filter(d => d.value != null))
    }

    c.timeScale().fitContent()

    const onResize = () => c.applyOptions({ width: el.clientWidth })
    const ro = new ResizeObserver(onResize)
    ro.observe(el)

    return () => {
      ro.disconnect()
      c.remove()
    }
  }, [chart, ticker])

  return (
    <section className="market-card rounded-[24px] p-5 md:p-6">
      <div className="section-divider">
        <div>
          <div className="text-[11px] uppercase tracking-[0.28em] text-text-muted">Price Structure</div>
          <h2 className="mt-2 font-display text-3xl text-white">{ticker} Price Chart &amp; Technical Overlay</h2>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-line-subtle bg-bg-panel/70 p-3">
        <div ref={containerRef} />
      </div>

      <div className="mt-4 flex flex-wrap gap-4 text-xs text-text-muted">
        {[
          { label: 'SMA 20', color: '#6fa8dc' },
          { label: 'SMA 50', color: '#caa76a' },
          { label: 'SMA 200', color: '#f0d9a3' },
          { label: 'BB Upper', color: 'rgba(224,179,90,0.65)' },
          { label: 'BB Lower', color: 'rgba(111,168,220,0.65)' },
        ].map(({ label, color }) => (
          <span key={label} className="flex items-center gap-1.5">
            <span className="inline-block w-4 h-0.5" style={{ background: color }} />
            {label}
          </span>
        ))}
      </div>
    </section>
  )
}
