import { useMemo } from 'react'
import PlotlyChart from './PlotlyChart'

interface Heatmap3DProps {
  matrix: number[][]
  xLabels: string[]
  yLabels: string[]
  title?: string
  colorscale?: string
  height?: number
  mode?: '2d' | '3d'
  showValues?: boolean
}

// ColorBrewer YlOrRd from low to high, so the darkest red is the largest value.
// Plotly's named "YlOrRd" runs the other way (low values came out dark red).
const YL_OR_RD = ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
const YL_OR_RD_ESCALA = YL_OR_RD.map((c, i) => [i / (YL_OR_RD.length - 1), c])

function hexToRgb(hex: string): number[] {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16))
}

// Same linear RGB interpolation Plotly uses between colorscale stops.
function corNaEscala(escala: string[], t: number): number[] {
  const pos = Math.min(Math.max(t, 0), 1) * (escala.length - 1)
  const i = Math.min(Math.floor(pos), escala.length - 2)
  const a = hexToRgb(escala[i])
  const b = hexToRgb(escala[i + 1])
  return a.map((c, k) => c + (b[k] - c) * (pos - i))
}

function luminancia(rgb: number[]): number {
  const [r, g, b] = rgb.map((c) => {
    const s = c / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

// Near-black or white, whichever contrasts more with the cell (always >= 4.5:1).
function corDoTexto(fundo: number[]): string {
  const l = luminancia(fundo)
  return (l + 0.05) / 0.05 >= 1.05 / (l + 0.05) ? '#0a0a0a' : '#ffffff'
}

export default function Heatmap3D({
  matrix,
  xLabels,
  yLabels,
  title = '',
  colorscale = 'YlOrRd',
  height = 450,
  mode = '2d',
  showValues = true,
}: Heatmap3DProps) {
  const escala = colorscale === 'YlOrRd' ? YL_OR_RD_ESCALA : colorscale
  // Values go in the trace only for scales without per-cell label colors (see rotulos).
  const valoresNoTrace = showValues && !Array.isArray(escala)

  const data = useMemo(() => {
    if (mode === '3d') {
      return [
        {
          type: 'surface' as const,
          z: matrix,
          x: xLabels,
          y: yLabels,
          colorscale: escala,
          hovertemplate:
            'De: %{y}<br>Para: %{x}<br>Peso: %{z:.3f}<extra></extra>',
          contours: {
            z: { show: true, usecolormap: true, highlightcolor: '#ffffff', project: { z: true } },
          },
        },
      ] as unknown as Plotly.Data[]
    }
    return [
      {
        type: 'heatmap' as const,
        z: matrix,
        x: xLabels,
        y: yLabels,
        colorscale: escala,
        hovertemplate:
          'De: %{y}<br>Para: %{x}<br>Peso: %{z:.3f}<extra></extra>',
        text: valoresNoTrace ? matrix.map((row) => row.map((v) => v.toFixed(2))) : undefined,
        texttemplate: valoresNoTrace ? '%{text}' : undefined,
        textfont: { size: 10, color: '#ffffff' },
      },
    ] as unknown as Plotly.Data[]
  }, [matrix, xLabels, yLabels, escala, mode, valoresNoTrace])

  // Value labels as annotations so each cell gets its own readable text color.
  const rotulos = useMemo(() => {
    if (mode === '3d' || !showValues || !Array.isArray(escala)) return []
    const valores = matrix.flat()
    const min = Math.min(...valores)
    const faixa = Math.max(...valores) - min || 1
    return matrix.flatMap((row, i) =>
      row.map((v, j) => ({
        x: xLabels[j],
        y: yLabels[i],
        text: v.toFixed(2),
        showarrow: false,
        font: { size: 11, color: corDoTexto(corNaEscala(YL_OR_RD, (v - min) / faixa)) },
      })),
    )
  }, [matrix, xLabels, yLabels, escala, mode, showValues])

  const layout = useMemo(() => {
    if (mode === '3d') {
      return {
        title: { text: title, font: { color: '#111827', size: 14 } },
        scene: {
          xaxis: { title: 'Keys', color: '#6b7280', gridcolor: 'rgba(209,213,219,0.8)' },
          yaxis: { title: 'Queries', color: '#6b7280', gridcolor: 'rgba(209,213,219,0.8)' },
          zaxis: { title: 'Peso', color: '#6b7280', gridcolor: 'rgba(209,213,219,0.8)' },
          bgcolor: 'rgba(255,255,255,0)',
        },
      } as Partial<Plotly.Layout>
    }
    return {
      title: { text: title, font: { color: '#111827', size: 14 } },
      xaxis: { title: 'Key (para onde olha)', side: 'bottom' as const, tickangle: -45 },
      yaxis: { title: 'Query (de onde olha)', autorange: 'reversed' as const },
      margin: { l: 80, r: 40, t: 50, b: 80 },
      annotations: rotulos,
    } as Partial<Plotly.Layout>
  }, [title, mode, rotulos])

  return <PlotlyChart data={data} layout={layout} height={height} />
}
