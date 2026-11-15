import { useEffect, useMemo, useRef, useState } from 'react'
import { GRID, useMission } from './store'

const PX = 3.5 // canvas pixels per cell
const SIZE = Math.round(GRID * PX)
const SENSOR_RANGE_CELLS = 8 / 0.25
const SENSOR_FOV_DEG = 45

// soil, bedrock, sand, big rock
const CLASS_RGB: [number, number, number][] = [
  [196, 160, 112],
  [140, 140, 150],
  [236, 200, 70],
  [220, 50, 50],
]

function hillshade(elev: number[][]): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = c.height = GRID
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(GRID, GRID)
  let min = Infinity
  let max = -Infinity
  for (const row of elev) for (const v of row) { if (v < min) min = v; if (v > max) max = v }
  const az = (315 * Math.PI) / 180
  const alt = (45 * Math.PI) / 180
  for (let r = 0; r < GRID; r++) {
    for (let col = 0; col < GRID; col++) {
      const r0 = Math.max(r - 1, 0), r1 = Math.min(r + 1, GRID - 1)
      const c0 = Math.max(col - 1, 0), c1 = Math.min(col + 1, GRID - 1)
      const dzdx = ((elev[r][c1] - elev[r][c0]) / ((c1 - c0) * 0.25)) * 1
      const dzdy = ((elev[r1][col] - elev[r0][col]) / ((r1 - r0) * 0.25)) * 1
      const slope = Math.atan(Math.hypot(dzdx, dzdy))
      const aspect = Math.atan2(dzdy, -dzdx)
      let shade = Math.cos(alt) * Math.cos(slope) + Math.sin(alt) * Math.sin(slope) * Math.cos(az - aspect)
      shade = Math.max(0, Math.min(1, shade))
      const h = (elev[r][col] - min) / (max - min || 1)
      const base = 0.55 + 0.45 * h
      const i = (r * GRID + col) * 4
      img.data[i] = 150 * shade * base + 40
      img.data[i + 1] = 85 * shade * base + 25
      img.data[i + 2] = 55 * shade * base + 18
      img.data[i + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  return c
}

export function MissionMap() {
  const ref = useRef<HTMLCanvasElement>(null)
  const overlay = useMemo(() => {
    const c = document.createElement('canvas')
    c.width = c.height = GRID
    return c
  }, [])
  const { last, trail, cls, unc, blocked, truth, world, layers, beliefVersion, setTarget } = useMission()
  const shade = useMemo(() => (world ? hillshade(world.elevation) : null), [world])
  const [hover, setHover] = useState<[number, number] | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    ctx.imageSmoothingEnabled = false
    ctx.fillStyle = '#1c120c'
    ctx.fillRect(0, 0, SIZE, SIZE)
    if (layers.hillshade && shade) ctx.drawImage(shade, 0, 0, SIZE, SIZE)

    // Per-cell overlays rendered at grid resolution, then scaled up.
    const octx = overlay.getContext('2d')!
    const img = octx.createImageData(GRID, GRID)
    const d = img.data
    for (let i = 0; i < GRID * GRID; i++) {
      let r = 0, g = 0, b = 0, a = 0
      if (layers.truth && truth) {
        const t = truth[i]
        if (t < 4) { [r, g, b] = CLASS_RGB[t]; a = 200 }
      } else if (layers.classes) {
        const k = cls[i]
        if (k < 4) { [r, g, b] = CLASS_RGB[k]; a = 150 }
        else { r = g = b = 10; a = 110 } // unknown: dark
      }
      if (layers.uncertainty && cls[i] < 4) {
        const u = unc[i] / 255
        r = r * (1 - u) + 170 * u; g = g * (1 - u) + 60 * u; b = b * (1 - u) + 230 * u
        a = Math.max(a, 60 + 180 * u)
      }
      if (layers.blocked && blocked[i]) {
        const x = i % GRID, y = (i / GRID) | 0
        if ((x + y) % 4 < 2) { r = 255; g = 30; b = 30; a = 220 }
      }
      d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = a
    }
    octx.putImageData(img, 0, 0)
    ctx.drawImage(overlay, 0, 0, SIZE, SIZE)

    // tile grid
    ctx.strokeStyle = 'rgba(255,255,255,0.07)'
    ctx.lineWidth = 1
    for (let i = 0; i <= 10; i++) {
      const p = i * 20 * PX
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, SIZE); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(SIZE, p); ctx.stroke()
    }

    const cx = (x: number) => (x + 0.5) * PX
    // driven trail
    ctx.strokeStyle = 'rgba(125,211,252,0.8)'
    ctx.lineWidth = 2
    ctx.beginPath()
    trail.forEach(([x, y], i) => (i ? ctx.lineTo(cx(x), cx(y)) : ctx.moveTo(cx(x), cx(y))))
    ctx.stroke()

    if (last && layers.path && last.path.length > 1) {
      ctx.strokeStyle = '#4ade80'
      ctx.lineWidth = 2.5
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      last.path.forEach(([x, y], i) => (i ? ctx.lineTo(cx(x), cx(y)) : ctx.moveTo(cx(x), cx(y))))
      ctx.stroke()
      ctx.setLineDash([])
    }
    if (last?.goal) {
      const [gx, gy] = last.goal
      ctx.strokeStyle = '#4ade80'
      ctx.lineWidth = 2
      ctx.beginPath(); ctx.arc(cx(gx), cx(gy), 8, 0, Math.PI * 2); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(cx(gx) - 12, cx(gy)); ctx.lineTo(cx(gx) + 12, cx(gy))
      ctx.moveTo(cx(gx), cx(gy) - 12); ctx.lineTo(cx(gx), cx(gy) + 12); ctx.stroke()
    }

    if (last && layers.rover) {
      const x = cx(last.pose.x), y = cx(last.pose.y)
      const h = ((last.pose.heading - 90) * Math.PI) / 180 // canvas 0 = east; heading 0 = north
      const half = ((SENSOR_FOV_DEG / 2) * Math.PI) / 180
      ctx.fillStyle = 'rgba(250,204,21,0.15)'
      ctx.strokeStyle = 'rgba(250,204,21,0.5)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(x, y)
      ctx.arc(x, y, SENSOR_RANGE_CELLS * PX, h - half, h + half)
      ctx.closePath(); ctx.fill(); ctx.stroke()
      ctx.fillStyle = last.state === 'SAFE_STOP' ? '#ef4444' : '#facc15'
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill()
      ctx.strokeStyle = '#111'
      ctx.lineWidth = 2
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + Math.cos(h) * 13, y + Math.sin(h) * 13); ctx.stroke()
    }

    if (layers.truth) {
      ctx.fillStyle = 'rgba(239,68,68,0.9)'
      ctx.fillRect(8, 8, 74, 22)
      ctx.fillStyle = 'white'
      ctx.font = 'bold 13px ui-monospace, monospace'
      ctx.fillText('TRUTH', 22, 24)
    }
  }, [last, trail, cls, unc, blocked, truth, layers, shade, overlay, beliefVersion])

  const toCell = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * GRID)
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * GRID)
    return [Math.min(GRID - 1, Math.max(0, x)), Math.min(GRID - 1, Math.max(0, y))]
  }

  return (
    <div className="relative shrink-0">
      <canvas
        ref={ref}
        width={SIZE}
        height={SIZE}
        className="max-w-full cursor-crosshair rounded border border-white/10"
        onMouseMove={(e) => setHover(toCell(e))}
        onMouseLeave={() => setHover(null)}
        onClick={async (e) => {
          const [x, y] = toCell(e)
          setErr(await setTarget(x, y))
        }}
      />
      <div className="mt-1 flex justify-between font-mono text-xs text-white/50">
        <span>{hover ? `cell x=${hover[0]} y=${hover[1]}` : 'click the map to set a target'}</span>
        {err && <span className="text-red-400">{err}</span>}
      </div>
    </div>
  )
}
