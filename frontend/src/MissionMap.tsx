import { useEffect, useRef } from 'react'
import { useMission } from './store'

const GRID = 200
const PX = 3 // canvas pixels per cell

// Placeholder 2D map: grid + rover trail + rover. Stream B adds layers
// (elevation shade, belief classes, uncertainty, path) per docs/STREAM_B.md.
export function MissionMap() {
  const ref = useRef<HTMLCanvasElement>(null)
  const { last, trail } = useMission()

  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#3a2418'
    ctx.fillRect(0, 0, GRID * PX, GRID * PX)
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'
    for (let i = 0; i <= 10; i++) {
      ctx.beginPath(); ctx.moveTo(i * 20 * PX, 0); ctx.lineTo(i * 20 * PX, GRID * PX); ctx.stroke()
      ctx.beginPath(); ctx.moveTo(0, i * 20 * PX); ctx.lineTo(GRID * PX, i * 20 * PX); ctx.stroke()
    }
    ctx.strokeStyle = '#7dd3fc'
    ctx.beginPath()
    trail.forEach(([x, y], i) => (i ? ctx.lineTo(x * PX, y * PX) : ctx.moveTo(x * PX, y * PX)))
    ctx.stroke()
    if (last) {
      ctx.fillStyle = '#facc15'
      ctx.beginPath(); ctx.arc(last.pose.x * PX, last.pose.y * PX, 6, 0, Math.PI * 2); ctx.fill()
    }
  }, [last, trail])

  return <canvas ref={ref} width={GRID * PX} height={GRID * PX} className="rounded border border-white/10" />
}
