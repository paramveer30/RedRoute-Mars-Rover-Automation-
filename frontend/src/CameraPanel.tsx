import { useState } from 'react'
import { useMission } from './store'

// Real AI4Mars NAVCAM frame for the tile the camera is looking at, with Stream A's model
// class overlay on top. Frame ids come from the tick's frame_image_id (m3_NNNN).
export function CameraPanel() {
  const frame = useMission((s) => s.last?.frame_image_id ?? null)
  const mode = useMission((s) => s.last?.perception_mode)
  const [opacity, setOpacity] = useState(0.45)

  return (
    <section className="rounded border border-white/10 p-3 text-sm">
      <div className="mb-1 flex items-center justify-between">
        <h2 className="font-semibold">Camera</h2>
        <span className="font-mono text-xs text-white/50">{frame ?? 'no frame yet'}</span>
      </div>
      <div className="relative aspect-square w-full overflow-hidden rounded bg-black">
        {frame ? (
          <>
            <img src={`/api/frames/${frame}`} alt={`NAVCAM frame ${frame}`} className="absolute inset-0 h-full w-full object-cover" />
            <img
              src={`/api/overlays/${frame}`}
              alt=""
              className="absolute inset-0 h-full w-full object-cover mix-blend-normal"
              style={{ opacity }}
            />
          </>
        ) : (
          <div className="flex h-full items-center justify-center text-white/40">waiting for first frame</div>
        )}
      </div>
      <label className="mt-2 flex items-center gap-2 text-xs">
        <span className="text-white/50">model overlay</span>
        <input type="range" min={0} max={1} step={0.05} value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))} className="flex-1" />
        <span className="w-8 text-right font-mono">{Math.round(opacity * 100)}%</span>
      </label>
      <p className="mt-1 text-[10px] text-white/40">
        Real NASA AI4Mars frame{mode === 'oracle' ? '; overlay is model output, map is ORACLE truth' : ''}.
      </p>
    </section>
  )
}
