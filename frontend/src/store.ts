import { create } from 'zustand'
import type { MissionEvent, PerceptionMode, Tick, WorldLayers } from './types'

export const GRID = 200
export const TILE = 20

function b64bytes(s: string): Uint8Array {
  const bin = atob(s)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

function unpackBits(s: string, n: number): Uint8Array {
  const packed = b64bytes(s)
  const out = new Uint8Array(n)
  for (let i = 0; i < n; i++) out[i] = (packed[i >> 3] >> (7 - (i & 7))) & 1
  return out
}

export type LayerKey = 'hillshade' | 'classes' | 'uncertainty' | 'blocked' | 'path' | 'rover' | 'truth'

interface MissionStore {
  missionId: string
  connected: boolean
  last: Tick | null
  trail: [number, number][]
  events: MissionEvent[]
  // full-resolution buffers the backend updates by delta
  cls: Uint8Array // GRID*GRID, 255 unknown
  unc: Uint8Array // GRID*GRID
  blocked: Uint8Array // GRID*GRID 0/1
  truth: Uint8Array | null
  beliefVersion: number
  world: WorldLayers | null
  layers: Record<LayerKey, boolean>
  connect: (missionId: string) => () => void
  toggle: (k: LayerKey) => void
  command: (body: Record<string, unknown>) => Promise<string | null>
  setTarget: (x: number, y: number) => Promise<string | null>
  resume: () => Promise<string | null>
  stop: () => Promise<string | null>
  reset: (seed?: number, mode?: PerceptionMode) => Promise<string | null>
  setMode: (mode: PerceptionMode) => Promise<string | null>
}

const emptyBuffers = () => ({
  cls: new Uint8Array(GRID * GRID).fill(255),
  unc: new Uint8Array(GRID * GRID).fill(128),
  blocked: new Uint8Array(GRID * GRID),
})

async function loadWorld(seed: number, set: (p: Partial<MissionStore>) => void) {
  const [w, t] = await Promise.all([
    fetch(`/api/world/${seed}`).then((r) => r.json() as Promise<WorldLayers>),
    fetch(`/api/world/${seed}/truth`).then((r) => r.json() as Promise<{ truth: string }>),
  ])
  set({ world: w, truth: b64bytes(t.truth) })
}

export const useMission = create<MissionStore>((set, get) => ({
  missionId: 'demo',
  connected: false,
  last: null,
  trail: [],
  events: [],
  ...emptyBuffers(),
  truth: null,
  beliefVersion: 0,
  world: null,
  layers: { hillshade: true, classes: true, uncertainty: false, blocked: true, path: true, rover: true, truth: false },

  connect: (missionId) => {
    set({ missionId })
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    let ws: WebSocket | null = null
    let closed = false
    let retry: ReturnType<typeof setTimeout> | undefined

    const open = () => {
      ws = new WebSocket(`${proto}://${location.host}/ws/missions/${missionId}`)
      ws.onopen = () => set({ connected: true })
      ws.onclose = () => {
        set({ connected: false })
        if (!closed) retry = setTimeout(open, 1000)
      }
      ws.onmessage = (e) => {
        const t = JSON.parse(e.data) as Tick
        if (t.type !== 'tick') return
        const s = get()
        let { cls, unc, blocked, trail, events } = s
        if (t.reset || (s.last && s.last.seed !== t.seed)) {
          ;({ cls, unc, blocked } = emptyBuffers())
          trail = []
          events = []
        }
        if (!s.world || s.world.seed !== t.seed) loadWorld(t.seed, set)
        for (const [id, d] of Object.entries(t.belief_delta.tiles)) {
          const [, tr, tc] = id.split('_').map(Number)
          const c = b64bytes(d.cls)
          const u = b64bytes(d.unc)
          for (let r = 0; r < TILE; r++) {
            const off = (tr * TILE + r) * GRID + tc * TILE
            cls.set(c.subarray(r * TILE, (r + 1) * TILE), off)
            unc.set(u.subarray(r * TILE, (r + 1) * TILE), off)
          }
        }
        if (t.blocked) blocked = unpackBits(t.blocked, GRID * GRID)
        const lastPt = trail[trail.length - 1]
        if (!lastPt || lastPt[0] !== t.pose.x || lastPt[1] !== t.pose.y) {
          trail = [...trail, [t.pose.x, t.pose.y] as [number, number]].slice(-2000)
        }
        if (t.events.length) events = [...t.events.filter((e) => e.kind !== 'state'), ...events].slice(0, 300)
        set({ last: t, cls, unc, blocked, trail, events, beliefVersion: s.beliefVersion + 1 })
      }
    }
    open()
    return () => {
      closed = true
      clearTimeout(retry)
      ws?.close()
    }
  },

  toggle: (k) => set({ layers: { ...get().layers, [k]: !get().layers[k] } }),

  command: async (body) => {
    const res = await fetch(`/api/missions/${get().missionId}/command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (res.ok) return null
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    return String(err.detail ?? res.statusText)
  },
  setTarget: (x, y) => get().command({ type: 'target', x, y }),
  resume: () => get().command({ type: 'resume' }),
  stop: () => get().command({ type: 'stop' }),
  reset: (seed, mode) => get().command({ type: 'reset', seed, mode }),
  setMode: (mode) => get().command({ type: 'mode', mode }),
}))
