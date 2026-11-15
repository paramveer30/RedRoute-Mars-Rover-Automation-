import { create } from 'zustand'
import type { Tick } from './types'

interface MissionStore {
  connected: boolean
  last: Tick | null
  trail: [number, number][]
  connect: (missionId: string) => () => void
}

export const useMission = create<MissionStore>((set, get) => ({
  connected: false,
  last: null,
  trail: [],
  connect: (missionId) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/missions/${missionId}`)
    ws.onopen = () => set({ connected: true })
    ws.onclose = () => set({ connected: false })
    ws.onmessage = (e) => {
      const t = JSON.parse(e.data) as Tick
      if (t.type !== 'tick') return
      const trail = [...get().trail, [t.pose.x, t.pose.y] as [number, number]].slice(-400)
      set({ last: t, trail })
    }
    return () => ws.close()
  },
}))
