// Mirrors the WebSocket tick schema (Stream B owns; keep in sync with backend/app/api.py)
export type RoverState = 'IDLE' | 'SCANNING' | 'PLANNING' | 'DRIVING' | 'REPLANNING' | 'SAFE_STOP'
export type PerceptionMode = 'oracle' | 'model' | 'geometry_only' | 'offline'

export interface Tick {
  type: 'tick'
  mission_time: number
  seq: number
  rover_id: string
  pose: { x: number; y: number; heading: number } // grid cells (x = col, y = row), heading deg
  state: RoverState
  battery: number
  route_version: number
  path: [number, number][]
  belief_delta: { tiles: string[] }
  perception_mode: PerceptionMode
  link: { status: 'CONNECTED' | 'BLACKOUT'; delay_s: number }
  events: { t: number; level: 'info' | 'warn' | 'error'; msg: string; late?: boolean }[]
}
