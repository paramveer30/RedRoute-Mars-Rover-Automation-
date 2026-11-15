// Mirrors the WebSocket tick schema (Stream B owns; keep in sync with backend/app/rover.py snapshot()
// and backend/app/api.py Mission.tick_payload()).
export type RoverState = 'IDLE' | 'SCANNING' | 'PLANNING' | 'DRIVING' | 'REPLANNING' | 'SAFE_STOP'
export type PerceptionMode = 'oracle' | 'model' | 'geometry_only' | 'offline'

export interface MissionEvent {
  mission_time: number
  level: 'info' | 'warn' | 'error'
  kind: string
  msg: string
  data?: Record<string, unknown>
  late?: boolean
}

export interface TileDelta {
  cls: string // base64, 400 bytes row-major: argmax class 0..3, 255 = unknown
  unc: string // base64, 400 bytes row-major: uncertainty 0..255
}

export interface Tick {
  type: 'tick'
  mission_id: string
  seed: number
  reset?: boolean
  mission_time: number
  seq: number
  rover_id: string
  pose: { x: number; y: number; heading: number } // grid cells (x = col, y = row), heading deg
  state: RoverState
  safe_stop_reason: string
  battery: number
  odometer_m: number
  since_contact_m: number
  route_version: number
  path: [number, number][] // [x, y] remaining route
  goal: [number, number] | null // [x, y]
  belief_delta: { tiles: Record<string, TileDelta> }
  blocked?: string // base64 packbits of the 200x200 blocked mask, sent only when it changes
  perception_mode: PerceptionMode
  frame_image_id: string | null
  link: { status: 'CONNECTED' | 'BLACKOUT'; delay_s: number }
  events: MissionEvent[]
}

export interface WorldLayers {
  seed: number
  grid: number
  elevation: number[][]
  slope_deg: number[][]
  tile_image: string[][]
}
