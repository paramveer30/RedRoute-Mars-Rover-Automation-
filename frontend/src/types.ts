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
  ground?: boolean // known at the ground immediately (not via downlink)
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
  link: LinkInfo
  late?: boolean
  initial?: boolean
  events: MissionEvent[]
}

export interface LinkInfo {
  status: 'CONNECTED' | 'BLACKOUT'
  delay_s: number // compressed, what the demo actually waits
  real_delay_s?: number
  real_delay_min?: number
  compression?: number
  source?: string
  date?: string | null
  blackout_remaining_s?: number
  buffered?: number
  uplink_in_flight?: number
  rover_time?: number
  ground_time?: number
}

export interface LinkMsg {
  type: 'link'
  link: LinkInfo
}

export interface GroundEvent {
  type: 'ground_event'
  level: 'info' | 'warn' | 'error'
  msg: string
  mission_time: number
  data?: Record<string, unknown>
}

export type FaultType =
  | 'hidden_rock'
  | 'drop_frame'
  | 'corrupt_frame'
  | 'low_confidence'
  | 'model_offline'
  | 'blackout'
  | 'stale_command'

export interface WorldLayers {
  seed: number
  grid: number
  elevation: number[][]
  slope_deg: number[][]
  tile_image: string[][]
}
