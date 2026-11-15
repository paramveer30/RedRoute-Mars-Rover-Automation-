import { useEffect, useState } from 'react'
import { MissionMap } from './MissionMap'
import { useMission, type LayerKey } from './store'
import type { FaultType, PerceptionMode, RoverState } from './types'

const STATE_COLOR: Record<RoverState, string> = {
  IDLE: 'bg-slate-600',
  SCANNING: 'bg-sky-600',
  PLANNING: 'bg-indigo-600',
  DRIVING: 'bg-green-600',
  REPLANNING: 'bg-amber-600',
  SAFE_STOP: 'bg-red-600',
}

const LAYERS: [LayerKey, string][] = [
  ['hillshade', 'Elevation'],
  ['classes', 'Belief classes'],
  ['uncertainty', 'Uncertainty'],
  ['blocked', 'Blocked'],
  ['path', 'Path'],
  ['rover', 'Rover + sensor'],
  ['truth', 'TRUTH (debug)'],
]

const LEGEND: [string, string][] = [
  ['#c4a070', 'soil'],
  ['#8c8c96', 'bedrock'],
  ['#ecc846', 'sand'],
  ['#dc3232', 'big rock'],
  ['#0a0a0a', 'unknown'],
]

const FAULTS: [FaultType, string, string][] = [
  ['hidden_rock', 'Hidden rock', 'Rock appears ~4 m ahead on the route'],
  ['drop_frame', 'Drop frames', 'Camera stops delivering frames: watchdog'],
  ['corrupt_frame', 'Corrupt frame', 'One frame fails its checksum'],
  ['low_confidence', 'Low confidence', 'Blur the next frames (model mode)'],
  ['model_offline', 'Model offline', 'Geometry-only, half speed'],
  ['blackout', 'Blackout 15 s', 'Earth link lost'],
  ['stale_command', 'Stale command', 'Send a command built on an old route'],
]

const COMPRESSION = [1, 10, 60, 300, 600]

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-white/50">{k}</span>
      <span>{v}</span>
    </div>
  )
}

export default function App() {
  const { connected, last, link, events, layers, toggle, connect, resume, stop, reset, setMode, setCompression, fault } =
    useMission()
  const [err, setErr] = useState<string | null>(null)
  const [seed, setSeed] = useState('1')
  useEffect(() => connect('demo'), [connect])

  const run = async (p: Promise<string | null>) => setErr(await p)
  const mode = last?.perception_mode

  return (
    <div className="flex min-h-full flex-col">
      {mode === 'oracle' && (
        <div className="bg-amber-500 px-4 py-1 text-center text-sm font-bold text-black">
          ORACLE MODE: perception is reading ground truth, not model output
        </div>
      )}
      {(mode === 'geometry_only' || mode === 'offline') && (
        <div className="bg-red-700 px-4 py-1 text-center text-sm font-bold">
          MODEL OFFLINE: geometry-only perception, half speed
        </div>
      )}
      {link?.status === 'BLACKOUT' && (
        <div className="bg-purple-700 px-4 py-1 text-center text-sm font-bold">
          COMMS BLACKOUT: {link.blackout_remaining_s?.toFixed(0)} s left · {link.buffered ?? 0} messages buffered
          onboard · uplink commands are dropped
        </div>
      )}
      {last?.state === 'SAFE_STOP' && (
        <div className="bg-red-600 px-4 py-1 text-center text-sm font-bold">
          SAFE_STOP: {last.safe_stop_reason || 'awaiting operator'} — operator must resume
        </div>
      )}
      <header className="flex flex-wrap items-center gap-4 border-b border-white/10 px-4 py-2">
        <h1 className="text-lg font-semibold tracking-wide">RedRoute Mission Control</h1>
        <span className={connected ? 'text-green-400' : 'text-red-400'}>
          {connected ? '● backend connected' : '● backend offline'}
        </span>
        {last && (
          <span className={`rounded px-2 py-0.5 font-mono text-sm font-bold ${STATE_COLOR[last.state]}`}>
            {last.state}
          </span>
        )}
      </header>

      <main className="flex flex-1 flex-wrap gap-4 p-4">
        <MissionMap />

        <aside className="flex w-80 flex-col gap-4 text-sm">
          <section className="space-y-1 rounded border border-white/10 p-3 font-mono">
            <h2 className="mb-1 font-sans font-semibold">Telemetry</h2>
            <Row k="mission time" v={`${last?.mission_time.toFixed(1) ?? '-'} s`} />
            <Row k="battery" v={`${last?.battery.toFixed(1) ?? '-'} %`} />
            <Row k="odometer" v={`${last?.odometer_m.toFixed(1) ?? '-'} m`} />
            <Row k="since Earth contact" v={`${last?.since_contact_m.toFixed(1) ?? '-'} m`} />
            <Row k="route" v={last ? `v${last.route_version}` : '-'} />
            <Row k="pose" v={last ? `${last.pose.x}, ${last.pose.y} @ ${last.pose.heading}°` : '-'} />
            <Row k="perception" v={mode ?? '-'} />
            <Row k="seed / seq" v={last ? `${last.seed} / ${last.seq}` : '-'} />
          </section>

          <section className="space-y-1 rounded border border-white/10 p-3 font-mono">
            <h2 className="mb-1 font-sans font-semibold">Comms link</h2>
            <Row k="status" v={<span className={link?.status === 'BLACKOUT' ? 'text-purple-300' : 'text-green-400'}>{link?.status ?? '-'}</span>} />
            <Row k="real delay" v={link?.real_delay_min != null ? `${link.real_delay_min.toFixed(1)} min` : '-'} />
            <Row k="on-screen delay" v={link ? `${link.delay_s.toFixed(1)} s (×${link.compression})` : '-'} />
            <div className="text-right text-[10px] text-white/40">{link?.source}{link?.date ? `, ${link.date}` : ''}</div>
            <Row k="rover clock" v={`${link?.rover_time?.toFixed(1) ?? '-'} s`} />
            <Row k="ground clock" v={`${link?.ground_time?.toFixed(1) ?? '-'} s`} />
            <Row k="lag" v={link?.rover_time != null && link.ground_time != null ? `${(link.rover_time - link.ground_time).toFixed(1)} s` : '-'} />
            <Row k="buffered / uplink" v={`${link?.buffered ?? 0} / ${link?.uplink_in_flight ?? 0}`} />
            <div className="flex items-center justify-between gap-2 pt-1 font-sans">
              <span className="text-white/50">time compression</span>
              <select className="rounded bg-white/10 px-2 py-0.5" value={link?.compression ?? 60}
                onChange={(e) => run(setCompression(Number(e.target.value)))}>
                {COMPRESSION.map((c) => <option key={c} value={c}>×{c}</option>)}
              </select>
            </div>
          </section>

          <section className="space-y-2 rounded border border-white/10 p-3">
            <h2 className="font-semibold">Fault injection</h2>
            <div className="grid grid-cols-2 gap-1">
              {FAULTS.map(([type, label, title]) => (
                <button key={type} title={title} onClick={() => run(fault(type))}
                  className="rounded bg-red-900/60 px-2 py-1 text-left text-xs hover:bg-red-800">{label}</button>
              ))}
            </div>
          </section>

          <section className="space-y-2 rounded border border-white/10 p-3">
            <h2 className="font-semibold">Operator</h2>
            <div className="flex flex-wrap gap-2">
              <button className="rounded bg-green-700 px-2 py-1 hover:bg-green-600 disabled:opacity-40"
                disabled={last?.state !== 'SAFE_STOP'} onClick={() => run(resume())}>Resume</button>
              <button className="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600" onClick={() => run(stop())}>Stop</button>
              <button className="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600"
                onClick={() => run(reset(Number(seed)))}>Reset</button>
              <input className="w-14 rounded bg-white/10 px-2 py-1 font-mono" value={seed}
                onChange={(e) => setSeed(e.target.value)} title="world seed" />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-white/50">perception</span>
              <select className="rounded bg-white/10 px-2 py-1" value={mode ?? 'oracle'}
                onChange={(e) => run(setMode(e.target.value as PerceptionMode))}>
                <option value="oracle">oracle</option>
                <option value="model">model</option>
                <option value="geometry_only">geometry_only</option>
              </select>
            </div>
            {err && <div className="text-red-400">{err}</div>}
          </section>

          <section className="rounded border border-white/10 p-3">
            <h2 className="mb-1 font-semibold">Layers</h2>
            <div className="grid grid-cols-2 gap-x-2">
              {LAYERS.map(([k, label]) => (
                <label key={k} className={`flex items-center gap-1 ${k === 'truth' ? 'text-red-300' : ''}`}>
                  <input type="checkbox" checked={layers[k]} onChange={() => toggle(k)} /> {label}
                </label>
              ))}
            </div>
            <div className="mt-2 flex flex-wrap gap-2 text-xs">
              {LEGEND.map(([c, n]) => (
                <span key={n} className="flex items-center gap-1">
                  <span className="inline-block h-3 w-3 rounded-sm border border-white/20" style={{ background: c }} />
                  {n}
                </span>
              ))}
            </div>
          </section>
        </aside>

        <section className="flex min-w-72 flex-1 flex-col rounded border border-white/10 p-3 text-sm">
          <h2 className="mb-1 font-semibold">Events</h2>
          <ol className="max-h-[640px] flex-1 space-y-0.5 overflow-y-auto font-mono text-xs">
            {events.map((e, i) => (
              <li key={i} className={e.level === 'error' ? 'text-red-400' : e.level === 'warn' ? 'text-amber-300' : 'text-white/80'}>
                <span className="text-white/40">{e.mission_time.toFixed(1)}s</span> {e.msg}
                {e.ground && <span className="ml-1 rounded bg-sky-900 px-1">ground</span>}
                {e.late && <span className="ml-1 rounded bg-purple-800 px-1">late</span>}
              </li>
            ))}
          </ol>
        </section>
      </main>
    </div>
  )
}
