import { useEffect } from 'react'
import { MissionMap } from './MissionMap'
import { useMission } from './store'

export default function App() {
  const { connected, last, connect } = useMission()
  useEffect(() => connect('demo'), [connect])

  return (
    <div className="flex h-full flex-col">
      {last?.perception_mode === 'oracle' && (
        <div className="bg-amber-500 px-4 py-1 text-center text-sm font-bold text-black">
          ORACLE MODE: perception is reading ground truth, not model output
        </div>
      )}
      <header className="flex items-center gap-4 border-b border-white/10 px-4 py-2">
        <h1 className="text-lg font-semibold tracking-wide">RedRoute Mission Control</h1>
        <span className={connected ? 'text-green-400' : 'text-red-400'}>
          {connected ? '● backend connected' : '● backend offline'}
        </span>
      </header>
      <main className="flex flex-1 gap-4 p-4">
        <MissionMap />
        <aside className="w-72 space-y-1 font-mono text-sm">
          <div>state: {last?.state ?? '-'}</div>
          <div>mission t: {last?.mission_time.toFixed(1) ?? '-'} s</div>
          <div>seq: {last?.seq ?? '-'}</div>
          <div>pose: {last ? `${last.pose.x}, ${last.pose.y} @ ${last.pose.heading}°` : '-'}</div>
          <div>tiles seen: {last?.belief_delta.tiles.join(' ') ?? '-'}</div>
          <div>link: {last?.link.status ?? '-'}</div>
        </aside>
      </main>
    </div>
  )
}
