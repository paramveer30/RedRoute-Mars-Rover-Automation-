import { useEffect, useMemo, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { Line, OrbitControls } from '@react-three/drei'
import type * as THREE from 'three'
import { useMission } from './store'
import { EXAGGERATION, buildGeometry, cellToWorld, heightAt, paintColors } from './three/terrainMesh'

function Scene() {
  const world = useMission((s) => s.world)
  const path = useMission((s) => s.last?.path)
  const built = useMemo(() => (world ? buildGeometry(world.elevation) : null), [world])
  const rover = useRef<THREE.Group>(null)

  // colours + rover pose update in place on store changes; geometry is built once per world
  useEffect(() => {
    if (!built) return
    let painted = -1
    const apply = () => {
      const s = useMission.getState()
      if (s.beliefVersion !== painted) {
        painted = s.beliefVersion
        paintColors(built.geo, s.cls)
      }
      const p = s.last?.pose
      if (p && rover.current) {
        const [x, z] = cellToWorld(p.x, p.y)
        rover.current.position.set(x, heightAt(built.heights, p.x, p.y) + 0.15, z)
        rover.current.rotation.y = -(p.heading * Math.PI) / 180 // heading 0 = north (-z), clockwise
      }
    }
    apply()
    return useMission.subscribe(apply)
  }, [built])

  const linePts = useMemo(() => {
    if (!built || !path || path.length < 2) return null
    return path.map(([cx, cy]) => {
      const [x, z] = cellToWorld(cx, cy)
      return [x, heightAt(built.heights, cx, cy) + 0.2, z] as [number, number, number]
    })
  }, [built, path])

  return (
    <>
      <ambientLight intensity={0.8} />
      {built && (
        <mesh geometry={built.geo}>
          <meshBasicMaterial vertexColors />
        </mesh>
      )}
      <group ref={rover}>
        <mesh>
          <boxGeometry args={[0.6, 0.3, 0.9]} />
          <meshStandardMaterial color="#ffffff" />
        </mesh>
        <mesh position={[0, 0.3, -0.8]} rotation={[-Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.2, 0.6, 12]} />
          <meshStandardMaterial color="#3b82f6" />
        </mesh>
      </group>
      {linePts && <Line points={linePts} color="#22d3ee" lineWidth={2} />}
      <OrbitControls makeDefault target={[0, 0, 0]} maxPolarAngle={Math.PI / 2.05} />
    </>
  )
}

export default function Terrain3D() {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: 300 }}>
      <Canvas camera={{ position: [0, 35, 38], fov: 50, near: 0.1, far: 300 }}>
        <Scene />
      </Canvas>
      <div style={{ position: 'absolute', left: 8, bottom: 8, fontSize: 12, color: '#ddd', pointerEvents: 'none' }}>
        {EXAGGERATION}x vertical exaggeration
      </div>
    </div>
  )
}
