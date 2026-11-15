import * as THREE from 'three'

export const N = 200
export const CELL = 0.25
export const HALF = 25
export const EXAGGERATION = 3

// class palette: soil, bedrock, sand, big rock
const PALETTE = ['#c8a165', '#8a8f98', '#f2d16b', '#e5484d'].map((h) => new THREE.Color(h))
const UNKNOWN = new THREE.Color('#1b1d22')
const LIGHT = new THREE.Vector3(-0.5, 0.8, -0.35).normalize()

export const cellToWorld = (col: number, row: number): [number, number] => [col * CELL - HALF, row * CELL - HALF]

export function buildGeometry(elevation: number[][]): { geo: THREE.PlaneGeometry; heights: Float32Array } {
  const geo = new THREE.PlaneGeometry(50, 50, N - 1, N - 1)
  geo.rotateX(-Math.PI / 2) // row 0 (first vertex row) lands at z = -25 (north)
  let min = Infinity
  for (const row of elevation) for (const v of row) if (v < min) min = v
  const pos = geo.attributes.position as THREE.BufferAttribute
  const heights = new Float32Array(N * N)
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const i = r * N + c
      const h = (elevation[r][c] - min) * EXAGGERATION
      heights[i] = h
      pos.setXYZ(i, c * CELL - HALF, h, r * CELL - HALF)
    }
  }
  geo.computeVertexNormals()
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(N * N * 3), 3))
  return { geo, heights }
}

/** Recolour in place from belief classes blended with hillshade. Never rebuilds geometry. */
export function paintColors(geo: THREE.BufferGeometry, cls: Uint8Array) {
  const col = geo.attributes.color as THREE.BufferAttribute
  const nor = geo.attributes.normal as THREE.BufferAttribute
  const a = col.array as Float32Array
  for (let i = 0; i < N * N; i++) {
    const shade = 0.55 + 0.45 * Math.max(0, nor.getX(i) * LIGHT.x + nor.getY(i) * LIGHT.y + nor.getZ(i) * LIGHT.z)
    const base = PALETTE[cls[i]] ?? UNKNOWN
    a[i * 3] = base.r * shade
    a[i * 3 + 1] = base.g * shade
    a[i * 3 + 2] = base.b * shade
  }
  col.needsUpdate = true
}

export function heightAt(heights: Float32Array, col: number, row: number): number {
  const c = Math.min(N - 1, Math.max(0, Math.round(col)))
  const r = Math.min(N - 1, Math.max(0, Math.round(row)))
  return heights[r * N + c]
}
