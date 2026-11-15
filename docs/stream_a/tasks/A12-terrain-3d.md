# A12: 3D terrain view  (P1, the first Stream A item to cut if behind)

**Wave 4, after the P0 freeze (hour 16).** Frontend, about 2 hours.

## You own
- `frontend/src/Terrain3D.tsx` (+ any `frontend/src/three/*` helpers)

## Do not touch
`frontend/src/App.tsx`, `store.ts`, `types.ts` and `MissionMap.tsx` (Stream B). Ask B to mount `<Terrain3D />`, and give them the one-line import.

## Inputs
- `GET /api/world/{seed}` → `elevation` (200 × 200 metres); B owns the endpoint
- The zustand store from `frontend/src/store.ts` (`last.pose`, `last.path`, belief buffer once B adds it)

## Spec
- `@react-three/fiber` `<Canvas>` + drei `<OrbitControls>`.
- `PlaneGeometry(50, 50, 199, 199)` rotated flat. Vertex height = (elevation - min) × 3 (exaggeration, labelled in the UI).
- Vertex colours: belief class colours (same palette as A11), dark for unknown, blended with a hillshade.
- Rover: a small box with a heading arrow at `pose`; path: drei `<Line>` raised 0.2 m above the surface.
- Grid cell (row, col) → world: `x = col*0.25 - 25`, `z = row*0.25 - 25`.
- Update colours only for tiles in `belief_delta` (don't rebuild the geometry every tick).
- `npx tsc -b` passes.

## Acceptance
- [ ] The 3D view mirrors the 2D map (same rover position and path) and runs at 60 fps on the laptop
