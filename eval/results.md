# Mission evaluation

Rover: `real` | git `4ab17cd7442fb5d59fea2a3bf604eeed55810028` | seeds 0..199 (n=200)

| Config | Success rate [95% CI] | Hazard entries / mission [95% CI] | Route ratio (reached) | Replan p50 / p95 ms | n (errors) |
| --- | --- | --- | --- | --- | --- |
| geometry_only | 0.04 [0.01, 0.07] | 2.27 [2.10, 2.44] | 0.858 (n=192) | 37.2 / 78.4 | 200 (0) |
| redroute_no_u | 0.23 [0.17, 0.29] | 0.71 [0.61, 0.82] | 1.167 (n=74) | 23.6 / 75.9 | 200 (0) |
| redroute | 0.34 [0.28, 0.41] | 0.64 [0.54, 0.74] | 1.140 (n=100) | 67.1 / 140.4 | 200 (0) |
| oracle | 0.54 [0.47, 0.61] | 0.47 [0.40, 0.56] | 1.125 (n=129) | 56.9 / 136.4 | 200 (0) |

Throughput: 800 missions in 46.4 s on 23 worker(s) = 1034.6 missions/min, n/a observe() calls/s.

## Definitions

- **hazard cell**: truth sand, truth big rock, or slope > 20 deg; truth 255 is not a hazard
- **hazard entry**: footprint (disk radius 2 cells = 0.5 m) overlaps a hazard cell; the trajectory is densified by straight-line interpolation; each contiguous run counts once
- **success**: final trajectory cell within 1 m of goal AND 0 hazard entries
- **route ratio**: trajectory length / shortest hazard-free 8-connected route on truth (same footprint mask, no corner cutting); reached missions only
- **ci95**: percentile bootstrap of the mean, 10000 resamples, seed 0
- **scenario**: seeded start/goal >= 30 m apart on non-hazard truth cells, goal reachable by the safe route; identical across configs

## Conditions

- **prediction cache**: `world/preds20.npz`, temperature 1.5186748519290354, weights sha256 `a7c5272b640bdf5f133eec6b36cc02518dc896da2d31631084edfebf141cd20e`
- **cost map blocking**: known cells with belief P(sand) >= 0.5 are blocked (blobs < 4 cells dropped), inflated by 0.5 m; big rock P > 0.5, slope > 20 deg and sensed geometry inflated by 1.5 m; sub-threshold sand keeps soft cost w_s*risk
