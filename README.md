# RedRoute: Uncertainty-Aware Mars Rover Navigation

Mission control plus an onboard autonomy stack. It drives a simulated Perseverance-class rover across real Jezero crater elevation (USGS HiRISE DTM), using a terrain model trained on real NASA rover images (AI4Mars), and stays safe when the AI is unsure or Earth goes silent.

- **Stream A plan (perception, world, eval):** [docs/STREAM_A.md](docs/STREAM_A.md)
- **Stream B plan (autonomy, reliability, mission control):** [docs/STREAM_B.md](docs/STREAM_B.md)
- **Shared contract:** [backend/app/contracts.py](backend/app/contracts.py). Change it only together.

## Setup

Requires Python 3.11+ (3.13 tested) and Node 20+. No API keys.

```bash
python -m venv .venv
# NVIDIA GPU (RTX 50-series needs cu128):
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CPU only: .venv/Scripts/python -m pip install torch torchvision
.venv/Scripts/python -m pip install -r requirements.txt
cd frontend && npm install
```
On macOS/Linux use `.venv/bin/python`. GDAL CLI tools are **not** required: rasterio's wheel bundles GDAL.

## Run

```bash
.venv/Scripts/python -m uvicorn backend.app.api:app --reload --port 8000   # backend
cd frontend && npm run dev                                                 # http://localhost:5173
.venv/Scripts/python -m pytest                                             # tests
```

## Data scripts

```bash
.venv/Scripts/python ml/check_dataset.py          # AI4Mars from Hugging Face
.venv/Scripts/python world/crop_dtm.py            # 50 m Jezero DTM crop -> terrain/jezero_crop.tif
.venv/Scripts/python scripts/fetch_light_delay.py # JPL Horizons -> terrain/light_delay.json
```

## Layout

```
ml/        Stream A  training, evaluation, calibration
world/     Stream A  DTM processing, tile warping, world builder (stub_world.py until hour 6)
eval/      Stream A  harness, baselines, results
backend/app/
  contracts.py   SHARED
  perception.py  A   observe(): oracle stub now, model at hour 12
  mapping.py     A   cost_grid()
  planner.py     B   A*
  rover.py       B   simulator + state machine
  link.py        B   comms link, delay, blackout, HMAC
  faults.py      B   fault injection
  api.py         B   REST + WebSocket
frontend/  Stream B  (A owns the 3D view and vision-panel components)
terrain/   jezero_crop.tif, light_delay.json (committed); tiles/, preds/ (generated, gitignored)
```

## Data sources
- AI4Mars (NASA JPL), Hugging Face `hassanjbara/AI4MARS`
- USGS Mars 2020 TRN HiRISE DTM Mosaic, Jezero, 1 m
- JPL Horizons API, Earth to Mars light time
