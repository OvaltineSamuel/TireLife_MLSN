# TireLife

TireLife predicts remaining tire life from vehicle, tire, and driving-style inputs.

![TireLife Demo QR Code](assets/TireLife_UI_QRCode.png)

The demo app is organized as:

```text
backend/            FastAPI API and ML inference service
src/                React frontend source
artifacts/          Local dataset and trained model artifacts, gitignored
requirements.txt    Runtime Python dependencies
requirements-dev.txt Notebook/training dependencies
package.json        Frontend dependencies and Vite scripts
```

The React app calls FastAPI endpoints under `/api`. In local development Vite proxies
those calls to `http://127.0.0.1:8000`. In production, FastAPI can serve the built
React files from `dist/`, so a single web service can host the full demo.

## Local Setup

From the repo root:

```bash
pip install -r requirements.txt
npm install
```

For notebook/model experimentation, install the larger optional dependency set:

```bash
pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env.local` if you need local overrides.

## Required Artifacts

The backend expects the cleaned training data at:

```text
artifacts/tyre_rul_cleaned.csv
```

On first startup with the dataset present, the backend builds a feature profile and
trained model artifacts in:

```text
artifacts/trained_models/
```

Important files for deployment without the large CSV:

```text
artifacts/trained_models/feature_profile.joblib
artifacts/trained_models/lightgbm_normal_non_leakage.joblib
artifacts/trained_models/deeplearning_test_mlp.joblib  # optional
```

`artifacts/` is intentionally ignored by git because the source CSV and training
outputs can be large. For a hosted demo, provide the prebuilt artifacts through the
hosting platform or intentionally force-add only the small model/profile files.

## Run Locally

Start the backend:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Or on macOS/Linux:

```bash
./backend/start_backend.sh
```

Start the frontend:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

## Production Build

Build React:

```bash
npm run build
```

Start FastAPI from the repo root:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

When `dist/index.html` exists, FastAPI serves the frontend and API from the same
origin.

## API

- `GET /api/health`
- `POST /api/predict`
- `POST /api/chat`
- `POST /api/config/gemini`

Example prediction:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lightgbm",
    "features": {
      "current_tread_depth(mm)": 4.5,
      "kilometers_driven(km)": 28000,
      "average_inflation_pressure(psi)": 32,
      "tyre_age(years)": 3
    }
  }'
```
