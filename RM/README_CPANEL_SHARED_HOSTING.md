# cPanel Shared Hosting Deployment Guide
## Land Matter Prediction System (Flask + Trained ML Models)

This guide explains how to deploy this project on **cPanel shared hosting** using the **Setup Python App** feature (Passenger/WSGI).

---

## 1. Important Reality Check

Before deploying, confirm these points with your hosting provider:

1. Your cPanel has **Setup Python App** enabled.
2. Python **3.10+** is available.
3. You can install Python packages inside the app virtual environment.

If Setup Python App is not available, this Flask app cannot run on normal shared hosting.

### Ollama note
Running Ollama directly on shared hosting is usually not possible (no root/system service access).  
Use one of these options:

1. Disable Ollama explanations in production UI.
2. Host Ollama on another server/VPS and set `OLLAMA_BASE_URL` to that remote endpoint.

---

## 2. Prepare Files Locally First

From your project root:

```bash
python src/main_training.py --data_file data/structured/cases.csv data/structured/scn-cases.csv --output_dir data/models
```

You must have these files after training:

1. `data/models/best_model.pkl`
2. `data/models/inference_artifacts.pkl`
3. `data/models/model_metadata.json`
4. `data/models/training_results.json`
5. `data/models/cv_results.json`
6. `data/models/case_reference_records.json` (for related-case citation in predictions)

Do not train on shared hosting if you can avoid it. Train locally, upload artifacts.

---

## 3. Build Upload Package

Create a zip of the project without local virtual environment files.

Include:

1. `src/`
2. `data/models/`
3. `requirements.txt`
4. `README.md` and this deployment guide (optional)

Exclude:

1. `venv/`
2. `__pycache__/`
3. local logs and temporary files

---

## 4. Upload to cPanel

1. Log in to cPanel.
2. Open **File Manager**.
3. Go to your home directory, usually: `/home/CPANEL_USERNAME/`.
4. Create a folder, for example: `land-system`.
5. Upload your project zip into `land-system`.
6. Extract zip so your code is in:

`/home/CPANEL_USERNAME/land-system/src`

---

## 5. Create Python App in cPanel

1. Open **Setup Python App** in cPanel.
2. Click **Create Application**.
3. Recommended values:

- Python version: `3.10` (or closest available)
- Application root: `land-system`
- Application URL: your preferred route (for example `/land`)
- Application startup file: `passenger_wsgi.py`
- Application entry point: `application`

4. Create the app.
5. Copy the virtualenv activation command shown by cPanel (you will use it in Terminal).

---

## 6. Install Python Dependencies on Server

Open **Terminal** in cPanel (or SSH), then:

```bash
cd /home/CPANEL_USERNAME/land-system
# Use the exact activation command shown in Setup Python App
source /home/CPANEL_USERNAME/virtualenv/land-system/3.10/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `spacy` model is required in your flow:

```bash
python -m spacy download en_core_web_sm
```

---

## 7. Create `passenger_wsgi.py`

In `/home/CPANEL_USERNAME/land-system/passenger_wsgi.py` add:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_app.app import app as application
```

This is required so Passenger can start Flask correctly.

---

## 8. Configure Environment Variables in cPanel

In **Setup Python App** -> your app -> **Environment variables**, set:

1. `LAND_MODELS_DIR=/home/CPANEL_USERNAME/land-system/data/models`
2. `LAND_MODEL_PATH=/home/CPANEL_USERNAME/land-system/data/models/best_model.pkl`
3. `LAND_ARTIFACTS_PATH=/home/CPANEL_USERNAME/land-system/data/models/inference_artifacts.pkl`
4. `LAND_ACT_PDF_PATH=/home/CPANEL_USERNAME/land-system/data/legal/Nigerian-land-use-act-2004.pdf`
5. `EVIDENCE_ACT_PDF_PATH=/home/CPANEL_USERNAME/land-system/data/legal/evidence-act-2011.pdf`

Optional for Ollama:

6. `OLLAMA_BASE_URL=http://YOUR_OLLAMA_HOST:11434`
7. `OLLAMA_MODEL=llama3`
8. `OLLAMA_TIMEOUT=60`

Optional for statute online fallback:

9. `LAND_ACT_URLS=https://lawsofnigeria.placng.org/print.php?sn=228`
10. `EVIDENCE_ACT_URLS=https://example.com/evidence-act`

Optional hybrid controls:

11. `TOP_MODELS_COUNT=3`
12. `HYBRID_ML_WEIGHT=0.8`
13. `HYBRID_OLLAMA_WEIGHT=0.2`

If you are not using Ollama, skip the Ollama variables.

---

## 9. Confirm Required Files on Server

Ensure these paths exist exactly:

1. `/home/CPANEL_USERNAME/land-system/src/web_app/app.py`
2. `/home/CPANEL_USERNAME/land-system/src/predict.py`
3. `/home/CPANEL_USERNAME/land-system/data/models/best_model.pkl`
4. `/home/CPANEL_USERNAME/land-system/data/models/inference_artifacts.pkl`
5. `/home/CPANEL_USERNAME/land-system/passenger_wsgi.py`

---

## 10. Restart and Test

1. In **Setup Python App**, click **Restart**.
2. Open your app URL.
3. Test health endpoint:

`https://your-domain.com/your-app-path/health`

Expected JSON should include `"status": "ok"`.

---

## 11. Common cPanel Issues and Fixes

### Problem: `ModuleNotFoundError`
Cause: package not installed in app virtualenv.  
Fix: activate the cPanel app environment and run `pip install -r requirements.txt` again.

### Problem: 500 Internal Server Error immediately
Cause: wrong `passenger_wsgi.py` import path or wrong startup/entrypoint settings.  
Fix: confirm startup file is `passenger_wsgi.py` and entry point is `application`.

### Problem: Predictor not ready in `/health`
Cause: model/artifact path wrong or files missing.  
Fix: verify `LAND_MODEL_PATH` and `LAND_ARTIFACTS_PATH` environment variables and file existence.

### Problem: Ollama error in UI
Cause: Ollama not reachable from shared host.  
Fix: disable Ollama usage or set `OLLAMA_BASE_URL` to reachable remote host.

---

## 12. Recommended Production Flow

1. Train locally.
2. Upload only code + trained artifacts.
3. Keep `--n_jobs 1` if retraining on constrained hosts.
4. Use remote Ollama service if explanations are required.

---

## 13. Quick Deployment Checklist

1. App created in Setup Python App.
2. `requirements.txt` installed in app venv.
3. `passenger_wsgi.py` exists and imports `web_app.app`.
4. Model and artifact files uploaded.
5. Environment variables set.
6. App restarted.
7. `/health` returns ok.
