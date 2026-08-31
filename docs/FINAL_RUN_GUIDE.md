# SwapShield AI v1.0 final run guide

This guide assumes the repository is at
`D:\buildathon\new\swapshield-ai` and the existing virtual environment folder
is named `venv`.

## 1. Install the final dependencies once

Open PowerShell in the repository root:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e "./ml[gpu,train,dev]"
npm ci
```

## 2. Configure the optional LLM API

```powershell
Copy-Item .env.example .env
notepad .env
```

To enable the independent API audit, set:

```dotenv
SWAPSHIELD_AUDITOR_URL=https://your-provider.example/v1/chat/completions
SWAPSHIELD_AUDITOR_MODEL=your-model-name
SWAPSHIELD_AUDITOR_API_KEY=your-secret-key
```

Leave all three empty to use the safe deterministic fallback. Never put the key
in frontend code or commit `.env`.

## 3. Start the GPU API

PowerShell terminal 1:

```powershell
.\venv\Scripts\Activate.ps1
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
$env:SWAPSHIELD_DEVICE="cuda"
$env:SWAPSHIELD_VLM_4BIT="true"
python -m uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`. It should report the trained fusion model
and either `llm_api` or `deterministic_fallback` for the auditor. `/` returning a
small JSON status is normal; Swagger is at `http://127.0.0.1:8000/docs`.

## 4. Start the dashboard

PowerShell terminal 2:

```powershell
cd D:\buildathon\new\swapshield-ai
npm run dev
```

Open `http://127.0.0.1:5173` and select **Live verify**.

## 5. Run three demo cases

Use clear photos and the same SKU when no SKU mismatch is intended.

1. Same mouse, different angle: should usually approve or safely request
   recapture when image quality is low.
2. Mouse versus another product: should route a supported mismatch to human
   review.
3. Blurry or obstructed return image: should request recapture rather than guess.

After each run, check the Qwen observations, calibrated probability, auditor
source, and human decision controls. The print button creates a reviewer report.

## 6. Verify the source checkpoint

```powershell
python -m unittest discover -s ml/tests -v
npm run lint
npm run build
```

The benchmark results in the Evaluation tab are the locked 30-pair ABO test.
The live LLM audit is an advisory post-processing layer and must not be described
as improving those precision or recall results.
