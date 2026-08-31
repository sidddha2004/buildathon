# SwapShield AI

SwapShield is a defense-only multimodal verifier for return-substitution loss.
It compares dispatch evidence with the item received back, reports calibrated
risk and uncertainty, and gives a human reviewer a source-bound investigation
brief.

## Proof of concept

The current vertical slice includes:

- A working merchant review queue with genuine, substitution, and insufficient-evidence cases.
- Transparent fusion of visual similarity, VLM discrepancy, serial, weight, and quality signals.
- A mandatory recapture path when evidence quality is too low.
- A deterministic seeded evaluation with precision, recall, F1, and false-positive cost.
- An editable review threshold that exposes the cost trade-off.
- A grounded investigator contract with no autonomous adverse action.
- A lazy-loaded DINOv2 pair encoder for real image embeddings.
- A 4-bit Qwen3-VL evidence extractor with strict JSON, bounded visual tokens,
  one lower-resolution CUDA-memory retry, and citation validation.
- A local FastAPI endpoint designed for the RTX 5050's 8 GB VRAM envelope.
- A strict real-image manifest that rejects item identity leakage across splits.
- A bounded ABO subset builder that downloads only selected 360-degree views.
- Resumable GPU extraction and validation-only threshold selection.
- A trained, cross-validated and JSON-serializable fusion classifier.
- Held-out cost, calibration, bootstrap interval, latency, and slice reporting.
- A live browser upload flow connected to the local GPU API.
- An optional API-based LLM evidence auditor with strict JSON validation,
  prompt-injection containment, deterministic fallback, and advisory-only authority.
- A human-owned final decision and printable reviewer report.

The browser keeps seeded cases available without a GPU and includes a separate
**Live verify** tab for real uploads. The local API contains the real model
integration, loads the versioned fusion model by default, and optionally calls
an independent LLM evidence auditor. The auditor never changes the calibrated
probability or recommendation. Synthetic sensitivity controls, locked benchmark
metrics, and live inference are explicitly separated in the interface.

## Locked benchmark result

The item-disjoint ABO test contains 30 unseen product pairs after two disclosed
hardware-smoke exclusions. At the validation-selected 0.92 threshold:

| Metric | Result |
|---|---:|
| Precision | 1.000 |
| Recall | 0.867 |
| F1 | 0.929 |
| Average precision | 0.996 |
| False positives | 0 |
| False negatives | 2 |
| Recapture rate | 0.400 |
| p50 / p95 latency | 11.5 s / 65.4 s |

Both misses were lamps, and one third of genuine returns requested recapture.
These limitations remain visible in the dashboard and model card. The result is
a furniture-only POC benchmark, not a production-performance claim.

## Architecture

```text
Dispatch + return evidence
        │
        ├─> DINOv2 similarity
        ├─> Qwen3-VL structured observations
        └─> ID, weight, and image quality
                     │
                     └─> calibrated fusion + quality gate
                                      │
                                      ├─> bounded recommendation
                                      └─> independent LLM evidence audit
                                                       │
                                                       └─> human decision
```

## Run the final local application

Windows PowerShell, terminal 1:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -e "./ml[gpu,train,dev]"
Copy-Item .env.example .env
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -m uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Windows PowerShell, terminal 2:

```powershell
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`, choose **Live verify**, upload dispatch and
return images, then enter both SKUs. The UI displays the model recommendation,
the independent audit, and human-only action controls.

For the complete Windows checklist, use
[`docs/FINAL_RUN_GUIDE.md`](docs/FINAL_RUN_GUIDE.md).

## Optional LLM API evidence audit

SwapShield accepts an OpenAI-compatible chat-completions endpoint. Keep the key
server-side in `.env`:

```dotenv
SWAPSHIELD_AUDITOR_URL=https://your-provider.example/v1/chat/completions
SWAPSHIELD_AUDITOR_MODEL=your-model-name
SWAPSHIELD_AUDITOR_API_KEY=your-secret-key
```

Restart Uvicorn after editing `.env`. If these values are empty, unreachable,
or the response violates the schema, verification still succeeds with a visible
deterministic fallback. The auditor is a second-opinion consistency checker; it
is not part of the reported held-out metrics and cannot change the score.

## Run only the seeded site

```bash
npm run dev
```

## Run the reproducible baseline

```bash
make evaluate
make test-ml
```

The baseline has no Python dependencies. The GPU stack is optional so safety,
cost, and contract tests run without downloading model weights.

## Run real local inference

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e 'ml[gpu,train,dev]'
cp .env.example .env
set -a && source .env && set +a
make api
```

Then open `http://127.0.0.1:8000/docs` or run a pair directly:

```bash
python -m ml.scripts.verify_pair dispatch.jpg return.jpg \
  --dispatch-sku SKU-100 --return-sku SKU-100
```

See [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md) for the RTX 5050 checklist.

## Run the real-image benchmark

Build the default 120-product ABO subset directly from the official public
objects. This downloads about 91 MB of metadata plus 360 selected images—not
the 40 GB spin archive:

```powershell
python -m ml.scripts.build_abo_subset `
  --output data\real\abo `
  --items 120 `
  --views-per-item 3 `
  --workers 6
```

The builder creates 240 balanced pairs across chair, sofa, table, and lamp
categories, assigns product identities to splits before pairing, validates every
downloaded path, and writes the required attribution.

### 1. Extract train and validation features

This is the long RTX stage. It processes 176 training and 32 validation pairs,
writing every completed case immediately so `--resume` is interruption-safe:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -m ml.scripts.extract_real_features `
  data\real\abo\manifest.jsonl `
  artifacts\fusion-features.jsonl `
  --split train `
  --split validation `
  --resume
```

The 8 GB profile caps each Qwen image at 262,144 pixels and retries an isolated
CUDA memory spike at 131,072 pixels. A second failure is recorded as a safe
recapture case instead of stopping the batch.

### 2. Train and calibrate the fusion model

The large encoders stay frozen. A logistic fusion classifier is fitted on train,
Platt-calibrated using deterministic out-of-fold train predictions, and its
cost-sensitive threshold is selected on validation:

```powershell
python -m ml.scripts.train_fusion_model `
  artifacts\fusion-features.jsonl `
  artifacts\fusion-model.json `
  --report-output artifacts\fusion-training-report.md
```

The JSON artifact contains the scaler, coefficients, calibration, and locked
threshold. Loading it for inference does not require pickle.

### 3. Touch test only after the model is locked

```powershell
python -m ml.scripts.extract_real_features `
  data\real\abo\manifest.jsonl `
  artifacts\locked-test-features.jsonl `
  --split test `
  --resume

python -m ml.scripts.score_fusion_model `
  artifacts\fusion-model.json `
  artifacts\fusion-features.jsonl `
  artifacts\locked-test-features.jsonl `
  --output artifacts\fusion-scored.jsonl `
  --exclude-case ABO-TES-B07124WCZY-G `
  --exclude-case ABO-TES-B07124WCZY-S

python -m ml.scripts.evaluate_real_predictions `
  artifacts\fusion-scored.jsonl `
  --json-output artifacts\real-report.json `
  --markdown-output artifacts\real-report.md
```

The two excluded cases were used for the RTX smoke test before training and are
therefore disclosed and kept outside the final reported test metrics.

See [`data/real/README.md`](data/real/README.md) for the generated files,
licensing, reruns, and the separate self-captured workflow.

## Repository map

```text
app/                         reviewer dashboard
lib/swapshield.ts            browser risk baseline and seeded cases
data/contracts/              canonical case schema
data/real/                   ABO builder guide and self-captured manifest template
evaluation/results/          locked report and portable fusion model
ml/swapshield_ml/            Python scoring and evaluation package
ml/scripts/                  reproducible commands
ml/tests/                    baseline, grounding, and pipeline tests
service/app.py               local FastAPI model service
docs/EVALUATION.md           held-out evaluation contract
docs/GPU_SETUP.md            RTX 5050 setup and smoke test
docs/LLM_GUARDRAILS.md       grounded-agent safety contract
docs/MODEL_CARD.md           intended use, results, and limitations
docs/THREAT_MODEL.md         scope and defense-only boundary
```

## Post-hackathon model roadmap

1. Add a new development dataset with more fine-grained lamp negatives.
2. Validate serial, barcode, weight, empty-box, and accessory signals on data
   that actually varies those fields.
3. Reduce genuine recapture friction without changing the frozen v1 test claim.
4. Profile and reduce the 65.4-second p95 local inference latency.

## Safety boundary

SwapShield verifies evidence. It does not infer intent, label people as
fraudulent, reject refunds automatically, block accounts, submit disputes, or
move money. See `docs/THREAT_MODEL.md` for the full boundary.
