# RTX 5050 local inference setup

SwapShield runs DINOv2 and Qwen3-VL locally because the hosted review dashboard
does not provide a GPU. Qwen is loaded in 4-bit NF4 mode; DINOv2 uses BF16 when
the installed CUDA stack supports it.

## 1. Verify the NVIDIA runtime

Use a current NVIDIA driver and a CUDA-enabled PyTorch build. On Windows, WSL2
with Ubuntu is the most reproducible environment.

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The second command must print `True` and your RTX 5050 before model loading.

## 2. Create the environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'ml[gpu,train,dev]'
cp .env.example .env
```

If PyTorch's default wheel does not detect the card, install the CUDA wheel
recommended by the current PyTorch selector, then repeat the editable install.

## 3. Start the API

```bash
set -a && source .env && set +a
make api
```

Models load lazily on the first verification. The first request downloads model
weights and will take longer; later requests reuse them in VRAM.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/verify \
  -F dispatch_image=@examples/dispatch.jpg \
  -F return_image=@examples/return.jpg \
  -F dispatch_sku=SKU-100 \
  -F return_sku=SKU-100 \
  -F dispatch_serial=ABC-123 \
  -F return_serial=ABC-123 \
  -F dispatch_weight_grams=1000 \
  -F return_weight_grams=1010
```

Swagger is available locally at `http://127.0.0.1:8000/docs`.

To use the browser workflow, start a second PowerShell terminal, run `npm ci`
and `npm run dev`, then open `http://127.0.0.1:5173` and choose **Live verify**.
The page connects to the local API, accepts the two images, shows the structured
model evidence, and leaves the final action to the reviewer.

## Optional independent LLM audit

Edit `.env` before starting Uvicorn:

```dotenv
SWAPSHIELD_AUDITOR_URL=https://your-provider.example/v1/chat/completions
SWAPSHIELD_AUDITOR_MODEL=your-model-name
SWAPSHIELD_AUDITOR_API_KEY=your-secret-key
SWAPSHIELD_AUDITOR_TIMEOUT_SECONDS=45
```

The endpoint must support the OpenAI-compatible chat-completions response shape.
The API key stays in FastAPI and is never sent to the browser. With no API
configuration—or if the provider fails schema validation—the response uses a
clearly labelled deterministic audit instead. The auditor cannot modify the
model score or decision.

If Qwen emits malformed or truncated JSON during development, set
`SWAPSHIELD_DEBUG_MODEL_OUTPUT=true`, restart the service, and inspect the raw
generation. The default response budget is 768 tokens and the prompt limits the
assessment to four observations. Batch extraction fills only omitted empty-list
fields, retries one genuine schema failure, and safely records recapture evidence
if the retry also fails; one malformed response never terminates the full run.

## 4. Memory-safe defaults

- Keep `SWAPSHIELD_VLM_4BIT=true` for an 8 GB card.
- `SWAPSHIELD_VLM_MAX_PIXELS=262144` caps each Qwen input image near 512 x 512 pixels.
- A CUDA out-of-memory error is cleared and retried once at
  `SWAPSHIELD_VLM_OOM_RETRY_PIXELS=131072`; a second failure becomes a safe
  recapture result instead of terminating batch extraction.
- The service serializes GPU requests to prevent demo-time VRAM spikes.
- Do not run the Next.js build and model warm-up simultaneously if system RAM is limited.
- On Windows, set `$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` before
  starting a long extraction to reduce allocator fragmentation.
- If the first request still runs out of memory, close other GPU applications and retry.

## What a successful smoke test proves

It proves that images reach both models, Qwen output passes the strict schema,
unsupported claims are removed, and the risk layer emits only `approve`,
`recapture`, or `review`. The first RTX 5050 smoke test passed this path. It is
not the held-out accuracy benchmark; that uses an item-disjoint Amazon Berkeley
Objects split and versioned predictions.

## 5. Build and run the first real benchmark

The dataset download itself uses CPU, disk, and internet—not the GPU. From the
repository root, with the environment active:

```powershell
python -m ml.scripts.build_abo_subset --output data\real\abo --items 120 --views-per-item 3
python -m ml.scripts.extract_real_features data\real\abo\manifest.jsonl artifacts\fusion-features.jsonl --split train --split validation --resume
python -m ml.scripts.train_fusion_model artifacts\fusion-features.jsonl artifacts\fusion-model.json --report-output artifacts\fusion-training-report.md
```

The feature-extraction command is the GPU-heavy stage. It caches each completed prediction,
so an interruption does not discard finished DINOv2 and Qwen3-VL work.

On native Windows, Qwen weight loading also needs sufficient committed memory.
Use a system-managed page file or a 32 GB initial / 64 GB maximum page file on
an SSD with enough free capacity. `low_cpu_mem_usage` is enabled in the model
loader, but it cannot compensate for a disabled or undersized Windows page file.

The repository root `README.md` contains the v1.0 train, validation, test, and
smoke-case exclusion commands in their required order. The completed portable
model is versioned at `evaluation/results/fusion-model.json` and loads by default.
