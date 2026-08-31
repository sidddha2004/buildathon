# Real-image benchmark workspace

This directory holds local benchmark manifests, not model weights or a bundled
third-party dataset. Downloaded images and generated predictions stay untracked
by default; only the builder, manifest contract, and documentation belong in
the repository.

## Recommended: build the bounded ABO subset

From the repository root in PowerShell:

```powershell
python -m ml.scripts.build_abo_subset `
  --output data\real\abo `
  --items 120 `
  --views-per-item 3 `
  --workers 6
```

The command downloads approximately 91 MB of official ABO metadata and only
the selected product views. It does **not** download the 40 GB spin archive.
The exact image total and size depend on the selected products; the default
creates 120 identities, 360 images, and 240 balanced genuine/substitution pairs.

Generated locally under `data/real/abo/`:

- `manifest.jsonl` — ready for SwapShield's feature extractor.
- `images/` — selected 360-degree product views.
- `dataset_summary.json` — exact selection, seed, split, and validation report.
- `ATTRIBUTION.md` — CC BY 4.0 source attribution that must stay with the data.
- `.cache/` — reusable listing and spin metadata.

The defaults use chairs, sofas, tables, and lamps because ABO's 360-degree
collection contains enough identities in those categories for item-disjoint,
same-category substitutions. Product identity is assigned to a split before
pairs are created. SKU, serial, and weight fields are deliberately omitted so
the visual benchmark cannot cheat using metadata.

If a previous manifest exists, rerun the exact dataset with `--overwrite`.
Existing metadata and image downloads are reused:

```powershell
python -m ml.scripts.build_abo_subset --output data\real\abo --overwrite
```

Validate independently at any time:

```powershell
python -m ml.scripts.validate_real_dataset data\real\abo\manifest.jsonl --check-files
```

ABO source and license: [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html), CC BY 4.0.

## Alternative: start with your own products

1. Copy `manifest.template.jsonl` to `manifest.jsonl`.
2. Put the referenced images under `data/real/images/`.
3. Give every physical product a stable `*_item_id`.
4. Assign identities to `train`, `validation`, or `test` before creating pairs.
5. For genuine pairs, use the same identity from a different angle.
6. For substitutions, pair two different identities from the same category.

Do not use a different SKU merely because a pair is fraudulent. A dishonest
benchmark would let the identifier rule reveal the label without inspecting the
images. Omit SKU/serial/weight fields when measuring the visual system alone.

Validate before running either GPU model:

```powershell
python -m ml.scripts.validate_real_dataset data\real\manifest.jsonl --check-files
```

Cache train and validation features first. `--resume` prevents a long RTX run
from starting over after an interruption:

```powershell
python -m ml.scripts.extract_real_features `
  data\real\manifest.jsonl `
  artifacts\fusion-features.jsonl `
  --split train `
  --split validation `
  --resume
```

Fit the classifier and select its threshold on validation before opening test:

```powershell
python -m ml.scripts.train_fusion_model `
  artifacts\fusion-features.jsonl `
  artifacts\fusion-model.json
```

The six template cases prove only that the workflow is wired correctly. They
are not a publishable benchmark. The generated ABO subset supplies the first
reproducible visual benchmark; self-captured mouse and packaging cases remain
valuable demo and domain-shift slices.

## Trained-fusion protocol

Do not use the original seeded risk probability as the final benchmark. Cache
train and validation features into a separate file, train the fusion artifact,
and only then extract test. The two `B07124WCZY` cases used during the initial
RTX smoke test must be excluded from final test metrics. The complete command
sequence is maintained in the repository root `README.md`.
