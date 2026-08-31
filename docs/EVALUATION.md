# Evaluation contract

SwapShield's positive class is **return substitution**: the returned item is
not the item or variant recorded at dispatch. An empty parcel is treated as the
null-item endpoint of the same class. Damage-only and customer-intent claims
are outside scope.

## Split policy

1. Product identities are assigned to train, validation, and test before pair generation.
2. Multiple views of one identity stay in the same split.
3. Hard negatives are different identities from the same product category.
4. Test labels are not used for feature selection, threshold selection, or calibration.
5. Seeded synthetic results are reported separately from real-image results.

## Required reporting

- Precision, recall, F1, PR-AUC and confusion matrix.
- Calibration error and a reliability diagram.
- Bootstrap 95% confidence intervals.
- Category, blur, occlusion, viewpoint and serial-number slices.
- False-positive review/friction cost and false-negative merchandise loss.
- DINOv2-only, VLM-only, objective-signals-only, and fused-model ablations.
- p50 and p95 latency on the declared hardware.

The operating threshold is selected on validation data using declared merchant
cost assumptions. Accuracy is not a headline metric.

## Implemented real-image protocol

The manifest validator rejects duplicate cases, label/identity contradictions,
unsafe paths, missing files, unsupported labels, and any product identity that
appears in more than one split. Source and licence are mandatory for every pair.

The ABO subset builder assigns identities to splits before it creates pairs,
then makes one different-angle genuine pair and one same-category substitution
pair per selected identity. The default seed is 5050 and the exact source views
are recorded in `dataset_summary.json`. No SKU, serial, or weight shortcut is
included in this vision-only benchmark.

The GPU extractor writes one prediction at a time and supports `--resume`, so a
long local run remains recoverable. The evaluator selects its threshold on
validation only, then reports the untouched test split with:

- Precision, recall, F1, average precision, and confusion matrix.
- Expected calibration error and deterministic bootstrap 95% intervals.
- False-positive review cost and missed-substitution merchandise loss.
- Recapture rate and genuine-return recapture friction.
- p50/p95 end-to-end latency and category/corruption slices.

Synthetic POC metrics and real-image reports are stored and presented
separately. The six-case manifest template validates the workflow only and must
never be presented as the final benchmark. The generated ABO test report also
needs to be labelled as an ABO visual benchmark, not evidence of production
performance on merchant returns.

## Fusion training protocol

The first trained baseline uses five bounded signals: DINOv2 dissimilarity,
Qwen3-VL mismatch, identifier mismatch, clipped weight deviation, and image
quality deficit. For the ABO visual-only benchmark, identifier and weight
signals remain zero by design.

The classifier is a class-balanced logistic regression. Standardization and
base-model fitting use only train. Platt calibration is fitted on deterministic
five-fold out-of-fold train scores. The financial operating threshold is then
selected on validation. The final scaler, coefficients, calibration parameters,
cost assumptions, quality-abstention threshold, and decision threshold are
stored in a portable JSON artifact; no unsafe pickle is required.

Two test cases (`ABO-TES-B07124WCZY-G` and `ABO-TES-B07124WCZY-S`) were executed
as hardware smoke tests before model fitting. They are explicitly excluded from
the final test report. No model parameter or threshold may be changed after the
remaining test predictions are opened.

## Frozen v1 result

The final 30-pair test produced precision 1.000, recall 0.867, F1 0.929,
average precision 0.996, and calibration error 0.046 at the validation-selected
0.92 threshold. The confusion matrix is TP=13, FP=0, TN=15, FN=2. Both misses
were lamps, giving the eight-case lamp slice 0.500 recall. Recapture was requested
for 40% of all test pairs and 33.3% of genuine pairs. End-to-end RTX 5050 latency
was 11.5 seconds p50 and 65.4 seconds p95.

The report is frozen in `evaluation/results/real-report.json`; known limitations
and intended use are documented in `docs/MODEL_CARD.md`.
