# Grounded investigator contract

The vision-language model may extract observations. It does not decide fraud,
reject refunds, submit disputes, or move money.

## Allowed output

- Evidence sufficiency.
- Visual observations tied to an image identifier and region.
- Missing-evidence checklist.
- Reviewer narrative assembled downstream from verified facts.

The LLM itself cannot emit a recommendation. The deterministic fusion and
policy layer owns the bounded `approve`, `recapture`, or `review` recommendation.

## Enforcement

1. Uploaded text and OCR are treated as untrusted data, never instructions.
2. Model output must validate against a strict JSON schema.
3. Each generated claim must bind to an immutable evidence field.
4. Unsupported claims are removed before display.
5. Any extra field—including `action`, `fraud`, or `customer_intent`—invalidates the entire output.
6. Schema or grounding failure falls back to recapture or a deterministic template.
7. Only a human can confirm an adverse decision.

## Separate evaluation

Track schema-valid rate, citation correctness, unsupported-claim rate, policy
compliance, repeated-run agreement, and prompt-injection resistance.

## Independent API auditor

After the calibrated recommendation is fixed, a separate optional LLM API may
check whether the structured evidence is internally consistent with it. This
auditor receives only a compact evidence payload; it never receives raw policy
authority, cannot change the risk score, and cannot emit an operational action.

Its response must contain exactly six fields: recommendation support, evidence
consistency, contradictions, missing evidence, one neutral reviewer summary,
and citations copied from the available evidence IDs. Extra fields, unavailable
citations, accusatory language, provider errors, timeouts, or malformed JSON
cause the entire response to be discarded and replaced by a deterministic
fallback. The UI displays whether the API or fallback produced the audit.

The auditor is post-model assistance and is not included in the locked benchmark
precision, recall, calibration, or latency claims.
