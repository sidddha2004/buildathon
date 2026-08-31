# Defense-only threat model

## Protected asset

Merchant margin lost when a returned parcel contains a substituted, wrong, or
missing item and the refund is issued as though the original product arrived.

## In scope

- Wrong SKU or variant.
- Different serial-numbered unit.
- Visually similar cheaper substitute.
- Missing product or materially incomplete return.
- Unreliable evidence that requires recapture.

## Out of scope

- Inferring customer intent.
- Identifying how to bypass return controls.
- Customer profiling beyond the submitted order evidence.
- Automatic refund rejection, account blocking, or dispute submission.
- Damage assessment and counterfeit certification in the first release.

## Safety boundary

SwapShield produces evidence and a bounded recommendation. Uncertainty routes
to recapture or human review. It never labels a person as fraudulent.

The optional external LLM auditor is advisory-only and executes after the
calibrated recommendation is fixed. Uploaded/OCR strings remain untrusted data;
the auditor sees only a compact evidence payload, must cite known evidence IDs,
and cannot add actions or alter the score. Provider errors, invalid JSON,
unsupported citations, or accusatory language activate a deterministic fallback.
The API endpoint is operator-configured, and its secret remains server-side.
