# SwapShield real-image evaluation

Operating threshold: **0.92**, selected on validation only.

| Metric | Held-out test result |
|---|---:|
| Cases | 30 |
| Precision | 1.000 |
| Recall | 0.867 |
| F1 | 0.929 |
| Average precision (PR) | 0.996 |
| Calibration error | 0.046 |
| Recapture rate | 0.400 |
| Genuine recapture rate | 0.333 |
| p50 latency | 11,486 ms |
| p95 latency | 65,405 ms |

Confusion matrix: TP=13, FP=0, TN=15, FN=2.

At the declared ₹80 false-positive and ₹6,200 false-negative costs, the locked
test produced ₹0 false-positive cost and ₹12,400 missed-substitution loss.

Both false negatives were in the lamp category. Lamp recall was 0.500; chair,
sofa, and table recall was 1.000 on this small test set.

This report was generated from the locked item-disjoint ABO test split. Two
previously viewed hardware smoke cases were disclosed and excluded, leaving 30
test pairs. Synthetic results are separate. No threshold or model parameter was
changed after test predictions were opened.
