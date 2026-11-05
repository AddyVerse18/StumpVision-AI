# StumpVision AI Delivery Finder — Evaluation Report

**Split Evaluated:** `TEST`  
**Method Version:** `unknown`  
**Report Generated:** `2026-11-01T16:55:16.733652+00:00`  
**Reviewed Clips Included:** 5  
**Unreviewed Clips Excluded:** 0  

---

## 1. Primary Benchmark Metrics

| Metric | Tolerance ±0.5s (Primary) | Tolerance ±1.0s (Secondary) |
| :--- | :--- | :--- |
| **Recall** | **0 of 0 (N/A)** | 0 of 0 (N/A) |
| **Precision** | **0 of 0 (N/A)** | 0 of 0 (N/A) |
| **F1 Score** | **0.000** | 0.000 |
| True Positives (Hits) | 0 | 0 |
| False Negatives (Misses) | 0 | 0 |
| False Positives (False Alarms) | 0 | 0 |

## 2. Temporal Precision (Matched Hits)

- *No true positive matches to compute timestamp errors.*

## 3. Review Burden & Clip-Level Performance

- **Predictions per Clip:** 0.00 (max in a single clip: 0)
- **Predictions per Minute of Footage:** 0.00 preds/min
- **Clips with ≥1 Hit:** 0 of 0 (N/A)
- **Empty Clips with 0 Predictions:** 5 of 5 (100.0%)

## 4. Partial Deliveries (Excluded from Primary Score)

0 partials present in reviewed clips

## 5. Performance Breakdowns

### By Contiguous Block

| Block | Recall (Count) | Precision (Count) | F1 | Total Truths | Total Preds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `block_test` | 0 of 0 (N/A) | 0 of 0 (N/A) | 0.000 | 0 | 0 |

### By Camera View

| View | Recall (Count) | Precision (Count) | F1 | Total Truths | Total Preds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `batter_end` | 0 of 0 (N/A) | 0 of 0 (N/A) | 0.000 | 0 | 0 |

### By Distractor Tags

| Distractor Tag | Clips with Tag | Misses (FN) | False Alarms (FP) | Hits (TP) |
| :--- | :--- | :--- | :--- | :--- |
| `batter_walkpast` | 0 | 0 | 0 | 0 |
| `fielder` | 0 | 0 | 0 | 0 |
| `camera_moved` | 0 | 0 | 0 | 0 |
| `other` | 0 | 0 | 0 | 0 |

## 6. Uncertainty & Confidence Intervals (Clip-Level Bootstrap)

> ⚠️ **PRELIMINARY DATA WARNING:**  
> WARNING: Dataset contains only 5 reviewed clips and 0 deliveries (<20 clips or <30 deliveries threshold). Bootstrap confidence intervals are expected to be wide and results must be treated as PRELIMINARY ONLY.

- **Recall 95% CI:** [0.0%, 0.0%]
- **Precision 95% CI:** [0.0%, 0.0%]

## 7. Score-Threshold Operating Points

> 🔒 **SPLIT SAFEGUARD:** Threshold sweep and operating point selection are restricted to the DEV split. Test split must only be evaluated at a fixed, predetermined operating point.

## 8. Reproducibility & File Hashes

| File | SHA-256 Digest |
| :--- | :--- |
| `manifest_sha256` | `7c1b2a5bc1538067fcc4fd89ecf795993258c38ca7112a68ff3f06944707cdad` |
| `labels_sha256` | `26d9e3a2a9df0ea8ef77527389554e0c309248ea7386bedb4fb730dc2cd476d9` |
| `clip_status_sha256` | `4004d0bb17a17e5767b9bd87d1c7fb5260ee45598d73ecf06985460f33b28670` |
| `predictions_sha256` | `778d00783ea4deb25ede738f0c2fed2a396a1d203e787bb932229fbe7e5a4eda` |

**Packages:** Python 3.13.7, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0
