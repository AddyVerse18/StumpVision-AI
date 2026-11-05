# StumpVision AI Delivery Finder — Evaluation Report

**Split Evaluated:** `DEV`  
**Method Version:** `m3_motion_baseline_v1`  
**Report Generated:** `2026-10-30T16:10:45.113114+00:00`  
**Reviewed Clips Included:** 13  
**Unreviewed Clips Excluded:** 0  

---

## 1. Primary Benchmark Metrics

| Metric | Tolerance ±0.5s (Primary) | Tolerance ±1.0s (Secondary) |
| :--- | :--- | :--- |
| **Recall** | **12 of 13 (92.3%)** | 13 of 13 (100.0%) |
| **Precision** | **12 of 18 (66.7%)** | 13 of 18 (72.2%) |
| **F1 Score** | **0.774** | 0.839 |
| True Positives (Hits) | 12 | 13 |
| False Negatives (Misses) | 1 | 0 |
| False Positives (False Alarms) | 6 | 5 |

## 2. Temporal Precision (Matched Hits)

- **Mean Absolute Error:** 0.111 s (6.7 frames)
- **Max Absolute Error:** 0.400 s (24.0 frames)

## 3. Review Burden & Clip-Level Performance

- **Predictions per Clip:** 1.38 (max in a single clip: 6)
- **Predictions per Minute of Footage:** 1.38 preds/min
- **Clips with ≥1 Hit:** 8 of 8 (100.0%)
- **Empty Clips with 0 Predictions:** 4 of 5 (80.0%)

## 4. Partial Deliveries (Excluded from Primary Score)

0 partials present in reviewed clips

## 5. Performance Breakdowns

### By Contiguous Block

| Block | Recall (Count) | Precision (Count) | F1 | Total Truths | Total Preds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `block_1` | 2 of 3 (66.7%) | 2 of 7 (28.6%) | 0.400 | 3 | 7 |
| `block_2` | 3 of 3 (100.0%) | 3 of 3 (100.0%) | 1.000 | 3 | 3 |
| `block_3` | 7 of 7 (100.0%) | 7 of 8 (87.5%) | 0.933 | 7 | 8 |

### By Camera View

| View | Recall (Count) | Precision (Count) | F1 | Total Truths | Total Preds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `batter_end` | 0 of 0 (N/A) | 0 of 1 (0.0%) | 0.000 | 0 | 1 |
| `bowler_end` | 12 of 13 (92.3%) | 12 of 17 (70.6%) | 0.800 | 13 | 17 |

### By Distractor Tags

| Distractor Tag | Clips with Tag | Misses (FN) | False Alarms (FP) | Hits (TP) |
| :--- | :--- | :--- | :--- | :--- |
| `batter_walkpast` | 0 | 0 | 0 | 0 |
| `fielder` | 0 | 0 | 0 | 0 |
| `camera_moved` | 0 | 0 | 0 | 0 |
| `other` | 0 | 0 | 0 | 0 |

## 6. Uncertainty & Confidence Intervals (Clip-Level Bootstrap)

> ⚠️ **PRELIMINARY DATA WARNING:**  
> WARNING: Dataset contains only 13 reviewed clips and 13 deliveries (<20 clips or <30 deliveries threshold). Bootstrap confidence intervals are expected to be wide and results must be treated as PRELIMINARY ONLY.

- **Recall 95% CI:** [80.0%, 100.0%]
- **Precision 95% CI:** [45.4%, 100.0%]

## 7. Score-Threshold Operating Points

**Optimal Operating Point on Dev:** Threshold `0.1` → Recall 12 of 13 (92.3%), Precision 12 of 18 (66.7%), F1 `0.774`

| Threshold | Recall | Precision | F1 | Hits (TP) | False Alarms (FP) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.10 | 12 of 13 (92.3%) | 12 of 18 (66.7%) | 0.774 | 12 | 6 |
| 0.20 | 12 of 13 (92.3%) | 12 of 18 (66.7%) | 0.774 | 12 | 6 |
| 0.30 | 12 of 13 (92.3%) | 12 of 18 (66.7%) | 0.774 | 12 | 6 |
| 0.40 | 12 of 13 (92.3%) | 12 of 18 (66.7%) | 0.774 | 12 | 6 |
| 0.50 | 12 of 13 (92.3%) | 12 of 18 (66.7%) | 0.774 | 12 | 6 |
| 0.60 | 12 of 13 (92.3%) | 12 of 18 (66.7%) | 0.774 | 12 | 6 |
| 0.70 | 12 of 13 (92.3%) | 12 of 18 (66.7%) | 0.774 | 12 | 6 |
| 0.80 | 12 of 13 (92.3%) | 12 of 18 (66.7%) | 0.774 | 12 | 6 |
| 0.90 | 0 of 13 (0.0%) | 0 of 0 (N/A) | 0.000 | 0 | 0 |
| 0.95 | 0 of 13 (0.0%) | 0 of 0 (N/A) | 0.000 | 0 | 0 |
| 0.99 | 0 of 13 (0.0%) | 0 of 0 (N/A) | 0.000 | 0 | 0 |

## 8. Reproducibility & File Hashes

| File | SHA-256 Digest |
| :--- | :--- |
| `manifest_sha256` | `7c1b2a5bc1538067fcc4fd89ecf795993258c38ca7112a68ff3f06944707cdad` |
| `labels_sha256` | `26d9e3a2a9df0ea8ef77527389554e0c309248ea7386bedb4fb730dc2cd476d9` |
| `clip_status_sha256` | `4004d0bb17a17e5767b9bd87d1c7fb5260ee45598d73ecf06985460f33b28670` |
| `predictions_sha256` | `3939ef2860194c84c349deb1194bd637b70b55a2b95fda22259e3e825d97f220` |

**Packages:** Python 3.13.7, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0
