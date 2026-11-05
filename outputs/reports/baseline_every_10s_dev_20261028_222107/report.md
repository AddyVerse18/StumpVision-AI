# StumpVision AI Delivery Finder — Evaluation Report

**Split Evaluated:** `DEV`  
**Method Version:** `baseline_every_10s`  
**Report Generated:** `2026-10-28T16:51:07.353470+00:00`  
**Reviewed Clips Included:** 13  
**Unreviewed Clips Excluded:** 0  

---

## 1. Primary Benchmark Metrics

| Metric | Tolerance ±0.5s (Primary) | Tolerance ±1.0s (Secondary) |
| :--- | :--- | :--- |
| **Recall** | **1 of 13 (7.7%)** | 1 of 13 (7.7%) |
| **Precision** | **1 of 91 (1.1%)** | 1 of 91 (1.1%) |
| **F1 Score** | **0.019** | 0.019 |
| True Positives (Hits) | 1 | 1 |
| False Negatives (Misses) | 12 | 12 |
| False Positives (False Alarms) | 90 | 90 |

## 2. Temporal Precision (Matched Hits)

- **Mean Absolute Error:** 0.117 s (7.0 frames)
- **Max Absolute Error:** 0.117 s (7.0 frames)

## 3. Review Burden & Clip-Level Performance

- **Predictions per Clip:** 7.00 (max in a single clip: 7)
- **Predictions per Minute of Footage:** 7.00 preds/min
- **Clips with ≥1 Hit:** 1 of 8 (12.5%)
- **Empty Clips with 0 Predictions:** 0 of 5 (0.0%)

## 4. Partial Deliveries (Excluded from Primary Score)

0 partials present in reviewed clips

## 5. Performance Breakdowns

### By Contiguous Block

| Block | Recall (Count) | Precision (Count) | F1 | Total Truths | Total Preds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `block_1` | 0 of 3 (0.0%) | 0 of 28 (0.0%) | 0.000 | 3 | 28 |
| `block_2` | 0 of 3 (0.0%) | 0 of 28 (0.0%) | 0.000 | 3 | 28 |
| `block_3` | 1 of 7 (14.3%) | 1 of 35 (2.9%) | 0.048 | 7 | 35 |

### By Camera View

| View | Recall (Count) | Precision (Count) | F1 | Total Truths | Total Preds |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `batter_end` | 0 of 0 (N/A) | 0 of 35 (0.0%) | 0.000 | 0 | 35 |
| `bowler_end` | 1 of 13 (7.7%) | 1 of 56 (1.8%) | 0.029 | 13 | 56 |

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

- **Recall 95% CI:** [0.0%, 30.0%]
- **Precision 95% CI:** [0.0%, 3.3%]

## 7. Score-Threshold Operating Points

**Optimal Operating Point on Dev:** Threshold `0.1` → Recall 1 of 13 (7.7%), Precision 1 of 91 (1.1%), F1 `0.019`

| Threshold | Recall | Precision | F1 | Hits (TP) | False Alarms (FP) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.10 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.20 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.30 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.40 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.50 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.60 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.70 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.80 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.90 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.95 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |
| 0.99 | 1 of 13 (7.7%) | 1 of 91 (1.1%) | 0.019 | 1 | 90 |

## 8. Reproducibility & File Hashes

| File | SHA-256 Digest |
| :--- | :--- |
| `manifest_sha256` | `7c1b2a5bc1538067fcc4fd89ecf795993258c38ca7112a68ff3f06944707cdad` |
| `labels_sha256` | `26d9e3a2a9df0ea8ef77527389554e0c309248ea7386bedb4fb730dc2cd476d9` |
| `clip_status_sha256` | `4004d0bb17a17e5767b9bd87d1c7fb5260ee45598d73ecf06985460f33b28670` |
| `predictions_sha256` | `6c9bf4d7934a6a086f2ee9e2bd13b82c9e073655096fbe0de41574904164b656` |

**Packages:** Python 3.13.7, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0
