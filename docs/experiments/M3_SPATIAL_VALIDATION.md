# M3 Spatial Refinement — Validation Record

## Dataset
Development split only: 13 reviewed clips and 13 labeled deliveries.

## Evaluation configuration
Confidence threshold: 0.80
Primary matching tolerance: +/- 0.5 seconds
Secondary matching tolerance: +/- 1.0 seconds

## Results

| Metric | M3 Baseline | M3 Spatial |
|---|---:|---:|
| Candidates | 18 | 15 |
| True positives | 12 | 12 |
| False positives | 6 | 3 |
| False negatives | 1 | 1 |
| Precision | 66.7% | 80.0% |
| Recall | 92.3% | 92.3% |
| F1 | 0.774 | 0.857 |
| Recall at +/- 1.0s | 100% | 100% |
| Review burden (predictions/min) | 1.38 | 1.15 |
| Mean absolute timestamp error | 0.111s | 0.111s |

## Interpretation
The spatial refinement reduced false positives from 6 to 3
while preserving the 12 true positives detected by the baseline.

Precision improved by 13.3 percentage points and F1 improved
from 0.774 to 0.857.

## Limitations
These results are preliminary development-set results.
The 0.80 confidence threshold was selected using development data.
Therefore, these results are not independent validation.

The locked test split contains no positive delivery labels.
It cannot establish delivery recall or validate generalization.

No claims of production readiness or independent test performance
should be made from this experiment.
