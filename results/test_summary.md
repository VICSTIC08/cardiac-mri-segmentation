# Test results: final model

Final model: `row4_dice` (U-Net with residual blocks and attention gates, cross-entropy + Dice loss) with largest-component post-processing, chosen on validation before the test set was used. Evaluated once on ACDC patients 101-150 (50 patients, 10 per diagnosis group), in each scan's original grid. Per-patient values: [test_metrics_row4_postprocessed.csv](test_metrics_row4_postprocessed.csv) (final) and [test_metrics_row4.csv](test_metrics_row4.csv) (no post-processing).

## Segmentation

| | RV | Myocardium | LV |
|---|---|---|---|
| Dice (mean of ED and ES) | 0.909 | 0.903 | 0.943 |
| Dice ED | 0.935 | 0.895 | 0.966 |
| Dice ES | 0.883 | 0.910 | 0.921 |
| HD95 mean (mm) | 5.8 | 3.1 | 3.5 |
| HD95 median (mm) | 3.8 | 1.7 | 1.7 |
| HD95 mean ED (mm) | 5.1 | 2.4 | 2.4 |
| HD95 mean ES (mm) | 6.5 | 3.8 | 4.6 |

## Clinical measurements

Model vs the same measurement from the experts' masks; positive bias = model reads higher.

| | Bias | Mean absolute error | Correlation r | 95% limits of agreement |
|---|---|---|---|---|
| LV ejection fraction | +1.5 points | 2.5 points | 0.988 | -5.2 to +8.2 points |
| RV ejection fraction | +1.3 points | 5.2 points | 0.842 | -13.9 to +16.5 points |
| LV end-diastolic volume | -3.3 mL | 4.7 mL | 0.997 | -13.2 to +6.6 mL |
| Myocardial mass | +6.7 g | 8.4 g | 0.989 | -8.7 to +22.0 g |

LV EF mean absolute error by group: DCM 2.0, NOR 2.5, MINF 2.8, HCM 3.4, RV 1.8 points.

## Without post-processing

Dice RV 0.909 / Myo 0.902 / LV 0.945; HD95 mean 5.7 / 3.1 / 3.3 mm. LV EF bias +1.3 points, MAE 2.3, r 0.989 (limits -5.1 to +7.7). RV EF MAE 5.2, r 0.841. LV EDV bias -3.3 mL (MAE 4.7, r 0.997). Myocardial mass bias +6.8 g (MAE 8.5, r 0.989).

See [EXPERIMENTS.md](../EXPERIMENTS.md#test-results) for individual cases and the comparison with the 2017 challenge winner.
