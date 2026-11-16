# Experiments

The full record behind the numbers in the [README](README.md): how the data was split, how every run was trained, how the final model was chosen, and every result on validation and test. All numbers come from the per-patient CSVs written by `scripts/evaluate.py`. The ones behind the final results are in [results/](results/).

## Protocol

| Split | Patients | Used for |
|---|---|---|
| Train | 80 (ACDC 001-100 minus validation) | fitting the weights |
| Validation | 20 (4 per diagnosis group) | choosing each run's checkpoint, comparing runs, choosing the final model |
| Test | 50 (ACDC 101-150, 10 per group) | one final evaluation |

- Validation patients (seed 42, [splits/acdc.json](splits/acdc.json)): 001, 004, 008, 009, 023, 024, 025, 028, 041, 042, 054, 059, 063, 067, 068, 077, 081, 087, 098, 100.
- Diagnosis groups: DCM (dilated cardiomyopathy), HCM (hypertrophic cardiomyopathy), MINF (previous myocardial infarction), RV (abnormal right ventricle), NOR (normal).
- Rules fixed in advance: compare runs on validation Dice and HD95, never on loss; choose the final configuration before looking at the test set; run the test set once and report it whatever it shows.
- Validation numbers are slightly optimistic, because the same 20 patients also picked each run's best epoch.

## Training runs

Shared settings: Adam, learning rate 1e-3, batch size 8, 50 epochs, seed 0, Apple Silicon laptop GPU (PyTorch MPS). Augmentation: one paired random affine per slice (rotation ±15°, scale 0.9-1.1, shift ±10 px), then contrast (×0.9-1.1) and brightness (±0.1) and, half of the time, Gaussian noise (sd 0.05) on the image only. The checkpoint with the lowest validation loss is kept.

| Run | Blocks | Attention gates | Loss | Parameters | Best epoch | Best val loss | Wall-clock |
|---|---|---|---|---|---|---|---|
| `row1_plain` | plain | no | cross-entropy | 7,762,564 | 33 | 0.0182 | 96 min |
| `row2_residual` | residual | no | cross-entropy | 8,113,700 | 24 | 0.0177 | 150 min |
| `row3_attention` | residual | yes | cross-entropy | 8,201,464 | 42 | 0.0184 | 153 min |
| `row4_dice` | residual | yes | cross-entropy + Dice | 8,201,464 | 45 | 0.1210 | 213 min |
| `extra_plain_dice` | plain | no | cross-entropy + Dice | 7,762,564 | 46 | 0.1201 | 76 min |

- Validation losses are only comparable between runs with the same loss: adding the Dice term changes the scale (about 0.02 vs about 0.12).
- Wall-clock times are rough. The laptop was sometimes in use during training, which slowed some epochs considerably, so they are not a speed benchmark.
- Residual blocks add a 1x1 convolution shortcut wherever a block changes its number of channels (every block here does): +351,136 parameters. Attention gates add +87,764.
- Each run's exact settings are saved in `checkpoints/<run>/config.json`; per-epoch losses in `history.csv` ([loss curves](figures/loss_curves.png)).

## Ablation (validation)

Means over the 20 validation patients and over ED and ES. EF error = mean absolute error in percentage points vs EF from the experts' masks.

| Change | Dice RV | Dice Myo | Dice LV | HD95 RV (mm) | HD95 Myo | HD95 LV | LV EF error | RV EF error |
|---|---|---|---|---|---|---|---|---|
| Plain U-Net, cross-entropy | 0.879 | 0.895 | 0.947 | 10.7 | 3.7 | 2.7 | 2.1 | 6.0 |
| + residual blocks | 0.878 | 0.894 | 0.942 | 7.1 | 3.4 | 4.1 | 2.7 | 6.5 |
| + attention gates | 0.877 | 0.899 | 0.945 | 10.2 | 3.1 | 4.0 | 2.8 | 6.4 |
| + Dice loss term | 0.887 | 0.902 | 0.949 | 6.9 | 4.2 | 3.3 | 2.2 | 5.9 |
| + largest-component post-processing | 0.888 | 0.902 | 0.950 | 6.8 | 4.2 | 3.0 | 2.0 | 5.9 |

### Extra run: plain U-Net with Dice loss

To check whether the gain in row 4 came from the loss or from the architecture, a plain U-Net (no residual blocks, no attention) was trained with cross-entropy + Dice:

| Run | Dice RV | Dice Myo | Dice LV | HD95 RV (mm) | HD95 Myo | HD95 LV | LV EF error | RV EF error |
|---|---|---|---|---|---|---|---|---|
| Plain U-Net, cross-entropy (row 1) | 0.879 | 0.895 | 0.947 | 10.7 | 3.7 | 2.7 | 2.1 | 6.0 |
| Plain U-Net, cross-entropy + Dice | 0.883 | 0.902 | 0.946 | 9.0 | 3.4 | 3.5 | 2.7 | 6.4 |
| Residual + attention, cross-entropy + Dice (row 4) | 0.887 | 0.902 | 0.949 | 6.9 | 4.2 | 3.3 | 2.2 | 5.9 |

The Dice term raised RV and myocardium Dice on the plain U-Net too, while LV Dice was unchanged and LV EF error got worse (2.1 to 2.7). Row 4 is ahead of the plain + Dice run by at most 0.004 Dice, which is within single-seed noise.

### Findings

1. **Residual blocks and attention gates gave no measurable improvement.** Dice changed by less than 0.01 between rows 1, 2 and 3. A U-Net of this depth has little vanishing-gradient problem for residual connections to solve, and its skip connections already give gradients a short path; the 224 x 224 crop already removes most of the background that attention gates would filter.
2. **The Dice loss term was the only change with a clear Dice gain,** largest on the right ventricle (row 3 to row 4: about +0.01). Cross-entropy averages over all pixels, about 90% of which are background; soft Dice weights each structure equally, so the smaller RV and myocardium count for more.
3. **Trade-off:** myocardium HD95 rose from 3.1 to 4.2 mm with the Dice term.
4. **Single seed.** None of these runs were repeated with other seeds, so differences of a few thousandths should not be over-interpreted.

## Reading the validation results (row 1)

These patterns, seen with the first model, explain most of what follows:

- **Dice depends on object size.** From ED to ES the ventricles shrink and get harder (LV 0.964 to 0.931, RV 0.916 to 0.842), while the myocardium thickens and gets easier (0.886 to 0.904). The same boundary error costs more Dice on a small object.
- **The hardest group at end-systole is HCM.** LV Dice at ES by group: DCM 0.962, MINF 0.946, RV 0.945, NOR 0.939, HCM 0.861. In HCM the genetically thickened muscle squeezes the cavity to a slit at end-systole. MINF patients had the hardest right ventricles at ES (RV Dice 0.778).
- **Correlation is not agreement.** LV EF correlation was 0.991, but the Bland-Altman analysis showed a bias of -1.3 points with 95% limits of -6.7 to +4.1 points: a high correlation can hide a consistent offset.
- **Errors matter most near a clinical cut-off.** patient087's LV EF read 38.3% vs 46.4%, crossing the 40% threshold for reduced EF, while patient009's 11.0% vs 15.0% was a larger relative error that changes nothing clinically.

## Post-processing

`keep_largest_blob` keeps the largest 3D connected component of each structure. On the first model (row 1, validation) it:

- fixed a stray-prediction outlier: patient009's RV HD95 went from 93.7/87.1 mm to 11.4/15.7 mm (ED/ES), bringing the mean RV HD95 from 10.7 to 6.8 mm. The median stayed at 5.2 mm, so the gain is from a few outliers, not from typical cases;
- slightly hurt cases where a structure is legitimately in separate pieces: myocardium HD95 3.7 to 3.9 mm, myocardium Dice down 0.007-0.011 on patients 004 and 042, and patient001's ED RV HD95 up from 2.2 to 10.0 mm.

The myocardium and RV can appear as separate pieces at the base of the heart and across the 10 mm slice gaps, so "largest component only" can delete real tissue. Both raw and post-processed numbers are therefore reported throughout. Applying it to the left ventricle only is a possible refinement that was not tried.

## Choosing the final model

Final configuration: `row4_dice` (residual blocks + attention gates, cross-entropy + Dice, 8,201,464 parameters) with largest-component post-processing, the last row of the ablation table. It had the best validation numbers overall, and was fixed before the test set was evaluated.

It keeps residual blocks and attention gates even though the ablation shows they did not measurably help. They did not hurt either, and the simpler plain U-Net + Dice is within noise of it; with a single seed there was no evidence-based reason to prefer one over the other, so the configuration with the best validation numbers was kept as planned.

Final model on validation, by phase (post-processed): Dice LV 0.968 / 0.932, RV 0.921 / 0.855, myocardium 0.897 / 0.907 (ED / ES).

## Test results

One evaluation of the final model on ACDC patients 101-150, run on 16 November 2026. The raw and post-processed versions came from the same run.

### Final (post-processed)

| | RV | Myocardium | LV |
|---|---|---|---|
| Dice (mean of ED and ES) | 0.909 | 0.903 | 0.943 |
| Dice ED | 0.935 | 0.895 | 0.966 |
| Dice ES | 0.883 | 0.910 | 0.921 |
| HD95 mean (mm) | 5.8 | 3.1 | 3.5 |
| HD95 median (mm) | 3.8 | 1.7 | 1.7 |
| HD95 mean ED (mm) | 5.1 | 2.4 | 2.4 |
| HD95 mean ES (mm) | 6.5 | 3.8 | 4.6 |

| Measurement | Bias | Mean absolute error | Correlation r | 95% limits of agreement |
|---|---|---|---|---|
| LV ejection fraction | +1.5 points | 2.5 points | 0.988 | -5.2 to +8.2 points |
| RV ejection fraction | +1.3 points | 5.2 points | 0.842 | -13.9 to +16.5 points |
| LV end-diastolic volume | -3.3 mL | 4.7 mL | 0.997 | -13.2 to +6.6 mL |
| Myocardial mass | +6.7 g | 8.4 g | 0.989 | -8.7 to +22.0 g |

LV EF mean absolute error by group (10 patients each): DCM 2.0, NOR 2.5, MINF 2.8, HCM 3.4, RV 1.8 points.

### Raw vs post-processed test results

| | Raw | Post-processed |
|---|---|---|
| Dice RV / Myo / LV | 0.909 / 0.902 / 0.945 | 0.909 / 0.903 / 0.943 |
| Dice ED, LV / RV / Myo | 0.965 / 0.935 / 0.895 | 0.966 / 0.935 / 0.895 |
| Dice ES, LV / RV / Myo | 0.925 / 0.883 / 0.909 | 0.921 / 0.883 / 0.910 |
| HD95 mean RV / Myo / LV (mm) | 5.7 / 3.1 / 3.3 | 5.8 / 3.1 / 3.5 |
| LV EF: bias, MAE, r | +1.3, 2.3, 0.989 | +1.5, 2.5, 0.988 |
| LV EF 95% limits of agreement | -5.1 to +7.7 | -5.2 to +8.2 |
| RV EF: MAE, r | 5.2, 0.841 | 5.2, 0.842 |
| LV EDV: bias, MAE, r | -3.3, 4.7, 0.997 | -3.3, 4.7, 0.997 |
| Myocardial mass: bias, MAE, r | +6.8, 8.5, 0.989 | +6.7, 8.4, 0.989 |

On the test set, post-processing was roughly neutral on average (LV Dice 0.945 to 0.943, LV EF error 2.3 to 2.5 points). Per patient, it showed both sides of the trade-off seen on validation:

- **Fixed:** patient135 (MINF), ES RV: a stray piece removed, HD95 31.9 to 12.7 mm, Dice 0.768 to 0.782.
- **Hurt:** patient142 (HCM), ES LV: the model's prediction was in separate pieces and the filter deleted a real one. Dice 0.899 to 0.673, HD95 1.9 to 30.5 mm, LV end-systolic volume 17.5 to 10.3 mL (experts: 19.9 mL), so LV EF 83.1% to 90.1% (experts: 81.6%). This one patient accounts for the entire drop in mean LV Dice.
- **Hurt, smaller:** patient108 (HCM), ES RV: HD95 10.0 to 20.1 mm, Dice 0.832 to 0.811.

A variant that filters only some structures could avoid the patient142 failure, but choosing it now would mean tuning on the test set, so the pre-registered configuration stands. Any such change would need to be chosen on validation and confirmed on new data.

### Individual cases

- **Clinical cut-off.** 18 of 50 patients have an LV EF below 40% by the experts' masks. The model puts 49 of 50 on the correct side of 40%; the exception is patient115 (MINF), 43.5% vs 36.5%.
- **Largest LV EF error:** patient138 (HCM), 78.0% vs 69.0% (9.0 points).
- **Largest RV EF errors:** patient120 (MINF) 77.5% vs 46.5%; patient144 (NOR) 68.4% vs 48.4%; patient135 (MINF) 34.0% vs 50.2%. For patient120 the model found 11.0 mL of RV at end-systole where the experts drew 32.0 mL (ES RV Dice 0.472). EF divides by the volumes, so an error in a small end-systolic RV becomes a large EF error. Without patient120, the RV EF correlation is 0.883 instead of 0.842.
- **HD95 granularity.** For the final model, 24 of the 100 RV HD95 values (50 patients x 2 phases) are exactly 10.0 mm, one slice gap; 10 LV and 9 myocardium values are too. They come from an extra or missing slice at the base or apex, not from in-plane errors.

### Comparison with the 2017 challenge winner

Bernard et al. 2018 report the winning entry (Isensee et al.) on the same 50 test patients (Tables III and IV):

| | This model | Winner | Gap |
|---|---|---|---|
| Dice LV ED | 0.966 | 0.968 | 0.002 |
| Dice LV ES | 0.921 | 0.931 | 0.010 |
| Dice RV ED | 0.935 | 0.946 | 0.011 |
| Dice RV ES | 0.883 | 0.899 | 0.016 |
| Dice Myo ED | 0.895 | 0.902 | 0.007 |
| Dice Myo ES | 0.910 | 0.919 | 0.009 |
| LV EF correlation | 0.988 | 0.991 | 0.003 |

Caveats: the metrics here come from this project's own code, not the organisers' evaluation platform. The challenge reported the full Hausdorff distance, not HD95, so distance metrics are not compared. The gaps are small but consistent: the winner is ahead on every row.

### Validation vs test

Test RV Dice (0.909) was higher than validation (0.888) and test LV Dice slightly lower (0.943 vs 0.950). These are different patients of different difficulty; the validation set has only 20 patients, including some very hard right ventricles. The two sets agree overall, and the higher test RV Dice should not be read as evidence of anything.

## Results files

| File | Contents |
|---|---|
| [results/ablation_table.md](results/ablation_table.md) | the ablation table, as written by `scripts/ablation_table.py` |
| [results/test_summary.md](results/test_summary.md) | final test tables |
| [results/test_metrics_row4_postprocessed.csv](results/test_metrics_row4_postprocessed.csv) | per-patient test metrics, final model |
| [results/test_metrics_row4.csv](results/test_metrics_row4.csv) | per-patient test metrics, final model without post-processing |
| [results/val_metrics_row4_postprocessed.csv](results/val_metrics_row4_postprocessed.csv) | per-patient validation metrics, final model |

Columns: patient, diagnosis group, Dice and HD95 (mm) per structure and phase, LV and RV EDV/ESV (mL) and EF (%), and myocardial mass (g), each as `_pred` (model) and `_true` (experts' masks). They contain derived numbers only, no images.
