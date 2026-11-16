| Change | Dice RV | Dice Myo | Dice LV | HD95 RV (mm) | HD95 Myo | HD95 LV | LV EF error (%) | RV EF error (%) |
|---|---|---|---|---|---|---|---|---|
| Plain U-Net, cross-entropy | 0.879 | 0.895 | 0.947 | 10.7 | 3.7 | 2.7 | 2.1 | 6.0 |
| + residual blocks | 0.878 | 0.894 | 0.942 | 7.1 | 3.4 | 4.1 | 2.7 | 6.5 |
| + attention gates | 0.877 | 0.899 | 0.945 | 10.2 | 3.1 | 4.0 | 2.8 | 6.4 |
| + Dice loss term | 0.887 | 0.902 | 0.949 | 6.9 | 4.2 | 3.3 | 2.2 | 5.9 |
| + largest-blob post-processing | 0.888 | 0.902 | 0.950 | 6.8 | 4.2 | 3.0 | 2.0 | 5.9 |
