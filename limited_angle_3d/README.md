# Limited-angle CT in 3D: chest runs

R²-Gaussian on a synthetic chest volume (50 cone-beam views, TIGRE projections, no noise). Reports:
[chest pilot](../reports/chest_limited_angle_pilot.pdf),
[reproduction and controls](../reports/chest_reproduction_and_controls.pdf),
[TV controls](../reports/chest_tv_controls.pdf).

Copy this folder into the root of the official R²-Gaussian repository and run the scripts from there. The chest
volume and the generated datasets are not included.

| Folder | Content |
|---|---|
| `limited_angle_120/` | matched 360° and 120° cases, rotated arc, FDK and final-volume comparisons, error by projection angle; |
| `harness/` | case manifest, dataset checks, job runner and evaluation (volume PSNR/SSIM, boundary error) shared by the runs |
| `tv_controls/` | ordinary TV controls with weights 0.025, 0.05 and 0.1 at start angles 0° and 90° |

Small result files of the 120° experiment are kept in `limited_angle_120/results/`.
