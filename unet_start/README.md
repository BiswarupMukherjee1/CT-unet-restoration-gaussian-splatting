# U-Net start: starting volumes for R²-Gaussian

Replaces the FBP start of R²-Gaussian by a SIRT, SIRT+TV or U-Net volume, converts it to initial Gaussians with the
standard rule or with a fitted conversion, and scores every stage (starting volume, initial Gaussians, result after
training). Results: [`reports/unet_starting_points_limited_angle_ct.pdf`](../reports/unet_starting_points_limited_angle_ct.pdf).

Run from the root of the official R²-Gaussian repository after applying
[`patches/r2_gaussian_parallel_beam_scaling.patch`](../patches/r2_gaussian_parallel_beam_scaling.patch); the scripts use
the helpers in [`../tv_weight`](../tv_weight). Set `PROJECT_ROOT` as described there.

| File | Role |
|---|---|
| `prepare_cases.py`, `init_lib.py` | test cases, physics-only starts (FDK, SIRT, SIRT+TV), shared helpers |
| `unet_init.py` | residual 2D U-Net: training (three seeds per arc, other poses only) and starting volumes |
| `make_inits.py` | initial point clouds from a starting volume (standard rule, ensemble-guided placement) |
| `prefit_gaussians.py` | fitted conversion: optimise the Gaussians until they reproduce the starting volume |
| `score_starts.py`, `score_gaussian_starts.py` | scores of the starting volume and of the initial Gaussians |
| `analyze_init.py` | tables and figures of the study |
