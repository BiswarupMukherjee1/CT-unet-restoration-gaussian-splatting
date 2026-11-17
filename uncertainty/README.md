# Uncertainty of the Gaussian reconstruction

Can a Gaussian reconstruction report where it is unsure when views are missing or noisy? The scripts fit Gaussians to
worm sinograms and to known 3D phantoms, build posteriors over the Gaussian parameters, and check calibration as the
coverage of 95 % intervals on held-back views. Results:
[`reports/gaussian_uncertainty_limited_angle_ct.pdf`](../reports/gaussian_uncertainty_limited_angle_ct.pdf).

Each experiment has a script, a `*_protocol_v1.json` with its fixed settings, a `plot_*.py` for the figures and a
`test_*.py` with unit tests that import the scripts next to them.

| Group | Scripts |
|---|---|
| Data and projector | `prepare.py` (clean and noisy sinogram with a measured-view mask), `native_parallel.py` (parallel-beam adapter and check) |
| Baselines | `check_fbp.py`, `compare_noise.py`, `compare_sart.py` (FBP and SART under noise and missing views) |
| Gaussian fits | `fit_gaussian_slab.py`, `limited_gaussian.py`, `measured_initializer.py`, `measured_init_study.py`, `measured_init_fallback.py` |
| Density posterior | `bayes_linear.py`, `conditional_density.py` (positions and shapes fixed) |
| Joint posterior on known phantoms | `diagnose_3d.py`, `stress_3d.py`, `phantom_variability.py` (density plus centre uncertainty, coverage and interval width) |
| Shape uncertainty | `shape_gaussian.py`, `cholesky_gaussian.py`, `cholesky_sensitivity.py`, `rotation_sensitivity.py`, `run_shape_controls.py`, `worm_shape_pilot.py` |
| Worm fits | `worm_fit_controls.py`, `audit_multistart.py` (starts, priors, number of Gaussians, stability across starts) |
