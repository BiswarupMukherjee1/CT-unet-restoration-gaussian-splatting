# TV weight: sweeps and selection rules

Limited-angle reconstruction of the worm sinograms with R²-Gaussian, and a classical 2D TV solver (parallel and fan
beam) used for dense sweeps. The scripts replay the chest pilot on the worm (full scan against limited arc, two arc
orientations, ordinary TV), sweep the TV weight over four angular ranges and noise levels, and compare ways of
choosing the weight without ground truth. Results: [`reports/tv_weight_vs_angular_range.pdf`](../reports/tv_weight_vs_angular_range.pdf).

Run from the root of the official R²-Gaussian repository after applying
[`patches/r2_gaussian_parallel_beam_scaling.patch`](../patches/r2_gaussian_parallel_beam_scaling.patch). Set
`PROJECT_ROOT` to the folder that holds the data and the outputs, and `WORM_ZIP` to the sinogram archive.

| File | Role |
|---|---|
| `common.py`, `prepare_case.py` | worm sinograms, slabs, limited-arc and noisy copies, R²-Gaussian datasets |
| `build_phantom.py` | digital phantom with exact voxel truth (Track B) |
| `train_worm.py`, `run_queue.py` | training wrapper (adds seeds and run records) and a restartable job queue |
| `evaluate_run.py` | metrics of one run (see below) |
| `tv2d.py`, `tv_sweep2d.py`, `fan2d.py`, `fan_sweep2d.py` | classical TV solver and dense sweeps, parallel and fan beam |
| `analyze_p1.py`, `analyze_sweep.py`, `analyze_fan.py` | tables, best weight against angular range, selection rules, transfer to fan beam |
| `make_report_figures.py`, `make_report_tables.py` | figures and LaTeX tables of the report |
| `test_projector_parity.py` | check of the parallel-beam rasteriser against TIGRE |

## Data

* Source: `sino_selected.zip` (not included in this repository), 144 TIFF stacks
  (z = 510, angle = 100 over [0°, 180°), detector = 310), parallel beam, 1 µm sampling,
  values mg/cm³·µm. Members are read directly from the zip (unpacked size 18.2 GB).
* Preprocessing (`common.py`): drop trailing empty slices, divide by 1000 (g/cm³·µm),
  64-slice slabs at fixed relative positions (`SLAB_POS`). The development slab is
  worm_004 z ∈ [223, 287), which contains slice 255.
* Limited-angle scans are created by keeping the views of an arc [start, start + span)
  on the native 1.8° grid. Withheld views are kept clean for evaluation only.
* Noise: additive Gaussian, σ = f × 6.2101 g/cm³·µm (RMS of the clean worm_004 z = 255
  sinogram), drawn on the full 100-view grid so that arcs share the same realisation.

## Two kinds of reference

* **Track A:** the supplied sinograms are the measurements. Projection-domain
  scores on withheld views are exact. Voxel scores use the Track B phantom as an
  approximate reference.
* **Track B:** a digital phantom with exact voxel truth (`build_phantom.py`). It is
  made from a 2,000-iteration SIRT reconstruction of all 100 clean views, segmented
  at 0.5 µm into five density levels (1-D k-means), with each level's density fitted
  by least squares to the supplied sinogram. Measurements are simulated at
  0.5 µm (0.5 µm detector, binned 2×2) and the truth is the 2×2×2 average at 1 µm.
  For worm_004/mid the phantom reproduces the supplied sinogram to 2.99%
  relative L2 (the SIRT reconstruction itself: 1.99%).

## Changes to the official R²-Gaussian code

1. **Parallel-beam projection scaling** (`graphics_utils.getProjectionMatrix`,
   `cameras.Camera`, `render_query.render`). The original parallel path used an
   identity projection and tan(fov/2) = 1, which assumes the scaled detector spans
   exactly [−1, 1] in both directions. For a 310 × 64 detector, views were squeezed
   ~4.8× vertically. With the initial Gaussians of the development case, renders
   disagreed with TIGRE by 88% (mass ratio 0.22). After the fix: 0.94% median, 1.95%
   maximum, mass ratio 1.006 (`test_projector_parity.py`). The fix reduces to the
   original for a detector spanning [−1, 1]; cone beam is untouched.
2. **TIGRE detector offset of 0.01 px** (`common.DET_EPS`, also written to each case's
   `offDetector`). With zero offset on even grids, TIGRE's parallel projector returns
   all-zero views at exactly 0° and 90°.
3. TIGRE 2.3's own iterative solvers fail with the installed NumPy (`np.int`), and its
   back-projector is not the exact adjoint of its forward projector. Reference
   reconstructions therefore use the short SIRT in `common.py`.
4. Training and initialisation are the unchanged native scripts, called with the
   chest-pilot arguments (`--n_points 50000 --density_thresh 0.05 --density_rescale 0.15`;
   30,000 iterations; λ_TV given per run). `train_worm.py` only adds `--seed` and a run
   record. Intermediate checkpoints are not written (disk budget).

## Metrics (`evaluate_run.py`)

All in g/cm³ and µm, restricted to the worm (the worm fills ~2.5% of the box).

| Question | Metric |
|---|---|
| Contours correct? | Dice of worm/air and of each density class; mean / 95% surface distance; boundary F-score at 1 and 2 µm (worm outline and compartment borders) |
| Homogeneous inside? | std and CV inside each class (pure voxels, 1 µm erosion) |
| Quantitative? | mean bias per class (g/cm³, %); MAE / RMSE / PSNR inside the worm (peak = max true density) |
| Streaks? | RMS density in air ≥ 3 µm from the worm |
| Data consistency | relative L2 of reprojected volume vs clean data on measured and withheld views |
| Continuity with the chest pilot | native whole-box PSNR (peak 1); high-gradient boundary MAE; edge error by edge orientation |

## Running

```bash
# phantom for one pose and slab (once)
python build_phantom.py --worm 4 --slab mid --out "$PROJECT_ROOT/runs_worm/phantoms"
# restartable job queue, skips finished jobs, stops at a disk limit
python run_queue.py --jobs jobs.jsonl --root "$PROJECT_ROOT/runs_worm" --disk-stop-gb 15.5
```
