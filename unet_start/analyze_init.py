"""Tables and figures of the initialisation study.

Reads runs_worm/results/*_init*.json (one per R2 run) and the starting-point diagnostics in
each case directory. Writes CSV tables and PNG figures to --out.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import init_lib as L

ROOT = L.c.PROJECT_ROOT / "runs_worm"
ARMS = ["fdk", "sirt", "sirttv", "sirttv_fit", "unet", "unetuq", "unet_fit"]
LABEL = {"fdk": "FBP start (current)", "sirt": "SIRT start", "sirttv": "SIRT+TV start",
         "sirttv_fit": "SIRT+TV start, fitted", "unet": "U-Net start", "unetuq": "U-Net + uncertainty start",
         "unet_fit": "U-Net start, fitted"}
COL = {"fdk": "#6b6b6b", "sirt": "#2a78b5", "sirttv": "#1b9e77", "sirttv_fit": "#0b5e43", "unet": "#d95f02",
       "unetuq": "#b8860b", "unet_fit": "#8c2d04"}
METRICS = [("worm_psnr_db", "Worm PSNR (dB)", "voxel"), ("interior_cv_pct_wmean", "Interior CV (%)", "voxel"),
           ("abs_rel_bias_pct_wmean", "Density bias (%)", "voxel"), ("worm_bf1_1um", "Outline F1 (1 µm)", "voxel"),
           ("worm_mean_surface_dist_um", "Surface dist. (µm)", "voxel"), ("dice_class_mean", "Class Dice", "voxel"),
           ("air_rms", "Air RMS (g/cm³)", "voxel"),
           ("proj_rel_l2_withheld", "Hidden-view error", "projection"),
           ("proj_rel_l2_measured", "Measured-view error", "projection")]


def collect():
    rows = []
    for f in sorted((ROOT / "results").glob("A_w*_init*.json")):
        r = json.loads(f.read_text())
        init = r["run_spec"].get("init", "fdk")
        cdir = ROOT / "cases" / r["case"]
        d = cdir / f"init_{init}_diag.json"
        diag = json.loads(d.read_text()) if d.exists() else {}
        row = dict(worm=r["worm"], span=r["span"], init=init, seed=r["run_spec"]["seed"],
                   gaussians=r["run_record"]["final_gaussians"], seconds=r["run_record"]["elapsed_seconds"],
                   init_psnr=diag.get("init_worm_psnr_db"), init_inside=diag.get("points_inside_worm_pct"),
                   start_hidden=diag.get("start_hidden_err"),
                   conv_psnr=diag.get("fit_worm_psnr_db" if init.endswith("_fit") else "gauss_worm_psnr_db"),
                   conv_hidden=diag.get("fit_hidden_err" if init.endswith("_fit") else "gauss_hidden_err"),
                   run=r["run"])
        for k, _, grp in METRICS:
            row[k] = r[grp][k]
        rows.append(row)
    return rows


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def summarise(rows, out):
    write_csv(out / "all_runs.csv", rows)
    base = {(r["worm"], r["span"]): r for r in rows if r["init"] == "fdk" and r["seed"] == 0}
    summ, paired = [], []
    for span in (120, 90, 60):
        for arm in ARMS:
            sel = [r for r in rows if r["span"] == span and r["init"] == arm and r["seed"] == 0]
            if not sel:
                continue
            s = dict(span=span, init=arm, n_poses=len(sel),
                     init_psnr=np.mean([r["init_psnr"] for r in sel if r["init_psnr"] is not None] or [np.nan]),
                     init_inside=np.mean([r["init_inside"] for r in sel if r["init_inside"] is not None] or [np.nan]),
                     gaussians=np.mean([r["gaussians"] for r in sel]))
            for k in ("start_hidden", "conv_psnr", "conv_hidden"):
                v = [r[k] for r in sel if r[k] is not None]
                s[k] = float(np.mean(v)) if v else np.nan
            for k, _, _ in METRICS:
                s[k] = float(np.mean([r[k] for r in sel]))
            d = [r["worm_psnr_db"] - base[(r["worm"], span)]["worm_psnr_db"] for r in sel if (r["worm"], span) in base]
            h = [r["proj_rel_l2_withheld"] - base[(r["worm"], span)]["proj_rel_l2_withheld"] for r in sel
                 if (r["worm"], span) in base]
            s["d_psnr_vs_fdk"] = float(np.mean(d)) if d else np.nan
            s["d_psnr_min"] = float(np.min(d)) if d else np.nan
            s["d_psnr_max"] = float(np.max(d)) if d else np.nan
            s["d_hidden_vs_fdk_pts"] = float(100 * np.mean(h)) if h else np.nan
            summ.append(s)
            for r in sel:
                if (r["worm"], span) in base:
                    paired.append(dict(worm=r["worm"], span=span, init=arm,
                                       d_psnr=r["worm_psnr_db"] - base[(r["worm"], span)]["worm_psnr_db"],
                                       d_cv=r["interior_cv_pct_wmean"] - base[(r["worm"], span)]["interior_cv_pct_wmean"],
                                       d_hidden_pts=100 * (r["proj_rel_l2_withheld"] - base[(r["worm"], span)]["proj_rel_l2_withheld"])))
    seeds = []  # same case and start, second training seed: the run-to-run spread
    for r in rows:
        if r["seed"] == 1:
            r0 = next(x for x in rows if x["seed"] == 0 and (x["worm"], x["span"], x["init"]) == (r["worm"], r["span"], r["init"]))
            seeds.append(dict(worm=r["worm"], span=r["span"], init=r["init"],
                              d_psnr=abs(r["worm_psnr_db"] - r0["worm_psnr_db"]),
                              d_hidden_pts=abs(100 * (r["proj_rel_l2_withheld"] - r0["proj_rel_l2_withheld"]))))
    write_csv(out / "seed_repeats.csv", seeds)
    write_csv(out / "summary_by_arm.csv", summ)
    write_csv(out / "paired_vs_fdk.csv", paired)
    return summ


def figures(rows, summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = COL
    arms = [a for a in ARMS if any(s["init"] == a for s in summ)]
    spans = [120, 90, 60]
    # 1. main metrics against arc
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for a_, (k, lab) in zip(ax, [("worm_psnr_db", "Worm PSNR (dB) ↑"), ("interior_cv_pct_wmean", "Interior CV (%) ↓"),
                                 ("abs_rel_bias_pct_wmean", "Density bias (%) ↓"),
                                 ("proj_rel_l2_withheld", "Hidden-view error (%) ↓")]):
        for arm in arms:
            y = [next((s[k] for s in summ if s["span"] == sp and s["init"] == arm), np.nan) for sp in spans]
            if k == "proj_rel_l2_withheld":
                y = [100 * v for v in y]
            a_.plot(spans, y, "o-", color=col[arm], label=LABEL[arm])
        a_.set_xlabel("measured arc (degrees)"); a_.set_title(lab, fontsize=10); a_.invert_xaxis(); a_.grid(alpha=.3)
    ax[0].legend(fontsize=7.5)
    fig.suptitle("R²-Gaussian after 30,000 iterations, mean of 3 worm poses", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_metrics_vs_arc.png", dpi=170); plt.close(fig)
    # 2. starting-point quality vs final quality
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    mk = {120: "o", 90: "s", 60: "^"}
    for r in rows:
        if r["seed"] != 0 or r["init_psnr"] is None:
            continue
        ax[0].scatter(r["init_psnr"], r["worm_psnr_db"], color=col[r["init"]], marker=mk[r["span"]], s=28)
        ax[1].scatter(r["init_inside"], r["worm_psnr_db"], color=col[r["init"]], marker=mk[r["span"]], s=28)
    ax[0].set_xlabel("worm PSNR of the starting volume (dB)"); ax[1].set_xlabel("starting Gaussians inside the worm (%)")
    for a_ in ax:
        a_.set_ylabel("final worm PSNR (dB)"); a_.grid(alpha=.3)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=col[a], marker="o", ls="", label=LABEL[a]) for a in arms]
    h += [Line2D([], [], color="k", marker=mk[s], ls="", label=f"{s}°") for s in spans]
    ax[1].legend(handles=h, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_start_vs_final.png", dpi=170); plt.close(fig)


def gauss_start(cdir, arm):
    """The initial Gaussians of a run (native rule, or the fitted checkpoint), voxelised (z, y, x)."""
    import os
    import sys
    import argparse as ap_
    repo = L.HERE.parents[1]
    cwd = os.getcwd(); os.chdir(repo); sys.path.insert(0, str(repo))
    import torch
    from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
    from r2_gaussian.dataset import Scene
    from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query

    base = arm.removesuffix("_fit")
    p = ap_.ArgumentParser(); lp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    args = p.parse_args([]); args.source_path = str(cdir)
    args.ply_path = str(cdir / ("init.npy" if base == "fdk" else f"init_{base}.npy"))
    ma, oa, pa = lp.extract(args), op.extract(args), pp.extract(args)
    sc = Scene(ma, False).scanner_cfg
    g = GaussianModel(np.array([ma.scale_min, ma.scale_max]) * max(sc["sVoxel"]))
    if arm.endswith("_fit"):
        g.restore(torch.load(cdir / f"init_{base}_fit.ckpt", weights_only=False)[0], oa)
    else:
        initialize_gaussian(g, ma, None)
    with torch.no_grad():
        vol = query(g, sc["offOrigin"], sc["nVoxel"], sc["sVoxel"], pa)["vol"].cpu().numpy()
    os.chdir(cwd); del g; torch.cuda.empty_cache()
    return np.transpose(vol, (2, 1, 0))


def slice_figure(out, worm=4, span=90, z=32, arms=("fdk", "sirttv", "unet", "unet_fit")):
    """One slice of each start followed through the three stages, with the error after training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cdir = ROOT / "cases" / f"A_w{worm:04d}_mid_span{span}_start0_f0_ns0"
    gt = np.load(cdir / "vol_gt.npy")[:, :, z].T
    ys, xs = np.nonzero(gt > 0)
    cy, cx = int(ys.mean()), int(xs.mean()); h = 48
    crop = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
    fig, ax = plt.subplots(4, len(arms) + 1, figsize=(2.2 * (len(arms) + 1), 9.0))
    ax[0, 0].imshow(gt[crop], cmap="gray", vmin=0, vmax=.65); ax[0, 0].set_title("reference", fontsize=9)
    for i in range(1, 4):
        ax[i, 0].axis("off")
    rows = ["starting volume", "as Gaussians, iteration 0", "after 30,000 iterations", "error after training"]
    for j, arm in enumerate(arms, 1):
        iv = np.load(cdir / f"initvol_{arm.removesuffix('_fit')}.npz")["vol"].astype(np.float32)[z]  # (z, y, x)
        gv = gauss_start(cdir, arm)[z]
        rv = np.load(ROOT / "results" / f"A_w{worm:04d}_mid_span{span}_start0_f0_ns0_tv0.05_seed0_it30000_init{arm}.vol.npz")
        rv = rv[rv.files[0]].astype(np.float32)
        sl = rv[:, :, z].T if rv.shape[0] != 64 else rv[z]
        for i, im in enumerate((iv, gv, sl)):
            ax[i, j].imshow(im[crop], cmap="gray", vmin=0, vmax=.65)
        ax[3, j].imshow((sl - gt)[crop], cmap="RdBu_r", vmin=-.3, vmax=.3)
        ax[0, j].set_title(LABEL[arm].replace(" (current)", "").replace(", fitted", "\nfitted conversion"), fontsize=8.5)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([])
    for i, t in enumerate(rows):
        ax[i, 1].set_ylabel(t, fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"fig_slices_w{worm}_span{span}.png", dpi=170); plt.close(fig)


def pipeline_figure(out):
    """Schematic of the study: only the starting volume differs between the compared runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 3.3)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.3); ax.axis("off")

    def box(x, y, w, h, txt, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec="#444", lw=.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#444", lw=.9))

    box(0.1, 1.25, 1.7, 0.8, "measured views\n(120°, 90° or 60° arc\nof the supplied sinogram)", "#eeeeee")
    rows = [("FBP (current start)", "#dddddd", 2.55), ("SIRT, 200 iterations", "#cfe2f3", 1.95),
            ("SIRT + TV", "#d0ece3", 1.35), ("U-Net ensemble on the SIRT volume", "#fbe0cc", 0.75),
            ("same + 30% of points where\nthe U-Nets disagree", "#f6e9c4", 0.1)]
    for t, fc, y in rows:
        box(2.4, y, 2.6, 0.5, t, fc); arrow(1.8, 1.65, 2.4, y + 0.25); arrow(5.0, y + 0.25, 5.6, 2.1)
    ax.text(3.7, 3.15, "starting volume", ha="center", fontsize=8.5, weight="bold")
    ax.text(3.7, -0.12, "Exp. 1: FBP / SIRT / SIRT+TV      Exp. 2: U-Net variants", ha="center", fontsize=7.5, color="#555")
    box(5.6, 1.75, 1.6, 0.7, "50,000 Gaussians,\nnative rule (voxels > 0.05,\ndensity × 0.15)", "#eeeeee")
    box(5.6, 0.85, 1.6, 0.7, "same 50,000 Gaussians,\nthen fitted to the\nstarting volume", "#e8dcd2")
    arrow(6.4, 1.75, 6.4, 1.55)
    box(7.6, 1.25, 1.6, 0.8, "R²-Gaussian\n30,000 iterations\nλ_TV = 0.05", "#eeeeee")
    box(9.5, 1.25, 1.4, 0.8, "scores: reference\nvolume + hidden views", "#eeeeee")
    arrow(7.2, 2.1, 7.6, 1.75); arrow(7.2, 1.2, 7.6, 1.55); arrow(9.2, 1.65, 9.5, 1.65)
    fig.tight_layout(); fig.savefig(out / "fig_pipeline.png", dpi=180); plt.close(fig)


def unet_figure(out, unet_dir=ROOT / "init_study" / "unet"):
    """Training curves of the U-Nets (validation poses) and one validation example."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.3))
    col = {120: "#2a78b5", 90: "#d95f02", 60: "#1b9e77"}
    for span in (120, 90, 60):
        for s in (0, 1, 2):
            f = unet_dir / "models" / f"unet_span{span}_seed{s}.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            ax.plot([l["step"] for l in m["log"]], [l["val_worm_mae"] for l in m["log"]], color=col[span], alpha=.8,
                    label=f"{span}°" if s == 0 else None)
            if s == 0:
                ax.axhline(m["val_worm_mae_input"], color=col[span], ls=":", lw=1)
    ax.set_xlabel("training step"); ax.set_ylabel("error inside worm (g/cm³)"); ax.set_yscale("log")
    ax.set_title("validation poses: U-Net (solid) vs SIRT input (dotted)", fontsize=8.5); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "fig_unet_training.png", dpi=170); plt.close(fig)


def convergence_figure(rows, out):
    """R2's own whole-box PSNR against iteration, mean over poses, one panel per arc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = COL
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for a_, span in zip(ax, (120, 90, 60)):
        for arm in ARMS:
            curves = [json.loads((ROOT / "results" / f"{r['run']}.json").read_text())["native_psnr3d_curve"]
                      for r in rows if r["span"] == span and r["init"] == arm and r["seed"] == 0]
            if not curves:
                continue
            its = sorted(int(k) for k in curves[0])
            y = np.mean([[c[str(i)] for i in its] for c in curves], 0)
            a_.plot(its, y, "o-", ms=3, color=col[arm], label=f"{LABEL[arm]} (n={len(curves)})")
        a_.set_xscale("log"); a_.set_xlabel("iteration"); a_.set_title(f"{span}° arc", fontsize=10); a_.grid(alpha=.3)
    ax[0].set_ylabel("whole-box PSNR (dB)"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_convergence.png", dpi=170); plt.close(fig)


def stages_figure(summ, out, arms=("fdk", "sirttv", "sirttv_fit", "unet", "unet_fit")):
    """Each start followed through three stages: starting volume, as Gaussians at iteration 0,
    after 30,000 iterations. Mean over poses; one column per arc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ["starting\nvolume", "Gaussians,\niteration 0", "after\ntraining"]
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.8), sharey="row")
    for c, span in enumerate((120, 90, 60)):
        for arm in arms:
            s = next((x for x in summ if x["init"] == arm and x["span"] == span), None)
            if s is None:
                continue
            ls = "--" if arm.endswith("_fit") else "-"
            ax[0, c].plot(range(3), [100 * s["start_hidden"], 100 * s["conv_hidden"], 100 * s["proj_rel_l2_withheld"]],
                          "o", ls=ls, color=COL[arm], lw=1.4, label=LABEL[arm])
            ax[1, c].plot(range(3), [s["init_psnr"], s["conv_psnr"], s["worm_psnr_db"]], "o", ls=ls, color=COL[arm], lw=1.4)
        ax[0, c].set_title(f"{span}° arc", fontsize=10); ax[0, c].set_yscale("log")
        for a_ in ax[:, c]:
            a_.set_xticks(range(3)); a_.set_xticklabels(stages, fontsize=8.5); a_.grid(alpha=.3)
    ax[0, 0].set_ylabel("hidden-view error (%)  ↓"); ax[1, 0].set_ylabel("worm PSNR (dB)  ↑")
    ax[0, 2].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "fig_stages.png", dpi=170); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "init_study" / "analysis")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = collect()
    summ = summarise(rows, a.out)
    figures(rows, summ, a.out)
    pipeline_figure(a.out)
    convergence_figure(rows, a.out)
    stages_figure(summ, a.out)
    unet_figure(a.out)
    for span in (120, 90, 60):
        slice_figure(a.out, 4, span)
    for s in summ:
        print(f"{s['span']:>3} {s['init']:10s} n={s['n_poses']} start {s['init_psnr']:.2f} dB/{100 * s['start_hidden']:.1f}% gauss {s['conv_psnr']:.2f}/{100 * s['conv_hidden']:.1f}% -> "
              f"PSNR {s['worm_psnr_db']:.2f} (Δ {s['d_psnr_vs_fdk']:+.2f} [{s['d_psnr_min']:+.2f},{s['d_psnr_max']:+.2f}]) "
              f"CV {s['interior_cv_pct_wmean']:.1f} bias {s['abs_rel_bias_pct_wmean']:.1f} "
              f"hidden {100 * s['proj_rel_l2_withheld']:.2f}% (Δ {s['d_hidden_vs_fdk_pts']:+.2f}) F1 {s['worm_bf1_1um']:.3f}")


if __name__ == "__main__":
    main()
