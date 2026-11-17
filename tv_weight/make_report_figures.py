"""Figures and table numbers for the worm TV report (all values read from result files).

Writes <out>/fig*.png and <out>/report_numbers.json.
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import common as c

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "figure.dpi": 110})
ARCS = [180, 120, 90, 60]
LAMS = [0, 0.025, 0.05, 0.1, 0.2, 0.5]
COL = {180: "#1b6ca8", 150: "#5aa1d6", 120: "#2e9e5b", 90: "#e08a1e", 60: "#c4372f"}


def res(R, name):
    p = R / "results" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def a_name(span, lam, f=0, start=0, tag=""):
    return f"A_w0004_mid_span{span}_start{start}_f{f:g}_ns0_tv{lam:g}_seed0_it30000{tag}"


def xpos(l):
    return 0.01 if l == 0 else l


def fig_data(R, out, N):
    stack, info = c.load_stack(4)
    ph = np.load(R / "phantoms" / "worm0004_mid" / "phantom.npz")
    truth = ph["truth"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25, 1.1]))
    for a_, v, t in ((ax[0], 0, "(a) radiograph, view 0°"), (ax[1], 50, "(b) radiograph, view 90°")):
        im = a_.imshow(stack[:, v, :], cmap="gray", aspect="auto")
        a_.axhspan(223, 287, color="#ffcc00", alpha=0.25)
        a_.set_title(t); a_.set_xlabel("detector position (µm)"); a_.set_ylabel("height z (µm)")
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="projected density (g/cm³·µm)")
    im = ax[2].imshow(stack[255], cmap="gray", aspect="auto", extent=(0, 310, 178.2, 0))
    ax[2].axhline(120, color="#ffcc00", lw=1.5, ls="--")
    ax[2].text(5, 113, "a 120° scan keeps views above the line", color="#ffcc00", fontsize=8.5)
    ax[2].set_title("(c) sinogram of slice z = 255"); ax[2].set_xlabel("detector position (µm)"); ax[2].set_ylabel("view angle (°)")
    z = 255 - 223
    ys, xs = np.nonzero(truth[z] > 0)
    y0, y1, x0, x1 = ys.min() - 6, ys.max() + 7, xs.min() - 6, xs.max() + 7
    im = ax[3].imshow(truth[z, y0:y1, x0:x1], cmap="gray", vmin=0, vmax=0.65, extent=(x0, x1, y1, y0))
    ax[3].set_title("(d) reference phantom, slice z = 255", pad=8); ax[3].set_xlabel("x (µm)"); ax[3].set_ylabel("y (µm)")
    fig.colorbar(im, ax=ax[3], shrink=0.8, label="density (g/cm³)")
    fig.tight_layout()
    fig.savefig(out / "fig1_data.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    rep = json.loads((R / "phantoms" / "worm0004_mid" / "phantom_report.json").read_text())
    N["phantom_levels"] = rep["class_density_g_cm3"]
    N["phantom_fit_rel_l2"] = rep["phantom_vs_supplied_sino_rel_l2"]
    N["phantom_fit_all"] = sorted(json.loads(p.read_text())["phantom_vs_supplied_sino_rel_l2"]
                                  for p in (R / "phantoms").glob("*/phantom_report.json"))
    N["sirt_rel_l2"] = rep["sirt_reprojection_residual"]


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(17, 5.6))
    ax.set_xlim(0, 17); ax.set_ylim(0, 5.6); ax.axis("off")
    box(ax, 0.1, 2.2, 2.2, 1.4, "Worm sinograms\n144 poses, parallel beam\n100 views over 180°\n1 µm pixels", "#e8f1fb")
    box(ax, 2.7, 2.2, 2.3, 1.4, "Preparation\nremove empty slices\nmg/cm³ → g/cm³\n64-slice block", "#e8f1fb")
    box(ax, 5.4, 3.5, 2.6, 1.8, "Limited-angle scan\nkeep views in one arc\n(180°, 150°, 120°, 90°, 60°)\nstarting at 0° or 90°\n+ Gaussian noise (0–5%)", "#fdf0dc")
    box(ax, 5.4, 0.3, 2.6, 1.8, "Reference phantom\nfull-angle reconstruction\n→ 5 density levels\n→ densities fitted to data\n(sinograms reproduced to 3%)", "#eef7ea")
    box(ax, 8.6, 3.9, 2.6, 1.3, "R²-Gaussian (GPU)\nGaussian blobs + TV(λ)", "#fce8e6")
    box(ax, 8.6, 2.2, 2.6, 1.3, "Classical TV (CPU)\nparallel and fan beam\n8 poses, many λ", "#fce8e6")
    box(ax, 11.8, 0.6, 2.4, 4.6, "Evaluation\n\nerror on unmeasured\nviews (exact)\n\noutlines\ninterior uniformity\ndensity bias\n(against the phantom)", "#f1eafa")
    box(ax, 14.6, 1.6, 2.3, 2.6, "TV analysis\n\nbest λ vs\nangular range\n\nautomatic λ rules\n(tuned on 4 poses,\ntested on 4 others)", "#f1eafa")
    arrow(ax, 2.3, 2.9, 2.7, 2.9)
    arrow(ax, 5.0, 3.2, 5.4, 4.0); arrow(ax, 5.0, 2.6, 5.4, 1.6)
    arrow(ax, 8.0, 4.5, 8.6, 4.5); arrow(ax, 8.0, 3.8, 8.6, 3.0)
    arrow(ax, 8.0, 1.9, 8.6, 2.5)
    ax.text(8.32, 2.05, "simulated\nscans", fontsize=8.5, color="#2b7a2b", ha="right", va="center")
    arrow(ax, 8.0, 0.8, 11.8, 0.8)
    ax.text(9.0, 0.95, "voxel truth used for scoring", fontsize=8.5, color="#2b7a2b")
    arrow(ax, 11.2, 4.5, 11.8, 4.5); arrow(ax, 11.2, 2.85, 11.8, 2.85)
    arrow(ax, 14.2, 2.9, 14.6, 2.9)
    fig.savefig(out / "fig2_pipeline.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_recons(R, out):
    ph = np.load(R / "phantoms" / "worm0004_mid" / "phantom.npz")
    z = 255 - 223
    truth = ph["truth"][z]
    ys, xs = np.nonzero(truth > 0)
    y0, y1, x0, x1 = ys.min() - 5, ys.max() + 6, xs.min() - 5, xs.max() + 6
    fig = plt.figure(figsize=(13.5, 11.2))
    gs = fig.add_gridspec(4, 6, width_ratios=[1, 1, 1, 1, 1, 0.07], wspace=0.06, hspace=0.12)
    rows = [("reconstruction\nno TV (λ = 0)", 0, "rec"), ("error\nno TV (λ = 0)", 0, "err"),
            ("reconstruction\nTV λ = 0.2", 0.2, "rec"), ("error\nTV λ = 0.2", 0.2, "err")]
    for r, (lab, lam, kind) in enumerate(rows):
        a0 = fig.add_subplot(gs[r, 0])
        if kind == "rec":
            a0.imshow(truth[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=0.65)
            if r == 0:
                a0.set_title("reference (truth)", fontsize=11)
            a0.plot([3, 23], [y1 - y0 - 4] * 2, "w", lw=2.5)
            a0.text(3, y1 - y0 - 7, "20 µm", color="w", fontsize=8.5)
        else:
            a0.axis("off")
        a0.set_xticks([]); a0.set_yticks([])
        a0.set_ylabel(lab, fontsize=10.5)
        if kind == "err":
            a0.text(-0.05, 0.5, lab, transform=a0.transAxes, ha="right", va="center", fontsize=10.5, rotation=90)
        for j_, sp in enumerate(ARCS, 1):
            a = fig.add_subplot(gs[r, j_])
            v = np.load(R / "results" / f"{a_name(sp, lam)}.vol.npz")["pred"].astype(np.float32)[z]
            if kind == "rec":
                im = a.imshow(v[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=0.65)
            else:
                er = a.imshow((v - truth)[y0:y1, x0:x1], cmap="RdBu_r", vmin=-0.25, vmax=0.25)
            if r == 0:
                a.set_title(f"{sp}° scan", fontsize=11)
            a.set_xticks([]); a.set_yticks([])
        cax = fig.add_subplot(gs[r, 5])
        if kind == "rec":
            fig.colorbar(im, cax=cax, label="density (g/cm³)")
        else:
            fig.colorbar(er, cax=cax, label="error (g/cm³)")
    fig.savefig(out / "fig3_reconstructions.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_r2_sweep(R, out, N):
    tab = {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for s in ARCS:
        rows = []
        for l in LAMS:
            d = res(R, a_name(s, l))
            if d is None:
                continue
            v, p = d["voxel"], d["projection"]["proj_rel_l2_withheld"]
            rows.append(dict(lam=l, psnr=v["worm_psnr_db"], cv=v["interior_cv_pct_wmean"], bias=v["abs_rel_bias_pct_wmean"],
                             dice=v["dice_class_mean"], f1=v["worm_bf1_1um"], msd=v["worm_mean_surface_dist_um"],
                             withheld=None if p is None else 100 * p))
        tab[s] = rows
        x = [xpos(r["lam"]) for r in rows]
        ax[0].plot(x, [r["psnr"] for r in rows], "o-", color=COL[s], label=f"{s}°")
        ax[1].plot(x, [r["cv"] for r in rows], "o-", color=COL[s], label=f"{s}°")
        if rows[0]["withheld"] is not None:
            ax[2].plot(x, [r["withheld"] for r in rows], "o-", color=COL[s], label=f"{s}°")
    titles = ["(a) accuracy inside the worm: PSNR (dB) ↑", "(b) interior variation (%) ↓", "(c) error on unmeasured views (%) ↓"]
    for a_, t in zip(ax, titles):
        a_.set_xscale("log"); a_.set_xticks([0.01, 0.025, 0.05, 0.1, 0.2, 0.5]); a_.minorticks_off()
        a_.set_xticklabels(["0", "0.025", "0.05", "0.1", "0.2", "0.5"]); a_.set_xlabel("TV weight λ")
        a_.set_title(t); a_.grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, [f"{x} scan" for x in l], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=10, frameon=False)
    ax[2].text(0.5, 0.97, "(a 180° scan has no unmeasured views)", transform=ax[2].transAxes, ha="center", va="top", fontsize=8.5, color="#555")
    fig.tight_layout()
    fig.savefig(out / "fig4_r2_tv_sweep.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    N["r2_sweep"] = tab


def best_lams_classical(R, sub, pattern, key="worm_psnr_db"):
    D = defaultdict(list)
    for f in glob.glob(str(R / sub / pattern)):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        pl = d["per_lambda"]
        b = max(pl, key=lambda p: p["voxel"][key])
        D[(int(d["span"]), int(d["start"]), d["noise_fraction"])].append(
            dict(best=b["lam"], val=b["voxel"][key],
                 at005=next((p["voxel"][key] for p in pl if p["lam"] == 0.05), np.nan),
                 at0=next((p["voxel"][key] for p in pl if p["lam"] == 0), np.nan),
                 cv_best=b["voxel"]["interior_cv_pct_wmean"],
                 cv0=next((p["voxel"]["interior_cv_pct_wmean"] for p in pl if p["lam"] == 0), np.nan),
                 per_crit={k: max(pl, key=lambda p: s * p["voxel"][k])["lam"] for k, s in
                           (("worm_psnr_db", 1), ("dice_class_mean", 1), ("interior_cv_pct_wmean", -1),
                            ("abs_rel_bias_pct_wmean", -1))}))
    return D


def fig_best_lambda(R, out, N):
    par = best_lams_classical(R, "tv2d", "w*.json")
    fan = best_lams_classical(R, "fan2d", "fan_w*.json")
    r2 = {s: max(N["r2_sweep"][s], key=lambda r: r["psnr"])["lam"] for s in ARCS}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for f, ls, lab in ((0.0, "-", "noise 0% and 1% (identical)"), (0.03, ":", "noise 3%")):
        spans = sorted({k[0] for k in par if k[1] == 0 and k[2] == f})
        ax.plot(spans, [np.median([e["best"] for e in par[(s, 0, f)]]) for s in spans], "o" + ls, color="#1b6ca8",
                label=f"classical TV, parallel beam, {lab}")
    spans = sorted({k[0] for k in fan if k[1] == 0 and k[2] == 0.01})
    ax.plot(spans, [np.median([e["best"] for e in fan[(s, 0, 0.01)]]) for s in spans], "s-", color="#2e9e5b",
            label="classical TV, fan beam, noise 1%")
    ax.plot(ARCS, [r2[s] for s in ARCS], "D-", color="#c4372f", ms=7, label="R²-Gaussian, the supplied sinograms")
    ax.set_yscale("log"); ax.set_xlabel("angular range of the scan (°)"); ax.set_ylabel("best TV weight λ (log scale)")
    ax.set_title("Best TV weight against angular range (classical: median of 8 poses)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.5)
    ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(out / "fig5_best_lambda.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    N["r2_best"] = r2
    summ = {}
    for (s, st, f), L in sorted(par.items()):
        summ[f"{s}_{st}_{f:g}"] = dict(n=len(L), best=float(np.median([e["best"] for e in L])),
                                       psnr_best=float(np.median([e["val"] for e in L])),
                                       psnr_005=float(np.nanmedian([e["at005"] for e in L])),
                                       psnr_0=float(np.nanmedian([e["at0"] for e in L])),
                                       cv_best=float(np.median([e["cv_best"] for e in L])),
                                       cv0=float(np.nanmedian([e["cv0"] for e in L])),
                                       crit={k: float(np.median([e["per_crit"][k] for e in L])) for k in L[0]["per_crit"]})
    N["classical_parallel"] = summ
    N["classical_fan"] = {f"{s}_{st}_{f:g}": dict(n=len(L), best=float(np.median([e["best"] for e in L])),
                                                  psnr_best=float(np.median([e["val"] for e in L])))
                          for (s, st, f), L in sorted(fan.items())}


def fig_selection(R, out, N):
    sel = json.loads((R / "analysis" / "sweep_tv2d" / "tv2d_selection_worm_psnr_db.json").read_text())
    from analyze_sweep import TEST
    rules = [("angle_noise_rule", "angle rule"), ("fixed_dev_best", "one fixed λ"), ("cross_validation", "cross-validation"),
             ("l_curve", "L-curve"), ("discrepancy", "discrepancy principle")]
    test = [k for k in sel if k["worm"] in TEST]
    vals = {r: float(np.mean([k["regret"][r] for k in test])) for r, _ in rules}
    by_noise = {f: {r: float(np.mean([k["regret"][r] for k in test if k["f"] == f])) for r, _ in rules}
                for f in sorted({k["f"] for k in test})}
    fan = json.loads((R / "analysis" / "fan" / "fan_transfer.json").read_text())
    fr = next(x for x in fan if x["metric"] == "worm_psnr_db")["rows"]
    frules = [("parallel angle rule", "angle rule from parallel beam"), ("parallel fixed λ", "fixed λ from parallel beam"),
              ("fan-native fixed λ", "fixed λ re-tuned on fan"), ("L-curve", "L-curve"), ("discrepancy", "discrepancy principle")]
    fvals = {n: float(np.mean([r[f"regret:{n}"] for r in fr if r[f"regret:{n}"] is not None])) for n, _ in frules}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].barh([l for _, l in rules][::-1], [vals[r] for r, _ in rules][::-1], color=["#c4372f", "#999", "#999", "#999", "#2e9e5b"][::-1])
    ax[0].set_xlabel("PSNR lost vs best possible λ (dB, lower is better)")
    ax[0].set_title(f"(a) parallel beam: {len(test)} scans of 4 poses never used for tuning")
    ax[1].barh([l for _, l in frules][::-1], [fvals[n] for n, _ in frules][::-1], color=["#c4372f", "#999", "#999", "#999", "#999"][::-1])
    ax[1].set_xlabel("PSNR lost vs best possible λ (dB, lower is better)")
    ax[1].set_title(f"(b) fan beam: rules applied to {len(fr)} fan-beam scans")
    for a_ in ax:
        a_.grid(axis="x", alpha=0.3)
        for p in a_.patches:
            a_.text(p.get_width() + 0.03, p.get_y() + p.get_height() / 2, f"{p.get_width():.2f}", va="center", fontsize=9)
    ax[0].patches[-5].set_color("#888")
    fig.tight_layout(); fig.savefig(out / "fig6_selection.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    N["selection_test"] = dict(n=len(test), all=vals, by_noise={f"{k:g}": v for k, v in by_noise.items()})
    N["fan_transfer"] = dict(n=len(fr), regret=fvals)


def fig_comparison(R, out, N):
    rows = json.loads((R / "analysis" / "p4" / "p4_comparison.json").read_text())
    N["p4"] = rows
    pick = lambda m, s, f: next((r for r in rows if r["method"].startswith(m) and r["span"] == s and r["noise_pct"] == f), None)
    groups = [(120, 0), (120, 5), (90, 0), (90, 5)]
    meths = [("FBP (re-run)", "FBP", "#bbb"), ("SART (re-run)", "SART", "#888"),
             ("Gaussian slab", "uncertainty-study Gaussian fit", "#e08a1e"), ("R²-Gaussian", "R²-Gaussian + TV (this work)", "#c4372f")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for i, (m, lab, colr) in enumerate(meths):
        ys = []
        for s, f in groups:
            r = pick(m, s, f)
            ys.append(np.nan if r is None else r["skimage_withheld"])
        ax.bar(np.arange(4) + (i - 1.5) * w, ys, w, color=colr, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{s}° scan\nnoise {f}%" for s, f in groups])
    ax.set_ylabel("error on unmeasured views (%)"); ax.set_ylim(0, 30)
    ax.text(-0.5, 28.5, "FBP: 80–86% (off scale)", fontsize=8.5, color="#555")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.93)); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Unmeasured-view error, slice z = 255, common scoring operator")
    fig.tight_layout(); fig.savefig(out / "fig7_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def replay_numbers(R, N):
    rows = {}
    for s, st in ((180, 0), (120, 0), (120, 90), (90, 0), (90, 90)):
        d = res(R, a_name(s, 0.05, start=st))
        v, p = d["voxel"], d["projection"]["proj_rel_l2_withheld"]
        rows[f"{s}_{st}"] = dict(views=d["n_measured"], psnr=v["worm_psnr_db"], dice_worm=v["dice_worm"], f1=v["worm_bf1_1um"],
                                 msd=v["worm_mean_surface_dist_um"], cv=v["interior_cv_pct_wmean"], bias=v["abs_rel_bias_pct_wmean"],
                                 withheld=None if p is None else 100 * p, native_psnr=d["native_eval3d"]["psnr_3d"],
                                 native_ssim=d["native_eval3d"]["ssim_3d"])
    N["replay"] = rows
    tv = {}
    for st in (0, 90):
        for l in (0.025, 0.05, 0.1):
            d = res(R, a_name(120, l, start=st))
            v = d["voxel"]
            tv[f"{st}_{l:g}"] = dict(native_psnr=d["native_eval3d"]["psnr_3d"], native_ssim=d["native_eval3d"]["ssim_3d"],
                                     bmae=v["highgrad_boundary_mae"], psnr=v["worm_psnr_db"], dice=v["dice_class_mean"],
                                     cv=v["interior_cv_pct_wmean"], bias=v["abs_rel_bias_pct_wmean"],
                                     withheld=100 * d["projection"]["proj_rel_l2_withheld"])
    N["pilot_tv"] = tv
    a, b = res(R, a_name(120, 0.05)), res(R, a_name(120, 0.05) .replace("seed0", "seed1"))
    N["seed_noise"] = dict(psnr=abs(a["voxel"]["worm_psnr_db"] - b["voxel"]["worm_psnr_db"]),
                           withheld=100 * abs(a["projection"]["proj_rel_l2_withheld"] - b["projection"]["proj_rel_l2_withheld"]),
                           cv=abs(a["voxel"]["interior_cv_pct_wmean"] - b["voxel"]["interior_cv_pct_wmean"]),
                           native_psnr=abs(a["native_eval3d"]["psnr_3d"] - b["native_eval3d"]["psnr_3d"]))
    noisy = {}
    for s in (180, 120, 90):
        for l in (0.05, 0.2):
            d = res(R, a_name(s, l, f=0.05, tag="_keep"))
            if d:
                noisy[f"{s}_{l:g}"] = dict(psnr=d["voxel"]["worm_psnr_db"], cv=d["voxel"]["interior_cv_pct_wmean"],
                                           withheld=None if d["projection"]["proj_rel_l2_withheld"] is None else
                                           100 * d["projection"]["proj_rel_l2_withheld"])
    N["noisy_5pct"] = noisy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="runs_worm directory")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    N = {}
    fig_data(a.root, a.out, N)
    fig_pipeline(a.out)
    fig_recons(a.root, a.out)
    fig_r2_sweep(a.root, a.out, N)
    fig_best_lambda(a.root, a.out, N)
    fig_selection(a.root, a.out, N)
    fig_comparison(a.root, a.out, N)
    replay_numbers(a.root, N)
    c.write_json(a.out / "report_numbers.json", N)
    print("figures and numbers written to", a.out)


if __name__ == "__main__":
    main()
