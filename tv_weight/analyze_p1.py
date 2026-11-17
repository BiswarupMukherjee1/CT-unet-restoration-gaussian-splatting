"""P1 analysis: chest-pilot replay on the worm (tables + figures, both tracks).

Writes  <out>/p1_summary.md, <out>/p1_tables.json and figures. Reads only result JSONs.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c

ARCS = [(180, 0), (120, 0), (120, 90), (90, 0), (90, 90)]
COLS = [("worm_psnr_db", "worm PSNR (dB)", "{:.2f}"), ("dice_worm", "Dice worm/air", "{:.3f}"),
        ("dice_class_mean", "Dice classes", "{:.3f}"), ("worm_bf1_1um", "outline F1@1µm", "{:.3f}"),
        ("compartment_bf1_1um", "compartment F1@1µm", "{:.3f}"),
        ("worm_mean_surface_dist_um", "surface dist (µm)", "{:.2f}"),
        ("interior_cv_pct_wmean", "interior CV (%)", "{:.2f}"), ("abs_rel_bias_pct_wmean", "|bias| (%)", "{:.2f}"),
        ("air_rms", "air RMS (g/cm³)", "{:.4f}"), ("highgrad_boundary_mae", "high-grad MAE", "{:.4f}")]


def rid(track, span, start, lam=0.05, seed=0):
    return f"{track}_w0004_mid_span{span}_start{start}_f0_ns0_tv{lam:g}_seed{seed}_it30000"


def load(res, r):
    p = res / f"{r}.json"
    return json.loads(p.read_text()) if p.exists() else None


def row(d):
    v, p = d["voxel"], d["projection"]
    out = {k: v[k] for k, _, _ in COLS}
    out["withheld_pct"] = None if p["proj_rel_l2_withheld"] is None else 100 * p["proj_rel_l2_withheld"]
    out["measured_pct"] = 100 * p["proj_rel_l2_measured"]
    out["native_psnr"] = d["native_eval3d"]["psnr_3d"]
    out["native_ssim"] = d["native_eval3d"]["ssim_3d"]
    out["minutes"] = d["run_record"]["elapsed_seconds"] / 60
    out["gaussians"] = d["run_record"]["final_gaussians"]
    if p.get("slice"):
        out["z255_withheld_pct"] = None if p["slice"]["withheld"] is None else 100 * p["slice"]["withheld"]
    return out


def md_table(rows, keys, heads, fmts):
    s = "| " + " | ".join(heads) + " |\n|" + "---|" * len(heads) + "\n"
    for r in rows:
        s += "| " + " | ".join(f.format(r[k]) if isinstance(r.get(k), (int, float)) else str(r.get(k, "—"))
                               for k, f in zip(keys, fmts)) + " |\n"
    return s


def pilot_rule(tab):
    """pilot rule (native metrics): a non-default weight is eligible only if neither
    orientation has worse boundary MAE or SSIM than its 0.05 run; among eligible weights and
    0.05 choose the highest mean PSNR; ties within 0.01 dB keep 0.05."""
    base = {s: tab[(s, 0.05)] for s in (0, 90)}
    elig = [0.05]
    for lam in (0.025, 0.1):
        if all(tab[(s, lam)]["highgrad_boundary_mae"] <= base[s]["highgrad_boundary_mae"]
               and tab[(s, lam)]["native_ssim"] >= base[s]["native_ssim"] for s in (0, 90)):
            elig.append(lam)
    mean = {lam: np.mean([tab[(s, lam)]["native_psnr"] for s in (0, 90)]) for lam in (0.025, 0.05, 0.1)}
    best = max(elig, key=lambda l: mean[l])
    if best != 0.05 and mean[best] - mean[0.05] <= 0.01:
        best = 0.05
    return dict(eligible=elig, mean_native_psnr=mean, selected=best)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    md, tables = ["# P1 — chest-pilot replay on the worm (worm_004, mid slab z 223–287)\n"], {}
    for track, label in (("A", "Track A: supplied sinograms (voxel scores vs fitted phantom, approximate)"),
                         ("B", "Track B: simulated from phantom (exact voxel truth)")):
        rows = []
        for span, start in ARCS:
            d = load(a.results, rid(track, span, start))
            if d:
                rows.append(dict(arc=f"{span}° from {start}°", n=d["n_measured"], **row(d)))
        tables[f"{track}_arcs"] = rows
        md.append(f"\n## {label}\n\n### Full scan vs limited arcs (λ_TV = 0.05, noise-free, seed 0)\n\n")
        keys = ["arc", "n"] + [k for k, _, _ in COLS] + ["withheld_pct", "measured_pct", "native_psnr", "native_ssim", "minutes"]
        heads = ["arc", "views"] + [h for _, h, _ in COLS] + ["withheld-view err (%)", "measured-view err (%)",
                                                             "native box PSNR", "native SSIM", "min"]
        fmts = ["{}", "{}"] + [f for _, _, f in COLS] + ["{:.2f}", "{:.2f}", "{:.2f}", "{:.4f}", "{:.1f}"]
        md.append(md_table(rows, keys, heads, fmts))
        d0, d1 = load(a.results, rid(track, 120, 0)), load(a.results, rid(track, 120, 0, seed=1))
        if d0 and d1:
            r0, r1 = row(d0), row(d1)
            diff = {k: abs(r0[k] - r1[k]) for k in r0 if isinstance(r0[k], float) and r1.get(k) is not None}
            tables[f"{track}_seed_noise"] = diff
            md.append("\n**Seed noise** (120° from 0°, seed 0 vs 1, absolute difference): " +
                      ", ".join(f"{k} {v:.4g}" for k, v in diff.items() if k in
                                ("worm_psnr_db", "dice_class_mean", "withheld_pct", "interior_cv_pct_wmean",
                                 "abs_rel_bias_pct_wmean", "native_psnr")) + "\n")
        tab = {}
        for s in (0, 90):
            for lam in (0.025, 0.05, 0.1):
                d = load(a.results, rid(track, 120, s, lam))
                if d:
                    tab[(s, lam)] = dict(start=s, lam=lam, **row(d))
        if len(tab) == 6:
            md.append("\n### pilot TV grid at 120° (both starts)\n\n")
            rows = [tab[k] for k in sorted(tab)]
            keys = ["start", "lam", "native_psnr", "native_ssim", "highgrad_boundary_mae", "worm_psnr_db",
                    "dice_class_mean", "compartment_bf1_1um", "interior_cv_pct_wmean", "abs_rel_bias_pct_wmean",
                    "withheld_pct"]
            heads = ["start", "λ_TV", "native PSNR", "native SSIM", "boundary MAE", "worm PSNR", "Dice classes",
                     "compartment F1", "interior CV %", "|bias| %", "withheld err %"]
            fmts = ["{}", "{:g}", "{:.3f}", "{:.4f}", "{:.5f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.2f}", "{:.2f}", "{:.2f}"]
            md.append(md_table(rows, keys, heads, fmts))
            rule = pilot_rule(tab)
            tables[f"{track}_pilot_rule"] = {k: (v if k != "mean_native_psnr" else {str(l): x for l, x in v.items()})
                                             for k, v in rule.items()}
            md.append(f"\npilot selection rule (native metrics): eligible {rule['eligible']}, "
                      f"mean native PSNR { {str(k): round(v, 3) for k, v in rule['mean_native_psnr'].items()} }, "
                      f"**selected λ_TV = {rule['selected']}**.\n")
            wm = {lam: np.mean([tab[(s, lam)]["worm_psnr_db"] for s in (0, 90)]) for lam in (0.025, 0.05, 0.1)}
            md.append(f"Same comparison on worm-restricted PSNR: { {str(k): round(v, 3) for k, v in wm.items()} }.\n")
        # figures: per-view withheld error vs angle, and edge error vs edge orientation
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
        for (span, start), col in zip(ARCS, ["k", "C0", "C1", "C2", "C3"]):
            d = load(a.results, rid(track, span, start))
            if not d:
                continue
            pv = 100 * np.array(d["projection"]["per_view_rel_l2"])
            ax[0].plot(c.ANGLES_DEG, pv, color=col, label=f"{span}° from {start}°")
            e = d["voxel"]["edge_err_by_normal_deg"]
            ax[1].plot([int(k.split("-")[0]) + 7.5 for k in e], list(e.values()), "o-", color=col,
                       label=f"{span}° from {start}°")
            cur = d["native_psnr3d_curve"]
            its = sorted(int(k) for k in cur)
            ax[2].plot(its, [cur[str(i)] if str(i) in cur else cur[i] for i in its], "o-", color=col, ms=3,
                       label=f"{span}° from {start}°")
        ax[0].set_xlabel("view angle (deg)"); ax[0].set_ylabel("relative error of reprojection (%)")
        ax[0].set_title("error per view (measured + withheld)"); ax[0].set_yscale("log"); ax[0].grid(alpha=.3)
        ax[1].set_xlabel("in-plane edge-normal direction (deg)"); ax[1].set_ylabel("gradient error / true gradient")
        ax[1].set_title("edge error by edge direction"); ax[1].grid(alpha=.3)
        ax[2].set_xlabel("iteration"); ax[2].set_ylabel("native box PSNR (dB)"); ax[2].set_title("convergence")
        ax[2].grid(alpha=.3)
        for x in ax:
            x.legend(fontsize=8)
        fig.suptitle(f"Track {track}: worm_004 mid slab, λ_TV = 0.05")
        fig.savefig(a.out / f"p1_{track}_angles_edges_convergence.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
    (a.out / "p1_summary.md").write_text("".join(md))
    c.write_json(a.out / "p1_tables.json", {k: v for k, v in tables.items()})
    print("".join(md))


if __name__ == "__main__":
    main()
