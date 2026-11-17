"""P2/P3 analysis: TV sweep curves, best lambda vs angular coverage, automatic selection.

  --r2 <results dir>   R2-Gaussian sweep (worm_004 mid): curves and best-lambda tables;
  --tv2d <dir>         classical sweep (8 poses): curves, best lambda, and automatic rules
                       evaluated by regret on held-out test poses.
Every number is read from result JSONs; nothing is tuned on the test poses.
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c

DEV, TEST = [4, 8916, 1035, 293], [12, 8945, 1031, 6896]
METRICS = [("worm_psnr_db", "worm PSNR (dB)", +1), ("dice_class_mean", "Dice of density classes", +1),
           ("compartment_bf1_1um", "compartment-boundary F1 @1µm", +1),
           ("interior_cv_pct_wmean", "interior CV (%)", -1), ("abs_rel_bias_pct_wmean", "|density bias| (%)", -1),
           ("withheld_pct", "withheld-view error (%)", -1)]


def metric(rec, key):
    if key == "withheld_pct":
        w = rec.get("proj_rel_l2_withheld", rec.get("projection", {}).get("proj_rel_l2_withheld"))
        return None if w is None else 100 * w
    return rec["voxel"][key]


def xlam(l):
    return max(l, 1e-4)  # lambda=0 drawn at the left edge of log axes


# ----------------------------------------------------------------------------- R2 sweep
def r2_groups(res_dir):
    g = defaultdict(dict)
    for f in glob.glob(str(Path(res_dir) / "*_it30000.json")):
        d = json.loads(Path(f).read_text())
        if d.get("worm") != 4 or d["run_spec"].get("seed", 0) != 0:
            continue
        lam = d["run_spec"]["lambda_tv"]
        g[(d["track"], d["noise_fraction"], int(d["span"]), int(d["start"]))][lam] = d
    return g


def best_band(curve, sign, tol):
    lams = sorted(curve)
    vals = np.array([curve[l] for l in lams])
    ib = int(np.argmax(sign * vals))
    band = [l for l, v in zip(lams, vals) if sign * (vals[ib] - v) <= tol]
    return lams[ib], band


def analyze_r2(res_dir, out, seed_tol):
    g = r2_groups(res_dir)
    rows, md = [], ["# R²-Gaussian TV sweep (worm_004 mid slab)\n\n",
                    "Best λ per metric; the band lists every λ within seed noise of the best "
                    f"(tolerances from the P1 seed repeat: {seed_tol}).\n\n"]
    for track in ("B", "A"):
        for f in sorted({k[1] for k in g if k[0] == track}):
            keys = sorted([k for k in g if k[0] == track and k[1] == f], key=lambda k: (-k[2], k[3]))
            if not keys:
                continue
            fig, ax = plt.subplots(2, 3, figsize=(17, 9))
            for k in keys:
                runs = g[k]
                if len(runs) < 3:
                    continue
                lams = sorted(runs)
                ls = "-" if k[3] == 0 else "--"
                for a_, (mk, mlab, sgn) in zip(ax.ravel(), METRICS):
                    ys = [metric(runs[l], mk) for l in lams]
                    if any(y is None for y in ys):
                        continue
                    a_.plot([xlam(l) for l in lams], ys, ls, marker="o", ms=3, label=f"{k[2]}° from {k[3]}°")
                    a_.set_xscale("log"); a_.set_title(mlab); a_.set_xlabel("λ_TV (λ=0 at 1e-4)"); a_.grid(alpha=.3)
                r = dict(track=track, noise=f, span=k[2], start=k[3], n_lambda=len(runs))
                for mk, _, sgn in METRICS:
                    cur = {l: metric(runs[l], mk) for l in lams if metric(runs[l], mk) is not None}
                    if cur:
                        b, band = best_band(cur, sgn, seed_tol.get(mk, 0))
                        r[f"best_{mk}"], r[f"band_{mk}"], r[f"val_{mk}"] = b, band, cur[b]
                        r[f"at005_{mk}"] = cur.get(0.05)
                rows.append(r)
            ax[0, 0].legend(fontsize=8)
            fig.suptitle(f"R²-Gaussian, Track {track}, noise {100 * f:g}% (solid: start 0°, dashed: start 90°)")
            fig.savefig(out / f"r2_sweep_{track}_f{f:g}.png", dpi=85, bbox_inches="tight")
            plt.close(fig)
    md.append("| track | noise | arc | runs | best λ (worm PSNR) [band] | PSNR best / at 0.05 | best λ (Dice) | "
              "best λ (interior CV) | best λ (|bias|) | best λ (withheld err) |\n|" + "---|" * 10 + "\n")
    for r in rows:
        def b(mk):
            return f"{r.get('best_' + mk, '—')} {r.get('band_' + mk, '')}"
        md.append(f"| {r['track']} | {100 * r['noise']:g}% | {r['span']}° from {r['start']}° | {r['n_lambda']} | "
                  f"{b('worm_psnr_db')} | {r.get('val_worm_psnr_db', float('nan')):.2f} / "
                  f"{(r.get('at005_worm_psnr_db') or float('nan')):.2f} | {b('dice_class_mean')} | "
                  f"{b('interior_cv_pct_wmean')} | {b('abs_rel_bias_pct_wmean')} | {b('withheld_pct')} |\n")
    # best lambda vs arc span
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for track, a_ in zip(("B", "A"), ax):
        for f in sorted({r["noise"] for r in rows if r["track"] == track}):
            for start, mk in ((0, "o"), (90, "s")):
                rr = sorted([r for r in rows if r["track"] == track and r["noise"] == f and r["start"] == start
                             and "best_worm_psnr_db" in r], key=lambda r: r["span"])
                if rr:
                    a_.plot([r["span"] for r in rr], [xlam(r["best_worm_psnr_db"]) for r in rr], marker=mk,
                            label=f"noise {100 * f:g}%, start {start}°")
        a_.set_yscale("log"); a_.set_xlabel("arc span (deg)"); a_.set_ylabel("best λ_TV (worm PSNR)")
        a_.set_title(f"Track {track}"); a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.savefig(out / "r2_best_lambda_vs_span.png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    c.write_json(out / "r2_sweep_table.json", rows)
    return "".join(md)


# ----------------------------------------------------------------------- classical sweep
def l_curve_corner(res, tv, lams):
    pts = [(np.log(r), np.log(t), l) for r, t, l in zip(res, tv, lams) if l > 0 and r > 0 and t > 0]
    if len(pts) < 3:
        return None
    x, y, ls = map(np.array, zip(*pts))
    k = []
    for i in range(1, len(x) - 1):
        a, b, cc = np.array([x[i - 1], y[i - 1]]), np.array([x[i], y[i]]), np.array([x[i + 1], y[i + 1]])
        ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(cc - b), np.linalg.norm(a - cc)
        cross = (b - a)[0] * (cc - b)[1] - (b - a)[1] * (cc - b)[0]
        k.append(2 * cross / max(ab * bc * ca, 1e-12))
    return float(ls[1 + int(np.argmax(np.abs(k)))])


def analyze_tv2d(d, out, key="worm_psnr_db"):
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(Path(d) / "*.json")))]
    if not recs:
        return "No classical sweep results yet.\n"
    sign = dict((m, s) for m, _, s in METRICS)[key]
    cases = []
    for r in recs:
        lams = [p["lam"] for p in r["per_lambda"]]
        val = np.array([metric(p, key) for p in r["per_lambda"]], float)
        res = np.array([p["residual_ms"] for p in r["per_lambda"]])
        tv = np.array([p["tv"] for p in r["per_lambda"]])
        cv = np.array([p["cv_heldout_ms"] for p in r["per_lambda"]])
        ib = int(np.argmax(sign * val))
        sel = {"oracle": lams[ib]}
        sh = r.get("sigma_hat_neg", r["sigma_hat"])  # negative-value estimator when available
        s2 = sh ** 2
        ok = [l for l, rr in zip(lams, res) if rr <= s2]
        sel["discrepancy"] = max(ok) if ok and r["noise_fraction"] > 0 else min([l for l in lams if l > 0])
        sel["cross_validation"] = lams[int(np.argmin(cv))]
        sel["l_curve"] = l_curve_corner(res, tv, lams) or lams[ib]
        cases.append(dict(worm=r["worm"], span=int(r["span"]), start=int(r["start"]), f=r["noise_fraction"],
                          sigma_hat=sh, sigma=r["noise_sigma"], lams=lams, val=val.tolist(), sel=sel))
    dev = [k for k in cases if k["worm"] in DEV]
    test = [k for k in cases if k["worm"] in TEST]
    # fixed lambda: best average metric over development cases
    grid = sorted(set.intersection(*[set(k["lams"]) for k in dev])) if dev else []
    if dev:
        avg = [np.mean([k["val"][k["lams"].index(l)] for k in dev]) * sign for l in grid]
        lam_fixed = grid[int(np.argmax(avg))]
        # angle/noise-aware rule: least squares log(lambda*) ~ a + b*log(missing fraction+0.05) + c*log(sigma_hat+eps)
        X, y = [], []
        for k in dev:
            lo = k["sel"]["oracle"]
            X.append([1.0, np.log(1 - k["span"] / 180 + 0.05), np.log(k["sigma_hat"] + 1e-3)])
            y.append(np.log(xlam(lo)))
        coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    for k in cases:
        if dev:
            k["sel"]["fixed_dev_best"] = lam_fixed
            lr = np.exp(coef @ [1.0, np.log(1 - k["span"] / 180 + 0.05), np.log(k["sigma_hat"] + 1e-3)])
            k["sel"]["angle_noise_rule"] = min(k["lams"], key=lambda l: abs(np.log(xlam(l)) - np.log(lr)))
        best = max(k["val"]) if sign > 0 else min(k["val"])
        k["regret"] = {m: sign * (best - k["val"][k["lams"].index(l)]) for m, l in k["sel"].items()}
    rules = ["oracle", "fixed_dev_best", "angle_noise_rule", "discrepancy", "cross_validation", "l_curve"]
    md = [f"# Classical TV sweep and automatic λ selection (metric: {key})\n\n",
          f"Development poses {DEV}, test poses {TEST}; regret = best achievable − achieved "
          f"(in {key} units; 0 is perfect). Rules fitted only on development poses.\n\n"]
    if dev:
        md.append(f"Fixed λ from development poses: **{lam_fixed}**. Angle/noise rule: log λ = "
                  f"{coef[0]:.3f} + {coef[1]:.3f}·log(missing fraction + 0.05) + {coef[2]:.3f}·log(σ̂ + 0.001).\n\n")
    md.append("| test subset | n | " + " | ".join(rules) + " |\n|" + "---|" * (len(rules) + 2) + "\n")
    for name, sub in [("all test", test)] + [(f"noise {100 * f:g}%", [k for k in test if k["f"] == f])
                                              for f in sorted({k["f"] for k in test})]:
        if sub:
            md.append(f"| {name} | {len(sub)} | " + " | ".join(
                f"{np.mean([k['regret'][r] for k in sub if r in k['regret']]):.3f}" for r in rules) + " |\n")
    # best lambda vs span
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for f in sorted({k["f"] for k in cases}):
        for start, mk in ((0, "o"), (90, "s")):
            spans = sorted({k["span"] for k in cases if k["start"] == start})
            med = [np.median([xlam(k["sel"]["oracle"]) for k in cases if k["f"] == f and k["span"] == s
                              and k["start"] == start]) for s in spans]
            ax[0].plot(spans, med, marker=mk, label=f"noise {100 * f:g}%, start {start}°")
    ax[0].set_yscale("log"); ax[0].set_xlabel("arc span (deg)"); ax[0].set_ylabel("best λ (median over 8 poses)")
    ax[0].set_title("classical TV: best λ vs coverage"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    sh = [(k["sigma"], k["sigma_hat"]) for k in cases]
    ax[1].plot(*zip(*sh), "k.")
    ax[1].plot([0, max(s for s, _ in sh) or 1], [0, max(s for s, _ in sh) or 1], "r--", lw=0.8)
    ax[1].set_xlabel("true noise σ"); ax[1].set_ylabel("σ̂ estimated from the noisy sinogram"); ax[1].grid(alpha=.3)
    ax[1].set_title("noise-level estimate (no truth needed)")
    fig.savefig(out / f"tv2d_best_lambda_{key}.png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    c.write_json(out / f"tv2d_selection_{key}.json", cases)
    return "".join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r2", type=Path)
    ap.add_argument("--tv2d", type=Path)
    ap.add_argument("--p1-tables", type=Path, help="p1_tables.json (seed-noise tolerances)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    md = []
    if a.r2:
        tol = {}
        if a.p1_tables and a.p1_tables.exists():
            t = json.loads(a.p1_tables.read_text()); sn = t.get("B_seed_noise") or t.get("A_seed_noise", {})
            tol = {k: sn[k] for k in ("worm_psnr_db", "dice_class_mean", "compartment_bf1_1um",
                                      "interior_cv_pct_wmean", "abs_rel_bias_pct_wmean", "withheld_pct") if k in sn}
        md.append(analyze_r2(a.r2, a.out, tol))
    if a.tv2d:
        for key in ("worm_psnr_db", "dice_class_mean", "interior_cv_pct_wmean"):
            md.append(analyze_tv2d(a.tv2d, a.out, key))
    (a.out / "sweep_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
