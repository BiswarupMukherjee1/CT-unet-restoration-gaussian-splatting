"""Does TV tuning transfer from parallel beam to fan beam (2-D cone beam)?

Reads the parallel sweep (tv2d/) and the fan sweep (fan2d/). For each metric:
  * best lambda vs arc span, fan vs parallel (median over poses);
  * transfer test on fan scans of all 8 poses: rules fitted ONLY on parallel-beam
    development poses (fixed lambda; angle/noise rule) applied unchanged to fan scans,
    plus truth-free rules (discrepancy principle with the negative-value noise estimate,
    L-curve) and a fan-native fixed lambda (fitted on fan development poses) for reference.
For fan scans, the missing fraction uses the complete fan-beam range (180 deg + fan angle).
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as c
import fan2d
from analyze_sweep import DEV, TEST, l_curve_corner, metric, xlam

FULL_FAN = 180 + 2 * np.degrees(np.arctan(fan2d.S_DET / 2 / fan2d.DSD))  # ~212 deg


def load(d, full):
    cases = []
    for f in sorted(glob.glob(str(Path(d) / "*.json"))):
        r = json.loads(Path(f).read_text())
        cases.append(dict(r=r, worm=r["worm"], span=float(r["span"]), start=float(r["start"]),
                          f=r["noise_fraction"], miss=max(0.0, 1 - float(r["span"]) / full),
                          sh=r.get("sigma_hat_neg", r["sigma_hat"])))
    return cases


def vals(case, key):
    pl = case["r"]["per_lambda"]
    return [p["lam"] for p in pl], np.array([metric(p, key) for p in pl], float)


def fit_rules(cases, key, sign):
    dev = [k for k in cases if k["worm"] in DEV]
    grid = sorted(set.intersection(*[set(vals(k, key)[0]) for k in dev]))
    avg = [np.mean([vals(k, key)[1][vals(k, key)[0].index(l)] for k in dev]) * sign for l in grid]
    fixed = grid[int(np.argmax(avg))]
    X, y = [], []
    for k in dev:
        lams, v = vals(k, key)
        X.append([1.0, np.log(k["miss"] + 0.05), np.log(k["sh"] + 1e-3)])
        y.append(np.log(xlam(lams[int(np.argmax(sign * v))])))
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return fixed, coef


def regret(case, key, sign, lam):
    lams, v = vals(case, key)
    best = v.max() if sign > 0 else v.min()
    return float(sign * (best - v[lams.index(lam)]))


def nearest(lams, x):
    return min(lams, key=lambda l: abs(np.log(xlam(l)) - np.log(xlam(x))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parallel", type=Path, required=True)
    ap.add_argument("--fan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fan-arcs", default=None, help="e.g. 120:0,90:0,60:0 (restrict the fan cases analysed)")
    ap.add_argument("--fan-noise", type=float, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    par, fan = load(a.parallel, 180.0), load(a.fan, FULL_FAN)
    if a.fan_arcs:
        keep = {tuple(float(v) for v in x.split(":")) for x in a.fan_arcs.split(",")}
        fan = [k for k in fan if (k["span"], k["start"]) in keep and (a.fan_noise is None or k["f"] == a.fan_noise)]
    md = [f"# Parallel → fan-beam transfer of TV tuning (complete fan range {FULL_FAN:.1f}°)\n\n",
          f"Parallel cases: {len(par)}, fan cases: {len(fan)}. Regret = best achievable − achieved on each fan scan "
          "(0 is perfect), mean over all 8 poses.\n\n"]
    out_rows = []
    for key, label, sign in (("worm_psnr_db", "worm PSNR (dB)", +1), ("dice_class_mean", "Dice of classes", +1),
                             ("interior_cv_pct_wmean", "interior CV (%)", -1),
                             ("abs_rel_bias_pct_wmean", "|density bias| (%)", -1)):
        fixed_p, coef_p = fit_rules(par, key, sign)
        fixed_f, _ = fit_rules(fan, key, sign)
        rows = []
        for k in fan:
            lams, v = vals(k, key)
            pl = k["r"]["per_lambda"]
            res = np.array([p["residual_ms"] for p in pl])
            tv = np.array([p["tv"] for p in pl])
            sel = {"oracle": lams[int(np.argmax(sign * v))],
                   "parallel fixed λ": nearest(lams, fixed_p),
                   "parallel angle rule": nearest(lams, float(np.exp(coef_p @ [1.0, np.log(k["miss"] + 0.05),
                                                                              np.log(k["sh"] + 1e-3)]))),
                   "fan-native fixed λ": nearest(lams, fixed_f),
                   "L-curve": l_curve_corner(res, tv, lams) or lams[0]}
            ok = [l for l, r in zip(lams, res) if r <= k["sh"] ** 2]
            sel["discrepancy"] = max(ok) if (ok and k["f"] > 0) else None
            rows.append(dict(worm=k["worm"], span=k["span"], start=k["start"], f=k["f"],
                             **{f"lam:{n}": l for n, l in sel.items()},
                             **{f"regret:{n}": (regret(k, key, sign, l) if l is not None else None)
                                for n, l in sel.items()}))
        names = ["oracle", "parallel fixed λ", "parallel angle rule", "fan-native fixed λ", "discrepancy", "L-curve"]
        md.append(f"## {label}\n\nParallel-fitted fixed λ = {fixed_p}; fan-native fixed λ = {fixed_f}.\n\n"
                  "| subset | n | " + " | ".join(names) + " |\n|" + "---|" * (len(names) + 2) + "\n")
        for name, sub in [("all fan scans", rows)] + [(f"noise {100 * f:g}%", [r for r in rows if r["f"] == f])
                                                      for f in sorted({r["f"] for r in rows})]:
            cells = []
            for n in names:
                xs = [r[f"regret:{n}"] for r in sub if r[f"regret:{n}"] is not None]
                cells.append(f"{np.mean(xs):.3f}" if xs else "—")
            md.append(f"| {name} | {len(sub)} | " + " | ".join(cells) + " |\n")
        md.append("\n")
        out_rows.append(dict(metric=key, parallel_fixed=fixed_p, fan_fixed=fixed_f, rows=rows))
        # figure: best lambda vs span, fan vs parallel
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for cases, geo, mk in ((par, "parallel", "o"), (fan, "fan", "s")):
            for f in sorted({k["f"] for k in cases}):
                spans = sorted({k["span"] for k in cases if k["f"] == f})
                med = [np.median([xlam(vals(k, key)[0][int(np.argmax(sign * vals(k, key)[1]))])
                                  for k in cases if k["f"] == f and k["span"] == s]) for s in spans]
                ax.plot(spans, med, marker=mk, ls="-" if geo == "fan" else "--", label=f"{geo}, noise {100 * f:g}%")
        ax.axvline(180, color="0.6", lw=0.8); ax.axvline(FULL_FAN, color="0.6", lw=0.8, ls=":")
        ax.set_yscale("log"); ax.set_xlabel("arc span (deg)"); ax.set_ylabel(f"best λ ({label})")
        ax.set_title("best TV weight vs coverage: parallel vs fan beam"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.savefig(a.out / f"fan_vs_parallel_best_lambda_{key}.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
    c.write_json(a.out / "fan_transfer.json", out_rows)
    (a.out / "fan_transfer.md").write_text("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
