"""LaTeX tables and number macros for the worm TV report, generated from report_numbers.json."""

import argparse
import json
from pathlib import Path


def f(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--numbers", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sweep-summary", type=Path, help="analysis/sweep_tv2d/sweep_summary.md (rule coefficients)")
    a = ap.parse_args()
    N = json.loads(a.numbers.read_text())
    o = a.out
    o.mkdir(parents=True, exist_ok=True)
    M = {}

    # ---- replay table
    rp = N["replay"]
    rows = [("180_0", r"$180^\circ$ (full)"), ("120_0", r"$120^\circ$ from $0^\circ$"), ("120_90", r"$120^\circ$ from $90^\circ$"),
            ("90_0", r"$90^\circ$ from $0^\circ$"), ("90_90", r"$90^\circ$ from $90^\circ$")]
    t = [r"\begin{tabular}{lcccccccc}\toprule",
         r"Scan & Views & PSNR (dB) & Dice & Outline F1 & Surface dist. (\textmu m) & Interior CV (\%) & $|$Bias$|$ (\%) & Unmeasured-view error (\%)\\\midrule"]
    for k, lab in rows:
        r = rp[k]
        t.append(f"{lab} & {r['views']} & {f(r['psnr'])} & {f(r['dice_worm'],3)} & {f(r['f1'],3)} & {f(r['msd'])} & "
                 f"{f(r['cv'],1)} & {f(r['bias'],1)} & {f(r['withheld'])}\\\\")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_replay.tex").write_text("\n".join(t) + "\n")

    # ---- pilot TV table
    w = N["pilot_tv"]
    t = [r"\begin{tabular}{llcccccc}\toprule",
         r"Start & $\lambda_{\mathrm{TV}}$ & Volume PSNR (dB) & SSIM & Boundary MAE & Worm PSNR (dB) & Interior CV (\%) & Unmeasured-view error (\%)\\\midrule"]
    for st in (0, 90):
        for l in ("0.025", "0.05", "0.1"):
            r = w[f"{st}_{l}"]
            t.append(f"${st}^\\circ$ & {l} & {f(r['native_psnr'],3)} & {f(r['native_ssim'],4)} & {f(r['bmae'],4)} & "
                     f"{f(r['psnr'])} & {f(r['cv'],2)} & {f(r['withheld'])}\\\\")
        if st == 0:
            t.append(r"\midrule")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_pilot.tex").write_text("\n".join(t) + "\n")

    # ---- R2 sweep table
    sw = N["r2_sweep"]
    lams = [0, 0.025, 0.05, 0.1, 0.2, 0.5]
    t = [r"\begin{tabular}{l" + "c" * len(lams) + "c}\\toprule",
         "Scan & " + " & ".join(f"$\\lambda={l:g}$" for l in lams) + r" & Best $\lambda$\\\midrule"]
    for s in ("180", "120", "90", "60"):
        rr = {r["lam"]: r for r in sw[s]}
        best = max(rr, key=lambda l: rr[l]["psnr"])
        cells = [("\\textbf{" + f(rr[l]["psnr"]) + "}") if l == best else f(rr[l]["psnr"]) for l in lams]
        t.append(f"${s}^\\circ$ & " + " & ".join(cells) + f" & {best:g}\\\\")
    t.append(r"\midrule \multicolumn{8}{l}{\emph{Interior variation CV (\%)}}\\")
    for s in ("180", "120", "90", "60"):
        rr = {r["lam"]: r for r in sw[s]}
        best = min(rr, key=lambda l: rr[l]["cv"])
        cells = [("\\textbf{" + f(rr[l]["cv"], 1) + "}") if l == best else f(rr[l]["cv"], 1) for l in lams]
        t.append(f"${s}^\\circ$ & " + " & ".join(cells) + f" & {best:g}\\\\")
    t.append(r"\midrule \multicolumn{8}{l}{\emph{Unmeasured-view error (\%)}}\\")
    for s in ("120", "90", "60"):
        rr = {r["lam"]: r for r in sw[s]}
        best = min(rr, key=lambda l: rr[l]["withheld"])
        cells = [("\\textbf{" + f(rr[l]["withheld"]) + "}") if l == best else f(rr[l]["withheld"]) for l in lams]
        t.append(f"${s}^\\circ$ & " + " & ".join(cells) + f" & {best:g}\\\\")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_r2sweep.tex").write_text("\n".join(t) + "\n")

    # ---- classical best lambda table (no noise)
    cp = N["classical_parallel"]
    t = [r"\begin{tabular}{lccccccc}\toprule",
         r"Scan & Best $\lambda$ & PSNR at best & PSNR at $\lambda=0.05$ & PSNR, no TV & CV at best (\%) & CV, no TV (\%) & Best $\lambda$ at 3\% noise\\\midrule"]
    for s in ("180", "150", "120", "90", "60"):
        r = cp[f"{s}_0_0"]
        r3 = cp.get(f"{s}_0_0.03")
        t.append(f"${s}^\\circ$ & {r['best']:g} & {f(r['psnr_best'])} & {f(r['psnr_005'])} & {f(r['psnr_0'])} & "
                 f"{f(r['cv_best'],1)} & {f(r['cv0'],1)} & {r3['best']:g}\\\\" if r3 else "")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_classical.tex").write_text("\n".join(t) + "\n")

    # ---- criteria table
    t = [r"\begin{tabular}{lcccc}\toprule",
         r"Scan & Overall accuracy (PSNR) & Part overlap (Dice) & Uniform interior (CV) & Correct density (bias)\\\midrule"]
    for s in ("180", "150", "120", "90", "60"):
        cr = cp[f"{s}_0_0"]["crit"]
        t.append(f"${s}^\\circ$ & {cr['worm_psnr_db']:g} & {cr['dice_class_mean']:g} & {cr['interior_cv_pct_wmean']:g} & "
                 f"{cr['abs_rel_bias_pct_wmean']:.3g}\\\\")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_criteria.tex").write_text("\n".join(t) + "\n")

    # ---- selection table
    sel = N["selection_test"]
    names = [("angle_noise_rule", "Angle rule (calibrated once on tuning poses)", "scan angle"),
             ("fixed_dev_best", "One fixed $\\lambda$ (best on tuning poses)", "nothing"),
             ("cross_validation", "Cross-validation on held-out measured views", "nothing (2$\\times$ compute)"),
             ("l_curve", "L-curve corner", "nothing"),
             ("discrepancy", "Discrepancy principle", "noise level")]
    t = [r"\begin{tabular}{llcccc}\toprule", r"Rule & Needs & All & No noise & 1\% & 3\%\\\midrule"]
    for k, lab, need in names:
        t.append(f"{lab} & {need} & {f(sel['all'][k])} & {f(sel['by_noise']['0'][k])} & {f(sel['by_noise']['0.01'][k])} & "
                 f"{f(sel['by_noise']['0.03'][k])}\\\\")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_selection.tex").write_text("\n".join(t) + "\n")

    # ---- fan tables
    cf = N["classical_fan"]
    t = [r"\begin{tabular}{lccc}\toprule", r"Scan & Best $\lambda$, parallel beam & Best $\lambda$, fan beam & PSNR at best, parallel / fan (dB)\\\midrule"]
    for s in ("120", "90", "60"):
        p, q = cp[f"{s}_0_0.01"], cf[f"{s}_0_0.01"]
        t.append(f"${s}^\\circ$ & {p['best']:g} & {q['best']:g} & {f(p['psnr_best'])} / {f(q['psnr_best'])}\\\\")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_fan_best.tex").write_text("\n".join(t) + "\n")
    ft = N["fan_transfer"]["regret"]
    t = [r"\begin{tabular}{lc}\toprule", r"Rule applied to fan-beam scans & PSNR lost (dB)\\\midrule"]
    for k, lab in (("parallel angle rule", "Angle rule from parallel beam, unchanged"), ("parallel fixed λ", "Fixed $\\lambda$ from parallel beam, unchanged"),
                   ("fan-native fixed λ", "Fixed $\\lambda$ re-tuned on fan scans"), ("L-curve", "L-curve"), ("discrepancy", "Discrepancy principle")):
        t.append(f"{lab} & {f(ft[k])}\\\\")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_fan_transfer.tex").write_text("\n".join(t) + "\n")

    # ---- comparison table
    rows = N["p4"]
    def pick(m, s, n):
        return next((r for r in rows if r["method"].startswith(m) and r["span"] == s and r["noise_pct"] == n), None)
    t = [r"\begin{tabular}{llcccc}\toprule",
         r" & & \multicolumn{2}{c}{No noise} & \multicolumn{2}{c}{5\% noise}\\\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Method & Operator & $120^\circ$ & $90^\circ$ & $120^\circ$ & $90^\circ$\\\midrule"]
    for m, lab, op in (("FBP (re-run)", "FBP", "matched"), ("SART (re-run)", "SART, 10 sweeps", "matched"),
                       ("Gaussian slab", "Gaussian fit, uncertainty study$^\\dagger$", "own renderer"),
                       ("R²-Gaussian", "R\\textsuperscript{2}-Gaussian + TV 0.05", "independent")):
        cells = []
        for n in (0, 5):
            for s in (120, 90):
                r = pick(m, s, n)
                cells.append(f(r["skimage_withheld"]) if r and r.get("skimage_withheld") is not None else "--")
        t.append(f"{lab} & {op} & " + " & ".join(cells) + "\\\\")
    t.append(r"\midrule \multicolumn{6}{l}{\emph{Independent TIGRE operator}}\\")
    for m, lab in (("SART (re-run)", "SART, 10 sweeps"), ("R²-Gaussian", "R\\textsuperscript{2}-Gaussian + TV 0.05")):
        cells = []
        for n in (0, 5):
            for s in (120, 90):
                r = pick(m, s, n)
                cells.append(f(r.get("tigre_withheld")) if r else "--")
        t.append(f"{lab} & independent & " + " & ".join(cells) + "\\\\")
    t.append(r"\bottomrule\end{tabular}")
    (o / "tab_comparison.tex").write_text("\n".join(t) + "\n")

    # ---- inline macros
    r2b = N["r2_best"]
    M["phantomfit"] = f"{100 * N['phantom_fit_rel_l2']:.1f}"
    M["phantomfitmin"] = f"{100 * min(N['phantom_fit_all']):.1f}"
    M["phantomfitmax"] = f"{100 * max(N['phantom_fit_all']):.1f}"
    M["sirtfit"] = f"{100 * N['sirt_rel_l2']:.1f}"
    M["seedpsnr"] = f"{N['seed_noise']['psnr']:.2f}"
    M["seedwithheld"] = f"{N['seed_noise']['withheld']:.2f}"
    M["regangle"] = f(sel["all"]["angle_noise_rule"])
    M["regfixed"] = f(sel["all"]["fixed_dev_best"])
    M["regcv"] = f(sel["all"]["cross_validation"])
    M["reglcurve"] = f(sel["all"]["l_curve"])
    M["regdisc"] = f(sel["all"]["discrepancy"])
    M["ntest"] = str(sel["n"])
    M["fanangle"] = f(ft["parallel angle rule"])
    M["fanfixed"] = f(ft["parallel fixed λ"])
    M["nfan"] = str(N["fan_transfer"]["n"])
    for s in ("180", "120", "90", "60"):
        M[f"rtwobest{ {'180': 'A', '120': 'B', '90': 'C', '60': 'D'}[s] }"] = f"{r2b[s]:g}"
    if a.sweep_summary and a.sweep_summary.exists():
        import re
        m = re.search(r"log λ = (-?[0-9.]+) \+ (-?[0-9.]+)·log\(missing", a.sweep_summary.read_text())
        if m:
            M["rulea"], M["ruleb"] = f"{float(m.group(1)):.2f}", f"{float(m.group(2)):.2f}"
    tex = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items())
    (o / "numbers.tex").write_text(tex + "\n")

    # ---- Markdown snippets and plain values for the explanation document
    md = {}
    t = ["| Scan | Best λ | PSNR at best | PSNR at 0.05 | PSNR without TV | Interior CV at best / without TV |", "|---|---:|---:|---:|---:|---:|"]
    for s_ in ("180", "150", "120", "90", "60"):
        r = cp[f"{s_}_0_0"]
        t.append(f"| {s_}° | {r['best']:g} | {f(r['psnr_best'])} | {f(r['psnr_005'])} | {f(r['psnr_0'])} | {f(r['cv_best'],1)}% / {f(r['cv0'],1)}% |")
    md["md_classical"] = "\n".join(t)
    t = ["| Scan | Overall (PSNR) | Part overlap (Dice) | Uniform insides (CV) | Correct density (bias) |", "|---|---:|---:|---:|---:|"]
    for s_ in ("180", "120", "90", "60"):
        cr = cp[f"{s_}_0_0"]["crit"]
        t.append(f"| {s_}° | {cr['worm_psnr_db']:g} | {cr['dice_class_mean']:g} | {cr['interior_cv_pct_wmean']:.2g} | {cr['abs_rel_bias_pct_wmean']:.2g} |")
    md["md_criteria"] = "\n".join(t)
    t = ["| Scan | Best λ, parallel | Best λ, fan | PSNR at best, parallel / fan |", "|---|---:|---:|---:|"]
    for s_ in ("120", "90", "60"):
        p_, q_ = cp[f"{s_}_0_0.01"], cf[f"{s_}_0_0.01"]
        t.append(f"| {s_}° | {p_['best']:g} | {q_['best']:g} | {f(p_['psnr_best'])} / {f(q_['psnr_best'])} dB |")
    md["md_fanbest"] = "\n".join(t)
    vals = dict(M)
    vals["ruleb_abs"] = vals.get("ruleb", "").lstrip("-")
    vals["cvgap"] = f(sel["all"]["cross_validation"] - sel["all"]["angle_noise_rule"])
    vals.update(md)
    (o / "explanation_values.json").write_text(json.dumps(vals, indent=1))
    print("tables written:", sorted(p.name for p in o.glob("*.tex")))


if __name__ == "__main__":
    main()
