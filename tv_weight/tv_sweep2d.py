"""Dense classical TV sweep on Track B phantoms, recording what automatic TV rules need.

For each (pose, arc, noise) and each lambda:
  * full fit on all measured views -> truth-based metrics (same evaluator as the R2 runs,
    on a contiguous block of slices), withheld-view error, residual to the noisy data,
    TV value of the reconstruction;
  * cross-validation fit with every 5th measured view held out -> error on those views
    (computed against the *noisy* held-out data, as would be possible on a real scan).
The noise level is also estimated from the noisy sinogram alone (robust median absolute
deviation of detector-neighbour differences; most of the sinogram is air).
One JSON per (pose, arc, noise); selection rules are applied afterwards (analysis).
Re-running with more lambda values or arcs extends existing JSONs instead of redoing them.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common as c
import tv2d
from evaluate_run import evaluate

_A = None


def matrix():
    global _A
    if _A is None:
        _A = tv2d.build_matrix(c.ANGLES_DEG)
    return _A


def rows_of(views):
    return (np.asarray(views)[:, None] * c.N_DET + np.arange(c.N_DET)).ravel()


def tv_value(X):
    g = tv2d.grad(X, c.N_DET)
    return float(np.sqrt((g**2).sum(0)).sum())


def sigma_hat(noisy_meas):
    d = np.diff(noisy_meas, axis=-1).ravel()
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2))


def sigma_hat_neg(noisy_meas):
    """Noise level from negative values only. The clean sinogram is >= 0 and exactly 0 in
    air, so negative values come (almost) only from noise on air rays; for those, the mean
    of b^2 over b < 0 equals sigma^2 (half-normal). Object rays (values >> sigma) rarely
    go negative, so they barely contaminate the estimate."""
    neg = noisy_meas[noisy_meas < 0]
    return float(np.sqrt((neg**2).mean())) if neg.size else 0.0


def job(args):
    pdir, block, span, start, f, lams, iters, out = args
    out = Path(out)
    old = json.loads(out.read_text()) if out.exists() else None
    if old is not None:
        lams = [l for l in lams if l not in old["lams"]]
        if not lams and "sigma_hat_neg" in old:
            return str(out), "skip"
    t0 = time.time()
    prep = json.loads((Path(pdir) / "phantom_report.json").read_text())
    ph = np.load(Path(pdir) / "phantom.npz")
    z0, z1 = block
    truth, labels, pure = ph["truth"][z0:z1], ph["labels"][z0:z1], ph["pure"][z0:z1]
    clean = np.load(Path(pdir) / "sino_sim_clean.npz")["sino"][z0:z1].astype(np.float64)
    noisy = c.add_noise(clean, f, 0)
    A = matrix()
    meas = np.flatnonzero(c.arc_mask(start, span))
    held = np.flatnonzero(~c.arc_mask(start, span))
    cv_out = meas[2::5]
    cv_in = np.setdiff1d(meas, cv_out)
    Am, Ain, Aout = A[rows_of(meas)], A[rows_of(cv_in)], A[rows_of(cv_out)]
    Bm = tv2d.sino_to_cols(noisy, meas)
    Bin = tv2d.sino_to_cols(noisy, cv_in)
    Bout = tv2d.sino_to_cols(noisy, cv_out)
    Cfull = tv2d.sino_to_cols(clean, np.arange(c.N_VIEWS))
    rec = dict(phantom=str(pdir), worm=prep["worm"], block=[z0, z1], span=span, start=start, noise_fraction=f,
               noise_sigma=f * c.NOISE_SCALE, sigma_hat=sigma_hat(noisy[:, meas, :]), iters=iters,
               n_measured=int(meas.size), n_cv_heldout=int(cv_out.size), lams=lams, per_lambda=[])
    rec["sigma_hat_neg"] = sigma_hat_neg(noisy[:, meas, :])
    edges = np.array(prep["class_edges_g_cm3"])
    for lam in lams:
        X = tv2d.tv_recon(Am, Bm, lam, iters=iters)
        vol = tv2d.cols_to_slab(X)
        m, _ = evaluate(vol, truth, labels, pure, edges, prep["class_density_g_cm3"])
        P = A @ X
        Pv = P.reshape(c.N_VIEWS, c.N_DET, -1)
        Cv = Cfull.reshape(c.N_VIEWS, c.N_DET, -1)
        r = dict(lam=lam, voxel=m,
                 proj_rel_l2_measured=c.rel_l2(Pv[meas], Cv[meas]),
                 proj_rel_l2_withheld=c.rel_l2(Pv[held], Cv[held]) if held.size else None,
                 residual_ms=float(((Am @ X - Bm) ** 2).mean()), tv=tv_value(X))
        Xcv = tv2d.tv_recon(Ain, Bin, lam, iters=iters)
        r["cv_heldout_ms"] = float(((Aout @ Xcv - Bout) ** 2).mean())
        rec["per_lambda"].append(r)
    rec["seconds"] = time.time() - t0
    if old is not None:  # extend an existing result with the new lambda values
        old["per_lambda"] = sorted(old["per_lambda"] + rec["per_lambda"], key=lambda r: r["lam"])
        old["lams"] = [r["lam"] for r in old["per_lambda"]]
        old["sigma_hat_neg"] = rec["sigma_hat_neg"]
        old["seconds"] = old.get("seconds", 0) + rec["seconds"]
        rec = old
    out.parent.mkdir(parents=True, exist_ok=True)
    c.write_json(out, rec)
    return str(out), f"{rec['seconds']:.0f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phantoms", type=Path, required=True)
    ap.add_argument("--worms", type=int, nargs="+", required=True)
    ap.add_argument("--slab", default="mid")
    ap.add_argument("--block", type=int, nargs=2, default=[28, 36])
    ap.add_argument("--arcs", default="180:0,150:0,150:90,120:0,120:90,90:0,90:90,60:0,60:90")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.01, 0.03])
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arcs = [tuple(float(v) for v in s.split(":")) for s in a.arcs.split(",")]
    jobs = []
    for f in a.noise:  # noisy cases first within each pose ordering below
        for w in a.worms:
            for span, start in arcs:
                pdir = a.phantoms / f"worm{w:04d}_{a.slab}"
                out = a.out / f"w{w:04d}_{a.slab}_span{int(span)}_start{int(start)}_f{f:g}.json"
                jobs.append((str(pdir), tuple(a.block), span, start, f, a.lams, a.iters, str(out)))
    with ProcessPoolExecutor(a.workers) as ex:
        for out, msg in ex.map(job, jobs):
            print(time.strftime("%F %T"), out, msg, flush=True)


if __name__ == "__main__":
    main()
