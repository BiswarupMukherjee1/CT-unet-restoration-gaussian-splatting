"""Fan-beam (2-D cone-beam) TV sweep on the Track B phantoms, CPU.

Question: does the best TV weight, and the angle rule fitted on parallel-beam data,
carry over to divergent-beam geometry? Same phantoms, slice block, solver, iteration
count, lambda grid and evaluator as the parallel sweep (tv_sweep2d.py); only the
geometry changes (fan2d.py). Measurements are simulated with fan2d.project_fine
(independent discretisation) and cached per phantom. Noise as in the parallel study
(sigma = f x 6.2101). Cross-validation refits are omitted (they were only needed for
the selection study, done on the parallel data).
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import common as c
import fan2d
import tv2d
from evaluate_run import evaluate
from tv_sweep2d import sigma_hat_neg, tv_value

_A = None


def matrix():
    global _A
    if _A is None:
        _A = fan2d.build_matrix()
    return _A


def rows_of(views):
    return (np.asarray(views)[:, None] * fan2d.N_BINS + np.arange(fan2d.N_BINS)).ravel()


def simulate(args):
    pdir, block = args
    out = Path(pdir) / f"sino_fan_clean_z{block[0]}_{block[1]}.npz"
    if out.exists():
        return str(out), "cached"
    truth = np.load(Path(pdir) / "phantom.npz")["truth"][block[0]:block[1]]
    P = fan2d.project_fine(tv2d.slab_to_cols(truth))  # (views, bins, z)
    np.savez_compressed(out, sino=np.transpose(P, (2, 0, 1)).astype(np.float32), angles_deg=fan2d.ANGLES_FAN,
                        geometry=json.dumps(dict(DSO=fan2d.DSO, DSD=fan2d.DSD, n_bins=fan2d.N_BINS,
                                                 s_det=fan2d.S_DET)))
    return str(out), "simulated"


def job(args):
    pdir, block, span, start, f, lams, iters, out = args
    out = Path(out)
    old = json.loads(out.read_text()) if out.exists() else None
    if old is not None:
        lams = [l for l in lams if l not in old["lams"]]
        if not lams:
            return str(out), "skip"
    t0 = time.time()
    prep = json.loads((Path(pdir) / "phantom_report.json").read_text())
    ph = np.load(Path(pdir) / "phantom.npz")
    z0, z1 = block
    truth, labels, pure = ph["truth"][z0:z1], ph["labels"][z0:z1], ph["pure"][z0:z1]
    clean = np.load(Path(pdir) / f"sino_fan_clean_z{z0}_{z1}.npz")["sino"].astype(np.float64)  # (z, view, bin)
    noisy = c.add_noise(clean, f, 0)
    A = matrix()
    meas = np.flatnonzero(fan2d.arc_mask_fan(start, span))
    held = np.flatnonzero(~fan2d.arc_mask_fan(start, span))
    Am = A[rows_of(meas)]
    Bm = np.ascontiguousarray(noisy[:, meas, :].reshape(noisy.shape[0], -1).T).astype(np.float32)
    Cv = np.transpose(clean, (1, 2, 0))  # (view, bin, z)
    rec = dict(geometry="fan", phantom=str(pdir), worm=prep["worm"], block=[z0, z1], span=span, start=start,
               noise_fraction=f, noise_sigma=f * c.NOISE_SCALE, sigma_hat_neg=sigma_hat_neg(noisy[:, meas, :]),
               iters=iters, n_measured=int(meas.size), lams=lams, per_lambda=[])
    rec["sigma_hat"] = rec["sigma_hat_neg"]
    edges = np.array(prep["class_edges_g_cm3"])
    for lam in lams:
        X = tv2d.tv_recon(Am, Bm, lam, iters=iters)
        m, _ = evaluate(tv2d.cols_to_slab(X), truth, labels, pure, edges, prep["class_density_g_cm3"])
        Pv = (A @ X).reshape(fan2d.N_VIEWS_FAN, fan2d.N_BINS, -1)
        rec["per_lambda"].append(dict(
            lam=lam, voxel=m, proj_rel_l2_measured=c.rel_l2(Pv[meas], Cv[meas]),
            proj_rel_l2_withheld=c.rel_l2(Pv[held], Cv[held]) if held.size else None,
            residual_ms=float(((Am @ X - Bm) ** 2).mean()), tv=tv_value(X)))
    rec["seconds"] = time.time() - t0
    if old is not None:
        old["per_lambda"] = sorted(old["per_lambda"] + rec["per_lambda"], key=lambda r: r["lam"])
        old["lams"] = [r["lam"] for r in old["per_lambda"]]
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
    ap.add_argument("--arcs", default="360:0,212:0,180:0,180:90,120:0,120:90,90:0,90:90,60:0")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.01])
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pdirs = [str(a.phantoms / f"worm{w:04d}_{a.slab}") for w in a.worms]
    with ProcessPoolExecutor(min(a.workers, len(pdirs))) as ex:
        for out, msg in ex.map(simulate, [(p, tuple(a.block)) for p in pdirs]):
            print(time.strftime("%F %T"), out, msg, flush=True)
    arcs = [tuple(float(v) for v in s.split(":")) for s in a.arcs.split(",")]
    jobs = []
    for f in a.noise:
        for p, w in zip(pdirs, a.worms):
            for span, start in arcs:
                out = a.out / f"fan_w{w:04d}_{a.slab}_span{int(span)}_start{int(start)}_f{f:g}.json"
                jobs.append((p, tuple(a.block), span, start, f, a.lams, a.iters, str(out)))
    with ProcessPoolExecutor(a.workers) as ex:
        for out, msg in ex.map(job, jobs):
            print(time.strftime("%F %T"), out, msg, flush=True)


if __name__ == "__main__":
    main()
