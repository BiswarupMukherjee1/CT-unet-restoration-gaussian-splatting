"""Faithful conversion of a starting volume into Gaussians (pre-fit).

The native rule (50,000 voxels drawn above 0.05 g/cm^3, density x 0.15, size from the
nearest neighbours) keeps where the object is but not what it looks like: rendered back,
a 25 dB starting volume becomes a 14 dB one. Here the same 50,000 initial Gaussians are
first fitted to the starting volume itself (L1 between R2-Gaussian's voxelised field and
the starting volume, no densification), then saved as an iteration-0 checkpoint with a
fresh optimiser. Projection training then starts from that checkpoint with the usual
settings. Only measured data enter the starting volume, so no extra information is used.

  python prefit_gaussians.py CASE --method unet [--iters 3000]
writes CASE/init_<method>_fit.ckpt and adds fit_* scores to CASE/init_<method>_fit_diag.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import init_lib as L
from evaluate_run import projection_metrics  # noqa: E402  (worm_tv is on the path via init_lib)

REPO = L.HERE.parents[1]


def prefit(cdir, method, iters=3000, verbose=True):
    os.chdir(REPO); sys.path.insert(0, str(REPO))
    import torch
    from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
    from r2_gaussian.dataset import Scene
    from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query

    cdir = Path(cdir)
    pf = cdir / ("init.npy" if method == "fdk" else f"init_{method}.npy")
    p = argparse.ArgumentParser(); lp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    args = p.parse_args([]); args.source_path = str(cdir); args.ply_path = str(pf)
    ma, oa, pa = lp.extract(args), op.extract(args), pp.extract(args)
    sc = Scene(ma, False).scanner_cfg
    vtw = max(sc["sVoxel"])
    g = GaussianModel(np.array([ma.scale_min, ma.scale_max]) * vtw)
    initialize_gaussian(g, ma, None)
    g.training_setup(oa)
    target = np.load(cdir / f"initvol_{method}.npz")["vol"].astype(np.float32)  # (z, y, x)
    tgt = torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 1, 0)))).cuda()  # (x, y, z)
    t0, log = time.time(), []
    for it in range(1, iters + 1):
        vol = query(g, sc["offOrigin"], sc["nVoxel"], sc["sVoxel"], pa)["vol"]
        loss = (vol - tgt).abs().mean()
        g.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        g.optimizer.step()
        if it % 500 == 0 or it == 1:
            log.append((it, float(loss)))
            if verbose:
                print(f"  prefit {it} L1 {float(loss):.5f}", flush=True)
    with torch.no_grad():
        vol = query(g, sc["offOrigin"], sc["nVoxel"], sc["sVoxel"], pa)["vol"].cpu().numpy()
    g.training_setup(oa)  # fresh optimiser state for projection training
    ckpt = cdir / f"init_{method}_fit.ckpt"
    torch.save((g.capture(), 0), ckpt)
    # scores of the fitted start (same definitions as the other start diagnostics)
    rec = json.loads((cdir / "case_record.json").read_text())
    stack, _ = L.c.load_stack(rec["worm"])
    z0, z1 = rec["z_range"]
    gt = np.load(cdir / "vol_gt.npy")
    err = (vol - gt)[gt > 0]
    pm = projection_metrics(np.ascontiguousarray(np.transpose(vol, (2, 1, 0))), stack[z0:z1], stack[z0:z1],
                            rec["measured_view_indices"])
    base = json.loads((cdir / f"init_{method}_diag.json").read_text())
    d = dict(base, method=f"{method}_fit", prefit_iters=iters, prefit_seconds=time.time() - t0, prefit_log=log,
             fit_worm_psnr_db=float(10 * np.log10(gt.max() ** 2 / (err**2).mean())),
             fit_measured_err=pm["proj_rel_l2_measured"], fit_hidden_err=pm["proj_rel_l2_withheld"],
             fit_mass_ratio=float(vol.sum() / gt.sum()))
    L.c.write_json(cdir / f"init_{method}_fit_diag.json", d)
    print(cdir.name, method, f"start {base['init_worm_psnr_db']:.2f} dB / hidden {100 * base['start_hidden_err']:.2f}% | "
          f"native {base['gauss_worm_psnr_db']:.2f} dB / {100 * base['gauss_hidden_err']:.2f}% | "
          f"fitted {d['fit_worm_psnr_db']:.2f} dB / {100 * d['fit_hidden_err']:.2f}% "
          f"(mass x{d['fit_mass_ratio']:.2f}, {d['prefit_seconds']:.0f}s)", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", type=Path)
    ap.add_argument("--method", nargs="+", default=["unet"])
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    for cdir in a.cases:
        for m in a.method:
            if not (cdir / f"init_{m}_fit.ckpt").exists():
                prefit(cdir, m, a.iters)


if __name__ == "__main__":
    main()
