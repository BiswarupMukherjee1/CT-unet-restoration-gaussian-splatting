"""Score the starting point *as Gaussians*, before any training.

The starting volume is turned into 50,000 Gaussians by the native rule. This script
renders those initial Gaussians back into a volume with R2-Gaussian's own voxelizer and
scores it like a finished run (worm PSNR against the reference, hidden-view error against
the supplied data). Comparing with the starting volume (score_starts.py) and with the final
run separates what is lost in the conversion from what changes during training.
Adds gauss_* keys to init_<method>_diag.json.
"""

import argparse
import json
import os
import sys

import numpy as np

import init_lib as L
from evaluate_run import projection_metrics  # noqa: E402  (worm_tv is on the path via init_lib)
from prepare_cases import ROOT, SPANS, TEST_WORMS
from score_starts import METHODS

REPO = L.HERE.parents[1]


def main():
    os.chdir(REPO); sys.path.insert(0, str(REPO))
    import torch
    from r2_gaussian.arguments import ModelParams, PipelineParams
    from r2_gaussian.dataset import Scene
    from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query

    for w in TEST_WORMS:
        stack, _ = L.c.load_stack(w)
        for span in SPANS:
            cdir = ROOT / "cases" / f"A_w{w:04d}_mid_span{span}_start0_f0_ns0"
            rec = json.loads((cdir / "case_record.json").read_text())
            z0, z1 = rec["z_range"]
            gt = np.load(cdir / "vol_gt.npy")
            worm = gt > 0
            for m in METHODS:
                df = cdir / f"init_{m}_diag.json"
                pf = cdir / ("init.npy" if m == "fdk" else f"init_{m}.npy")
                if not (df.exists() and pf.exists()):
                    continue
                p = argparse.ArgumentParser(); lp, pp = ModelParams(p), PipelineParams(p)
                args = p.parse_args([]); args.source_path = str(cdir); args.ply_path = str(pf)
                ma, pa = lp.extract(args), pp.extract(args)
                sc = Scene(ma, False).scanner_cfg
                vtw = max(sc["sVoxel"])
                g = GaussianModel(np.array([ma.scale_min, ma.scale_max]) * vtw)
                initialize_gaussian(g, ma, None)
                with torch.no_grad():
                    vol = query(g, sc["offOrigin"], sc["nVoxel"], sc["sVoxel"], pa)["vol"].cpu().numpy()  # (x, y, z)
                err = (vol - gt)[worm]
                pm = projection_metrics(np.ascontiguousarray(np.transpose(vol, (2, 1, 0))), stack[z0:z1], stack[z0:z1],
                                        rec["measured_view_indices"])
                d = json.loads(df.read_text())
                d.update(gauss_worm_psnr_db=float(10 * np.log10(gt.max() ** 2 / (err**2).mean())),
                         gauss_measured_err=pm["proj_rel_l2_measured"], gauss_hidden_err=pm["proj_rel_l2_withheld"],
                         gauss_mass_ratio=float(vol.sum() / gt.sum()))
                L.c.write_json(df, d)
                print(cdir.name, m, f"start vol {d['init_worm_psnr_db']:.2f} dB -> as Gaussians {d['gauss_worm_psnr_db']:.2f} dB; "
                      f"hidden {100 * d.get('start_hidden_err', np.nan):.2f}% -> {100 * d['gauss_hidden_err']:.2f}%; "
                      f"mass x{d['gauss_mass_ratio']:.2f}", flush=True)
                del g; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
