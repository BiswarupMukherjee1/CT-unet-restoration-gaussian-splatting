"""Parity of R2-Gaussian's parallel-beam rasterizer with TIGRE on the same Gaussians.

Loads a prepared case and its initial Gaussians, renders every training view with the
R2 rasterizer, voxelizes the same Gaussians with the R2 query function, projects that
volume with TIGRE (parallel, same scaled geometry, R2's v-flip convention) and reports
the relative L2 difference per view, plus the fit of each to the measured projections.
Pass criterion (pre-set): median relative difference < 5% and no view above 10%.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from r2_gaussian.arguments import ModelParams, PipelineParams  # noqa: E402
from r2_gaussian.dataset import Scene  # noqa: E402
from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query, render  # noqa: E402
from r2_gaussian.utils.ct_utils import get_geometry_tigre  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import tigre

    parser = __import__("argparse").ArgumentParser()
    lp, pp = ModelParams(parser), PipelineParams(parser)
    args = parser.parse_args(["--source_path", str(a.case), "--ply_path", str(a.ply)])
    margs, pargs = lp.extract(args), pp.extract(args)
    scene = Scene(margs, shuffle=False)
    cfg = scene.scanner_cfg
    vmax = max(cfg["sVoxel"])
    g = GaussianModel([margs.scale_min * vmax, margs.scale_max * vmax])
    initialize_gaussian(g, margs, None)
    cams = scene.getTrainCameras()
    with torch.no_grad():
        vol = query(g, cfg["offOrigin"], cfg["nVoxel"], cfg["sVoxel"], pargs)["vol"].cpu().numpy()
        renders = [render(cam, g, pargs)["render"][0].cpu().numpy() for cam in cams]
    geo = get_geometry_tigre(cfg)
    ang = np.array([float(cam.angle) for cam in cams], np.float32)
    tig = tigre.Ax(np.ascontiguousarray(np.transpose(vol, (2, 1, 0))), geo, ang)[:, ::-1, :]
    meas = np.stack([cam.original_image[0].cpu().numpy() for cam in cams])
    rel = [float(np.linalg.norm(r - t) / np.linalg.norm(t)) for r, t in zip(renders, tig)]
    rep = dict(
        case=str(a.case), n_views=len(cams),
        render_vs_tigre_rel_l2_median=float(np.median(rel)), render_vs_tigre_rel_l2_max=float(np.max(rel)),
        render_sum_over_tigre_sum=float(np.sum(renders) / np.sum(tig)),
        render_vs_measured_rel_l2=float(np.linalg.norm(np.stack(renders) - meas) / np.linalg.norm(meas)),
        tigre_vs_measured_rel_l2=float(np.linalg.norm(tig - meas) / np.linalg.norm(meas)),
    )
    rep["pass"] = rep["render_vs_tigre_rel_l2_median"] < 0.05 and rep["render_vs_tigre_rel_l2_max"] < 0.10
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    np.savez_compressed(a.out.with_suffix(".npz"), render0=renders[0], tigre0=tig[0], meas0=meas[0])


if __name__ == "__main__":
    main()
