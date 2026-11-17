"""Projection-domain scores of the starting volumes (same definition as evaluate_run.py).

For every test case and starting method, reproject the starting volume on all 100 views
and store its relative L2 error on the measured and on the withheld views in
init_<method>_diag.json. This score uses only the supplied data, no reference volume.
"""

import json
import sys
from pathlib import Path

import numpy as np

import init_lib as L
from evaluate_run import projection_metrics  # noqa: E402  (worm_tv is on the path via init_lib)
from prepare_cases import ROOT, SPANS, TEST_WORMS

METHODS = ("fdk", "sirt", "sirttv", "unet", "unetuq")


def main():
    for w in TEST_WORMS:
        stack, _ = L.c.load_stack(w)
        for span in SPANS:
            cdir = ROOT / "cases" / f"A_w{w:04d}_mid_span{span}_start0_f0_ns0"
            rec = json.loads((cdir / "case_record.json").read_text())
            z0, z1 = rec["z_range"]
            clean = stack[z0:z1]
            for m in METHODS:
                df, vf = cdir / f"init_{m}_diag.json", cdir / f"initvol_{m}.npz"
                if not (df.exists() and vf.exists()):
                    continue
                vol = np.load(vf)["vol"].astype(np.float32)  # (z, y, x)
                p = projection_metrics(vol, clean, clean, rec["measured_view_indices"])
                d = json.loads(df.read_text())
                d.update(start_measured_err=p["proj_rel_l2_measured"], start_hidden_err=p["proj_rel_l2_withheld"])
                L.c.write_json(df, d)
                print(cdir.name, m, f"measured {100 * d['start_measured_err']:.2f}%  hidden {100 * d['start_hidden_err']:.2f}%",
                      flush=True)


if __name__ == "__main__":
    sys.exit(main())
