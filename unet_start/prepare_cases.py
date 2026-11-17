"""Build the test cases of the initialisation study and their physics-only starting points.

Test poses: worm_004 (development pose of the TV study), worm_1031 and worm_8916, mid slab,
Track A (the supplied sinograms, no added noise), arcs of 120, 90 and 60 degrees from 0 deg.
For each case: dataset, native FDK start (unchanged initialize_pcd.py), then the SIRT and
SIRT+TV starts and the diagnostics of every start.
"""

import subprocess
import sys
from pathlib import Path

import init_lib as L
import make_inits
from prepare_case import build
from run_queue import INIT_ARGS, PY, REPO

ROOT = L.c.PROJECT_ROOT / "runs_worm"
TEST_WORMS = (4, 1031, 8916)
SPANS = (120, 90, 60)


def main():
    for w in TEST_WORMS:
        for span in SPANS:
            cdir, _ = build("A", w, "mid", span, 0, 0, 0, ROOT / "phantoms", ROOT / "cases")
            if not (cdir / "init.npy").exists():
                subprocess.run([PY, "-u", "initialize_pcd.py", "--data", cdir, "--output", cdir / "init.npy"]
                               + INIT_ARGS, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
            if not (cdir / "init_fdk_diag.json").exists():
                subprocess.run([PY, "-u", Path(__file__).with_name("make_inits.py"), cdir, "--fdk-diag"],
                               check=True)
            make_inits.build_case(cdir, sys.argv[1:] or ["sirt", "sirttv"])


if __name__ == "__main__":
    main()
