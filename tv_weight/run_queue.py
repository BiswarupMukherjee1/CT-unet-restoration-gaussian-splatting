"""Sequential job queue for the worm TV study (run under nohup).

Each line of the jobs file is a JSON object:
  {"phase": "P1", "track": "A", "worm": 4, "slab": "mid", "span": 120, "start": 0,
   "noise": 0, "noise_seed": 0, "lambda_tv": 0.05, "seed": 0, "iterations": 30000,
   "keep_volume": true, "figure": true}
For each job: prepare the case (once), run the native FDK initializer (once per case, so
all TV values / seeds of a case share the same initial Gaussians, as in the chest pilot), train,
evaluate, clean up. Finished jobs (result JSON present) are skipped, so the queue can be
restarted. Failed jobs are logged and their run directory renamed; the queue continues.
The jobs file is re-read after every job, so jobs can be appended while it runs.
Stops before starting a job if PROJECT_ROOT uses more than --disk-stop-gb.
Several copies may run at once (one per GPU slot): each run is claimed with a lock file
(`locks/<run>.lock`, holding host and pid; a lock whose process is gone on this host is
taken over), and dataset building / initialisation of a case is serialised with a file lock.
"""

import argparse
import fcntl
import json
import os
import socket
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import common as c  # noqa: E402
from prepare_case import build, case_id  # noqa: E402

PY = sys.executable
INIT_ARGS = ["--recon_method", "fdk", "--n_points", "50000", "--density_thresh", "0.05",
             "--density_rescale", "0.15"]


def run_id(j):
    return (case_id(j["track"], j["worm"], j["slab"], j["span"], j["start"], j["noise"], j["noise_seed"])
            + f"_tv{j['lambda_tv']:g}_seed{j['seed']}_it{j['iterations']}" + j.get("tag", ""))


def claim(root, rid):
    """Atomically claim a run; returns True if this process owns it."""
    lock = root / "locks" / f"{rid}.lock"
    me = f"{socket.gethostname()} {os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode()); os.close(fd)
            return True
        except FileExistsError:
            try:
                host, pid = lock.read_text().split()
            except (ValueError, FileNotFoundError):
                return False
            if host != socket.gethostname() or os.path.exists(f"/proc/{pid}"):
                return False
            lock.unlink(missing_ok=True)  # owner died on this host: take over
    return False


class CaseLock:
    def __init__(self, root, name):
        self.path = root / "locks" / f"case_{name}.lock"

    def __enter__(self):
        self.f = open(self.path, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def du_gb(path):
    out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True).stdout.split()[0]
    return int(out) / 1048576


def log(status, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(status, "a") as f:
        f.write(line + "\n")


def sh(cmd, logf, cwd=REPO):
    with open(logf, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="runs_worm directory")
    ap.add_argument("--disk-stop-gb", type=float, default=15.5)
    a = ap.parse_args()
    root = a.root
    for d in ("cases", "runs", "results", "logs", "locks"):
        (root / d).mkdir(parents=True, exist_ok=True)
    status = root / "status.log"
    log(status, f"QUEUE START jobs={a.jobs}")
    done_or_failed = set()
    while True:
        jobs = [json.loads(l) for l in a.jobs.read_text().splitlines() if l.strip() and not l.startswith("#")]
        todo = [j for j in jobs if run_id(j) not in done_or_failed
                and not (root / "results" / f"{run_id(j)}.json").exists()]
        if not todo:
            break
        j = todo[0]
        rid = run_id(j)
        done_or_failed.add(rid)
        if not claim(root, rid):
            continue
        gb = du_gb(c.PROJECT_ROOT)
        if gb > a.disk_stop_gb:
            log(status, f"STOP disk {gb:.2f} GB > {a.disk_stop_gb} GB before {rid}")
            return
        logf = root / "logs" / f"{rid}.log"
        t0 = time.time()
        log(status, f"START {j.get('phase', '')} {rid} (disk {gb:.2f} GB)")
        try:
            cid = case_id(j["track"], j["worm"], j["slab"], j["span"], j["start"], j["noise"], j["noise_seed"])
            with CaseLock(root, cid):
                cdir, rec = build(j["track"], j["worm"], j["slab"], j["span"], j["start"], j["noise"],
                                  j["noise_seed"], root / "phantoms", root / "cases")
                init = cdir / "init.npy"
                if not init.exists():
                    if sh([PY, "-u", "initialize_pcd.py", "--data", cdir, "--output", init] + INIT_ARGS, logf):
                        raise RuntimeError("initialize failed")
                ckpt = None
                if j.get("init", "fdk") != "fdk":  # starting points built by worm_init/make_inits.py
                    base = j["init"].removesuffix("_fit")
                    init = cdir / ("init.npy" if base == "fdk" else f"init_{base}.npy")
                    if not init.exists():
                        raise RuntimeError(f"missing {init.name}")
                    if j["init"].endswith("_fit"):  # same points, first fitted to the starting volume
                        ckpt = cdir / f"init_{base}_fit.ckpt"
                        if not ckpt.exists() and sh([PY, "-u", HERE.parent / "worm_init" / "prefit_gaussians.py",
                                                     cdir, "--method", base], logf, cwd=HERE.parent / "worm_init"):
                            raise RuntimeError("prefit failed")
            rdir = root / "runs" / rid
            if rdir.exists():
                rdir.rename(rdir.with_name(rid + f"__stale_{int(time.time())}"))
            rdir.mkdir(parents=True)
            (rdir / "run_spec.json").write_text(json.dumps(dict(j, run_id=rid, init_sha256=c.sha256_file(init)), indent=1))
            cmd = [PY, "-u", HERE / "train_worm.py", "--source_path", cdir, "--model_path", rdir,
                   "--ply_path", init, "--iterations", j["iterations"], "--lambda_tv", j["lambda_tv"],
                   "--seed", j["seed"], "--run_record", rdir / "run_record.json", "--quiet",
                   "--test_iterations", 1000, 2500, 5000, 10000, 15000, 20000, 25000]
            cmd += ["--start_checkpoint", ckpt] if ckpt else []
            if sh(cmd, logf):
                raise RuntimeError("train failed")
            ev = [PY, "-u", HERE / "evaluate_run.py", "--run", rdir, "--case", cdir, "--iteration",
                  j["iterations"], "--out", root / "results" / f"{rid}.json", "--cleanup"]
            ev += ["--keep-volume"] if j.get("keep_volume") else []
            ev += ["--figure"] if j.get("figure") else []
            if sh(ev, logf, cwd=HERE):
                raise RuntimeError("evaluate failed")
            r = json.loads((root / "results" / f"{rid}.json").read_text())
            v, p = r["voxel"], r["projection"]
            log(status, f"END   {rid} {time.time() - t0:.0f}s worm_psnr={v['worm_psnr_db']:.3f} "
                        f"dice_cls={v['dice_class_mean']:.3f} bias%={v['abs_rel_bias_pct_wmean']:.2f} "
                        f"cv%={v['interior_cv_pct_wmean']:.2f} withheld={p['proj_rel_l2_withheld']}")
        except Exception as e:  # keep going; the failure is recorded
            log(status, f"FAIL  {rid} after {time.time() - t0:.0f}s: {e}")
            rd = root / "runs" / rid
            if rd.exists():
                rd.rename(rd.with_name(rid + f"__failed_{int(time.time())}"))
    log(status, "QUEUE FINISHED")


if __name__ == "__main__":
    main()
