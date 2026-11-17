"""2-D fan-beam geometry (the central slice of a cone-beam scan) for the CPU TV study.

Geometry scaled from the chest cone-beam scanner of the chest pilot (source-origin 5,
source-detector 7, detector 4 for a volume of half-size 1) to the worm slab (half-size
155 um): DSO = 775 um, DSD = 1085 um, flat detector 620 um wide with 448 bins
(0.99 um at the rotation centre). Fan half-angle ~16 deg; the whole 310 x 310 slice is
inside the beam. Views every 1.8 deg over 360 deg (200 views); a complete fan-beam
scan needs 180 deg + fan angle (~212 deg).

* build_matrix: pixel-driven projector with 2x2 sub-samples, linear interpolation and
  the fan-beam ray-density weight, stored as an exactly adjoint sparse matrix
  (same image convention as tv2d.build_matrix).
* project_fine: an independent on-the-fly projector (4x4 sub-samples, 2x finer
  detector binned 2:1) used only to simulate measurements, so simulation and
  reconstruction do not share the same discretisation.
* disk_test: both projectors against analytic chord lengths of an off-centre disc.
"""

import numpy as np
import scipy.sparse as sp

import common as c

DSO, DSD = 775.0, 1085.0
N_BINS, S_DET = 448, 620.0
D_DET = S_DET / N_BINS
N_VIEWS_FAN = 200
ANGLES_FAN = np.arange(N_VIEWS_FAN) * 1.8


def arc_mask_fan(start, span):
    if not (0 <= start < 360 and 0 < span <= 360):
        raise ValueError("need 0 <= start < 360 and 0 < span <= 360")
    return ((ANGLES_FAN - start) % 360) < span - 1e-9


def _footprints(angles_deg, n, sub, n_bins, d_det):
    """Yield (angle index, pixel index, bin index, weight) arrays per angle."""
    off = (np.arange(sub) + 0.5) / sub - 0.5
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    pix = (ii * n + jj).ravel()
    for a_i, a in enumerate(np.deg2rad(angles_deg)):
        ca, sa = np.cos(a), np.sin(a)
        rows, cols, vals = [], [], []
        for dy in off:
            for dx in off:
                x = jj.ravel() + dx - (n - 1) / 2
                y = (n - 1) / 2 - (ii.ravel() + dy)
                t = x * ca + y * sa          # along the detector
                s = -x * sa + y * ca         # along the central ray, towards the detector
                depth = DSO + s              # distance from the source along the central ray
                u = t * DSD / depth          # position on the detector
                cosphi = DSD / np.sqrt(DSD**2 + u**2)
                w = DSD / (d_det * depth * cosphi) / sub**2  # area / perpendicular ray spacing
                b = u / d_det + (n_bins - 1) / 2
                b0 = np.floor(b).astype(int)
                f1 = b - b0
                for bb, ff in ((b0, 1 - f1), (b0 + 1, f1)):
                    ok = (bb >= 0) & (bb < n_bins) & (ff > 0)
                    rows.append(bb[ok]); cols.append(pix[ok]); vals.append((ff * w)[ok])
        yield a_i, np.concatenate(cols), np.concatenate(rows), np.concatenate(vals)


def build_matrix(angles_deg=ANGLES_FAN, n=c.N_DET, sub=2):
    R, C, V = [], [], []
    for a_i, cols, bins, vals in _footprints(angles_deg, n, sub, N_BINS, D_DET):
        R.append(a_i * N_BINS + bins); C.append(cols); V.append(vals)
    A = sp.csr_matrix((np.concatenate(V).astype(np.float32), (np.concatenate(R), np.concatenate(C))),
                      shape=(len(angles_deg) * N_BINS, n * n))
    A.sum_duplicates()
    return A


def project_fine(cols, angles_deg=ANGLES_FAN, n=c.N_DET, sub=4, up=2):
    """cols: (n*n, k) image columns -> (len(angles), N_BINS, k) projections, independent
    discretisation (sub x sub samples per pixel, detector `up` times finer, then binned)."""
    k = cols.shape[1]
    out = np.zeros((len(angles_deg), N_BINS * up, k))
    for a_i, pcols, bins, vals in _footprints(angles_deg, n, sub, N_BINS * up, D_DET / up):
        for j in range(k):
            out[a_i, :, j] = np.bincount(bins, weights=vals * cols[pcols, j], minlength=N_BINS * up)
    return out.reshape(len(angles_deg), N_BINS, up, k).mean(2)


def disk_test(radius=40.0, cx=60.0, cy=-30.0, n=c.N_DET):
    """Relative error of both projectors vs analytic chord lengths of a uniform disc."""
    yy, xx = np.mgrid[0:n, 0:n]
    X = xx - (n - 1) / 2
    Y = (n - 1) / 2 - yy
    sub = 8
    o = (np.arange(sub) + 0.5) / sub - 0.5
    img = np.zeros((n, n))
    for dy in o:  # anti-aliased disc (area fraction per pixel)
        for dx in o:
            img += ((X + dx - cx) ** 2 + (Y - dy - cy) ** 2 <= radius**2)
    img /= sub**2
    ang = ANGLES_FAN[::10]
    exact = np.zeros((len(ang), N_BINS))
    u = (np.arange(N_BINS) - (N_BINS - 1) / 2) * D_DET
    for i, a in enumerate(np.deg2rad(ang)):
        src = np.array([-DSO * -np.sin(a), -DSO * np.cos(a)])        # source at s = -DSO
        tdir, sdir = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        for j, uu in enumerate(u):
            d = (uu * tdir + DSD * sdir)
            d = d / np.linalg.norm(d)
            pc = np.array([cx, cy]) - src
            dist = abs(pc[0] * d[1] - pc[1] * d[0])
            exact[i, j] = 2 * np.sqrt(max(radius**2 - dist**2, 0))
    col = img.reshape(-1, 1)
    A = build_matrix(ang)
    pa = (A @ col).reshape(len(ang), N_BINS)
    pf = project_fine(col, ang)[:, :, 0]
    return dict(matrix_rel_err=float(np.linalg.norm(pa - exact) / np.linalg.norm(exact)),
                fine_rel_err=float(np.linalg.norm(pf - exact) / np.linalg.norm(exact)),
                matrix_vs_fine=float(np.linalg.norm(pa - pf) / np.linalg.norm(pf)),
                mass_ratio_matrix=float(pa.sum() / exact.sum()))
