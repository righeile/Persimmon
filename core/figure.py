"""Proportion and deviation on a marked figure or surface.

Three things live here, all projective so they survive the camera:

* ``metric_coords`` -- real distances along a marked axis, recovered from the
  axis's vanishing point.  This is the primitive; every proportion question
  (heads tall, limb ratios, where the knee falls) is a ratio of differences of
  its output.  Image distances are NOT proportions: a figure leaning away
  images its far half shorter, and reading the pixels gives the lean, not the
  build.
* ``midline`` -- symmetry, via the projective midpoints of left/right landmark
  pairs.  The pairs supply their own vanishing point, so no calibration is
  needed.
* ``homography`` / ``deviation`` -- the drawing against its reference, one
  planar surface at a time.  A homography absorbs the difference in viewpoint,
  so what is left over is the hand, not the camera.  Fitting needs 4 points and
  says nothing at 4; the leave-one-out residual is the honest number.

Every readout returns ``(value, conditioning, verdict)`` with verdict possibly
``"undetermined"``, per PLAN.md section 1.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cross_ratio", "metric_coords", "axis_check", "harmonic_mid",
           "pair_vp", "midline", "homography", "deviation", "transfer"]


# --- the line's own coordinate -------------------------------------------------

def _line_coords(pts, v):
    """Scalars along the marked axis for pts, plus the vanishing point's own.

    Returns ``(t, (tv, wv), p0, d)``.  The vanishing point comes back
    homogeneous because it is allowed to be at infinity -- an axis whose image
    is exactly parallel needs no correction, and that is the ``wv == 0`` case,
    not an error.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    v = np.asarray(v, float).ravel()
    if v.size == 2:
        v = np.array([v[0], v[1], 1.0])
    p0 = p.mean(0)
    # axis direction from the marked points, not from the vanishing point:
    # the marks are what the user actually drew.
    u, s, vt = np.linalg.svd(p - p0)
    d = vt[0]
    t = (p - p0) @ d
    if abs(v[2]) < 1e-12 * max(1.0, np.abs(v[:2]).max()):
        return t, (float(v[:2] @ d), 0.0), p0, d   # at infinity
    return t, (float((v[:2] / v[2] - p0) @ d), 1.0), p0, d


def cross_ratio(t1, t2, t3, t4):
    """Cross-ratio of four coordinates on a line; each is a scalar or (t, w)."""
    def h(x):
        return (float(x), 1.0) if np.isscalar(x) or np.ndim(x) == 0 else (float(x[0]), float(x[1]))
    a, b, c, e = (h(x) for x in (t1, t2, t3, t4))
    det = lambda i, j: i[0] * j[1] - j[0] * i[1]
    num = det(a, c) * det(b, e)
    den = det(b, c) * det(a, e)
    return float(num / den) if abs(den) > 1e-300 else float("inf")


def metric_coords(pts, v):
    """Real-world coordinates along the axis, up to an overall scale and offset.

    The projective map of the image line that sends the vanishing point to
    infinity is ``T = 1/(tv - t)``; any other such map differs from it by an
    affine one, so ratios of differences of T are exactly the real ratios.
    """
    t, (tv, wv), _, _ = _line_coords(pts, v)
    if wv == 0.0:
        T = t.copy()                       # already affine: no correction due
    else:
        gap = tv - t
        if np.abs(gap).min() < 1e-9:
            raise ValueError("a marked point coincides with the vanishing point")
        T = 1.0 / gap
    T = T - T.min()
    m = np.abs(T).max()
    return T / m if m > 0 else T


def axis_check(pts, v, span=(0, -1), unit=(0, 1), v_alt=()):
    """A proportion along a marked axis: span length in units of ``unit``.

    ``pts`` are marked in order along the axis; ``span`` and ``unit`` index into
    them (feet..head over chin..head gives heads tall).  Conditioning carries
    the naive image reading -- how much work the perspective correction is doing
    -- and the spread over any alternative vanishing points supplied in
    ``v_alt``, which is where a vanishing point's own uncertainty belongs.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    ratio = lambda T: float((T[span[1]] - T[span[0]]) / (T[unit[1]] - T[unit[0]]))
    val = ratio(metric_coords(p, v))
    t, _, _, _ = _line_coords(p, v)
    naive = float((t[span[1]] - t[span[0]]) / (t[unit[1]] - t[unit[0]]))
    alts = [ratio(metric_coords(p, a)) for a in v_alt]
    spread = float(np.ptp(alts + [val])) if alts else 0.0
    off_axis = float(np.abs(np.linalg.svd(p - p.mean(0))[1][1]) / max(np.ptp(t), 1e-12))
    cond = {"naive_image_ratio": naive,
            "correction": abs(val - naive),
            "vp_spread": spread,
            "off_axis_frac": off_axis,
            "n_marks": len(p)}
    if off_axis > 0.05:
        verdict = "undetermined"        # the marks are not on one straight axis
    elif alts and spread > 0.1 * abs(val):
        verdict = "undetermined"        # the vanishing point is not pinned well enough
    else:
        verdict = "ok"
    return val, cond, verdict


# --- symmetry ------------------------------------------------------------------

def harmonic_mid(a, b, v):
    """Image of the real midpoint of ab, given the vanishing point of its line.

    The midpoint is the harmonic conjugate of the point at infinity, so
    ``(a, b; m, v) = -1``.  Collapses to the image midpoint when v is far.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    v = np.asarray(v, float).ravel()
    if v.size == 3:
        if abs(v[2]) < 1e-12 * max(1.0, np.abs(v[:2]).max()):
            return (a + b) / 2
        v = v[:2] / v[2]
    d = b - a
    L = np.dot(d, d)
    if L < 1e-24:
        return (a + b) / 2
    ta, tb, tv = 0.0, 1.0, float(np.dot(v - a, d) / L)
    if abs(ta - tv) < 1e-12:
        return (a + b) / 2
    k = (tb - tv) / (ta - tv)
    return a + d * ((ta * k + tb) / (1.0 + k))


def pair_vp(pairs):
    """Vanishing point of the left-right body direction, from landmark pairs.

    Each pair joins the same landmark on the two sides; in the world those
    segments are parallel, so their images meet at one point.
    """
    P = np.asarray(pairs, float).reshape(-1, 2, 2)
    if len(P) < 2:
        raise ValueError("need two pairs to find their vanishing point")
    L = np.cross(np.c_[P[:, 0], np.ones(len(P))], np.c_[P[:, 1], np.ones(len(P))])
    L /= np.maximum(np.linalg.norm(L[:, :2], axis=1), 1e-12)[:, None]
    v = np.linalg.svd(L)[2][-1]
    resid = float(np.sqrt(np.mean((L @ v / max(abs(v[2]), 1e-12)) ** 2))) if abs(v[2]) > 1e-12 else 0.0
    return v, resid


def midline(pairs, v_axis=None, v_pairs=None, floor=0.03):
    """Symmetry: where the midline crosses each pair, as a fraction of its width.

    A symmetric figure is crossed at exactly 0.5 of every left-right pair.  That
    is the readout -- not the scatter of the midpoints about a fitted line, which
    throws most of the signal away: a displaced landmark moves its midpoint
    mostly *along* the midline, so perpendicular scatter saw 0.98 px of a 5.5 px
    shift, and got worse as pairs were added.  Measured this way the answer is in
    body units and reads directly: "the left shoulder sits 0.08 of shoulder-width
    off centre".

    The midline must be fitted without the odd pair or it leans toward it and the
    asymmetry hides -- with four pairs, leave-one-out leakage beat the real signal
    at every magnitude tested.  So the odd pair is identified first, as the one
    whose removal leaves the remaining midpoints collinear, and every pair is then
    measured against that clean line.

    ``floor`` is the smallest asymmetry worth believing, as a fraction of the
    pair's width.  Measured: four pairs clicked to +-1.5 px read 0.021 on a
    figure that is exactly symmetric, and the odd pair is named correctly 94% of
    the time at 0.071 and always at 0.126, so 0.03 is a fair default.  Below it
    the verdict is ``"symmetric"``, not a small number.
    """
    P = np.asarray(pairs, float).reshape(-1, 2, 2)
    n = len(P)
    if n < 4:
        raise ValueError("need four pairs: one is identified, the rest set the midline")
    v, pair_resid = pair_vp(P) if v_pairs is None else (np.asarray(v_pairs, float).ravel(), float("nan"))
    M = np.array([harmonic_mid(a, b, v) for a, b in P])

    def fit(X):
        c = X.mean(0)
        u, s, vt = np.linalg.svd(X - c)
        return c, vt[0], float(np.sqrt(np.mean(((X - c) @ vt[1]) ** 2)))

    # the odd pair: drop each in turn and see which leaves the rest straight
    score = np.array([fit(M[np.arange(n) != i])[2] for i in range(n)])
    odd = int(np.argmin(score))
    c, d, straight = fit(M[np.arange(n) != odd])
    rest = np.sort(score)
    margin = float(rest[1] / rest[0]) if rest[0] > 1e-12 else float("inf")

    asym = np.full(n, np.nan)
    angle = np.full(n, np.nan)
    L1 = np.cross([*c, 1.0], [*(c + d), 1.0])
    for i in range(n):
        a, b = P[i]
        e = b - a
        angle[i] = np.degrees(np.arccos(min(1.0, abs(float(d @ e / max(np.linalg.norm(e), 1e-12))))))
        x = np.cross(L1, np.cross([*a, 1.0], [*b, 1.0]))
        if abs(x[2]) < 1e-12:
            continue
        T = metric_coords([a, x[:2] / x[2], b], v)
        asym[i] = float((T[1] - T[0]) / (T[2] - T[0])) - 0.5

    cond = {"pair_vp_resid_px": pair_resid, "n_pairs": n,
            "odd_pair": odd, "odd_margin": margin,
            "midline_straight_px": straight,
            "worst_cross_angle_deg": float(np.nanmin(angle)),
            "asym_width_frac": asym.tolist()}
    if v_axis is not None:
        va = np.asarray(v_axis, float).ravel()
        va = va[:2] / va[2] if va.size == 3 and abs(va[2]) > 1e-12 else va[:2]
        da = va - c
        nn = np.linalg.norm(da)
        if nn > 1e-9:
            cond["axis_angle_deg"] = float(np.degrees(np.arccos(min(1.0, abs(float(da / nn @ d))))))
    worst = float(np.nanmax(np.abs(asym))) if not np.isnan(asym).all() else np.nan
    cond["worst_asym"] = worst
    cond["floor"] = floor
    if np.nanmin(angle) < 20.0 or np.isnan(asym).all():
        verdict = "undetermined"        # pairs run too nearly along the midline to cross it
    elif worst < floor:
        verdict = "symmetric"           # nothing above the clicking floor; no pair to attribute
    elif margin < 1.5:
        verdict = "undetermined"        # no pair stands out: two asymmetries can look like none
    else:
        verdict = "ok"
    return (c, d, asym), cond, verdict


# --- the drawing against its reference -----------------------------------------

def homography(src, dst):
    """Plane-to-plane map from marked correspondences, normalized before the fit.

    Hartley normalization for the same reason ``rhythm.conic`` needs it: the
    quadratic columns otherwise run to 1e5 against a column of ones.
    """
    s = np.asarray(src, float).reshape(-1, 2)
    d = np.asarray(dst, float).reshape(-1, 2)
    if len(s) != len(d) or len(s) < 4:
        raise ValueError("need four or more matched points")

    def norm(p):
        c = p.mean(0)
        k = np.sqrt(2) / max(np.linalg.norm(p - c, axis=1).mean(), 1e-12)
        return np.array([[k, 0, -k * c[0]], [0, k, -k * c[1]], [0, 0, 1.0]])

    Ts, Td = norm(s), norm(d)
    sn = (np.c_[s, np.ones(len(s))] @ Ts.T)[:, :2]
    dn = (np.c_[d, np.ones(len(d))] @ Td.T)[:, :2]
    A = []
    for (x, y), (X, Y) in zip(sn, dn):
        A += [[-x, -y, -1, 0, 0, 0, x * X, y * X, X],
              [0, 0, 0, -x, -y, -1, x * Y, y * Y, Y]]
    H = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / (H[2, 2] if abs(H[2, 2]) > 1e-12 else np.abs(H).max())


def transfer(pts, H):
    """Carry marked points through a homography -- stage to stage, or to a reference."""
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H, float).T
    w = np.where(np.abs(q[:, 2]) < 1e-12, 1e-12, q[:, 2])
    return q[:, :2] / w[:, None]


def deviation(src, dst, scale=None):
    """Per-point deviation of a drawing from its reference, on one flat surface.

    The fit residual is not the answer: four points fit exactly and prove
    nothing.  Each point is re-measured from a homography fitted without it, so
    the numbers are out-of-sample.  ``scale`` (e.g. a head length in the
    drawing) converts pixels into something a person can act on.
    """
    s = np.asarray(src, float).reshape(-1, 2)
    d = np.asarray(dst, float).reshape(-1, 2)
    n = len(s)
    H = homography(s, d)
    fit = np.linalg.norm(transfer(s, H) - d, axis=1)
    loo = np.full(n, np.nan)
    if n >= 6:
        for i in range(n):
            k = np.arange(n) != i
            loo[i] = np.linalg.norm(transfer(s[i:i + 1], homography(s[k], d[k]))[0] - d[i])
    cond = {"n_points": n,
            "fit_rms_px": float(np.sqrt(np.mean(fit ** 2))),
            "loo_rms_px": float(np.sqrt(np.nanmean(loo ** 2))) if n >= 6 else None,
            "scale_px": scale}
    if scale:
        cond["loo_rms_units"] = None if n < 6 else cond["loo_rms_px"] / scale
    verdict = "undetermined" if n < 6 else "ok"
    return (H, fit if n < 6 else loo), cond, verdict


# --- self-check ----------------------------------------------------------------

def _demo():
    """Every number in PLAN.md section 2's rows for this module, re-measured."""
    def rot(a, ax):
        s, c = np.sin(np.radians(a)), np.cos(np.radians(a))
        M = np.eye(3); i, j = [(1, 2), (2, 0), (0, 1)][ax]
        M[i, i] = c; M[j, j] = c; M[i, j] = -s; M[j, i] = s
        return M

    def cam(f=1200, c=(800, 600), pitch=0.0, yaw=0.0, dist=500.0):
        K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
        return K @ np.diag([1.0, -1.0, 1.0]) @ np.c_[rot(pitch, 0) @ rot(yaw, 1), [0, 0, dist]]

    def proj(P, X):
        X = np.atleast_2d(np.asarray(X, float))
        q = np.c_[X, np.ones(len(X))] @ P.T
        return q[:, :2] / q[:, 2:3]

    # proportion: 180 cm figure, 24 cm head, read through four cameras
    h = np.array([0.0, 49.0, 108.0, 156.0, 180.0])
    for yaw, pitch, dist in [(0, 0, 900), (35, 0, 500), (55, -14, 320), (72, -28, 200)]:
        P = cam(yaw=yaw, pitch=pitch, dist=dist)
        p = proj(P, np.c_[np.full(5, 10.0), h, np.zeros(5)])
        v = P[:, :3] @ np.array([0.0, 1.0, 0.0])
        val, cond, verd = axis_check(p, v, span=(0, 4), unit=(3, 4))
        assert abs(val - 7.5) < 1e-6, (yaw, val)
        assert np.allclose(metric_coords(p, v) / metric_coords(p, v)[4], h / 180.0, atol=1e-9)
    # ...and the pixels alone are wrong once the figure turns
    P = cam(yaw=72, pitch=-28, dist=200)
    p = proj(P, np.c_[np.full(5, 10.0), h, np.zeros(5)])
    _, cond, _ = axis_check(p, P[:, :3] @ np.array([0.0, 1.0, 0.0]), span=(0, 4), unit=(3, 4))
    assert cond["naive_image_ratio"] < 5.0, cond

    # symmetry: exact on a symmetric figure, exact and localised on a pushed shoulder
    P = cam(yaw=40, pitch=-8, dist=420); w = 18.0
    mk = lambda x, y: proj(P, [[x, y, 0.0]])[0]
    ph = [150.0, 120.0, 95.0, 60.0]
    pairs = np.array([[mk(10 - w, y), mk(10 + w, y)] for y in ph])
    vax = P[:, :3] @ np.array([0.0, 1.0, 0.0])
    (_, _, a), cond, verd = midline(pairs, v_axis=vax)
    assert np.nanmax(np.abs(a)) < 1e-9 and verd == "symmetric", (a, verd)
    assert cond["axis_angle_deg"] < 1e-3, cond          # midline runs to the vertical VP
    for dx in (6.0, 12.0, 24.0):
        bad = pairs.copy(); bad[1][1] = mk(10 + w + dx, ph[1])
        (_, _, a), cond, verd = midline(bad, v_axis=vax)
        assert cond["odd_pair"] == 1 and verd == "ok", (dx, cond, verd)
        assert abs(a[1] - (w / (2 * w + dx) - 0.5)) < 1e-9, (dx, a[1])
    # the projective midpoint is not halfway along the line
    assert max(np.linalg.norm((p0 + p1) / 2 - mk(10, y))
               for (p0, p1), y in zip(pairs, ph)) > 0.5

    # deviation: a faithful drawing from another viewpoint is zero; one bad point is found
    rng = np.random.default_rng(0)
    Pa, Pb = cam(dist=500), cam(f=1500, yaw=18, pitch=-6, dist=560)
    Q = np.c_[rng.uniform(-80, 80, 9), rng.uniform(-60, 60, 9), np.zeros(9)]
    ref, drw = proj(Pa, Q), proj(Pb, Q)
    (_, res), cond, verd = deviation(ref, drw)
    assert cond["loo_rms_px"] < 1e-6 and verd == "ok", cond
    bad = drw.copy(); bad[4] += [25.0, -14.0]
    (_, res), cond, verd = deviation(ref, bad, scale=100.0)
    assert int(np.argmax(res)) == 4 and abs(res[4] - 28.653) < 0.01, res
    assert deviation(ref[:5], bad[:5])[2] == "undetermined"

    # stage transfer: marks carried between two photographs of the same flat work
    P1, P2 = cam(f=1300, yaw=-9, pitch=4, dist=480), cam(f=1180, yaw=14, pitch=-7, dist=540)
    reg = np.c_[rng.uniform(-90, 90, 7), rng.uniform(-70, 70, 7), np.zeros(7)]
    mk2 = np.c_[rng.uniform(-70, 70, 12), rng.uniform(-50, 50, 12), np.zeros(12)]
    H = homography(proj(P1, reg), proj(P2, reg))
    assert np.abs(transfer(proj(P1, mk2), H) - proj(P2, mk2)).max() < 1e-7
    print("figure.py self-check passed")


if __name__ == "__main__":
    _demo()
