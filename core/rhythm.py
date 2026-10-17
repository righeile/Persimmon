"""Checks that need more than one line: spacing, roofs, figures, ellipses."""
import numpy as np
import cv2

from . import figure as F


def is_vertical(s, m, deg=60):
    d = np.abs(np.degrees(np.arctan2(s[m][:, 3] - s[m][:, 1], s[m][:, 2] - s[m][:, 0])))
    return np.median(np.minimum(d, 180 - d)) > deg


def horizon(s, fams):
    """The horizon is where the HORIZONTAL vanishing points line up.

    Picking the two VPs with the smallest |y| gave +26.6 deg on the whale
    sketch: it grabbed a roof-slope VP, which is exactly the thing that does
    not lie on the horizon. Fit the line the ground-plane VPs agree on, by
    weight of evidence, and whatever misses it is a tilted plane.
    """
    P = [(np.array([v[0] / v[2], v[1] / v[2]]), int(m.sum()))
         for v, m in fams if abs(v[2]) > 1e-12 and not is_vertical(s, m)]
    if len(P) < 2: return None, None, []
    best = (-1, None)
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            a, b = P[i][0], P[j][0]
            n = np.array([b[1] - a[1], a[0] - b[0]], float)
            nn = np.linalg.norm(n)
            if nn < 1e-9: continue
            n /= nn
            w = sum(k for p, k in P if abs(n @ (p - a)) < 40)
            if w > best[0]: best = (w, (a, b))
    a, b = best[1]
    n = np.array([b[1] - a[1], a[0] - b[0]], float); n /= np.linalg.norm(n)
    off = [(p, k, float(n @ (p - a))) for p, k in P if abs(n @ (p - a)) >= 40]
    return a, b, off


def tilted(slope_vp, vert_vp, rise_vp):
    """A tilted plane's slope VP, the vertical VP and the VP of the direction
    it rises along are COLLINEAR. No camera calibration needed for that --
    only for turning it into degrees. Returns the miss, in pixels."""
    a, b, c = (np.asarray(x, float) for x in (slope_vp, vert_vp, rise_vp))
    d = b - a; L = np.linalg.norm(d)
    if L < 1e-9: return np.nan
    return abs(d[0] * (c[1] - a[1]) - d[1] * (c[0] - a[0])) / L

def _coord(p, a, b):
    """Signed position of p along the line a->b, in pixels."""
    e = (b - a) / np.linalg.norm(b - a)
    return (np.asarray(p) - a) @ e

def row_frame(pts, vp):
    """The row's own line, and how far the marks and the VP sit off it.

    A vanishing point lies ON the image of every line running to it, so `off`
    is the test of whether the family named is the one the row runs along:
    name the cross direction and the answer is about some other direction.
    Both misses are pixels at the row's own ends, so a hand mark compares with
    them directly. Returns (mid, direction, bend, off).
    """
    q = np.asarray(pts, float).reshape(-1, 2)
    mid = q.mean(0)
    e = np.linalg.svd(q - mid)[2][0]
    half = float(np.abs((q - mid) @ e).max())
    d = np.asarray(vp, float)[:2] - mid              # numpy 2 has no 2-D cross
    off = abs(e[0] * d[1] - e[1] * d[0]) / max(float(np.linalg.norm(d)), 1e-9) * half
    bend = float(np.abs((q - mid) @ np.array([-e[1], e[0]])).max())
    return mid, e, bend, float(off)


def _theil_sen(x, y):
    """A straight line through points one of which may be badly wrong.

    Least squares with leave-one-out was tried first and is the wrong tool
    here: the model has two parameters, so with six posts one post really out
    of place threw the fit far enough to flag its two neighbours as strays as
    well. The median of all pairwise slopes does not care.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = [(y[j] - y[k]) / (x[j] - x[k])
         for j in range(len(x)) for k in range(j + 1, len(x)) if x[j] != x[k]]
    A = float(np.median(s))
    return A, float(np.median(y - A * x))


def _row_target(q, vp, idx):
    """Where the even row that the posts in `idx` agree on puts every post.

    The line is taken from those posts alone, which is what makes a second
    tidy a no-op: afterwards they lie exactly on it, so it fits itself.
    """
    mid = q[idx].mean(0)
    e = np.linalg.svd(q[idx] - mid)[2][0]
    t = (q - mid) @ e
    tv = float((np.asarray(vp, float)[:2] - mid) @ e)
    rank = np.empty(len(q), float)
    rank[np.argsort(t)] = np.arange(len(q))
    d = tv - t
    if np.any(np.abs(d) < 1e-6):
        return None
    A, B = _theil_sen(rank[idx], 1.0 / d[idx])
    return mid + (tv - 1.0 / (A * rank + B))[:, None] * e


def tidy_row(pts, vp, mark_px):
    """Pull the marks onto the even row the posts that are NOT strays agree on.

    The same bargain as `vp.tidy`: marking is coarser than drawing, so a post
    asking for less than a hand mark is the marker's wobble and is absorbed,
    and one asking for more is the drawing talking and is left exactly where
    it was. The target is fitted WITHOUT the strays -- aim at the whole row's
    own fit and the stray pulls the rest toward itself.

    Returns (posts, which moved, the largest move applied, which are strays),
    all indices in the order the posts were marked.
    """
    q = np.asarray(pts, float).reshape(-1, 2).copy()
    if len(q) < 4:
        return q, [], 0.0, []
    idx = np.arange(len(q))
    pred = _row_target(q, vp, idx)
    if pred is None:
        return q, [], 0.0, []
    step = np.hypot(*(pred - q).T)
    keep, stray = idx[step <= mark_px], [int(k) for k in idx[step > mark_px]]
    if len(keep) < 3:
        return q, [], 0.0, stray
    pred = _row_target(q, vp, keep)
    if pred is None:
        return q, [], 0.0, stray
    step = np.hypot(*(pred - q).T)
    # the move applied is the gate, and the number reported -- not the
    # residual that chose the post, which is a different figure
    moved = [int(k) for k in keep if 1e-9 < step[k] <= mark_px]
    q[moved] = pred[moved]
    return q, moved, (float(step[moved].max()) if moved else 0.0), stray


def spacing(posts, vp, a, b, px_per_mm):
    """Are these repeated uprights equally spaced in the world?

    Sending the vanishing point to infinity turns world-equal spacing into
    page-equal spacing: u = 1/(t_vp - t) is affine in the world coordinate,
    so the test is whether u comes out as an arithmetic sequence.
    """
    t = np.array([_coord(p, a, b) for p in posts])
    tv = _coord(vp, a, b)
    o = np.argsort(t); t = t[o]
    d = tv - t
    if np.any(np.abs(d) < 1e-6): return None
    u = 1.0 / d
    i = np.arange(len(u))
    A, B = np.polyfit(i, u, 1)                      # u should be a straight line in i
    pred = A * i + B
    t_pred = tv - 1.0 / pred
    return dict(order=o, t=t, t_pred=t_pred,
                err_mm=(t - t_pred) / px_per_mm,
                ratio=np.diff(u) / np.mean(np.diff(u)))

def roof_vp_target(slope_vp, rise_vp, a, b):
    """A tilted plane rising along a known horizontal direction puts its slope
    lines' vanishing point on the VERTICAL through that direction's VP.
    Returns how far off the slope VP sits, in pixels along the horizon."""
    return _coord(slope_vp, a, b) - _coord(rise_vp, a, b)

def figure_ratio(feet, head, a, b):
    """Where the horizon cuts a standing figure, as a fraction of its height.
    Same-height figures on one ground plane all give the same fraction --
    that is the whole check, and it needs no scale, no depth, no camera."""
    f, h = np.asarray(feet, float), np.asarray(head, float)
    n = np.array([b[1] - a[1], a[0] - b[0]], float)      # horizon normal
    df, dh = n @ (f - a), n @ (h - a)
    if abs(df - dh) < 1e-9: return np.nan
    return df / (df - dh)



def square_partner(K, v, hz):
    """The vanishing point of the horizontal direction square to this one.

    Two directions at right angles have vanishing points conjugate in
    w = inv(K Kt) -- u' w v = 0 -- so the partner of `v` lies on the line
    w v, which is the vanishing line of every plane square to it. A partner
    that is HORIZONTAL lies on the horizon as well, and two lines meet in one
    point. So a table standing askew to the room needs only one of its two
    directions marked: the camera says where the other one goes.

    None where `v` is the upright direction itself -- every horizontal is
    square to that, so the two lines are one line and nothing is picked out --
    and None where the partner runs parallel to the picture and has no point
    on the page at all.
    """
    K = np.asarray(K, float)
    l = np.linalg.inv(K @ K.T) @ _h(v)
    n, hz = float(np.hypot(l[0], l[1])), np.asarray(hz, float)
    nh = float(np.hypot(hz[0], hz[1]))
    if n < 1e-12 or nh < 1e-12:
        return None
    # both normalised, so q[2] is the sine of the angle between the two lines
    # and the test below is in radians rather than in whatever scale K carried
    q = np.cross(l / n, hz / nh)
    return None if abs(q[2]) < 1e-9 else q[:2] / q[2]


# ------------------------------------------------ a square, and the circle in it

def seg_line(s, through=None):
    """The line a marked segment lies on, as a homogeneous 3-vector whose first
    two numbers are a unit normal. Then the cross product of two of them has
    the sine of the angle between them in its last place, and `c` of a line
    against a point is that point's distance from it.

    `through` is a vanishing point: the line is then the one from it through
    the middle of the segment -- the mark made consistent with its family,
    which is the line a construction that assumes the family is concurrent
    has to be given."""
    s = np.asarray(s, float)
    p, q = s[:2], s[2:4]
    if through is not None:
        p, q = (p + q) / 2.0, np.asarray(through, float)[:2]
    l = _cross(_h(p), _h(q))
    n = float(np.hypot(l[0], l[1]))
    return l / n if n > 1e-9 else None


def _at(x):
    """Where a cross product of two unit-normal lines lies on the page, or None
    where they run parallel."""
    return None if abs(x[2]) < 1e-9 else x[:2] / x[2]


def _unit(l):
    n = float(np.hypot(l[0], l[1]))
    return l / n if n > 1e-12 else None


def square_completions(K, va, vb, a, bi, bj):
    """The lines that make a square with `a` and the two lines `bi`, `bj`.

    `a` runs to the vanishing point `va`, the other two to `vb`: the page's
    picture of two directions in a plane. Two vanishing points name the
    directions without saying the angle between them, so the camera is what
    says "square" -- the diagonal of a square lies along the SUM of its two
    sides' directions, and K sends that to a point on the page. Which side of
    `a` the square is on is the sign of the difference, so there are two.

    The far side is then the line from `va` through the place the diagonal
    out of `a`'s foot on `bi` meets `bj`. One cross product after another; no
    fit, nothing to converge.

    Two answers, one per side, as (line, corners) or None where that side has
    none -- a direction parallel to the picture, a diagonal through the foot
    itself. The corners run round: foot on bi, foot on bj, far corner on bj,
    far corner on bi.
    """
    try:
        Ki = np.linalg.inv(np.asarray(K, float))
    except np.linalg.LinAlgError:
        return [None, None]
    da, db = Ki @ _h(va), Ki @ _h(vb)
    na, nb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    pi, pj = _at(_cross(a, bi)), _at(_cross(a, bj))
    if na < 1e-12 or nb < 1e-12 or pi is None or pj is None:
        return [None, None]
    da, db = da / na, db / nb
    out = []
    for sgn in (1.0, -1.0):
        diag = _unit(_cross(_h(pi), np.asarray(K, float) @ (da + sgn * db)))
        qj = None if diag is None else _at(_cross(diag, bj))
        far = None if qj is None else _unit(_cross(_h(qj), _h(va)))
        qi = None if far is None else _at(_cross(far, bi))
        out.append(None if qi is None else (far, np.array([pi, pj, qj, qi])))
    return out


def _flank(segs, toward, p):
    """The two lines of one family nearest `p`, one on each side of it.

    Measured along the line from `p` to `toward` -- the OTHER family's
    vanishing point, which is the way that family's lines run -- so the walk
    stays inside the strip the other family bounds and the first line it meets
    each way is a line of THIS cell, not of one further off."""
    p = np.asarray(p, float)
    m = _unit(_cross(_h(p), _h(toward)))
    if m is None:
        return None
    u = np.array([m[1], -m[0]])
    lo = hi = None
    for s in segs:
        l = seg_line(s)
        x = None if l is None else _at(_cross(m, l))
        if x is None:
            continue
        t = float((x - p) @ u)
        if t > 1e-9 and (hi is None or t < hi[0]):
            hi = (t, l)
        if t < -1e-9 and (lo is None or t > lo[0]):
            lo = (t, l)
    return None if lo is None or hi is None else (lo[1], hi[1])


def square_cell(segs_a, segs_b, va, vb, p):
    """The four corners, in order round, of the cell of two families that the
    page point `p` sits in -- or None where it sits in none: outside the
    marked lines, or where the nearest lines cross each other before they
    close the cell."""
    a, b = _flank(segs_a, vb, p), _flank(segs_b, va, p)
    if a is None or b is None:
        return None
    q = [_at(_cross(x, y)) for x, y in
         ((a[0], b[0]), (a[0], b[1]), (a[1], b[1]), (a[1], b[0]))]
    if any(c is None for c in q):
        return None
    q = np.array(q)
    e = np.roll(q, -1, 0) - q
    z = e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)
    return q if bool((z > 0).all() or (z < 0).all()) else None


def _square_to_quad(q):
    """The homography taking the unit square (0,0) (1,0) (1,1) (0,1) to `q`."""
    A, b = [], []
    for (u, v), (x, y) in zip(((0, 0), (1, 0), (1, 1), (0, 1)), np.asarray(q, float)):
        A += [[u, v, 1, 0, 0, 0, -u * x, -v * x], [0, 0, 0, u, v, 1, -u * y, -v * y]]
        b += [x, y]
    return np.append(np.linalg.solve(np.array(A), np.array(b)), 1.0).reshape(3, 3)


def cell_ring(q, n=96):
    """The circle inscribed in a square, as the page sees it.

    The four corners fix one homography from the square to the page, and the
    circle is carried along with it -- so it touches the four sides at the
    images of their middles, which are NOT the middles of the sides on the
    page, and it needs no camera. What needs the camera is whether the cell
    is a square at all; that is `cell_reading`."""
    t = np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    return _apply(_square_to_quad(q),
                  np.column_stack([0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)]))


def cell_reading(K, q):
    """What the cell is on its own plane: its two sides' ratio and the angle
    between them -- 1 and 90 degrees for a square.

    The homography's first two columns are the two sides' directions on the
    page, and K^-1 turns each into its world direction, scaled by that side's
    length and by one factor they share. So the ratio of the two lengths is
    the cell's aspect and the angle between the two is its angle, with no
    plane to name and no distance to know.

    (aspect, angle_deg): aspect is side along the first pair of lines over the
    side along the second -- the way `corners` run, see `square_cell`.
    """
    M = np.linalg.inv(np.asarray(K, float)) @ _square_to_quad(q)
    m1, m2 = M[:, 0], M[:, 1]
    n1, n2 = float(np.linalg.norm(m1)), float(np.linalg.norm(m2))
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    c = float(np.clip(abs(m1 @ m2) / (n1 * n2), 0.0, 1.0))
    return n1 / n2, float(np.degrees(np.arccos(c)))


def vps_on_horizon(s, a, b, tol=1.5, min_inl=6, span=4e5):
    """Once the horizon is known, hunt vanishing points ALONG it.

    A free 2-D RANSAC has to find both coordinates of a VP at once and will
    miss a small family whose lines are nearly parallel; with the horizon
    fixed this is a 1-D scan, and weak facets show up. Returns a list of
    (vp, inlier_mask), greedily, strongest first.
    """
    from .vp import resid_deg
    a = np.asarray(a, float); d = np.asarray(b, float) - a
    d = d / np.linalg.norm(d)
    grid = np.concatenate([-np.geomspace(span, 20, 1500), np.geomspace(20, span, 1500)])
    # a facet parallel to the picture plane has its VP AT INFINITY on the horizon;
    # no finite grid point represents it, and the right wall of one painting was
    # missed entirely until this candidate was added.
    cand = [np.array([a[0] + t * d[0], a[1] + t * d[1], 1.0]) for t in grid]
    cand.append(np.array([d[0], d[1], 0.0]))
    free = np.ones(len(s), bool)
    out = []
    while True:
        best = (0, None, None)
        for v in cand:
            m = (resid_deg(s, v) < tol) & free
            if m.sum() > best[0]: best = (m.sum(), v, m)
        if best[0] < min_inl: return out
        v, m = best[1], best[2]
        for _ in range(3 if abs(v[2]) > 1e-12 else 0):
            # Refine ALONG the horizon, but never at the cost of support: minimising
            # the median residual in a wide window wanders off a weak family and
            # shrinks it below threshold, which is how a whole wall went missing.
            loc = np.linspace(-1, 1, 400) * 2e4 + ((v[:2] - a) @ d)
            best2 = (m.sum(), v, m)
            for t in loc:
                v2 = np.array([a[0] + t * d[0], a[1] + t * d[1], 1.0])
                m2 = (resid_deg(s, v2) < tol) & free
                if m2.sum() > best2[0]: best2 = (m2.sum(), v2, m2)
            if best2[1] is v: break
            v, m = best2[1], best2[2]
        if m.sum() < min_inl: return out
        free &= ~m
        out.append((v, m))


def _fit_1d_proj(u, t):
    """t = (a u + b)/(c u + d). Three points fix it; more give a check."""
    A = np.column_stack([u, np.ones_like(u), -t * u, -t])
    return np.linalg.svd(A)[2][-1]


def polygon_check(t, world_deg, span=None):
    """Do these edge-direction vanishing points belong to the polygon claimed?

    The map from a world direction to its vanishing point on the horizon is a
    1-D projectivity in tan(theta). Say "regular hexagon" and the world angles
    are known, so the map is over-determined and the mismatch is measurable.
    Needs 4 directions to say anything: 3 always fit exactly.
    """
    t = np.asarray(t, float); u = np.tan(np.radians(np.asarray(world_deg, float)))
    if len(t) < 4: return None
    h = _fit_1d_proj(u, t)
    pred = (h[0] * u + h[1]) / (h[2] * u + h[3])
    res = t - pred
    out = dict(pred=pred, resid_px=res, rms_px=float(np.sqrt((res ** 2).mean())))
    if span:                       # same error expressed as a world angle
        out["rms_deg"] = float(np.degrees(np.arctan(out["rms_px"] / span)))
    return out


def regular_polygon_angles(n):
    """World directions of the edges of a regular n-gon, in degrees."""
    k = n // 2 if n % 2 == 0 else n
    return [i * 180.0 / k for i in range(k)]


def polygon_vertices(pts, n, idx=None):
    """Is this the perspective view of a regular n-gon?

    Any perspective view of the ideal polygon is a homography away from it, so
    fit the homography over ALL the marked corners and look at what is left.
    Four corners fit anything -- five is the first that can fail.

    `idx` says which corner of the n-gon each click is, so a bay showing only
    part of the polygon still works. Default: consecutive from 0.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 5: return None
    k = np.arange(len(p)) if idx is None else np.asarray(idx, int)
    th = 2 * np.pi * k / n
    ideal = np.column_stack([np.cos(th), np.sin(th)])
    H, _ = cv2.findHomography(ideal, p, 0)
    if H is None: return None
    q = cv2.perspectiveTransform(ideal.reshape(-1, 1, 2), H).reshape(-1, 2)
    d = np.linalg.norm(p - q, axis=1)
    size = np.linalg.norm(p - p.mean(0), axis=1).mean()   # scale to judge px against
    return dict(fitted=q, miss_px=d, rms_px=float(np.sqrt((d ** 2).mean())),
                rms_frac=float(np.sqrt((d ** 2).mean()) / size), worst=int(np.argmax(d)))


def polygon_which(pts, ns=range(3, 13), idx=None):
    """Try every polygon and see which one the corners actually are.

    Ranked by what is left after the fit. A warning: with only part of the
    polygon marked, a bigger n bends less over the same corners and fits more
    easily -- so read the whole ranking, not just the winner.
    """
    out = []
    for n in ns:
        if n < len(np.asarray(pts).reshape(-1, 2)): continue
        r = polygon_vertices(pts, n, idx)
        if r: out.append((n, r["rms_px"], r["rms_frac"], r))
    return sorted(out, key=lambda t: t[1])


def conic(pts):
    """The conic through the marked arc, as a 3x3 matrix.

    Normalized first: on raw pixels the x^2 column runs to 1e5 while the
    constant column is 1, and the fit comes back worse than a straight line.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    c0 = p.mean(0)
    s = np.sqrt(2) / max(np.linalg.norm(p - c0, axis=1).mean(), 1e-12)
    T = np.array([[s, 0, -s * c0[0]], [0, s, -s * c0[1]], [0, 0, 1.0]])
    x, y = ((p - c0) * s).T
    A = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    a, b, c, d, e, f = np.linalg.svd(A)[2][-1]
    Cn = np.array([[a, b / 2, d / 2], [b / 2, c, e / 2], [d / 2, e / 2, f]])
    C = T.T @ Cn @ T
    return C / np.abs(C).max()


def conic_resid(pts, C=None):
    """RMS distance from the marked points to the fitted conic, in pixels."""
    p = np.asarray(pts, float).reshape(-1, 2); x, y = p.T
    if C is None: C = conic(p)
    v = C[0,0]*x*x + 2*C[0,1]*x*y + C[1,1]*y*y + 2*C[0,2]*x + 2*C[1,2]*y + C[2,2]
    gx = 2*(C[0,0]*x + C[0,1]*y + C[0,2]); gy = 2*(C[0,1]*x + C[1,1]*y + C[1,2])
    return float(np.sqrt(np.mean((np.abs(v) / np.maximum(np.hypot(gx, gy), 1e-12))**2)))


def _cross(u, v):
    return np.array([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]])





def calibrate(v1, v2, v3):
    """Focal length and center, from three mutually perpendicular vanishing points.

    This is what makes a shape question answerable at all. Concurrency and
    cross ratios need no camera, but "is that a circle or an oval" does: every
    ellipse is some circle seen by some camera, so without the camera there is
    nothing to decide. Two walls meeting at a right angle plus the uprights are
    enough -- the drawing carries its own calibration.
    Square pixels and no skew assumed, which leaves 4 unknowns and 3 equations.
    """
    v = [np.append(np.asarray(p, float)[:2], 1.0) for p in (v1, v2, v3)]
    # A vanishing point at infinity arrives here as inf or nan -- an exactly
    # level camera puts the vertical one there -- and the SVD below then raises
    # LinAlgError instead of returning. Refuse it like any other unusable input;
    # `checks.camera_level` is the solve that case actually wants.
    if not np.isfinite(np.concatenate(v)).all():
        return None
    A = np.array([[a[0] * b[0] + a[1] * b[1], a[0] * b[2] + a[2] * b[0],
                   a[1] * b[2] + a[2] * b[1], a[2] * b[2]]
                  for a, b in ((v[0], v[1]), (v[0], v[2]), (v[1], v[2]))])
    w = np.linalg.svd(A)[2][-1]
    u0, v0 = -w[1] / w[0], -w[2] / w[0]
    f2 = w[3] / w[0] - u0 ** 2 - v0 ** 2
    if f2 <= 0: return None                       # the three points are not a right-angled triple
    f = float(np.sqrt(f2))
    return np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])


def plane_frame(K, vanish_line):
    """A square grid on the plane -> the page. Needs the camera.

    With this, a shape question stops needing a projective trick: map the
    drawn conic back onto its own plane and just look at it.
    """
    n = K.T @ np.asarray(vanish_line, float)      # the plane's normal, in camera coords
    n = n / np.linalg.norm(n)
    e1 = np.array([1.0, 0, 0]) - n[0] * n
    if np.linalg.norm(e1) < 1e-6: e1 = np.array([0, 1.0, 0]) - n[1] * n
    e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    return K @ np.column_stack([e1, e2, n])


def horizon_diameter(end_a, end_b, vanish_line):
    """The drag turned parallel to the plane's vanishing line, about its middle.

    A circle on a horizontal plane has one diameter whose image runs parallel
    to the horizon: the one parallel to the picture plane. Its two ends sit at
    the same depth, so they recede by the same amount and the chord cannot
    tilt. Measured at 2e-6 deg across rolls, pitches and yaws -- exact, not a
    convention. Every other diameter leans, the ellipse's own long axis most
    of all: 5.7 deg on the centre line of the test camera and 15.0 deg out at
    the side.

    So a ring on a horizontal plane is drawn from a level drag, and the drag
    is levelled here instead of by eye. About the middle, because that is
    where the ring is: turning it about one end would carry the circle off as
    well as its diameter.
    """
    a, b = np.asarray(end_a, float), np.asarray(end_b, float)
    c, h = (a + b) / 2.0, float(np.linalg.norm(b - a)) / 2.0
    l = np.asarray(vanish_line, float)
    d = np.array([-l[1], l[0]])                   # along the vanishing line
    nd = float(np.linalg.norm(d))
    if nd < 1e-12 or h < 1e-9:
        return None
    d = d / nd
    if float(d @ (b - a)) < 0:                    # keep the drag's own sense
        d = -d
    return c - h * d, c + h * d


def square_on_circle(K, vanish_line, end_a, end_b, n=192):
    """The circle at those two dots, seen SQUARE ON -- the eye turned to face it.

    The two dots are the ends of the LONG diameter, which is what this is for.
    The circle a picture's own camera casts does not do that: its long diameter
    leans away from the centre of vision -- 8.7 degrees on an arm's-length ring
    in a watercolour, more at the edge -- so its ends sit 4 to 16 px from the
    dots. Worse, the dots are up to 3 px off that ring altogether, because
    pinning a circle to two page points is four equations in three unknowns and
    only the symmetric case is consistent.

    Turn the eye to face the ring and all of that goes away. What is left is
    the plain fact that a circle seen at a slant is foreshortened by the sine
    of the angle between the line of sight and its plane -- one dot product,
    and a rotation invariant, so it holds whatever the camera's roll or tilt.
    Moving the principal point sideways instead is NOT that turn: it asks for a
    different plane, and in a rolled camera it flattened a ring reading 2.26 to
    1.06.

    This is a drawing guide, not a measurement. Away from the centre of vision
    the picture plane stretches a circle further than the eye sees it -- 2.99
    against 1.68 at 56 degrees off axis -- so `plane_circle` is what a marked
    ring is checked against, and both are drawn.
    """
    a, b = np.asarray(end_a, float), np.asarray(end_b, float)
    c = (a + b) / 2.0
    l = np.asarray(vanish_line, float)
    d = np.linalg.inv(K) @ np.append(c, 1.0)
    nrm = K.T @ l
    kd, kn = np.linalg.norm(d), np.linalg.norm(nrm)
    if kd < 1e-12 or kn < 1e-12:
        return None
    sin_t = abs(float((d / kd) @ (nrm / kn)))     # the slant, a rotation invariant
    if sin_t < 1e-3:                              # edge on: the ring is at the horizon
        return None

    # The turned eye, drawn out: the optical axis on the ring, no roll, the
    # plane at the same slant. Its horizon then lies f*tan(slant) off the
    # middle, and the ring is whatever THAT camera casts -- a real projection,
    # not a circle squashed by the cosine. The two differ by 1.3 % on a ring
    # wide enough to matter, and the squashed one is the wrong 1.3 %.
    f = (abs(K[0, 0]) + abs(K[1, 1])) / 2.0
    u = (b - a) / 2.0
    h = float(np.hypot(*u))
    if f < 1e-9 or h < 1e-9:
        return None
    flat = plane_circle(np.array([[f, 0, 0.], [0, f, 0.], [0, 0, 1.]]),
                        np.array([0.0, np.sqrt(max(0.0, 1 - sin_t ** 2)) / f, sin_t]),
                        (-h, 0.0), (h, 0.0), n=n)
    if flat is None:
        return None

    # ...and laid back down on the drag. Across it, the way AWAY from the
    # vanishing line, because that is the way the turned view's own horizon
    # was put: get this backwards and the ring bulges the wrong side.
    e = u / h
    away = np.array([-e[1], e[0]])
    g = np.sign(float(l @ np.append(c, 1.0))) * l[:2]
    if float(away @ g) < 0:
        away = -away
    return c + flat @ np.vstack([e, away])


def plane_circle(K, vanish_line, end_a, end_b, n=192):
    """The image of the circle that lies in that plane, with `end_a` and
    `end_b` -- both points on the page -- at the two ends of a diameter.

    This is the ellipse an artist wants: not any ellipse through the two marks,
    but the one a real circle in that plane would cast. Two vanishing points
    alone cannot say it, because they name two directions without saying the
    angle between them; the camera does, which is why K is required.

    The two marks span the ring rather than running from its middle to its
    rim. A centre and a radius sound like the natural way to say a circle and
    are the wrong handle: the image's middle is not the image of the middle,
    so the drawn ring ran up to 17 % past the point it was dragged to.

    None when the circle reaches the plane's own vanishing line, where its
    image stops being an ellipse and no drawing of it exists.
    """
    H = plane_frame(K, vanish_line)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    pa, pb = np.asarray(end_a, float), np.asarray(end_b, float)
    u = pb - pa
    span = float(np.hypot(*u))
    if span < 1e-9:
        return None
    u = u / span

    # The ring must END at the two dots. On the plane that says: the circle
    # touches the line through each dot that crosses the drag, and touches it
    # AT that dot -- so its centre lies one radius along each dot's inward
    # normal. Two statements of the same centre, which is three unknowns in
    # four equations: exact when the dots really are some circle's ends, and
    # the nearest circle when the hand was not that precise.
    #
    # Taking the dots as the ends of a diameter instead is only right for a
    # small ring: on a steeply foreshortened one the image of a diameter's end
    # sits 85 % of the span inside the ring's real end.
    def foot(p):
        """A dot, and the inward normal of the line that crosses the drag
        there -- both on the plane."""
        q = Hi @ [p[0], p[1], 1.0]
        if abs(q[2]) < 1e-12:
            return None, None
        m = H.T @ np.array([u[0], u[1], -float(u @ p)])
        k = float(np.hypot(m[0], m[1]))
        return (q[:2] / q[2], m[:2] / k) if k > 1e-12 else (None, None)

    a1, n1 = foot(pa)
    a2, n2 = foot(pb)
    if a1 is None or a2 is None:
        return None
    # a line has no sign: point both normals at the other dot's side
    if n1 @ (a2 - a1) < 0:
        n1 = -n1
    if n2 @ (a1 - a2) < 0:
        n2 = -n2
    dn = n1 - n2
    den = float(dn @ dn)
    if den < 1e-12:                     # the two lines meet at infinity here
        return None
    rad = float(-((a1 - a2) @ dn) / den)
    c = ((a1 + rad * n1) + (a2 + rad * n2)) / 2.0
    if not np.isfinite(rad) or rad < 1e-9 or not np.all(np.isfinite(c)):
        return None
    t = np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    q = (H @ np.column_stack([c[0] + rad * np.cos(t), c[1] + rad * np.sin(t),
                              np.ones(len(t))]).T).T
    # a sign change in the third coordinate is the ring crossing the plane's
    # vanishing line: half of it would be drawn behind the viewer
    if np.ptp(np.sign(q[:, 2])) > 0 or np.min(np.abs(q[:, 2])) < 1e-12:
        return None
    return q[:, :2] / q[:, 2:3]


def _circumcentre(a, b, c):
    """The centre of the circle through three points of a plane.

    None where they lie in a line, which has no circle: the limb is straight
    there, and a straight limb is drawn as a straight limb.
    """
    a, b, c = (np.asarray(z, float) for z in (a, b, c))
    u, v = b - a, c - a
    k = 2.0 * (u[0] * v[1] - u[1] * v[0])                        # no 2-D cross
    if abs(k) < 1e-9 * (float(np.hypot(*u)) * float(np.hypot(*v)) + 1e-30):
        return None
    return a + np.array([v[1] * (u @ u) - u[1] * (v @ v),
                         u[0] * (v @ v) - v[0] * (u @ u)]) / k


def _bez_ctrl(c, p0, p1):
    """The four control points of the cubic Bezier that is the arc p0 -> p1.

    Used to sample the arc, and for the tangents it leaves its ends by. Not
    for drawing: the curve the page draws is fitted on the page by
    `fit_cubic`, because the image of a cubic is not a cubic.

    Over a quarter turn a single cubic lies 0.027 % of the radius off the
    true arc, 0.04 px on a 300 px arch, and 0.07 % at the 105 degrees a
    horseshoe's half needs; over a whole semicircle it is 1.84 %. Sampled at
    96 points a side either way, so what `plane_arch` returns as `pts` is the
    arc, not a drawing of it.
    """
    u0, u1 = np.asarray(p0, float) - c, np.asarray(p1, float) - c
    r0, r1 = float(np.hypot(*u0)), float(np.hypot(*u1))
    if r0 < 1e-9 or r1 < 1e-9:
        return None
    phi = np.arctan2(float(u0[0] * u1[1] - u0[1] * u1[0]),      # no 2-D cross
                     float(u0 @ u1))                            # the signed sweep
    if abs(phi) < 1e-9:
        return None
    sgn = 1.0 if phi > 0 else -1.0
    t0 = sgn * np.array([-u0[1], u0[0]]) / r0     # the tangents, pointing the way round
    t1 = sgn * np.array([-u1[1], u1[0]]) / r1
    k = 4.0 / 3.0 * np.tan(abs(phi) / 4.0) * (r0 + r1) / 2.0
    return np.array([p0, p0 + k * t0, p1 - k * t1, p1], float)


def bez_at(P, t):
    """A cubic Bezier's points at parameters t. Four control points in, n out."""
    t = np.asarray(t, float)[:, None]
    return np.column_stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                            3 * (1 - t) * t ** 2, t ** 3]) @ np.asarray(P, float)


def _bez_arc(c, p0, p1, n):
    P = _bez_ctrl(c, p0, p1)
    return None if P is None else bez_at(P, np.linspace(0.0, 1.0, int(n)))


def _dir_fwd(H, p, v):
    """The page direction a plane direction `v` at plane point `p` images as.

    Not the plane direction mapped by H's top-left block: a homography turns
    directions differently at every point, and at the springing of a steeply
    foreshortened arch the difference is degrees. Taken as the derivative, so
    it is the tangent the drawn curve really leaves at.
    """
    z = H @ [float(p[0]), float(p[1]), 1.0]
    dz = H[:, :2] @ np.asarray(v, float)
    d = dz[:2] * z[2] - z[:2] * dz[2]             # the 1/w**2 cancels below
    k = float(np.hypot(*d))
    return d / k if k > 1e-12 else None


def fit_cubic(arc, t0, t1, passes=6, m=120):
    """One cubic Bezier along `arc`, fitted ON THE PAGE. Four control points.

    `t0` and `t1` are the page directions the curve leaves its two ends by,
    `t1` pointing back along the curve.

    A cubic drawn in the arch's own plane can have its control points mapped
    through the homography, which is the obvious thing to do and is wrong
    twice: the image of a cubic is not a cubic, and that error adds to the
    cubic's own. The classic 4/3 semicircle construction, mapped, lands
    3.0-4.9 % of the rise off the true arc. Fitted here instead -- the same
    curve, the two handle LENGTHS chosen against the arc as it actually
    images -- it is 1.8-2.0 %, and steady under foreshortening rather than
    growing with it.

    That is still the price of asking for one cubic. The curve this replaced
    was the projected arc sampled, 0.055-0.128 px off the true one against
    this curve's 1.4-3.3 px; a pointed arch, which keeps a cubic per side,
    pays 0.4-1.3 px. `curve_off` travels with it so the page can say so.

    Only the lengths are free. The directions are the real arch's tangents at
    its springing, and an arch that leaves its springing at a slant is not the
    thing being drawn.

    A cubic's point is linear in those two lengths, so a pass is one
    least-squares solve; between passes every sample takes the parameter of
    the nearest point on the curve so far. Six passes: three leaves 0.15 %
    on the table, ten and twice the samples buy 0.02 % for 3x the time.
    """
    arc = np.asarray(arc, float)
    if len(arc) < 4 or t0 is None or t1 is None:
        return None
    p0, p3 = arc[0], arc[-1]
    t0, t1 = np.asarray(t0, float), np.asarray(t1, float)
    step = np.hypot(*np.diff(arc, axis=0).T)
    sl = np.concatenate([[0.0], np.cumsum(step)])
    if sl[-1] < 1e-9:
        return None
    u = sl / sl[-1]                               # by chord length, to start
    X = np.column_stack([np.interp(np.linspace(0, 1, m), u, arc[:, 0]),
                         np.interp(np.linspace(0, 1, m), u, arc[:, 1])])
    u = np.linspace(0.0, 1.0, m)
    grid = np.linspace(0.0, 1.0, 601)
    P = None
    for _ in range(int(passes)):
        b1, b2 = 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u ** 2
        rhs = (X - ((1 - u) ** 3 + b1)[:, None] * p0
                 - (b2 + u ** 3)[:, None] * p3).ravel()
        A = np.zeros((2 * m, 2))
        A[0::2, 0], A[1::2, 0] = b1 * t0[0], b1 * t0[1]
        A[0::2, 1], A[1::2, 1] = b2 * t1[0], b2 * t1[1]
        ab, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        P = np.array([p0, p0 + ab[0] * t0, p3 + ab[1] * t1, p3])
        q = bez_at(P, grid)
        u = grid[np.linalg.norm(X[:, None] - q[None], axis=2).argmin(1)]
        u[0], u[-1] = 0.0, 1.0
    return P


def curve_off(bez, arc, n=601):
    """How far a Bezier path strays from the arc it stands for, in px.

    The number that has to travel with a single-cubic round arch: it is a
    drawing of a circle, not the circle, and by how much is not a detail the
    page gets to leave out.
    """
    arc = np.asarray(arc, float)
    q = np.vstack([bez_at(P, np.linspace(0.0, 1.0, n)) for P in bez])
    return float(np.linalg.norm(q[:, None] - arc[None], axis=2).min(1).max())


def _h(p):
    """A page point as a homogeneous 3-vector."""
    return np.append(np.asarray(p, float).ravel()[:2], 1.0)


def shadow_tip(l, s, base, top):
    """Where an upright's shadow ends, under a light at `l` with its foot at `s`.

    The ray grazes the top and runs to the light; the shadow runs along the
    ground away from the light's own foot. Both are lines on the page and the
    tip is where they cross. Everything homogeneous, so the sun -- light and
    foot both infinitely far off -- is not a special case.
    """
    return np.cross(np.cross(l, top), np.cross(s, base))


def shadow_stray(T, cap=66):
    """Which object's shadow disagrees with the rest, and by how much.

    Not the worst miss inside one fit, and not leave one out either. Both were
    measured on a tip pushed 40 px sideways and both named the wrong object:
    the in-fit worst landed on object 1, and leaving one out landed on object
    5, because the blunder stayed inside the fit for every fold and the
    length-weighted solve swung toward it once the longest honest shadow was
    the one removed.

    So: two objects determine a light exactly, and the light whose MEDIAN miss
    over all of them is smallest is the one the majority drew. The culprit is
    then whoever misses that light farthest. Needs three objects -- with two
    there is no majority to be in.
    """
    from itertools import combinations
    T = np.asarray(T, float).reshape(-1, 3, 2)
    m = len(T)
    if m < 3:
        return None
    b = np.c_[T[:, 0], np.ones(m)]
    t = np.c_[T[:, 1], np.ones(m)]
    x = np.c_[T[:, 2], np.ones(m)]

    def off(pair):
        """Every tip's distance from the two lines that light puts through it."""
        i, j = pair
        l = np.cross(np.cross(t[i], x[i]), np.cross(t[j], x[j]))
        s = np.cross(np.cross(b[i], x[i]), np.cross(b[j], x[j]))
        d = []
        for v, p in ((l, t), (s, b)):
            L = np.cross(v, p)
            d.append(np.abs((L * x).sum(1))
                     / np.maximum(np.hypot(L[:, 0], L[:, 1]), 1e-12))
        return np.maximum(d[0], d[1])

    pairs = list(combinations(range(m), 2))[:cap]
    best = min((np.median(off(q)), k) for k, q in enumerate(pairs))
    d = off(pairs[best[1]])
    return int(np.argmax(d)), float(np.max(d))


def shadow_light(T, vert, hz=None, window=0.05):
    """The light these cast shadows come from, and its foot under it.

    A light stands over its own foot, so the light's vanishing point, the
    foot's image and the vertical vanishing point are COLLINEAR -- a lamp in
    the room and the sun alike. That leaves the light one degree of freedom:
    WHICH line through the vertical vanishing point it stands on. The sun is
    the same line with its foot pushed infinitely far off, which on the page
    means ON THE HORIZON.

    `T` is (m, 3, 2) -- base, top and marked shadow tip for each object.

    Priced in how far a marked tip is off the two lines it must lie on, each
    measured perpendicular to that line -- px of mark movement and nothing
    else. Two other currencies were measured and dropped: the swing about a
    line's own midpoint hides a blunder on a SHORT mark (a shadow pushed 40 px
    across its own direction swung 8 px and was called sunlight), and the
    distance to where the two lines cross inflates it instead, by 1/sin of the
    angle they cross at (a 9 px hand mark was worth 146 px out there). The
    crossing angle goes out as conditioning, and the tips as what the drawing
    would have to change -- never as the test.

    Not scanned. The points on the line are thousands of px off the page and
    the lever is savage -- 1e-5 rad of slack in that one angle costs 26 px of
    tip movement, so a grid fine enough would need 3e7 steps. The line comes
    out of a construction instead: hold the foot where the ground lines want
    it, and the line through the uprights' vanishing point follows. A true
    picture then comes back at 1e-9 px and the polish is only for drawings
    that are not true.
    """
    from scipy.optimize import minimize_scalar
    from .vp import fit, fit_on_line
    T = np.asarray(T, float).reshape(-1, 3, 2)
    b = np.c_[T[:, 0], np.ones(len(T))]
    t = np.c_[T[:, 1], np.ones(len(T))]
    x = T[:, 2]
    rays = np.c_[T[:, 1], T[:, 2]]
    grd = np.c_[T[:, 0], T[:, 2]]
    xh = np.c_[x, np.ones(len(T))]
    vert = np.asarray(vert, float)

    def lines(l, s):
        """The ray through each top, and the ground line through each base."""
        return np.cross(l, t), np.cross(s, b)

    def cost(l, s):
        d = []
        for L in lines(l, s):
            n = np.maximum(np.hypot(L[:, 0], L[:, 1]), 1e-12)
            d.append(np.abs((L * xh).sum(1)) / n)
        off = np.maximum(d[0], d[1])
        return float(np.max(off)), off

    def pack(l, s):
        c, off = cost(l, s)
        q = shadow_tip(l, s, b, t)
        w = q[:, 2]
        at = None if np.any(np.abs(w) < 1e-9) else q[:, :2] / w[:, None]
        # how squarely the ray and the ground line cross at each tip. Glancing,
        # the tip is not located at all and the movement below is the honest
        # reading while the tip position is not.
        u = [L / np.maximum(np.hypot(L[:, 0], L[:, 1]), 1e-12)[:, None]
             for L in lines(l, s)]
        sin = np.abs(u[0][:, 0] * u[1][:, 1] - u[0][:, 1] * u[1][:, 0])
        return {"worst_px": c, "move_px": off.tolist(), "light": l, "foot": s,
                "crossing_sin": sin.tolist(),
                "tip_at": None if at is None else at.tolist(),
                "tip_px": None if at is None else np.hypot(*(at - x).T).tolist()}

    l0, s0 = fit(rays), fit(grd)
    out = {"free": pack(l0, s0)}
    # the pencil of lines through the vertical vanishing point: every l with
    # l . vert = 0, which is what the two trailing right singular vectors span
    B = np.linalg.svd(vert[None, :] / max(float(np.linalg.norm(vert)), 1e-12))[2][1:]

    def at(th, pin):
        """The best light on the pencil line at angle `th`, and what it costs."""
        L = np.array([np.cos(th), np.sin(th)]) @ B
        s = np.cross(L, hz) if pin else fit_on_line(grd, L)
        if float(np.hypot(s[0], s[1])) < 1e-12:
            return {"worst_px": np.inf, "move_px": None, "tip_px": None,
                    "light": l0, "foot": s}
        return pack(fit_on_line(rays, L), s)

    def polish(pin, starts):
        """Each start is a line already THROUGH `vert`, so it has a pencil angle
        exactly -- projecting a line that misses `vert` into the pencil is a
        coefficient-space nearest, not a geometric one, and on a jittered lamp
        scene it started 43 px out and stayed there."""
        tries = []
        for L0 in starts:
            co = B @ np.asarray(L0, float)
            if float(np.hypot(co[0], co[1])) < 1e-12:
                continue
            th0 = float(np.arctan2(co[1], co[0]))
            tries.append(at(th0, pin))
            try:
                g = minimize_scalar(lambda th: at(th, pin)["worst_px"],
                                    bounds=(th0 - window, th0 + window),
                                    method="bounded", options={"xatol": 1e-13})
                tries.append(at(float(g.x), pin))
            except (ValueError, FloatingPointError):
                pass
        return min(tries, key=lambda d: d["worst_px"]) if tries else None

    # The foot wants to be where the ground lines put it, and the line through
    # the uprights' vanishing point follows from it; the light is then the
    # rays' best point on that line. The light's own free fit gives a second
    # line the same way, and whichever of the two ends up cheaper wins. For the
    # sun the foot is pinned to the horizon first, which is one linear solve
    # and not a search at all.
    for key, starts, pin in (
            ("lamp", [np.cross(s0, vert), np.cross(l0, vert)], False),
            ("sun", ([] if hz is None
                     else [np.cross(fit_on_line(grd, hz), vert),
                           np.cross(l0, vert)]), True)):
        if not len(starts):
            continue
        got = polish(pin, starts)
        if got is not None:
            out[key] = got
    return out


def span_vp_of(K, vert_vp, foot_a, foot_b):
    """The springing line's OWN vanishing point: where the line through the two
    feet meets the horizon.

    Name no wall and this is the plane the arch stands in: the vertical plane
    through its own two supports. Mark a springing line across the bay and you
    get a diagonal rib, where a named wall would have flattened it onto the
    wall's face. The two feet are taken to spring at one height, which is what
    a springing line is.

    None where the feet's line runs level with the horizon and so has no
    vanishing point at all -- there the wall's own has to be named.
    """
    hz = np.linalg.inv(K @ K.T) @ _h(vert_vp)             # the horizon
    q = np.cross(np.cross(_h(foot_a), _h(foot_b)), hz)
    return None if abs(q[2]) < 1e-9 else q[:2] / q[2]


def lean_line(K, vert_vp, span_vp, foot_a, foot_b, crown):
    """The plane of an arch that LEANS off its supports, and by how much.

    The springing line is a hinge: the arch can turn about it, as a gothic rib
    leaves its respond and goes over the bay instead of staying in the wall.
    Every plane through that line has its vanishing line through the span's
    vanishing point, and the arch's upright -- across the span, in its own
    plane -- has its vanishing point on the vanishing line of the plane the
    span is normal to. The crown picks it out: the line from the middle of the
    span through the crown IS that upright, drawn, so where it crosses gives
    the direction and the two together give the plane. One intersection, and
    one mark saying two things at once -- the lean and the rise -- which is
    the trick the vault uses to say the rise and the depth.

    Returns (vanishing line, degrees off the wall), or (None, None) where the
    crown sights along the springing line and nothing says how far it leans.
    """
    M = F.harmonic_mid(foot_a, foot_b, span_vp)           # the middle, on the page
    hinge = np.cross(_h(M), _h(span_vp))                  # the springing line, drawn
    if abs(hinge @ _h(crown)) < float(np.hypot(*hinge[:2])):
        return None, None          # the top is within a pixel of the hinge itself
    w = np.linalg.inv(K @ K.T)
    up = np.cross(np.cross(_h(M), _h(crown)), w @ _h(span_vp))
    l = np.cross(_h(span_vp), up)
    if np.linalg.norm(up) < 1e-12 or np.linalg.norm(l) < 1e-9 * float(np.abs(up).max()):
        return None, None
    Ki = np.linalg.inv(K)
    d, e = Ki @ up, Ki @ _h(vert_vp)
    d, e = d / np.linalg.norm(d), e / np.linalg.norm(e)
    return l, float(np.degrees(np.arccos(min(1.0, abs(float(d @ e))))))


def plane_arch(K, vanish_line, foot_a, foot_b, crown=None, pointed=False,
               span_vp=None, depth_vp=None, n=96, shape=None, with_bez=True,
               through=None):
    """The arch that two feet and a crown mark describe, in that plane.

    Two marks on the page, or three. The two feet are pulled onto one line of the
    springing family -- the arch's two supports end on a line of a family that
    is not their own -- and the crown is snapped over the middle of the span.
    That middle is the middle ON THE PLANE, which on the page is not halfway
    between the feet: in a steep view the nearer half of a true arch is much
    the wider of the two.

    With no crown mark the top goes where a semicircle's would -- half the
    span up, measured on the wall and not on the page, which is the arch
    everyone means by an arch.

    `pointed` is the whole difference between the two arches, and it is one
    line of geometry. A round arch has ONE centre, on the crown's own upright,
    so the two halves leave the top with the same tangent and the top is blunt;
    a pointed arch has TWO, both sitting on the springing line itself, so the
    halves arrive at an angle and the top is sharp. Either way it is drawn as
    two mirrored cubics meeting at the crown.

    Nothing here is a free ellipse: the curve is the image of a real circular
    arch standing in that plane, which needs the camera and not just two
    vanishing points -- two vanishing points name two directions without
    saying the angle between them.

    `through` is one more mark, ON the curve, and it takes the place of the
    round/pointed choice: three points name one circle, so the limb is that
    circle and the toggle has nothing left to decide. Straight sides ignore
    it, there being no circle in a roof.

    `shape` is "round", "pointed" or "straight", and STRAIGHT sides make it a
    roof: the gable is two straight limbs, and slid back by `depth_vp` it is a
    pitched roof -- near gable, far gable, ridge and eaves. Nothing about the
    rest changes, because a straight line in the plane images as a straight
    line on the page.

    None when the arch reaches the plane's own vanishing line, where half of it
    would be drawn behind the viewer and no drawing of it exists.
    """
    shape = shape or ("pointed" if pointed else "round")
    pointed = shape == "pointed"
    H = plane_frame(K, vanish_line)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    def back(p):
        q = Hi @ [float(p[0]), float(p[1]), 1.0]
        return q[:2] / q[2] if abs(q[2]) > 1e-12 else None

    def fwd(p):
        z = H @ [float(p[0]), float(p[1]), 1.0]
        return (z[:2] / z[2]) if abs(z[2]) > 1e-12 else None

    A, Bm = back(foot_a), back(foot_b)
    T = back(crown) if crown is not None else None
    if A is None or Bm is None or (crown is not None and T is None):
        return None
    d = Bm - A
    if span_vp is not None:
        # the springing family's direction, there. A vanishing point of a family
        # that lies in this plane comes back as a direction, not a point.
        q = Hi @ np.append(np.asarray(span_vp, float)[:2], 1.0)
        k = float(np.hypot(*q[:2]))
        if k > 1e-12:
            e = q[:2] / k
            d = e * float((Bm - A) @ e)           # the far foot, pulled onto the line
    span = float(np.hypot(*d))
    if span < 1e-9:
        return None
    d, s = d / span, span / 2.0
    B, M = A + span * d, A + s * d
    # Up is across the springing line. On a real wall that is the direction of
    # the supports too -- a wall's uprights and its horizontals are square to
    # each other in the world, and `plane_frame` is orthonormal, so they come
    # back square here.
    up = np.array([-d[1], d[0]])

    # A vault is an arch slid straight back. Name the depth vanishing point and
    # the crown mark is free to leave the wall: where it sits says the rise AND
    # the depth at once, because its wall position is where the depth line
    # through it crosses the upright over the middle of the span. The slide
    # itself is a homology about that vanishing point with the wall's own
    # vanishing line as axis -- one number, mu, and it is the same number the
    # arch CHECK reads a built vault by.
    mu, far_crown = 0.0, None
    lh = np.asarray(vanish_line, float) / np.linalg.norm(vanish_line)
    if depth_vp is not None and T is not None:
        v = np.append(np.asarray(depth_vp, float)[:2], 1.0)
        c = np.append(np.asarray(crown, float)[:2], 1.0)
        a0, a1 = fwd(M), fwd(M + up)
        if a0 is None or a1 is None:
            return None
        q = np.cross(np.cross(np.append(a0, 1.0), np.append(a1, 1.0)),
                     np.cross(c, v))
        if abs(q[2]) < 1e-9:                  # the crown sights along the wall
            return None
        q = q / q[2]
        T = back(q[:2])
        if T is None:
            return None
        g = (lh @ q) * np.cross(c, v)         # c ~ q + mu (l.q) v, linear in mu
        k = float(g @ g)
        mu = float(g @ -np.cross(c, q) / k) if k > 1e-18 else 0.0
        if abs(mu) * float(np.linalg.norm(v[:2] - q[:2])) < 1e-6:
            mu = 0.0                          # on the upright: a flat arch
        else:
            far_crown = c[:2] / c[2]

    if T is None:
        h = s                                     # no crown asked for: a semicircle
        a, b = fwd(M + s * up), fwd(M - s * up)
        if a is None or b is None:
            return None
        if a[1] > b[1]:                           # of the two, the one up the page
            up = -up
    else:
        h = float((T - M) @ up)
        if h < 0:
            up, h = -up, -h
        if h < 1e-9:                              # the crown is on the springing line
            return None
    apex = M + h * up
    th = np.array([0.0, 1 / 3, 2 / 3, 1.0])

    def flip(q):
        """A plane point across the arch's own upright. An arch is symmetric
        about it, which on the page is no mirror at all."""
        q = np.asarray(q, float) - M
        return M + (q @ up) * up - (q @ d) * d

    # ONE MORE POINT ON THE CURVE. Two feet and a top leave the arch no
    # freedom: the centre is forced onto the springing line (pointed) or onto
    # the crown's own upright (round), and which of the two it was is a
    # toggle. A point on the curve takes the toggle's place and MEASURES it --
    # three points name one circle, so the limb is that circle. It is marked
    # on either limb and mirrored onto the near one, because the other limb
    # is the mirror of this one and not a second thing to mark.
    #
    # On a vault it is read on the NEAR arch -- the wall one, which is what
    # `pts` is -- while the crown mark rides the far one. The two are the same
    # arch slid back, so either says the shape; only the same one must be
    # marked twice, and the mark the app puts down starts on the near curve.
    W = None if through is None or shape == "straight" else back(through)
    if W is not None:
        if (W - M) @ d > 0:
            W = flip(W)                           # marked on the far limb
        if min(np.hypot(*(W - A)), np.hypot(*(W - apex))) < 1e-6:
            W = None                              # on an end, so it says nothing
    cW = None if W is None else _circumcentre(A, W, apex)
    if shape == "straight" or (W is not None and cW is None):
        arcs = [A + np.outer(th, apex - A), apex + np.outer(th, B - apex)]
    elif W is not None:
        # split at the mark, so the curve goes THROUGH it and is not merely
        # near it: one cubic a quarter is also truer than one a half, and a
        # limb that sweeps past a half turn has no short way round to get wrong
        fW, fc = flip(W), flip(cW)
        arcs = [_bez_ctrl(cW, A, W), _bez_ctrl(cW, W, apex),
                _bez_ctrl(fc, apex, fW), _bez_ctrl(fc, fW, B)]
    elif pointed:
        t_ = (h * h - s * s) / (2 * s)            # |c - foot| = |c - apex|, on the line
        c1, c2 = M + t_ * d, M - t_ * d
        arcs = [_bez_ctrl(c1, A, apex), _bez_ctrl(c2, apex, B)]
    else:
        y = (h * h - s * s) / (2 * h)             # the same, up the crown's upright
        c1 = c2 = M + y * up                      # 0 semicircle, <0 segmental, >0 horseshoe
        arcs = [_bez_ctrl(c1, A, apex), _bez_ctrl(c2, apex, B)]
    if any(P is None for P in arcs):
        return None

    def page(q):
        z = (H @ np.column_stack([q, np.ones(len(q))]).T).T
        if np.ptp(np.sign(z[:, 2])) > 0 or np.min(np.abs(z[:, 2])) < 1e-12:
            return None
        return z[:, :2] / z[:, 2:3]

    t = np.linspace(0.0, 1.0, int(n))
    side = [page(bez_at(P, t)) for P in arcs]
    if any(q is None for q in side):
        return None
    pts = np.vstack([side[0]] + [q[1:] for q in side[1:]])
    # The curve DRAWN is a Bezier path, as asked: one cubic for a round
    # arch, two for a pointed one, which is where its corner is. Fitted
    # against a dense sampling of the arc as it really images, not against
    # this one -- `pts` is what the readouts use and is coarse enough that
    # its own sampling would show up in the fit.
    if not with_bez:
        # Asked only for the numbers. Fitting the drawn path is five sixths of
        # the cost of an arch (9.4 ms against 1.6), and the sensitivity run
        # that re-reads one arch under every camera the marks allow does it
        # nine times over -- 102 ms a frame on a drag, which is felt.
        bez, off = [], 0.0
    elif shape == "straight":
        # A straight limb images straight, so the path drawn IS the limb: the
        # control points are thirds along it and it costs nothing.
        q = [z[0] for z in side] + [side[-1][-1]]
        bez, off = [a + np.outer(th, b - a) for a, b in zip(q, q[1:])], 0.0
    else:
        td = np.linspace(0.0, 1.0, 801)
        dq = [page(bez_at(P, td)) for P in arcs]
        if any(z is None for z in dq):
            return None
        ends = [(_dir_fwd(H, P[0], P[1] - P[0]), _dir_fwd(H, P[3], P[2] - P[3]))
                for P in arcs]
        dense = np.vstack([dq[0]] + [z[1:] for z in dq[1:]])
        # one cubic for a round arch, two for a pointed one -- its corner is
        # where they meet -- and one a quarter once a mark on the curve says
        # where the curve goes through
        bez = ([fit_cubic(z, *e) for z, e in zip(dq, ends)] if len(arcs) > 2
               or pointed else [fit_cubic(dense, ends[0][0], ends[-1][1])])
        if any(b is None for b in bez):
            return None
        off = curve_off(bez, dense)
    # An arch is symmetric about its own upright, and on the page that is not
    # a mirror: the near limb images wider than the far one. This is that
    # symmetry as a map from page to page -- reflect across the span, on the
    # PLANE, with the homography either side of it. A handle moved on one
    # limb puts the other through this and lands where the wall says.
    F = np.eye(2) - 2.0 * np.outer(d, d)
    Rp = np.eye(3)
    Rp[:2, :2], Rp[:2, 2] = F, M - F @ M
    mir = H @ Rp @ Hi
    # `mid` is the springing line's middle ON THE PLANE, imaged: the foot of
    # the arch's own upright, and so the other end of the line the crown runs
    # along. Half way between the feet on the page is not it.
    #
    # The plane itself goes out beside the mirror, both ways round. A hand
    # that moves one handle is not always saying "this arch is fuller": on a
    # rib it is saying the plane the rib stands in bends, and a bend pinned
    # at the feet and the crown runs one way between the near foot and the
    # top and the other way beyond it. The page cannot tell those two apart,
    # so the page sends the plane and the hand's move is carried across it
    # rather than reflected in it. See `bezOf`.
    out = dict(pts=pts, feet=[fwd(A), fwd(B)], crown=fwd(apex), mid=fwd(M),
               rise_over_span=float(h / span), pointed=bool(pointed),
               bez=[b.tolist() for b in bez], bez_off=off, shape=shape,
               mirror=(mir / np.max(np.abs(mir))).tolist(), mu=mu,
               plane=[(H / np.max(np.abs(H))).tolist(),
                      (Hi / np.max(np.abs(Hi))).tolist()])
    if mu:
        # the far limb, exactly: the homology IS the slide, so the whole arc
        # goes through it and nothing is approximated here
        Hm = np.eye(3) + mu * np.outer(np.append(np.asarray(depth_vp,
                                                            float)[:2], 1.0), lh)
        out.update(homology=Hm.tolist(), far_crown=far_crown.tolist(),
                   far_feet=[_apply(Hm, f) for f in out["feet"]],
                   far_pts=_apply(Hm, pts).tolist())
        if shape == "straight":
            # Straight sides slid back are a pitched roof. Each rake and its
            # twin on the far gable are ONE direction in the world, so they
            # meet at the slope's vanishing point -- the roof check's first
            # input, read off the drawing instead of named by hand.
            out["slope_vps"] = [_meet(nf, out["crown"], ff, out["far_crown"])
                                for nf, ff in zip(out["feet"], out["far_feet"])]
    return out


def _meet(p, q, r, t):
    """Where the line p-q crosses the line r-t, on the page. None if they run
    parallel, which is where the vanishing point is at infinity."""
    def h(z):
        return np.append(np.asarray(z, float)[:2], 1.0)
    x = np.cross(np.cross(h(p), h(q)), np.cross(h(r), h(t)))
    return None if abs(x[2]) < 1e-9 else (x[:2] / x[2]).tolist()


def _apply(H, p):
    """Page points through a 3x3. One point or many."""
    q = np.atleast_2d(np.asarray(p, float))
    z = (H @ np.column_stack([q, np.ones(len(q))]).T).T
    r = z[:, :2] / z[:, 2:3]
    return r if np.ndim(p) > 1 else r[0].tolist()


def circle_check(pts, K, vanish_line):
    """Is that drawn arc a circle on its plane, or an oval?

    Answers in the only units an artist can act on: the ratio of the two axes
    of the shape as it really is, 1.000 for a circle, and which way the long
    one runs.
    """
    H = plane_frame(K, vanish_line)
    Cp = H.T @ conic(pts) @ H                     # the conic, back on its own plane
    w, V = np.linalg.eigh(Cp[:2, :2] / np.abs(Cp[:2, :2]).max())
    if w[0] * w[1] <= 0: return None              # not an ellipse there
    ax = 1.0 / np.sqrt(np.abs(w))                 # semi-axes, up to one common scale
    return dict(ratio=float(max(ax) / min(ax)), long_on_plane=V[:, int(np.argmin(np.abs(w)))])


def _conic_feet(C):
    """The conic's two extreme points along the span axis, in the wall frame.

    The springing of an arch is where its tangent runs along the rise, and the
    extreme MARK in span is a bad stand-in for it: there the curve is vertical,
    so span changes slowly with arc length and the winning mark slides freely in
    rise. Solved on the fitted conic instead, where every mark votes.

    This is not vacuous even though the conic is symmetric about its own axes:
    the feet are extreme in the WALL's span direction, so an ellipse tilted
    relative to the wall puts them at different heights -- which is exactly an
    arch whose two springings are not level.
    """
    a, b, c = C[0, 0], C[0, 1], C[1, 1]
    d, e, f = C[0, 2], C[1, 2], C[2, 2]
    if abs(c) < 1e-18:
        return None
    # tangent parallel to the rise axis  <=>  d/dy (x^T C x) = 0  <=>  b x + c y + e = 0
    m, k = -b / c, -e / c                                   # y = m x + k
    A = a + 2 * b * m + c * m * m
    B = 2 * (b * k + c * m * k + d + e * m)
    Cc = c * k * k + 2 * e * k + f
    disc = B * B - 4 * A * Cc
    if abs(A) < 1e-18 or disc < 0:
        return None
    r = np.sqrt(disc)
    xs = np.array([(-B - r) / (2 * A), (-B + r) / (2 * A)])
    return np.column_stack([xs, m * xs + k])


def arch_shape(pts, K, vert_vp, wall_vp):
    """The arch's real proportion: rise over span, on its own wall.

    0.500 is a semicircle, below that segmental, above it horseshoe. An axis
    ratio alone cannot tell too-flat from too-tall -- both read >1 -- so work
    in the wall's own frame, where the springing direction is known.
    """
    l = _cross(np.append(np.asarray(vert_vp, float)[:2], 1.0),
               np.append(np.asarray(wall_vp, float)[:2], 1.0))
    H = plane_frame(K, l / np.linalg.norm(l))
    p = np.asarray(pts, float).reshape(-1, 2)
    q = np.linalg.solve(H, np.column_stack([p, np.ones(len(p))]).T)
    q = (q[:2] / q[2]).T                                      # the arc, flat on its wall
    w = np.linalg.solve(H, np.append(np.asarray(wall_vp, float)[:2], 1.0))[:2]
    w = w / np.linalg.norm(w)                                 # the springing direction, there
    s = np.column_stack([q @ w, q @ np.array([-w[1], w[0]])])  # span axis, rise axis
    # Take the axes from the fitted conic, not from where the marks happen to
    # stop. An arch marked only from 10 to 170 degrees has its springing
    # missing, and reading rise and span off the marks alone called a true
    # semicircle 0.419.
    C = conic(s)
    cen = np.linalg.solve(C[:2, :2], -C[:2, 2])
    k = float(cen @ C[:2, :2] @ cen - C[2, 2])
    ev, V = np.linalg.eigh(C[:2, :2] / k)
    if np.any(ev <= 0): return None
    ax = 1.0 / np.sqrt(ev)                       # semi-axes, in the wall's own units
    along = int(np.argmax(np.abs(V[0])))         # which one lies along the springing
    span, rise = 2 * ax[along], ax[1 - along]
    return dict(rise_over_span=float(rise / span), pts_on_wall=s,
                center_on_wall=cen, semi_axes=ax, conic_on_wall=C)


def arch(near, far=None, *, K, vert_vp, wall_vp, normal_vp=None):
    """An arch, flat or projecting. Everything is read on the wall itself.

    Four things, and they fail independently:
      rise/span      -- 0.500 is a semicircle; this is the arch's proportion
      crown offset   -- the top has to sit over the middle of the span
      springing      -- the two feet have to sit at the same height
      depth miss     -- for a projecting arch, the far arc is the near one slid
                        straight back: in the image that is a homology about the
                        normal's vanishing point, one number for the whole arc.
                        This is what catches a far arc drawn the wrong size,
                        which the proportion checks let through.
    """
    l = _cross(np.append(np.asarray(vert_vp, float)[:2], 1.0),
               np.append(np.asarray(wall_vp, float)[:2], 1.0))
    l = l / np.linalg.norm(l)
    out = {"vanish_line": l, "rise_over_span": [], "crown_off_frac": [], "springing_off_frac": []}
    for a in (near, far):
        if a is None: continue
        s = arch_shape(a, K, vert_vp, wall_vp)
        q = s["pts_on_wall"]; span = np.ptp(q[:, 0])
        out["rise_over_span"].append(s["rise_over_span"])
        # Which way is up on the wall has an arbitrary sign, and argmax on it
        # picks a foot, not the crown. The chord between the two springing
        # points needs no sign: the crown is the point furthest off it.
        feet = _conic_feet(s["conic_on_wall"])
        if feet is None:
            feet = q[[np.argmin(q[:, 0]), np.argmax(q[:, 0])]]
        # ...but taking that furthest point from the MARKS is 48x noisier than
        # taking it from the fitted conic (std 0.0152 vs 0.0003 of span, 60
        # trials at +-2 px), because a round arch's apex is flat: the winning
        # mark wanders sqrt(4 sigma r) either side and nothing about denser
        # marking helps (721 marks read the same as 181). The conic's centre
        # sits under the crown by construction, and every mark votes on it.
        out["crown_off_frac"].append(float((s["center_on_wall"][0] - feet[:, 0].mean())
                                           / span))
        out["springing_off_frac"].append(float(abs(feet[0, 1] - feet[1, 1]) / span))
    if far is not None and normal_vp is not None:
        x = np.column_stack([np.asarray(near, float).reshape(-1, 2), np.ones(len(near))])
        y = np.column_stack([np.asarray(far, float).reshape(-1, 2), np.ones(len(far))])
        v = np.append(np.asarray(normal_vp, float)[:2], 1.0)
        # x' ~ x + mu (l.x) v : one mu for the whole arc. Fit it, then look at
        # what is left in pixels -- the ratio of scatter to mu blows up when mu
        # crosses zero and says nothing an artist can use.
        def miss(mu):
            z = x + mu * (x @ l)[:, None] * v
            return np.linalg.norm(z[:, :2] / z[:, 2:] - y[:, :2], axis=1)
        # x' ~ x + mu (l.x) v means cross(x', x + mu (l.x) v) = 0, which is
        # linear in mu -- no search needed.
        A = np.vstack([(x @ l)[:, None] * np.cross(yi, v) for yi in (y,)][0])
        b = -np.vstack([np.cross(y, x)])
        mu = float(np.linalg.lstsq(A.reshape(-1, 1), b.reshape(-1), rcond=None)[0][0])
        out["depth"] = mu
        out["depth_miss_px"] = miss(mu)
        out["depth_rms_px"] = float(np.sqrt((miss(mu) ** 2).mean()))
    return out


def segment_depth(x1, x2, v, K):
    """How far a segment of unit length is from the camera, at each end.

    The one thing a calibrated camera adds that no amount of marking can: a
    real size in the SCENE, turned into a real distance. A segment whose world
    direction has a known vanishing point ``v`` makes a triangle with the
    camera centre, and that triangle is rigid up to scale --

        t1 * r1 + s * d = t2 * r2

    with r1, r2 the unit rays through the endpoints and d the unit world
    direction, all in camera coordinates. Three coplanar vectors, so the null
    space is one-dimensional: fix s = 1 and t1, t2 come out as multiples of the
    segment's own length. Hand back a real length instead and they are
    distances.

    ``v`` is homogeneous, so a direction parallel to the picture plane -- the
    uprights of a level camera, which have no vanishing point on the page --
    arrives as [vx, vy, 0] and works like any other.

    Returns (t1, t2) per unit of segment length, or None when the triangle is
    degenerate: a segment pointing along its own line of sight has no depth to
    give, and says so rather than returning the large number the arithmetic
    would otherwise produce.
    """
    try:
        Ki = np.linalg.inv(np.asarray(K, float))
    except np.linalg.LinAlgError:
        return None
    r = []
    for x in (x1, x2):
        p = Ki @ np.append(np.asarray(x, float)[:2], 1.0)
        n = np.linalg.norm(p)
        if n < 1e-12 or not np.isfinite(n):
            return None
        r.append(p / n)
    d = np.asarray(v, float).ravel()
    d = Ki @ (np.append(d[:2], 1.0) if d.size == 2 else d)
    n = np.linalg.norm(d)
    if n < 1e-12 or not np.isfinite(n):
        return None
    d /= n
    M = np.column_stack([r[0], d, -r[1]])
    sv = np.linalg.svd(M)
    null = sv[2][-1]
    if abs(null[1]) < 1e-12:
        return None
    null = null / null[1]                       # s = 1: one unit of real length
    t1, t2 = float(null[0]), float(null[2])
    if not (np.isfinite(t1) and np.isfinite(t2)):
        return None
    # Both ends must be in front of the camera. The direction's sign is free --
    # a vanishing point names a line, not an arrow -- so flip it rather than
    # refuse a segment marked top-to-bottom.
    if t1 < 0 and t2 < 0:
        t1, t2 = -t1, -t2
    return (t1, t2) if (t1 > 0 and t2 > 0) else None
