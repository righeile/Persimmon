"""Linear-perspective check: lines -> vanishing points -> per-line residual."""
import cv2, numpy as np

def lines(img, min_len=40, clahe=True, join=0.015):
    """Straight segments, from a path or from pixels already in hand.

    It takes an array as well as a path because a trimmed picture exists only
    in memory -- the user's original file is never rewritten -- and detection
    has to run on what the user is looking at, not on the untrimmed file.

    LSD breaks a drawn line wherever the stroke thins or the grain shows
    through: on 4.jpg 81 % of the line it found lay in pieces under 40 px, and
    the minimum length threw them away. The pieces of one straight line are
    joined first, across gaps up to `join` picture diagonals (0 joins
    nothing), and the minimum length applies to what they make.
    """
    g = cv2.imread(img, cv2.IMREAD_GRAYSCALE) if isinstance(img, (str, bytes)) \
        else (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
    if clahe:
        g = cv2.createCLAHE(3.0, (8, 8)).apply(g)
    segs = cv2.createLineSegmentDetector().detect(g)[0]
    if segs is None: return np.zeros((0, 4)), g
    s = segs.reshape(-1, 4).astype(float)
    diag = float(np.hypot(*g.shape[:2]))
    # a piece under 0.6 % of the diagonal has no direction to join by
    s = s[_len(s) >= 0.006 * diag]
    if join > 0:
        s = _join(s, gap=join * diag)
    return s[_len(s) >= min_len], g


def _len(s):
    return np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])


def _pairs(s, deg):
    """Every pair of segments within `deg` of each other in direction, once."""
    th = np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])) % 180
    o = np.argsort(th)
    ext, oo = np.r_[th[o], th[o] + 180], np.r_[o, o]        # 179 deg is next to 0
    hi = np.searchsorted(ext, th[o] + deg, "right")
    I = np.repeat(o, hi - np.arange(len(o)) - 1)
    J = np.concatenate([oo[k + 1:h] for k, h in enumerate(hi)]) if len(o) else np.zeros(0, int)
    keep = I != J
    return I[keep], J[keep]


def _join(s, gap, off=1.0, overlap=2.0):
    """Pieces of one straight line, joined into it.

    Two pieces join when they run within 2 deg of each other, end to end with
    less than `gap` between them, and one straight line passes within `off` px
    of every end of every piece joined so far. It is every end, not the last
    pair, or a gentle arc is walked round one short step at a time. Pieces
    side by side do not join: the two edges of one stroke, 3-4 px apart, stay
    two lines. Each joined line is fitted through its pieces' ends, weighted
    by the pieces' length.
    """
    pts = [np.array([q[:2], q[2:]]) for q in s]
    wts = [np.full(2, l) for l in _len(s)]
    for _ in range(12):                      # a piece joins once a round: 2**12 pieces
        if len(s) < 2: break
        I, J = _pairs(s, 2.0)
        L = _len(s)
        u = (s[:, 2:] - s[:, :2]) / L[:, None]
        nrm = np.c_[-u[:, 1], u[:, 0]]
        p = s[I, :2]
        d1, d2 = ((s[J, :2] - p) * nrm[I]).sum(1), ((s[J, 2:] - p) * nrm[I]).sum(1)
        t1, t2 = ((s[J, :2] - p) * u[I]).sum(1), ((s[J, 2:] - p) * u[I]).sum(1)
        g = np.maximum(np.minimum(t1, t2) - L[I], -np.maximum(t1, t2))   # < 0 overlaps
        ok = (np.maximum(np.abs(d1), np.abs(d2)) < 3 * off + 1) & (g < gap) & (g > -overlap)
        used = np.zeros(len(s), bool)
        new = []
        for k in np.flatnonzero(ok)[np.argsort(g[ok])]:      # nearest first
            i, j = I[k], J[k]
            if used[i] or used[j]: continue
            P, w = np.r_[pts[i], pts[j]], np.r_[wts[i], wts[j]]
            c = (P * w[:, None]).sum(0) / w.sum()
            d = np.linalg.svd((P - c) * np.sqrt(w)[:, None], full_matrices=False)[2][0]
            if np.abs((P - c) @ np.array([-d[1], d[0]])).max() > off: continue
            t = (P - c) @ d
            new.append((np.r_[c + t.min() * d, c + t.max() * d], P, w))
            used[i] = used[j] = True
        if not new: break
        keep = np.flatnonzero(~used)
        s = np.r_[s[keep], [q for q, _, _ in new]]
        pts = [pts[i] for i in keep] + [P for _, P, _ in new]
        wts = [wts[i] for i in keep] + [w for _, _, w in new]
    return s

def resid_deg(s, vp):
    """Angle between each segment and the ray from its midpoint to vp, degrees."""
    mx, my = (s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2
    d = np.stack([s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]], 1)
    if abs(vp[2]) < 1e-9:                      # vp at infinity: parallel family
        r = np.broadcast_to(vp[:2], d.shape)
    else:
        r = np.stack([vp[0] / vp[2] - mx, vp[1] / vp[2] - my], 1)
    c = np.abs(d[:, 0] * r[:, 1] - d[:, 1] * r[:, 0])
    dot = np.abs(d[:, 0] * r[:, 0] + d[:, 1] * r[:, 1])
    return np.degrees(np.arctan2(c, dot))

def fit(s, idx=None):
    """The vanishing point of a set of segments, weighted by their length.

    Homogeneous: the third component is ~0 when the family is parallel on the
    page, so the caller divides through only after checking it. Lifted out of
    the fit_vp endpoint when the stray test needed the same solve -- two
    copies of a weighting rule is one copy too many.
    """
    return np.linalg.svd(_weighted(s, idx))[2][-1]


def _weighted(s, idx=None):
    """Each segment as a line, scaled by its length so long marks count more."""
    s = np.asarray(s, float)
    idx = np.arange(len(s)) if idx is None else np.asarray(idx)
    L = np.hypot(s[idx, 2] - s[idx, 0], s[idx, 3] - s[idx, 1])
    p1 = np.c_[s[idx, :2], np.ones(len(idx))]
    p2 = np.c_[s[idx, 2:], np.ones(len(idx))]
    H = np.cross(p1, p2)
    return H * (L / np.maximum(np.linalg.norm(H[:, :2], axis=1), 1e-12))[:, None]


def fit_on_line(s, l, idx=None):
    """The same fit, but with the point held on the line `l`.

    Every family of horizontals shares one horizon, whichever way it runs, so
    a third family's vanishing point belongs on the line the first two give.
    Fitting it free and then noting that it misses says the drawing is wrong;
    fitting it ON the horizon says what the smallest change would be, which is
    the question this app asks.

    `l` is homogeneous [a, b, c]. The solve is the free one restricted to the
    two-dimensional space l . v = 0: the right singular vectors of `l` past
    the first span exactly that space, so the constraint costs a 3x2 multiply
    and no penalty weight to choose.
    """
    s = np.asarray(s, float)
    idx = np.arange(len(s)) if idx is None else np.asarray(idx)
    H = _weighted(s, idx)
    B = np.linalg.svd(np.asarray(l, float)[None, :])[2][1:]
    return B.T @ np.linalg.svd(H @ B.T)[2][-1]


def horizon(vp_a, vp_b):
    """The line through two vanishing points, normalised."""
    l = np.cross(np.append(np.asarray(vp_a, float)[:2], 1.0),
                 np.append(np.asarray(vp_b, float)[:2], 1.0))
    n = float(np.linalg.norm(l[:2]))
    return None if n < 1e-12 else l / n


def strays(s, vp, mark_px, factor=4.0):
    """Which lines in a family do not belong to its vanishing point.

    A residual in degrees is the wrong yardstick. A short segment can be two
    degrees off and still be inside the slop of the hand that drew it, where a
    long one cannot: on the picture this was written for, a 409 px line at
    2.44 deg asks for 8.7 px of movement and a 1149 px line at 4.29 deg asks
    for 43 px. So what is measured is the swing the line's own ends must make
    about its midpoint -- half its length times the tangent of its residual.

    Measured LEAVE ONE OUT, against the point the other lines agree on. A
    stray judged inside its own fit drags that fit toward itself and inflates
    everyone else's residual with it, so it hides in the baseline it is being
    compared against: a 4 deg twist on the two longest lines of the test
    family scored 3.2 and 3.7 times the median in-fit, and 4.7 and 5.2 times
    it left out.

    A line is a stray when it needs more than a hand mark is worth AND more
    than `factor` times what the rest of the family needs. Both conditions,
    because a family where every line wants 3 px and one wants 12 holds no
    stray, only a hand; and a family of careless marks should not have its
    worst member singled out when they are all equally careless. The second
    condition sets the floor of what can be convicted: on a 6583 px diagonal
    a hand mark is worth 38 px, so a line under about 1200 px can be 4 deg out
    and still not be called -- its ends simply do not move far enough to tell.

    `vp` is the family's own fit, used only when there are too few lines to
    leave any out. Returns (move_px per line, indices of the strays).
    """
    s = np.asarray(s, float)
    half = 0.5 * np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
    if len(s) < 3:
        # Two lines meet at exactly one point: their residuals are zero by
        # construction and leaving one out leaves nothing to fit.
        return half * np.tan(np.radians(resid_deg(s, vp))), []
    move = np.array([
        half[i] * np.tan(np.radians(resid_deg(
            s[i:i + 1], fit(s, [j for j in range(len(s)) if j != i]))[0]))
        for i in range(len(s))])
    typical = max(float(np.median(move)), 1e-6)
    bad = (move > mark_px) & (move > factor * typical)
    return move, np.flatnonzero(bad).tolist()


def tidy(s, vp, mark_px, line=None):
    """Swing the lines that are only hand-shaky onto the point the rest agree on.

    Marking is coarser than drawing: a line drawn true can still be marked a
    degree off, and that wobble is the marker's, not the artist's. With three
    lines or more the family is over-determined, so the two can be told apart
    -- anything a line asks for that is within what a hand mark is worth here
    is marking noise and gets absorbed, and anything beyond it is left exactly
    as marked, because that is the drawing talking.

    Each line turns about its own midpoint and keeps its length, which is the
    same motion `strays` measures, so what is absorbed is exactly what was
    measured. Returns (lines, which moved, the largest swing applied).

    Afterwards those lines agree with the point by construction and their
    residuals say nothing about how well the drawing was made -- the caller
    has to carry that, which is what `tidied_why` in the readout is for.
    """
    s = np.asarray(s, float).copy()
    # Three lines is the floor for a free tidy: two meet at exactly one point,
    # so there is nothing to tell the hand's wobble from the drawing's. Given
    # a horizon there is -- two lines and a line they must meet on is one
    # constraint more than the point has freedom.
    if len(s) < (2 if line is not None else 3) or abs(vp[2]) < 1e-12:
        return s, [], 0.0
    move, bad = strays(s, vp, mark_px)
    # The target is the point the lines that are NOT strays agree on. Pulling
    # them onto the whole family's own fit lets the stray pull back: tidying
    # the test family five times over walked its vanishing point 174 px and
    # the stray stopped being flagged at all, laundered into agreement by the
    # lines that had been dragged toward it.
    keep = [i for i in range(len(s)) if i not in bad]
    if len(keep) < 2:
        target = vp
    elif line is None:
        target = fit(s, keep)
    else:
        # held on the horizon the OTHER families give. The horizon is never
        # this family's own fit, so this cannot walk: one pass lands on a
        # point that does not move when the pass is repeated.
        target = fit_on_line(s, line, keep)
    if abs(target[2]) < 1e-12:
        return s, [], 0.0
    p = np.asarray(target[:2], float) / target[2]
    moved, swing = [], 0.0
    for i in np.flatnonzero(move <= mark_px):
        mid = np.array([(s[i, 0] + s[i, 2]) / 2, (s[i, 1] + s[i, 3]) / 2])
        d = p - mid
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue                       # the mark sits on the point itself
        u = d / n * 0.5 * float(np.hypot(s[i, 2] - s[i, 0], s[i, 3] - s[i, 1]))
        # keep the ends in the order they were marked, so the handle a person
        # grabbed before the tidy is the same handle afterwards
        if (s[i, 2] - s[i, 0]) * d[0] + (s[i, 3] - s[i, 1]) * d[1] < 0:
            u = -u
        # the swing reported is the one applied, not the leave-one-out figure
        # that chose the line: with a horizon to aim at they differ, and the
        # page was printing 20 px for a move that turned out to be 13.
        # It is also the gate, so nothing is ever moved further than a hand
        # mark -- two lines have a zero residual by construction, and without
        # this the horizon could swing them any distance it liked.
        sw = float(np.hypot(*(mid + u - s[i, 2:])))
        if sw > mark_px:
            continue
        swing = max(swing, sw)
        s[i, :2], s[i, 2:] = mid - u, mid + u
        moved.append(int(i))
    return s, moved, swing


def _homog(s):
    p1 = np.c_[s[:, :2], np.ones(len(s))]
    p2 = np.c_[s[:, 2:], np.ones(len(s))]
    return np.cross(p1, p2)

def vps(s, size, n=3, tol=1.5, iters=3000, min_inl=4, floor=0.7, seed=0):
    """Sequential RANSAC. Returns [(vp, inlier_mask)]; leftovers stay unassigned.

    Lines are sampled and scored by their length, not one vote each, and the
    search ends at a family holding less than `floor` picture diagonals of
    line in all (`size` is the picture's (h, w)). Twelve lines at least, as it
    was, never offered the ten 166-255 px pole edges of shadowtest.png, 1.2
    diagonals of line. On 40 synthetic pictures the floor of 0.7 took the
    families matching no vanishing point from 18 to 5 and lost one of 105
    found. Each refit is weighted by length, as a marked family's is.
    """
    rng = np.random.default_rng(seed)
    L = _len(s)
    floor *= float(np.hypot(*size))
    free = np.ones(len(s), bool)
    out = []
    for _ in range(n):
        idx = np.flatnonzero(free)
        if len(idx) < min_inl: break
        H = _homog(s[idx])
        p = L[idx] / L[idx].sum()
        best = (0.0, None)
        for _ in range(iters):
            a, b = rng.choice(len(idx), 2, replace=False, p=p)
            v = np.cross(H[a], H[b])
            if np.linalg.norm(v) < 1e-12: continue
            sc = L[idx][resid_deg(s[idx], v) < tol].sum()
            if sc > best[0]: best = (sc, v)
        if best[1] is None: break
        v = best[1]
        for _ in range(5):                      # refit through the inliers
            m = resid_deg(s[idx], v) < tol
            if m.sum() < 2: break
            v = fit(s, idx[m])
        m = resid_deg(s[idx], v) < tol
        if m.sum() < min_inl or L[idx][m].sum() < floor: break
        full = np.zeros(len(s), bool); full[idx[m]] = True
        free &= ~full
        out.append((v / (np.linalg.norm(v[:2]) or 1), full))
    return out

def sheet(path, inset=0.03):
    """The drawing surface as a quad, shrunk inward to drop tape and edges.
    approxPolyDP is not reliable here -- on one photo it bottomed out at 5
    vertices for every tolerance -- so reduce the convex hull to 4 instead,
    dropping whichever vertex costs the least area each time."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, th = cv2.threshold(cv2.GaussianBlur(g, (9, 9), 0), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    c = max(cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            key=cv2.contourArea)
    h = cv2.convexHull(c).reshape(-1, 2).astype(float)
    while len(h) > 4:
        a = h[np.arange(len(h)) - 1] - h            # numpy 2.5 has no 2-D cross
        b = h[(np.arange(len(h)) + 1) % len(h)] - h
        tri = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        h = np.delete(h, int(np.argmin(tri)), axis=0)
    q = h[np.argsort(np.arctan2(h[:, 1] - h[:, 1].mean(), h[:, 0] - h[:, 0].mean()))]
    return q + (q.mean(0) - q) * inset


def inside(s, quad, frac=1.0):
    """Keep segments whose endpoints both sit inside the quad."""
    q = quad.astype(np.float32).reshape(-1, 1, 2)
    ok = lambda p: cv2.pointPolygonTest(q, (float(p[0]), float(p[1])), False) >= 0
    return np.array([ok(x[:2]) and ok(x[2:]) for x in s], bool) if len(s) else np.zeros(0, bool)

def merge(s, fams, size, tol=1.5, apart=3.0):
    """Two families are one if their pooled lines still share a vanishing point
    and their own points lie within `apart` degrees of one direction from the
    eye (0 merges nothing).

    RANSAC splits a wide family (the columns) in two; the pooled refit is the
    test, but not on its own: a big family keeps the pooled median low
    whatever joins it. It pooled a family of 2.jpg whose lines sat 4.2 deg
    from the pooled point, and on 4.jpg the folds of a curtain into the
    verticals, which moved them 1 deg. Directions are read as a lens as long
    as the picture's diagonal would see them (`size` is (h, w)); the halves
    of one split family came 2.4-2.9 deg apart, the curtain 15.
    """
    fams = list(fams)
    d = float(np.hypot(*size))
    Ki = np.linalg.inv([[d, 0, size[1] / 2], [0, d, size[0] / 2], [0, 0, 1]])
    changed = True
    while changed and len(fams) > 1:
        changed = False
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                a, c = Ki @ fams[i][0], Ki @ fams[j][0]
                if abs(a @ c) < np.cos(np.radians(apart)) * np.linalg.norm(a) * np.linalg.norm(c):
                    continue
                pool = fams[i][1] | fams[j][1]
                v = fit(s, np.flatnonzero(pool))
                if np.median(resid_deg(s[pool], v)) < tol:
                    fams[i] = (v / (np.linalg.norm(v[:2]) or 1), pool)
                    fams.pop(j); changed = True; break
            if changed: break
    L = _len(s)
    return sorted(fams, key=lambda f: -L[f[1]].sum())

def _reach(seg, quad):
    """How far the segment could run, along its own direction, before leaving the sheet.
    LSD chops one drawn line into fragments, so a fragment's own length understates
    the error: 2 deg over 47 px is 0.5 mm, the same 2 deg across the page is 8 mm."""
    m = np.array([(seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2])
    d = np.array([seg[2] - seg[0], seg[3] - seg[1]], float)
    d /= np.linalg.norm(d)
    best = np.hypot(seg[2] - seg[0], seg[3] - seg[1]) / 2
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        e = b - a
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-9: continue
        t = ((a[0] - m[0]) * e[1] - (a[1] - m[1]) * e[0]) / den
        u = ((a[0] - m[0]) * d[1] - (a[1] - m[1]) * d[0]) / den
        if 0 <= u <= 1: best = max(best, abs(t))
    return best


def report(s, fams, quad, px_per_mm):
    """Per-line angular error, and the gap it opens at the edge of the sheet."""
    rows = []
    for k, (v, m) in enumerate(fams):
        r = resid_deg(s[m], v)
        for seg, a in zip(s[m], r):
            gap = _reach(seg, quad) * np.tan(np.radians(a)) / px_per_mm
            rows.append((k, seg, a, gap))
    return sorted(rows, key=lambda t: -t[3])

def not_edge(s, quad, near_px=22, near_deg=6):
    """Drop segments lying along the sheet boundary: paper edge, tape, board.
    A blunt inward inset removes drawing too -- the vertical family's residual
    swung 0.51-1.27 deg across insets while the ceiling family never moved."""
    keep = np.ones(len(s), bool)
    mid = np.stack([(s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2], 1)
    d = np.stack([s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]], 1)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        e = b - a; L = np.linalg.norm(e); e = e / L
        r = mid - a
        t = np.clip(r @ e, 0, L)
        dist = np.linalg.norm(r - t[:, None] * e, axis=1)
        ang = np.degrees(np.arccos(np.clip(np.abs(d @ e), 0, 1)))
        keep &= ~((dist < near_px) & (ang < near_deg))
    return keep


def short_for_vp(median_len, vp, w, h):
    """True when segments this short cannot locate a vanishing point this far.

    Bug 9: 28-46 px mouldings put a vanishing point at x=-153 with a bootstrap
    of [-178, -49] while four long marked lines gave -18, and the short-segment
    answer was briefly believed.

    Both thresholds are fractions of the picture, not pixel counts. As absolute
    numbers read off one 1556 px diagonal scan they failed twice over: on a
    bigger scan every family looks long enough, and the distance test looked
    only at x, so a vertical family's vanishing point 59 000 px ABOVE a 884x1280
    picture counted as nearby and went out unflagged.
    """
    if vp is None:                      # a parallel family has no point to be far from
        return False
    diag = float(np.hypot(w, h))
    far = float(np.hypot(vp[0] - w / 2.0, vp[1] - h / 2.0)) > 3.0 * diag
    return bool(float(median_len) < 0.039 * diag and far)
