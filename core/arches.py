"""Arches in a picture, for the page to offer.

An edge finder cuts a curve into short straight pieces, each a little turned
from the last. Pieces that meet end to end and keep turning one way are a
curve, and it is an arch when it is big, smooth and bows upward.

Those same pieces, joined, are what the line families are made of, so a curve
is taken out of them here. Many curves over one point are the ribs of a vault
seen along its length, and they are sorted by whether each stands in one plane.
A regular vault repeats, so the rule it was built by can be deduced from the ribs
found, and the rest of it drawn. Any other vault repeats too: whatever its pattern,
curves that are copies of each other moved along it are found together, and the
picture itself says at what bay they stand.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .checks import TURN_MIN_DEG
from .rhythm import conic, conic_dist, conic_resid
from .vp import drawn, fit, resid_deg

GAP_PX = 5.0             # pieces further apart than this are not one curve
TURN_MAX_DEG = 40.0      # one join turns less than this, or it is a corner
TURN_BACK_DEG = 1.5      # a join may wobble this far against the turn
BOW_PX = 3.0             # a piece of a curve stands this far off it at most
LEAST_TURN_DEG = 60.0    # less than this is not a curve anybody would call one
LEAST_SPAN = 0.05        # picture diagonals across
MOST_RMS_PX = 1.5        # further from a smooth conic than this is ornament
SAME_PX = 8.0            # the two sides of one molding
WALL_MISS_DEG = 4.0      # an arch misses its own wall by less than this
WALL_APART_DEG = 2.0     # and the next wall by this much more, or it is not said

CHORD_TURN_DEG = 25.0    # a curve turning this far takes its lines out of the families
RIB_TURN_DEG = 20.0      # one turning less is taken only if a vault vouches for it
LINE_GAP_PX = 30.0       # the lines of one curve lie further apart than its pieces
LINE_CORNER_DEG = 30.0   # one join of lines turns less than this
LINE_MISS_PX = 4.0       # and they are coarser, so a curve through them misses by this
TRACE_BLUR = 2.0         # px: edges are followed at this scale, over the grain of the stone
TRACE_TOP = 85           # percent of the picture lies under a strong edge
TRACE_LEAST = 40.0       # and less than this is noise, however much of the picture is blank
TRACE_FEWEST = 8         # px: less of an edge than this is grain
TRACE_GAP_PX = 8.0       # the pieces of one followed edge lie further apart than the edge finder's
ON_CURVE_PX = 4.0        # a line of a curve stays this near it
ON_CURVE_PART = 0.6      # for this much of its length
NEXT_LINE = 0.035        # picture diagonals: the next line of a curve is this near
MOLDING = 0.006          # picture diagonals: the other edges of a molding are this near

FEWEST_RIBS = 12         # fewer curves than this are no vault
FEWEST_SIDE = 4          # ribs a side needs before it has a shape of its own
LEAST_SWEEP_DEG = 3.0    # a rib sweeps this far round the point, or says nothing
SHAPE_POWERS = (2, 3, 4) # the shape all the ribs of a side share
RIB_MISS = 0.006         # picture diagonals: a vault's ribs stay this near that shape
MEET_STEP = 0.03         # of the picture's sides: the grid of points where lines might meet
MEET_WIDE_DEG = 1.5      # a line through a cell of it passes this near
MEET_TRIES = 150         # meeting points the ribs are tried about
SAME_POINT = 0.01        # picture diagonals: meeting points this near are one
CLOSE_FIT = 1.3          # the ribs fit a point this many times the best, and cannot tell them apart
MEET_DEG = 0.5           # a line through the point passes this near
BUNDLE_DEG = 12.0        # lines this close in direction are one bundle
FEWEST_ACROSS = 5        # lines beyond it, or nothing says where the point is
LEAST_COVER = 0.4        # of that sweep: a curve crossing less is arcade or window, no rib
CROWN_GAP = 1 / 3        # of that sweep: a rib stopping further short has no crown to read
HALF_BAYS = np.arange(.04, .3, .005)  # half a bay over the depth of the nearest rib
BAND_STEP = 0.002        # as a log: how finely a side's flat ratio is sought
BAND_WIDTH = 0.012       # as a log: how far a rib lies off its line
SAMPLES_PER_FACT = 4     # readings half a degree apart are not independent
RUN_SD = 0.015           # a rib read no better than this says nothing: half a bay is 5-10 %
DRAWS = 400              # ratios with no lines in them, the ribs set against
MOST_CHANCE = 0.02       # this share of those may do as well, or the ribs show no lines

COMB_SPAN = .75, 1.4     # half a bay as the feet read it, against its real step
COMB_TOOTH = 0.2         # of a step: how far a crown lies off its tooth
FOOT_REACH = 90          # percent of the ribs a side stop short of the foot
SECTION_EYE = 0.6        # of the crown's height: the section is read round a point this high
FLAT_POWERS = 3          # the section's curve, each side
OFF_POWERS = 2           # a 3D rib's section against it, tied at crown and foot
TRIM = 2.5               # spreads: further off is a jog, where a curve changes edge of its molding
RULE_MISS = 0.009        # picture diagonals: a rib on the rule stays this near it
RULE_NEXT = 1.5          # and this many times nearer than to any other, or its family is not said
RULE_RUNS = np.linspace(-1.5, 1.5, 13)  # half bays: where a foot might stand with no rule to set it
RULE_BEAT = 3.3          # spreads: this far above the ribs that would follow it by chance, or it is not theirs
RULE_SEEN = 0.3          # of its own ribs in sight this many are found, or it is a fine comb and no vault
CROWN_STEPS = np.arange(.04, .3, .002)  # how far apart the crowns may repeat, of the nearest's depth
PEAK_APART = 0.06        # they repeat within this of the rule's step, or it is not their rule
RULE_POINTS = 40         # points along each rib built
FEWEST_ASKED = 20        # curves with no point: only this many are worth asking for one
RAYS = 90                # angles a vault's curves are read on, about its point
LEAST_RAYS = 6           # a curve crossing fewer says nothing
HOLE_RAYS = 4            # a curve's points are this far apart, and further it has broken
FOLLOW = .7              # of the rays it crosses, a curve follows its fellows over this much
PENCIL_MISS = .005       # picture diagonals: and stays this near them
PENCIL_TILT = .5         # of that: with no more of it in a tilt across its rays
SEEDS = 80               # the longest curves, tried in pairs
FEWEST_APART = 3         # copies of one curve that stand clear of each other, at the least
MOST_PENCILS = 8         # sets of copies read
BEHIND_DEG = 115.0       # from straight up: past this the curves are on the floor
LINE_PX = 1.0            # lines to a point are good to this: three can agree by luck alone
EDGE_BLUR = 1.2          # px
SWEEP_RAYS = 48          # of a set's rays, the most swept
COMB_STEP = .5           # px along the comb
FAR_PX = 40.0            # a copy this small about the point is past reading
BUMP_PX = 60.0           # broader than this is shading, not an edge
PITCH = 8.0              # px: copies nearer each other than this cannot be told apart
BAY_RANGE = .04, .7      # of the nearest rib's depth, as in CROWN_STEPS
SPIKE = 2.5              # spreads above a comb's level: a tooth
SPIKE_APART = 3.0        # px: the tallest within this is one tooth
FEWEST_PHASES = 4        # teeth a period is read from
LONGER_NEARLY = .6       # of the best: a longer period doing this well is the period
HARMONICS = 3            # of a period: what a comb of narrow teeth is tested on
HARMONIC_TRIES = 6       # and that many times as many periods it has tried, on teeth at random
SAME_BAY = .15           # as a log: families sharing rays keep one bay, this closely
MOST_ANY_BAY = .05       # in the whole picture: teeth lying this well on some bay by chance
MOST_THE_BAY = .02       # and on the one bay already found
ONE_BAY = 2 * SAME_BAY / np.log(BAY_RANGE[1] / BAY_RANGE[0])     # two sets read alone agree on their bay this often by luck
TOOTH_REACH = .12        # of a bay: a copy stands this near its place, a little uneven
IN_FRAME = .6            # of a copy, or it is not offered
FEWEST_FOUND = 3         # copies with a tooth, for a family to stand
FLAT_RUN = .2            # bays: a rib running less than this from foot to crown is flat
FLAT_LEAN = .15          # and one coming in from the wall less than this for each unit it rises
CALL_CLEAR = 2.0         # spreads of its run: a family stands this clear of that, or it is not called
MOST_RUN = 1.2           # bays: one running further is no rib of the vault


def _pieces(gray):
    """Each piece twice, once each way: i ^ 1 is i reversed."""
    s = cv2.createLineSegmentDetector().detect(gray)[0]
    if s is None:
        return np.zeros((0, 4))
    s = s.reshape(-1, 4).astype(float)
    s = s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= 6.0]
    o = np.empty((2 * len(s), 4))
    o[::2], o[1::2] = s, s[:, [2, 3, 0, 1]]
    return o


def _traced(gray):
    """Curves of edges followed pixel by pixel.

    The edge finder keeps what is straight, and so it loses a long shallow curve
    of soft or rough stone in short pieces too few or too far apart to chain. They
    are coarser than its curves, which only what repeats can afford: a curve off by
    a pixel is another copy, and the rule of §53 is read to the pixel.
    """
    g = cv2.GaussianBlur(gray, (0, 0), TRACE_BLUR)
    top = max(float(np.percentile(np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)), TRACE_TOP)), TRACE_LEAST)
    e = cv2.Canny(g, top / 2, top, L2gradient=True)
    seen, s = np.zeros(e.shape, bool), []

    def cut(run):
        if len(run) >= TRACE_FEWEST:
            p = cv2.approxPolyDP(np.array(run, np.float32).reshape(-1, 1, 2), 1.0, False)[:, 0]
            s.append(np.c_[p[:-1], p[1:]])

    # a contour of a thin edge goes along it and back, so what has been walked is left out
    for c in cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
        run = []
        for x, y in c[:, 0].tolist():
            if seen[y, x]:
                cut(run)
                run = []
            else:
                seen[y, x] = True
                run.append((x, y))
        cut(run)
    s = np.vstack(s).astype(float) if s else np.zeros((0, 4))
    s = s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= 3.0]
    o = np.empty((2 * len(s), 4))
    o[::2], o[1::2] = s, s[:, [2, 3, 0, 1]]
    return [c[0] for _, run in _runs(o, RIB_TURN_DEG, TRACE_GAP_PX, TURN_MAX_DEG, 3) for c in _portions(o[run], RIB_TURN_DEG)
            if conic_resid(c[0]) <= MOST_RMS_PX]


def _links(o, gap=GAP_PX, corner=TURN_MAX_DEG):
    """{piece: [(cost, next piece, turn in degrees)]} for pieces end to start."""
    th = np.arctan2(o[:, 3] - o[:, 1], o[:, 2] - o[:, 0])
    m = cKDTree(o[:, 2:]).sparse_distance_matrix(cKDTree(o[:, :2]), gap, output_type="coo_matrix")
    t = np.degrees((th[m.col] - th[m.row] + np.pi) % (2 * np.pi) - np.pi)
    # a piece touching its own reverse is no join, nor is a corner, and the
    # longer a straight piece the less a curve can turn at its end: a chord L
    # long bowing `b` lies on a radius of L^2/8b, and turns 8b/L in its length
    L = np.hypot(*(o[:, 2:] - o[:, :2]).T)
    sharp = np.degrees(8 * BOW_PX / np.maximum(L[m.row], L[m.col]))
    keep = (np.abs(t) <= np.minimum(corner, sharp)) & (m.row // 2 != m.col // 2)
    links = {}
    for i, j, d, a in zip(m.row[keep], m.col[keep], m.data[keep], t[keep]):
        links.setdefault(int(i), []).append((float(d + 0.2 * abs(a)), int(j), float(a)))
    return links


def _walk(i, sign, links, used):
    """The run from piece `i` turning one way, and how far it turned."""
    run, seen, total = [i], {i // 2}, 0.0
    while True:
        go = [c for c in links.get(run[-1], ()) if sign * c[2] >= -TURN_BACK_DEG
              and c[1] // 2 not in seen and not used[c[1] // 2]]
        if not go:
            return run, total
        _, j, a = min(go)
        run.append(j)
        seen.add(j // 2)
        total += a


def _runs(o, least=LEAST_TURN_DEG, gap=GAP_PX, corner=TURN_MAX_DEG, fewest=4):
    """Runs of pieces that keep turning one way, the most turned first."""
    links = _links(o, gap, corner)
    used = np.zeros(len(o) // 2, bool)
    long_first = np.argsort(-np.hypot(*(o[::2, 2:] - o[::2, :2]).T))
    out = []
    for sign in (1, -1):
        for k in long_first:
            if used[k]:
                continue
            head, a = _walk(2 * int(k), sign, links, used)
            tail, b = _walk(2 * int(k) + 1, -sign, links, used)
            run = [i ^ 1 for i in tail[:0:-1]] + head
            if abs(a) + abs(b) >= least and len(run) >= fewest:
                used[[i // 2 for i in run]] = True
                out.append((abs(a) + abs(b), run))
    return sorted(out, key=lambda r: -r[0])


def _along(q, step=6.0):
    """Points along pieces in order, every `step` pixels or so, and whose each is."""
    n = np.maximum((np.hypot(*(q[:, 2:] - q[:, :2]).T) / step).astype(int), 2)
    return np.vstack([np.linspace(a[:2], a[2:], k) for a, k in zip(q, n)]), np.repeat(np.arange(len(q)), n)


def _curve(q, miss=1.5 * MOST_RMS_PX):
    """What of a run is one smooth curve, how far it turns, and its first and last piece.

    A rim runs on into its straight jambs. The middle of the turn is curve if
    anything is, so the fit starts there and takes in what stays on it.
    """
    t = np.unwrap(np.arctan2(q[:, 3] - q[:, 1], q[:, 2] - q[:, 0]))
    p, of = _along(q)
    f = np.abs(t[of] - t[0]) / max(abs(t[-1] - t[0]), 1e-9)
    on = np.flatnonzero((f >= 0.2) & (f <= 0.8))
    if len(on) < 6:
        return None
    lo, hi = on[0], on[-1]
    for _ in range(12):
        far = np.flatnonzero(conic_dist(p, conic(p[lo:hi + 1])) > miss)
        was = lo, hi
        lo, hi = far[far < lo].max(initial=-1) + 1, far[far > hi].min(initial=len(p)) - 1
        if (lo, hi) == was:
            break
    return p[lo:hi + 1], float(np.degrees(abs(t[of[hi]] - t[of[lo]]))), int(of[lo]), int(of[hi])


def find(gray, keep=12):
    """Arches among the edges: [{"pts", "turn_deg", "rms_px", "span_px"}], biggest turn first.

    `pts` are a dozen points round the curve, as a hand would mark it.
    """
    o = _pieces(gray)
    if not len(o):
        return []
    least = LEAST_SPAN * float(np.hypot(*gray.shape[:2]))
    out, conics = [], []
    for _, run in _runs(o):
        got = _curve(o[run])
        if got is None:
            continue
        p, turn = got[:2]
        if turn < LEAST_TURN_DEG or max(np.ptp(p, axis=0)) < least:
            continue
        C = conic(p)
        rms = conic_resid(p, C)
        if rms > MOST_RMS_PX:
            continue
        # which side the curve bows to; y runs down the picture
        if (p[len(p) // 2] - (p[0] + p[-1]) / 2)[1] >= 0:
            continue
        if any(conic_resid(p, c) <= SAME_PX for c in conics):
            continue
        conics.append(C)
        pick = np.linspace(0, len(p) - 1, 12).round().astype(int)
        out.append({"pts": p[pick].round(1).tolist(), "turn_deg": round(float(turn), 1),
                    "rms_px": round(rms, 2), "span_px": round(float(max(np.ptp(p, axis=0))), 1)})
        if len(out) == keep:
            break
    return out


def wall(arch, up, walls):
    """Which of `walls` ({name: vanishing point}) an arch stands in, or None.

    An arch square in its wall makes the wall's two directions conjugate: the
    polar of the horizontal runs up the arch's axis and on to the uprights. That
    takes no lens and no idea of the arch's shape, so it prejudges nothing read
    from the arch afterwards. Below the turn a shape can be read from, a polar
    says no more: half a rim and its jamb chose the wrong wall in 4 of 11.
    """
    if arch["turn_deg"] < TURN_MIN_DEG or not walls:
        return None
    p = np.asarray(arch["pts"], float)
    C = conic(p)

    def miss(w):
        l = C @ w
        at = p[np.argmin(np.abs(p @ l[:2] + l[2]))]
        to = up[:2] - at * up[2]
        return np.degrees(np.arccos(min(1.0, abs(to @ [l[1], -l[0]]) / np.linalg.norm(to) / np.hypot(*l[:2]))))

    got = sorted((miss(w), n) for n, w in walls.items())
    if got[0][0] > WALL_MISS_DEG or (len(got) > 1 and got[1][0] - got[0][0] < WALL_APART_DEG):
        return None
    return got[0][1]


def _portions(q, least):
    """Each smooth curve of a run: a run can hold several, end to end."""
    got = _curve(q) if len(q) >= 3 else None
    if got is None:
        return []
    p, turn, a, b = got
    return _portions(q[:a], least) + ([(p, turn)] if turn >= least else []) + _portions(q[b + 1:], least)


def _seeds(s, gray, least):
    """[(points, turn)] of the curves turning `least` or more.

    From the edge finder's pieces, and from the lines themselves: a rib with a
    run of 150 px chords is too broken for the first and whole for the second.
    """
    o = _pieces(gray)
    out = [c for _, run in _runs(o, least) for c in _portions(o[run], least)
           if conic_resid(c[0]) <= MOST_RMS_PX]
    o = np.empty((2 * len(s), 4))
    o[::2], o[1::2] = s, s[:, [2, 3, 0, 1]]
    for _, run in _runs(o, least, LINE_GAP_PX, LINE_CORNER_DEG, 3):
        got = _curve(o[run], LINE_MISS_PX)
        if got is not None and got[1] >= least:
            out.append(got[:2])
    return out


def _grown(seed, pts, of, n, reach):
    """A curve with the lines that follow it: (points, which lines)."""
    p, on = seed, np.zeros(n, bool)
    count = np.bincount(of, minlength=n)
    for _ in range(6):
        i = np.flatnonzero(cKDTree(p).query(pts, distance_upper_bound=reach)[0] <= reach)
        near = i[conic_dist(pts[i], conic(p)) <= ON_CURVE_PX]
        new = np.bincount(of[near], minlength=n) >= ON_CURVE_PART * count
        if (new == on).all():
            break
        on = new
        p = np.vstack([seed, pts[on[of]]])
    return p, on


def _chords(p, on, pts, of, n, wide):
    """Those lines, and the ones running beside them along the same molding."""
    i = np.flatnonzero(cKDTree(p).query(pts, distance_upper_bound=wide + 6)[0] <= wide + 6)
    near = i[conic_dist(pts[i], conic(p)) <= wide]
    return on | (np.bincount(of[near], minlength=n) >= ON_CURVE_PART * np.bincount(of, minlength=n))


def _about(N, V):
    """The way up from the point N and the inverse of its signed distance to the
    vanishing point V of the uprights, if there is one (image up otherwise)."""
    if V is None:
        return np.array([0.0, -1.0]), 0.0
    d = V[:2] - N * V[2]
    sign = -1.0 if d[1] > 0 else 1.0
    return sign * d / np.linalg.norm(d), sign * float(V[2]) / np.linalg.norm(d)


def _rows(curves, N, V=None):
    """Each curve read round N, half a degree a step, split at its crown.

    {side: [(curve, angle from up, log distance, mean points)]}, side 1 right of
    the crown. Distance is what a copy of the rib further along the vault scales
    by; with the uprights converging it is 1/r - cos(angle)/D instead of 1/r,
    which is depth over radius whatever the tilt, and needs no lens.
    """
    N = np.asarray(N, float)
    up, inv = _about(N, V)
    right = np.array([-up[1], up[0]])
    rows = {1: [], -1: []}
    for i, p in enumerate(curves):
        d = p - N
        h = d @ up
        if (h <= 0).any():
            continue
        phi = np.degrees(np.arctan2(d @ right, h))
        y = np.log(np.hypot(*d.T)) - np.log(np.maximum(1 - h * inv, 1e-6))
        u, at = np.unique(np.round(phi * 2).astype(int), return_inverse=True)
        c = np.bincount(at)
        ph, lr = np.bincount(at, phi) / c, np.bincount(at, y) / c
        xy = np.c_[np.bincount(at, p[:, 0]), np.bincount(at, p[:, 1])] / c[:, None]
        k = int(np.argmax(lr))
        if ph[-1] - ph[k] >= LEAST_SWEEP_DEG:
            rows[1].append((i, ph[k:], lr[k:], xy[k:]))
        if ph[k] - ph[0] >= LEAST_SWEEP_DEG:
            rows[-1].append((i, ph[:k + 1], lr[:k + 1], xy[:k + 1]))
    return rows


def _joint(q):
    """log r = a + b t + g(t) over the ribs of a side, g shared: (fit, residuals).

    Ribs of one section differ only by their depth, so whatever is left when
    each has its own offset and slope is the section, and nothing but the slope
    is left to tell one rib's depth at the top from its depth at the foot.
    """
    n, P = len(q), len(SHAPE_POWERS)
    a = np.concatenate([z[1] for z in q])
    c, w = float(a.mean()), max(float(np.ptp(a)), 1.0) / 2
    A = np.zeros((len(a), 2 * n + P))
    y = np.concatenate([z[2] for z in q])
    k = 0
    for j, z in enumerate(q):
        e, t = k + len(z[1]), (z[1] - c) / w
        A[k:e, 2 * j] = 1
        A[k:e, 2 * j + 1] = t
        for d, m in enumerate(SHAPE_POWERS):
            A[k:e, 2 * n + d] = t ** m
        k = e
    x = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"x": x, "c": c, "w": w, "n": n}, y - A @ x, y


def _off(q):
    """Each rib's rms pixels off the shape its side shares."""
    res, y = _joint(q)[1:]
    e = np.cumsum([len(z[1]) for z in q])
    return np.array([np.sqrt(np.mean((res[a:b] * np.exp(y[a:b])) ** 2)) for a, b in zip(np.r_[0, e[:-1]], e)])


def _at(f, angle):
    """Every rib's fitted log distance at `angle`."""
    t = (angle - f["c"]) / f["w"]
    n, x = f["n"], f["x"]
    return x[0:2 * n:2] + x[1:2 * n:2] * t + sum(g * t ** m for g, m in zip(x[2 * n:], SHAPE_POWERS))


def _misfit(N, curves):
    """RMS pixels of the ribs off one shape a side, about N.

    A point seating fewer of the curves pays for it: 20 of Worcester's 120 fit
    about a point 1000 px off as well as all of them do about the true one.
    """
    rows = _rows(curves, N)
    if min(len(rows[1]), len(rows[-1])) < FEWEST_SIDE:
        return np.inf
    t = c = 0.0
    for q in rows.values():
        res, y = _joint(q)[1:]
        t += float(((res * np.exp(y)) ** 2).sum())
        c += len(res)
    return np.sqrt(t / c) * len(curves) / len({z[0] for q in rows.values() for z in q})


def _meets(s, size):
    """Where lines meet below the top of the picture, the strongest first.

    A grid, for every cell of it can be asked and none is used up by the piers,
    which are what the one best point of each round finds on a nave.
    """
    h, w = size
    L = np.hypot(*(s[:, 2:] - s[:, :2]).T)
    xs, ys = np.arange(-.6, 1.6, MEET_STEP) * w, np.arange(.2, 3, MEET_STEP) * h
    S = np.array([[L[resid_deg(s, np.array([x, y, 1.0])) < MEET_WIDE_DEG].sum() for x in xs] for y in ys])
    i, j = np.nonzero((S == cv2.dilate(S, np.ones((3, 3)))) & (S > 0))
    out = []
    for k in np.argsort(-S[i, j]):
        v = np.array([xs[j[k]], ys[i[k]], 1.0])
        for tol in (2.0, 1.0, 1.0):
            on = np.flatnonzero(resid_deg(s, v) < tol)
            if len(on) >= 2:
                v = fit(s, on)
        if abs(v[2]) > 1e-12 and all(np.hypot(*(v[:2] / v[2] - q)) > SAME_POINT * np.hypot(h, w) for q in out):
            out.append(v[:2] / v[2])
            if len(out) == MEET_TRIES:
                break
    return out


def _across(N, s):
    """Lines through N beyond their biggest bundle: (those, all of them).

    Lines of one direction pass through any point far enough along, as the piers
    do through every point under them; only other directions fix this one.
    """
    on = resid_deg(s, np.r_[N, 1.0]) < MEET_DEG
    d = (s[on, :2] + s[on, 2:]) / 2 - N
    a = np.sort(np.degrees(np.arctan2(d[:, 0], -d[:, 1])))
    return len(a) - int((np.searchsorted(a, a + BUNDLE_DEG) - np.arange(len(a))).max(initial=0)), len(a)


def _point(curves, s, size):
    """The point the vault runs to: (N, rms pixels, lines through it) or None.

    The ribs say which points are near, within a third of each other: alone they
    slid 660 px along the ridge, down which every point serves them about as well.
    The lines say which of those: of 17 versions of a nave this took the point on 14
    by 5 to 19 lines against 0 to 8; the other 3 were a small or cropped picture.
    And the curves have to follow it, which is what stops Worcester's wrong point
    at x.75 and on its side.
    """
    near = _meets(s, size)
    cost = np.array([_misfit(N, curves) for N in near])
    if not np.isfinite(cost).any():
        return None
    each = [_across(N, s) if c <= CLOSE_FIT * cost.min() else (0, 0) for N, c in zip(near, cost)]
    i = int(np.argmax([e[0] for e in each]))
    if each[i][0] < FEWEST_ACROSS or cost[i] > RIB_MISS * np.hypot(*size):
        return None
    return near[i], float(cost[i]), each[i][1]


def vault(s, gray):
    """The chords of curves among lines `s`, and any vault the curves make.

    {"chords": mask of `s`, "curves": [points], "point": [x, y] or None, "rms_px",
    "lines", "diag"}. Turning 25 degrees a curve is a curve anywhere: on 13 pictures
    that took none of 67 lines marked by hand, and table edges and window jambs
    stayed. At 20 degrees those went too, and it takes a vault's ribs all agreeing
    on one shape about one point to say that so slight a turn is a curve.
    """
    n, size = len(s), gray.shape[:2]
    pts, of = _along(s) if n else (np.zeros((0, 2)), np.zeros(0, int))
    diag = float(np.hypot(*size))
    reach, wide = NEXT_LINE * diag, MOLDING * diag
    drop, curves, each = np.zeros(n, bool), [], []
    for p, turn in _seeds(s, gray, RIB_TURN_DEG):
        got, on = _grown(p, pts, of, n, reach) if n else (p, np.zeros(0, bool))
        curves.append(got)
        each.append(_chords(got, on, pts, of, n, wide) if n else drop)
        if turn >= CHORD_TURN_DEG:
            drop |= each[-1]
    out = {"chords": drop, "curves": curves, "point": None, "rms_px": None, "lines": 0, "diag": diag}
    if len(curves) < FEWEST_RIBS or not n:
        return out
    got = _point(curves, s[~np.any(each, axis=0)], size)
    if got is None:
        return out
    out["point"], out["rms_px"], out["lines"] = got[0].tolist(), round(got[1], 2), got[2]
    # the ribs it vouches for: all of them on the shape, with their chords
    for q in _rows(curves, got[0]).values():
        for z, o in zip(q, _off(q)):
            if o <= RIB_MISS * diag:
                drop |= each[z[0]]
    return out


def beside(arches, ribs, wide):
    """The arches that are not ribs already sorted: those say more as ribs."""
    on = [np.asarray(r["pts"]) for r in ribs if r["verdict"] != "undetermined"]
    if not on:
        return arches
    near = cKDTree(np.vstack([_along(np.c_[q[:-1], q[1:]])[0] for q in on]))
    return [a for a in arches if (near.query(a["pts"])[0] <= wide).mean() < ON_CURVE_PART]


def _sweep(rows):
    """The crown's angle, each side's foot, and where each rib starts and stops."""
    near = {sd: np.array([z[1][0 if sd > 0 else -1] for z in q]) for sd, q in rows.items()}
    far = {sd: np.array([z[1][-1 if sd > 0 else 0] for z in q]) for sd, q in rows.items()}
    crown = (np.percentile(near[1], 15) + np.percentile(near[-1], 85)) / 2
    return crown, {sd: float(np.median(a)) for sd, a in far.items()}, near, far


def _one_edge(q, wide):
    """Of the edges along one molding, the longest."""
    out = []
    for z in sorted(q, key=lambda z: -len(z[3])):
        if not any((cKDTree(o[3]).query(z[3])[0] <= wide).mean() >= ON_CURVE_PART for o in out):
            out.append(z)
    return out


def _smooth(p, keep):
    """`keep` points along a rib, without the jogs where its pieces change edge."""
    t = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    c = np.polyfit(t, p, 4)
    return np.polynomial.polynomial.polyval(np.linspace(0, t[-1], keep), c[::-1]).T


def _read(q, crown, foot):
    """(log distance at crown and at foot, spread of their difference) for each rib."""
    f, res = _joint(q)[:2]
    span = abs(foot - crown) / f["w"]
    sxx = np.array([((z[1] - z[1].mean()) ** 2).sum() for z in q]) / f["w"] ** 2
    return _at(f, crown), _at(f, foot), np.sqrt(np.mean(res ** 2)) * span * np.sqrt(SAMPLES_PER_FACT / np.maximum(sxx, 1e-9))


def _bands(sides):
    """Each side's ratio for a flat rib, half a bay, the fit: ({side: a}, c, score), as logs.

    A flat rib's crown stands to its foot as every other flat rib's does, and a
    rib running half a bay along the vault, to the pier before its boss or the one
    after, reads that much further from it, by half a bay over its depth. So
    y = a + s c x, with s -1, 0 or 1 and x the size at the crown, the nearest rib's
    1: three lines closing on `a`, and `c` common to both sides of one vault.
    Near ribs count for more, for only they stand clear of the other lines.
    """
    best = 0.0, None, None
    for c in HALF_BAYS:
        score, flat = 0.0, {}
        for sd, (x, y) in sides.items():
            at = np.arange(y.min(), y.max() + BAND_STEP, BAND_STEP)
            d = np.abs(y - at[:, None])
            s = (np.exp(-.5 * (np.minimum(d, np.abs(d - c * x)) / BAND_WIDTH) ** 2) * x).sum(1)
            k = int(np.argmax(s))
            score, flat[sd] = score + float(s[k]), float(at[k])
        if score > best[0]:
            best = score, flat, float(c)
    return best[1], best[2], best[0]


def _flattened(N, V):
    """Picture to coordinates where a copy further along is a scaling about 0, and back."""
    up, inv = _about(N, V)
    right = np.array([-up[1], up[0]])

    def to(p):
        h = (p - N) @ up
        return np.c_[(p - N) @ right, h] / np.maximum(1 - h * inv, 1e-6)[:, None]

    return to, lambda q: N + (q[:, :1] * right + q[:, 1:] * up) / (1 + q[:, 1:] * inv)


def _comb(q, fl, d3, half):
    """(first, step) of 1 / depth at the crown, or None: flat ribs on the even teeth, 3D on the odd."""
    base = q[fl | d3].min()
    steps = base * half * np.linspace(*COMB_SPAN, 66)
    t = (q[:, None, None] - base - np.outer(steps, np.arange(-2, 2, .05))) / steps[:, None]
    even = lambda x: 2 * np.round(x / 2)
    tooth = lambda x: np.exp(-.5 * (np.abs(x - even(x)) / COMB_TOOTH) ** 2)
    i, j = np.unravel_index(np.argmax(tooth(t)[fl].sum(0) + tooth(t - 1)[d3].sum(0)), t.shape[1:])
    t = t[:, i, j]
    ks = even(t) * fl + (even(t - 1) + 1) * d3
    on = (fl | d3) & (np.abs(t - ks) < .5)
    return np.polyfit(ks[on], q[on], 1)[::-1] if len(np.unique(ks[on])) > 2 else None


def _trimmed(a, y):
    """Least squares, without what stands far off."""
    on = np.ones(len(y), bool)
    for _ in range(5):
        x = np.linalg.lstsq(a[on], y[on], rcond=None)[0]
        e = np.abs(y - a @ x)
        on = e <= TRIM * 1.4826 * np.median(e) + 1e-9
    return x


def _depth(x, k, s, S, sd):
    """Along a rib of boss k: its depth at x. s = 0 is flat, s = 1 or -1 runs straight in
    plan from the boss to the pier half a bay on or back, Z = Z_k + s h (X - X_c) / W."""
    h, W = S["h"], S["W"][sd]
    return (1 + k * h - s * h * S["Xc"] / W) / (1 - s * h * x / W)


def _radius(S, sd, s, t):
    """The section, round its eye at angle t: the flat ribs', or the 3D ribs' off it."""
    poly = np.polynomial.polynomial.polyval
    r = S["r0"] + t * poly(t, S["P"][sd])
    return r + t * (t - S["th"][sd]) * poly(t, S["Q"][sd]) if s else r


def _section(L, h, Xc, Cy, foot):
    """The shape every rib shares, from `L` [(rib, boss, s)], or None.

    Set at its true depth each flat rib falls on one curve, and each 3D rib on
    another that leaves it at the crown and meets it at the pier. Round a point
    below the crown both are single valued, from the ridge to the foot. The foot
    is where the flat curve meets the ray the ribs reach to.
    """
    C = np.array([Xc, Cy])

    def polar(p):
        p = p - C
        return np.arctan2(p[:, 0], p[:, 1]), np.hypot(*p.T)

    flat = [(d["sd"], polar(d["q"] * (1 + k * h))) for d, k, s in L if not s]
    if {sd for sd, _ in flat} != {1, -1}:
        return None
    th, r = np.hstack([p for _, p in flat])
    sds = np.concatenate([np.full(len(p[0]), sd) for sd, p in flat])
    x = _trimmed(np.column_stack([np.ones(len(th))] + [(sds == e) * th ** (j + 1) for e in (1, -1)
                                                       for j in range(FLAT_POWERS)]), r)
    S = {"r0": x[0], "P": {1: x[1:1 + FLAT_POWERS], -1: x[1 + FLAT_POWERS:]}, "C": C, "h": h, "Xc": Xc,
         "Yc": Cy + x[0], "th": {}, "W": {}, "Q": {}}
    for sd in (1, -1):
        t = sd * np.linspace(0, 2, 801)
        r = _radius(S, sd, 0, t)
        X, Y = Xc + r * np.sin(t), Cy + r * np.cos(t)
        g = X - foot[sd] * Y
        cross = np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
        i = cross[0] if len(cross) else int(np.argmin(np.abs(g)))
        if sd * (X[i] - Xc) <= 0:
            return None
        S["th"][sd], S["W"][sd] = float(t[i]), float(X[i] - Xc)
    for sd in (1, -1):
        got = [polar(d["q"] * _depth(d["q"][:, 0], k, s, S, sd)[:, None]) for d, k, s in L if s and d["sd"] == sd]
        S["Q"][sd] = np.zeros(OFF_POWERS)
        if got:
            th, r = np.hstack(got)
            tied = th * (th - S["th"][sd])
            S["Q"][sd] = _trimmed(np.column_stack([tied * th ** j for j in range(OFF_POWERS)]), r - _radius(S, sd, 0, th))
    return S


def _miss(d, k, s, S, cap):
    """Rms px from rib d to rib (k, s) of the rule, inf if most of it lies beyond that rib."""
    sd, q = d["sd"], d["q"]
    Z = _depth(q[:, 0], k, s, S, sd)
    p = q * Z[:, None] - S["C"]
    t = np.arctan2(p[:, 0], p[:, 1])
    e = (np.hypot(*p.T) - _radius(S, sd, s, t)) / Z
    e = e[(sd * t >= -.05) & (sd * t <= sd * S["th"][sd] + .05) & np.isfinite(e)]
    return float(np.sqrt(np.mean(np.minimum(np.abs(e), cap) ** 2))) if 2 * len(e) >= len(q) and len(e) else np.inf


def _on(ribs, S, ks, limit):
    """[(rib, boss, s)] of those within `limit` of the rule; each keeps its two nearest."""
    each = [(k, s) for k in ks for s in ((0,) if k % 2 == 0 else (-1, 1))]
    L = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for d in ribs:
            d["near"] = sorted((_miss(d, k, s, S, 2 * limit), k, s) for k, s in each)[:2]
            if d["near"][0][0] <= limit:
                L.append((d, d["near"][0][1], d["near"][0][2]))
    return L


def _by_chance(ribs, S, limit):
    """How many ribs would follow the rule with their feet anywhere, and how far that count wanders.

    Far ribs close up until every run lies within the limit, so a rib on the rule
    says only as much as the runs that would miss it. An odd boss allows two runs.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p = [(1 + d["near"][0][1] % 2) * np.mean([_miss(d, d["near"][0][1], s, S, 2 * limit) <= limit for s in RULE_RUNS])
             for d in ribs]
    p = np.minimum(p, 1)
    return float(p.sum()), float(np.sqrt((p * (1 - p)).sum()))


def _rib(S, back, k, s, sd):
    """Rib (k, s) of the rule on side sd, in the picture."""
    t = np.linspace(0, S["th"][sd], RULE_POINTS)
    r = _radius(S, sd, s, t)
    X, Y = S["C"][0] + r * np.sin(t), S["C"][1] + r * np.cos(t)
    Z = 1 + k * S["h"] + s * S["h"] * (X - S["Xc"]) / S["W"][sd]
    return back(np.c_[X, Y] / Z[:, None])


def _built(L, firm, S, back):
    """Every rib of the rule, from the first one in sight to the end of the last bay found."""
    xy = np.vstack([d["xy"] for d, _, _ in L])
    lo, hi = xy.min(0), xy.max(0)
    seen = lambda p: bool(((p >= lo) & (p <= hi)).all(1).any())
    ribs_of = lambda k: [(k, s, sd) for s in ((0,) if k % 2 == 0 else (-1, 1)) for sd in (1, -1)]
    first = min(k for _, k, _ in L)
    while 1 + (first - 1) * S["h"] > 0 and any(seen(_rib(S, back, *r)) for r in ribs_of(first - 1)):
        first -= 1
    last = max(k for _, k, _ in L)
    out = []
    for k in range(first, last + last % 2 + 1):
        for r in ribs_of(k):
            p = _rib(S, back, *r)
            if seen(p):
                out.append({"pts": p.round(1).tolist(), "family": "3D" if r[1] else "flat", "boss": k,
                            "side": "right" if r[2] > 0 else "left", "found": r in firm})
    return out


def _rule(ribs, N, V, half, crown, far, diag):
    """The regular vault these ribs belong to: every rib of it, and how far to believe it.

    A rib and its copy further along are one curve moved along the axis, so about
    the point depth is the only denominator, and the bosses standing one half bay
    apart put 1 / distance at the crown on a comb. An even boss takes a flat rib a
    side, an odd one two a side, one to the pier before it and one to the pier
    after. The ribs firmly sorted set the comb and the section; then every rib,
    whole or not, goes to the rib of the rule it follows, and that is refitted until
    they stop changing. Two things must agree that the fit cannot arrange: the
    crowns' own spacing, read with no regard to any family, and more of the ribs
    than would follow it with their feet anywhere. Curves off it prove nothing, since a
    molding has several edges and a picture has other curves; ribs of it that are not
    there do, so enough of them must be found.
    """
    refused = lambda why: {"verdict": "undetermined", "why": why, "ribs": []}
    to, back = _flattened(N, V)
    for d in ribs:
        d["q"] = to(d["xy"])
    whole = [d for d in ribs if d["whole"]]
    q = np.exp(-np.array([d["u"] for d in whole]))
    v = np.array([d["verdict"] for d in whole])
    fl, d3 = v == "flat", v == "3D"
    tooth = _comb(q, fl, d3, half) if (fl | d3).any() else None
    if tooth is None:
        return refused("too few ribs are firm enough to start from")
    first, step = tooth
    h = step / first
    Xc, Yc = np.sin(np.radians(crown)) / first, np.cos(np.radians(crown)) / first
    Cy = SECTION_EYE * Yc
    foot = {sd: np.tan(np.radians(np.percentile(far[sd], FOOT_REACH if sd > 0 else 100 - FOOT_REACH))) for sd in far}
    at, L = (q - first) / step, []
    for d, t in zip(whole, at):
        if d["verdict"] == "flat":
            L.append((d, 2 * round(t / 2), 0))
        elif d["verdict"] == "3D":
            L.append((d, 2 * round((t - 1) / 2) + 1, 1 if d["run"] > 0 else -1))
    ks = np.arange(int(np.floor(at.min())) - 2, int(np.ceil(at.max())) + 3)
    limit = RULE_MISS * diag
    S, old = _section(L, h, Xc, Cy, foot), None
    for _ in range(10):
        if S is None:
            break
        L = _on(ribs, S, ks, limit)
        now = [(id(d), k, s) for d, k, s in L]
        if now == old:
            break
        old, S = now, _section(L, h, Xc, Cy, foot)
    if S is None:
        return refused("the ribs give no one section on both sides")
    luck, wander = _by_chance(ribs, S, limit)
    if len(L) - luck < RULE_BEAT * max(wander, 1):
        return refused("%d of the %d ribs follow the nearest regular vault, and %.0f would with their feet anywhere"
                       % (len(L), len(ribs), luck))
    steps = q.min() * CROWN_STEPS
    peak = steps[np.argmax(np.abs(np.exp(2j * np.pi * q[None] / steps[:, None]).sum(1)))]
    gap = h / np.hypot(Xc, S["Yc"])
    if abs(peak / gap - 1) > PEAK_APART:
        return refused("the crowns repeat every %.3f of the nearest rib's depth, and a rule that fits needs %.3f"
                       % (peak / q.min(), gap / q.min()))
    held = []
    for b in {k for _, k, _ in L}:
        T = _section([z for z in L if z[1] != b], h, Xc, Cy, foot)
        if T is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                held += [_miss(d, k, s, T, 2 * limit) for d, k, s in L if k == b]
    held = [e for e in held if np.isfinite(e)]
    if not held:
        return refused("too few bosses to test a rule on")
    sure = {id(d): (s, k) for d, k, s in L if d["near"][0][0] * RULE_NEXT <= d["near"][1][0]}
    miss = [d["near"][0][0] for d, _, _ in L]
    built = _built(L, {(k, s, d["sd"]) for d, k, s in L if id(d) in sure}, S, back)
    seen = sum(r["found"] for r in built)
    if seen < RULE_SEEN * len(built):
        return refused("the nearest regular vault has %d ribs in sight and %d of them are found" % (len(built), seen))
    return {"verdict": "holds", "why": "", "half_bay": round(float(gap / q.min()), 3), "on": len(L), "of": len(ribs),
            "by_chance": round(luck), "miss_px": round(float(np.median(miss)), 1), "worst_px": round(float(np.max(miss)), 1),
            "held_out_px": round(float(np.median(held)), 1), "peak": round(float(abs(peak / gap - 1)), 3),
            "ribs": built, "calls": [sure.get(id(d)) for d in ribs], "h": float(h)}


def sort(found, V=None, keep=12):
    """Which ribs stand in one plane across the vault, and which run through depth.

    A transverse arch is one depth from foot to crown, so the ribs of one family
    are copies scaled about the point, and every copy has the same ratio of
    distance at the crown to distance at the foot. A diagonal's ratio is off by
    its run in depth, half a bay either way, so the ratios lie on three lines. Each
    rib is read to 2 or 3 % with the lines 5 to 10 % apart, so some three lines can
    always be drawn: the sort is refused when the same ribs do as well moved at
    random along their depth by up to a line's spacing, which keeps their spread and
    their closing and loses only the lines. Each rib is judged by its distance from
    the middle line of its side. `keep` is the points of each rib offered.

    Where a rule holds (`_rule`) it can vouch for ribs chance would refuse, and it
    decides the families: beyond the ribs read clearly, the lines close up and a rib's
    place on the comb says more than its ratio.
    """
    N, curves = found.get("point"), found["curves"]
    refused = lambda why: {"verdict": "undetermined", "why": why, "ribs": []}
    if N is None:
        return None
    rows = _rows(curves, N, V)
    if min(len(rows[1]), len(rows[-1])) < FEWEST_SIDE:
        return refused("ribs are needed on both sides of the crown")
    crown, foot, near, far = _sweep(rows)
    if foot[1] <= crown or foot[-1] >= crown:
        return refused("the ribs do not fall away on both sides of the crown")
    for sd, q in rows.items():
        hi = sd * (foot[sd] - crown)
        part = (np.minimum(sd * (far[sd] - crown), hi) - np.maximum(sd * (near[sd] - crown), 0)) / hi
        rows[sd] = _one_edge([z for z, o, c in zip(q, _off(q), part) if o <= RIB_MISS * found["diag"]
                              and c >= LEAST_COVER], MOLDING * found["diag"])
    if min(len(rows[1]), len(rows[-1])) < FEWEST_SIDE:
        return refused("too few of the curves follow the vault")
    crown, foot, near, far = _sweep(rows)
    read = {sd: _read(q, crown, foot[sd]) for sd, q in rows.items()}
    whole = {sd: np.abs(near[sd] - crown) <= CROWN_GAP * abs(foot[sd] - crown) for sd in rows}
    good = {sd: whole[sd] & (read[sd][2] <= 2 * RUN_SD) for sd in rows}
    sides = {sd: (read[sd][0][good[sd]], (read[sd][0] - read[sd][1])[good[sd]])
             for sd in rows if good[sd].sum() >= FEWEST_SIDE}
    if not sides:
        return refused("too few ribs run from the crown to the foot")
    top = max(u.max() for u, y in sides.values())
    sides = {sd: (np.exp(u - top), y) for sd, (u, y) in sides.items()}
    flat, half_bay, score = _bands(sides)
    pick = np.random.default_rng(0)
    # each rib moved along its depth by up to a line's spacing: the spread and the closing stay, the lines go
    chance = np.mean([_bands({sd: (x, y + x * pick.uniform(-half_bay, half_bay, len(y))) for sd, (x, y) in sides.items()})[2]
                      >= score for _ in range(DRAWS)])
    ribs, each = [], []
    for sd, q in rows.items():
        u, v, e = read[sd]
        off = u - v - flat.get(sd, np.nan)
        half = half_bay * np.exp(u - top) / 2
        for z, a, d, h, s, w in zip(q, u.tolist(), off.tolist(), half.tolist(), e.tolist(), whole[sd]):
            # nearer one line than the other by more than the rib can be misread
            verdict = "undetermined" if s > RUN_SD or not w or sd not in flat else "flat" if abs(d) + 2 * s <= h \
                else "3D" if abs(d) - 2 * s >= h else "undetermined"
            run = None if sd not in flat else round(float(np.exp(d) - 1), 3)
            ribs.append({"pts": _smooth(z[3], keep).round(1).tolist(), "side": "right" if sd > 0 else "left",
                         "run": run, "sd": round(s, 3), "verdict": verdict, "reach_px": round(float(np.hypot(*(z[3][-1] - z[3][0]))), 1)})
            each.append({"sd": sd, "xy": z[3], "u": a, "whole": bool(w), "verdict": verdict, "run": run})
    rule = _rule(each, np.asarray(N, float), V, half_bay, crown, far, found["diag"])
    holds = rule["verdict"] == "holds"
    if chance > MOST_CHANCE and not holds:
        return refused("the ribs lie on three lines no better than chance, and " + rule["why"])
    if holds:
        h = rule.pop("h")
        for r, c in zip(ribs, rule.pop("calls")):
            r["verdict"] = "undetermined" if c is None else "3D" if c[0] else "flat"
            if c is not None:           # how far its foot runs from its crown, as the rule has it
                r["run"] = round(c[0] * h / (1 + c[1] * h), 3)
    return {"verdict": "sorted", "why": "", "ribs": ribs, "point": [round(c, 1) for c in N],
            "lines": found["lines"], "rms_px": found["rms_px"], "half_bay": round(half_bay, 3), "chance": round(float(chance), 3),
            "ratio": {nm: round(float(np.exp(flat[sd])), 3) for sd, nm in ((-1, "left"), (1, "right")) if sd in flat},
            "rule": rule}


def _ordered(p):
    d = p - p.mean(0)
    return p[np.argsort(d @ np.linalg.svd(d, full_matrices=False)[2][0])]


def _branches(t, least):
    """A run of angles split where it turns back: a wall arch crosses its rays twice."""
    e = len(t) - 1
    for i in (int(np.argmax(t)), int(np.argmin(t))):
        if 3 < i < e - 3 and min(abs(t[i] - t[0]), abs(t[i] - t[e])) >= least:
            return _branches(t[:i], least) + [x + i for x in _branches(t[i:], least)]
    return [np.arange(len(t))]


def _table(curves, N, V):
    """1/distance from N, flattened, with a row for each curve and a column for each ray:
    (the table, the curve of each row, the rays in degrees from up, back to the picture)."""
    to, back = _flattened(np.asarray(N, float), V)
    q = [to(_ordered(p)) for p in curves]
    th = [np.degrees(np.arctan2(z[:, 0], z[:, 1])) for z in q]
    keep = [i for i, t in enumerate(th) if abs((np.median(t) + 180) % 360 - 180) <= BEHIND_DEG]
    if not keep:
        return np.zeros((0, 1)), np.zeros(0, int), np.zeros(1), back
    a = np.sort(np.concatenate([th[i] for i in keep]))
    gaps = np.diff(np.r_[a, a[0] + 360])
    i = int(np.argmax(gaps))
    start, span = a[(i + 1) % len(a)], 360 - gaps[i]
    step = float(np.clip(span / RAYS, .25, 1))
    rays = int(span / step) + 1
    rows, of = [], []
    for c in keep:
        t = (th[c] - start) % 360
        for x in _branches(np.convolve(np.pad(t, 2, "edge"), np.ones(5) / 5, "valid") if len(t) > 9 else t, 2 * step):
            if len(x) < 4:
                continue
            b = np.minimum((t[x] / step).astype(int), rays - 1)
            with np.errstate(all="ignore"):
                u = np.bincount(b, minlength=rays) / np.bincount(b, np.hypot(*q[c][x].T), minlength=rays)
            k = np.flatnonzero(np.isfinite(u))
            for i, j in zip(k[:-1], k[1:]):                     # its points stand apart near the point
                if 1 < j - i <= HOLE_RAYS:
                    u[i:j + 1] = np.linspace(u[i], u[j], j - i + 1)
            if np.isfinite(u).sum() >= LEAST_RAYS:
                rows.append(u)
                of.append(c)
    return (np.array(rows) if rows else np.zeros((0, rays))), np.array(of, int), start + step * (np.arange(rays) + .5), back


def _on_rays(U, pos):
    """u = a + pos b on each ray by itself, pixels counting alike: nan where two rows do not speak."""
    y, w, x = np.nan_to_num(U), np.isfinite(U) / np.where(np.isfinite(U), U, 1) ** 4, pos[:, None]
    S0, S1, S2, T0, T1 = w.sum(0), (w * x).sum(0), (w * x * x).sum(0), (w * y).sum(0), (w * x * y).sum(0)
    with np.errstate(all="ignore"):
        det = S0 * S2 - S1 ** 2
        b = (S0 * T1 - S1 * T0) / det
        a = (T0 - b * S1) / S0
    bad = (np.isfinite(U).sum(0) < 2) | (det <= 1e-9 * S0 * S2)
    a[bad] = b[bad] = np.nan

    def smooth(v, k=np.ones(5)):
        f = np.isfinite(v)
        with np.errstate(all="ignore"):
            o = np.convolve(np.where(f, v, 0), k, "same") / np.convolve(f * 1., k, "same")
        o[~f] = np.nan
        return o

    return smooth(a), smooth(b)


def _members(U, a, b, span, tol):
    """Which rows follow u = a + pos b, and at what pos: (mask, pos, misses in px). A row whose
    miss lies in a tilt across its rays has another run, however short it is and near it stays."""
    with np.errstate(all="ignore"):
        pos = np.nanmedian((U - a) / b, axis=1)
        bad = ~np.isfinite(pos)
        pos = np.where(bad, 0, pos)
        p = a + pos[:, None] * b
        e = 1 / U - 1 / np.where(p > 0, p, np.nan)
        f = np.isfinite(e)
        n = f.sum(1)
        rms = np.sqrt(np.nansum(e ** 2, 1) / np.maximum(n, 1))
        d = f * np.arange(U.shape[1]) - (f * np.arange(U.shape[1])).sum(1, keepdims=True) / np.maximum(n[:, None], 1) * f
        tilt = np.abs(np.nansum(e * d, 1)) / np.sqrt(np.maximum((d * d).sum(1) * n, 1e-9))
    return ~bad & (rms <= tol) & (tilt <= PENCIL_TILT * tol) & (n >= 4) & (n >= FOLLOW * span), pos, e


def _grow(U, i, j, tol, span):
    """The curves that are copies of these two, refitted until they stop changing."""
    s = np.isfinite(U[i]) & np.isfinite(U[j])
    if s.sum() < LEAST_RAYS:
        return None
    d = U[j] - U[i]
    if d[s].mean() < 0:
        i, j, d = j, i, -d
    with np.errstate(all="ignore"):
        if (d[s] <= 0).any() or np.median(np.abs(1 / U[i][s] - 1 / U[j][s])) < 2.5 * tol:
            return None
    a, b, old = np.where(s, U[i], np.nan), np.where(s, d, np.nan), None
    for _ in range(5):
        on, pos, e = _members(U, a, b, span, tol)
        m = np.flatnonzero(on)
        if len(m) < 2:
            return None
        a, b = _on_rays(np.where(np.abs(e[m]) <= 2 * tol, U[m], np.nan), pos[m])
        if np.isfinite(b).sum() < LEAST_RAYS or np.nanmedian(b) <= 0:
            return None
        if old is not None and old.size == m.size and (old == m).all():
            break
        old = m
    on, pos, _ = _members(U, a, b, span, tol)
    return {"a": a, "b": b, "on": np.flatnonzero(on), "pos": pos[on]} if on.sum() >= 2 else None


def _apart(P, U, tol):
    """The rays each separate curve of a set covers: rows this near each other are one curve."""
    ok = np.isfinite(P["a"]) & np.isfinite(P["b"])
    r = lambda x: 1 / (P["a"][ok] + x * P["b"][ok])
    o = np.argsort(P["pos"])
    pos, rows = P["pos"][o], P["on"][o]
    T = [[0]]
    for i in range(1, len(pos)):
        if np.median(np.abs(r(pos[i]) - r(pos[T[-1][0]]))) >= 1.5 * tol:
            T.append([i])
        else:
            T[-1].append(i)
    return [int(np.isfinite(U[rows[t]]).any(0).sum()) for t in T]


def _copies(U, tol):
    """Sets of curves that are copies of one curve moved along the vault, the fullest first,
    with no curve in two. A pair of curves gives a set on rays from the point alone."""
    span = np.isfinite(U).sum(1)
    pick, got = np.argsort(-span)[:SEEDS], []
    for x in range(len(pick)):
        for y in range(x + 1, len(pick)):
            P = _grow(U, int(pick[x]), int(pick[y]), tol, span)
            if P:
                t = _apart(P, U, tol)
                if len(t) >= FEWEST_APART:
                    P["score"] = sum(t)
                    got.append(P)
    got.sort(key=lambda P: -P["score"])
    out, used = [], set()
    for P in got:
        if len(set(P["on"].tolist()) & used) > .3 * len(P["on"]):
            continue
        keep = np.array([r not in used for r in P["on"].tolist()])
        Q = dict(P, on=P["on"][keep], pos=P["pos"][keep])
        if len(Q["on"]) >= 2 and len(_apart(Q, U, tol)) >= FEWEST_APART:
            used |= set(Q["on"].tolist())
            out.append(Q)
    return out[:MOST_PENCILS]


def _edges(gray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), EDGE_BLUR)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)


def _around(v, n):
    """The mean of v within n steps either way, where it is known."""
    f = np.isfinite(v)
    C, F, i = np.r_[0, np.cumsum(np.where(f, v, 0))], np.r_[0, np.cumsum(f * 1.)], np.arange(len(v))
    lo, hi = np.maximum(i - n, 0), np.minimum(i + n + 1, len(v))
    return (C[hi] - C[lo]) / np.maximum(F[hi] - F[lo], 1)


def _moved(P, ray, back, grad):
    """A set moved along the vault through the picture, the comb it leaves: x, the edge along
    the curve there in spreads over its level, px per x. x is linear in true depth, so every
    rib of the family is a tooth, found by the curve finder or not. It is stepped in pixels,
    or the far ribs would take all the steps."""
    gx, gy = grad
    h, w = gx.shape
    ok = np.flatnonzero(np.isfinite(P["a"]) & np.isfinite(P["b"]))
    ok = ok[::max(1, len(ok) // SWEEP_RAYS)]
    A, B, th = P["a"][ok], P["b"][ok], np.radians(ray[ok])
    Am, Bm = np.median(A), np.median(B)
    r = np.arange(2 / (Am + P["pos"].min() * Bm), FAR_PX, -COMB_STEP)   # from twice the nearest curve
    x = (1 / r - Am) / Bm
    u = A[None] + x[:, None] * B[None]
    on = u.min(1) > 0
    x, u, r = x[on], u[on], r[on]
    if len(x) < 30:
        return x, np.full(len(x), np.nan), r * r * Bm
    pt = back(np.dstack([np.sin(th) / u, np.cos(th) / u]).reshape(-1, 2)).reshape(len(u), len(th), 2)
    t = np.gradient(pt, axis=1)
    px, py = pt[..., 0], pt[..., 1]
    inn = (px >= 2) & (py >= 2) & (px < w - 3) & (py < h - 3)
    X, Y = np.where(inn, px, -5).astype(np.float32), np.where(inn, py, -5).astype(np.float32)
    G = (cv2.remap(gy, X, Y, cv2.INTER_LINEAR, borderValue=0) * t[..., 0]
         - cv2.remap(gx, X, Y, cv2.INTER_LINEAR, borderValue=0) * t[..., 1]) / np.maximum(np.hypot(t[..., 0], t[..., 1]), 1e-9)
    c = inn.sum(1)
    S = np.where(c >= .4 * len(ok), (np.abs(G) * inn).sum(1) / np.maximum(c, 1), np.nan)
    s = S - _around(S, int(BUMP_PX / COMB_STEP))        # edges are left of it, not a web's shading
    f = np.isfinite(s)
    if not f.any():
        return x, s, r * r * Bm
    m = np.median(s[f])
    return x, (s - m) / max(1.48 * np.median(np.abs(s[f] - m)), 1e-9), r * r * Bm


def _spikes(x, s, res):
    """The teeth of a comb: where, how tall, px per x there."""
    with np.errstate(invalid="ignore"):
        i = np.flatnonzero(np.r_[False, (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:]) & (s[1:-1] >= SPIKE), False])
    keep = []
    for j in i[np.argsort(-s[i])]:
        if all(abs(int(j) - k) * COMB_STEP >= SPIKE_APART for k in keep):
            keep.append(int(j))
    keep = np.array(sorted(keep), int)
    return x[keep], s[keep], res[keep]


def _power(teeth, p, harmonics=HARMONICS):
    """The power of the teeth clear of each other at period p, over its first harmonics, for a
    rib is a narrow tooth and has an edge or two to it: (Z, phase, teeth)."""
    X, W, R = teeth
    u = R * p >= PITCH
    if u.sum() < FEWEST_PHASES:
        return 0., 0., 0
    w = W[u]
    c = (w * np.exp(2j * np.pi * np.arange(1, harmonics + 1)[:, None] * X[u] / p)).sum(1) / w.sum()
    return float(w.sum() ** 2 / (w * w).sum() * (abs(c) ** 2).sum()), float(np.angle(c[0]) / (2 * np.pi) * p), int(u.sum())


def _period(teeth, depth, window=None):
    """The period the teeth of a comb lie on: {p, phase, Z, chance, teeth} or None.

    Each tooth is a phase, and on the true period the phases line up, a little
    unevenness costing little: Z, n R^2 summed over three harmonics, is a gamma with no
    period there, and the periods tried are counted. Teeth on p lie on p/2 and p/3 as
    well, so the longer is taken when it does nearly as well on its first harmonic alone.
    `depth` is the nearest rib's, in x.
    """
    X = teeth[0]
    if len(X) < FEWEST_PHASES:
        return None
    span = float(np.ptp(X))
    lo, hi = window or (BAY_RANGE[0] * depth, min(BAY_RANGE[1] * depth, span / 2))
    if hi <= lo or span <= 0:
        return None
    tries = max(span * (1 / lo - 1 / hi), 1.)
    fs = np.linspace(1 / hi, 1 / lo, int(np.clip(6 * tries, 40, 1500)))
    Z = np.array([_power(teeth, 1 / f)[0] for f in fs])
    i = int(np.argmax(Z))
    if not window:
        first = lambda j: _power(teeth, 1 / fs[j], 1)[0]
        for k in (3, 2):
            j = int(np.argmin(np.abs(fs - fs[i] / k)))
            if fs[i] / k >= fs[0] and first(j) >= LONGER_NEARLY * first(i):
                i = j
                break
    z, phase, n = _power(teeth, 1 / fs[i])
    return {"p": float(1 / fs[i]), "phase": phase, "Z": z, "teeth": n, "chance": float(1 - (1 - np.exp(-z) * (1 + z + z * z / 2)) ** (HARMONIC_TRIES * tries))}


def _sloped(a, b, n, ray):
    """A set's run along the vault, its foot from its crown, in x, and how far it comes in from
    the wall for each unit it rises, over the rays most of its curves hold. a/b is the rib's
    depth, one number in a rib flat across the vault; sin/b and cos/b are its section, the
    first true for any camera and one number in a rib flat along a wall."""
    with np.errstate(all="ignore"):
        z = a / b
    ok = np.flatnonzero(np.isfinite(z) & (n >= .5 * n.max()))
    if len(ok) < 4:
        return None
    sign = 1. if abs(ray[ok[-1]]) >= abs(ray[ok[0]]) else -1.       # the foot is the end further from up
    th, w = np.radians(ray[ok]), np.sqrt(n[ok])
    X, Y = np.sin(th) / b[ok], np.cos(th) / b[ok]
    return np.array([sign * np.polyfit(ray[ok], z[ok], 1, w=w)[0] * np.ptp(ray[np.isfinite(z)]),
                     -np.sign(X.mean()) * np.polyfit(Y, X, 1, w=w)[0]])


def _run(P, U, of, ray, curves, N, V, moved):
    """(run, lean), (how far off each may be), the run in bays. Off by what splitting its curves
    in two does, and by what the point moving `moved` px either way does: a flat rib read about
    a point 30 px to one side has a run of a third of a bay. Unknown with nothing to spare,
    and with no uprights, when a tilt runs every rib alike."""
    n = np.isfinite(U[P["on"]]).sum(0)
    got, scale = _sloped(P["a"], P["b"], n, ray), np.array([1 / P["p"], 1.])
    if got is None:
        return None, None
    unknown = got * scale, np.full(2, np.inf)
    if len(P["on"]) < 4 or moved is None or V is None:
        return unknown
    o = np.argsort(P["pos"])
    ends = []
    for half in (o[::2], o[1::2]):
        a, b = _on_rays(U[P["on"][half]], P["pos"][half])
        ends.append(_sloped(a, b, np.isfinite(U[P["on"][half]]).sum(0), ray))
    off = np.zeros(2) if ends[0] is None or ends[1] is None else np.abs(ends[0] - ends[1]) / 2
    ids = sorted(set(of[P["on"]].tolist()))
    by = dict(zip(of[P["on"]].tolist(), P["pos"].tolist()))
    for d in ((moved[0], 0), (-moved[0], 0), (0, moved[1]), (0, -moved[1])):
        W, at, rr, _ = _table([curves[i] for i in ids], np.asarray(N, float) + d, V)
        if len(W) < 2:
            return unknown
        pos = np.array([by[ids[i]] for i in at])
        a, b = _on_rays(W, pos)
        for _ in range(2):
            if np.isfinite(b).sum() < LEAST_RAYS:
                return unknown
            with np.errstate(all="ignore"):
                pos = np.nan_to_num(np.nanmedian((W - a) / b, axis=1))
            a, b = _on_rays(W, pos)
        there = _sloped(a, b, np.isfinite(W).sum(0), rr)
        if there is None:
            return unknown
        off = np.maximum(off, np.abs(there - got))
    off = off * scale
    return got * scale, np.where(off <= MOST_RUN, off, np.inf)      # past that the point moved has left the curves


def _called(got, off):
    """Flat if it stays in a plane across the vault or along a wall, 3D if clearly neither."""
    limit, clear = (FLAT_RUN, FLAT_LEAN), CALL_CLEAR * off
    return "flat" if (np.abs(got) + clear <= limit).any() else "3D" if (np.abs(got) - clear >= limit).all() else "undetermined"


def _one_section(P, Q):
    """The rays two sets share if they read one bay on them, which is T/radius, or None."""
    s = np.isfinite(P["a"] + P["b"] + Q["a"] + Q["b"])
    if s.sum() >= LEAST_RAYS and abs(np.log(np.median(P["b"][s] * P["p"] / (Q["b"][s] * Q["p"])))) <= SAME_BAY:
        return s
    return None


def _between(P, Q, ray):
    """How far two families on one section run apart over the longer sweep, in bays, and how
    loosely: None if they are not on one. b is the section, so on a ray they share, the
    difference of two ribs' u over one b takes out what the point and the planes across
    the vault do to each alike. Each rib is taken where its curves are, for a rib moved off
    them carries every wobble of its b, and the one b is the fuller set's. It says what they
    do between them, and so not which."""
    s = _one_section(P, Q)
    if s is None:
        return None
    rib = lambda R: R["a"][s] + np.median(R["pos"]) * R["b"][s]
    full = max(P, Q, key=lambda R: len(R["pos"]))
    d = (rib(P) - rib(Q)) / (full["b"][s] * full["p"])
    c = np.polyfit(ray[s], d, 1)
    sweep = max(np.ptp(ray[np.isfinite(R["a"] + R["b"])]) for R in (P, Q))
    return float(c[0] * sweep), float((d - np.polyval(c, ray[s])).std() * np.sqrt(12 * SAMPLES_PER_FACT / s.sum()) * sweep / np.ptp(ray[s]))


def _family(P, ray, back, size):
    """A family's ribs, on their teeth: [(points, found, depth along the vault)], from the first
    found to the last, and no further than one rib missed at a time. The picture goes on past
    the vault's end, and what is there has edges too."""
    ok = np.isfinite(P["a"]) & np.isfinite(P["b"])
    A, B, th = P["a"][ok], P["b"][ok], np.radians(ray[ok])
    Am, Bm, p, (X, W, _) = np.median(A), np.median(B), P["p"], P["spikes"]
    h, w = size
    ribs = []
    for k in range(int(np.floor((P["x"][0] - P["phase"]) / p)), int(np.ceil((P["x"][-1] - P["phase"]) / p)) + 1):
        x = P["phase"] + k * p
        if Am + x * Bm <= 0 or Bm / (Am + x * Bm) ** 2 * p < PITCH:
            continue
        near = np.flatnonzero(np.abs(X - x) <= TOOTH_REACH * p)
        if len(near):
            x = float(X[near[np.argmax(W[near])]])
        u = A + x * B
        if (u <= 0).any():
            continue
        pts = back(np.c_[np.sin(th) / u, np.cos(th) / u])
        if ((pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < w) & (pts[:, 1] < h)).mean() >= IN_FRAME:
            ribs.append((pts, len(near) > 0, x + Am / Bm))
    best, run = [], []
    for r in ribs:
        if not r[1] and (not run or not run[-1][1]):
            run = []
            continue
        run = run + [r]
        if sum(q[1] for q in run) > sum(q[1] for q in best):
            best = run
    while best and not best[-1][1]:
        best = best[:-1]
    return best


def repeats(found, V, gray, lines):
    """What repeats along a vault, whatever its pattern: families of ribs, each at one bay.

    Moved along the vault any object stays on its rays from the point, and on each ray
    1/r is affine in how far it moved, whatever the lens, the tilt, the section or the
    plan: u(ray, x) = a(ray) + x b(ray). So curves that are copies of each other are
    found with no pattern assumed (`_copies`), moved through the picture (`_moved`), and
    every rib of theirs is a tooth on a line linear in true depth, where a period is
    a plain question (`_period`). A family sharing rays with one already read has to
    stand on its bay, that bay being T/radius and so comparable on shared rays. Its ribs
    are flat when a/b is one number, or sin/b is, within what its curves and the point
    can tell: flat across the vault or flat along a wall. Two families on one section
    say more between them (`_between`). None when nothing repeats.
    """
    N, curves, diag = np.asarray(found["point"], float), found["curves"] + _traced(gray), found["diag"]
    U, of, ray, back = _table(curves, N, V)
    sets = _copies(U, max(PENCIL_MISS * diag, 2.)) if len(U) >= FEWEST_APART else []
    grad = _edges(gray)
    for P in sets:
        n = np.isfinite(U[P["on"]]).sum(0)
        P["a"], P["b"] = (np.where(n >= .5 * n.max(), P[k], np.nan) for k in "ab")     # its ends rest on a curve or two
        P["x"], P["s"], P["res"] = _moved(P, ray, back, grad)
        with np.errstate(all="ignore"):
            P["depth"] = float(np.nanmedian((P["a"] + P["pos"].min() * P["b"]) / P["b"]))
        P["spikes"] = _spikes(P["x"], P["s"], P["res"])
        P["read"] = _period(P["spikes"], P["depth"])
    paced = [P for P in sets if P["read"]]
    for P in paced:
        P["p"] = P["read"]["p"]
    for i, P in enumerate(paced):
        for Q in paced[i + 1:]:
            if _one_section(P, Q) is not None:
                P["read"]["chance"] = Q["read"]["chance"] = min(P["read"]["chance"], Q["read"]["chance"], P["read"]["chance"] * Q["read"]["chance"] * ONE_BAY)
    sets.sort(key=lambda P: -(P["read"]["Z"] if P["read"] else 0))
    at = drawn(lines, LINE_PX)
    moved, done = (np.std(at, axis=0).tolist() if len(at) else None), []
    for P in sets:
        R, shared = P["read"], None
        for Q in done:
            s = np.isfinite(P["b"]) & np.isfinite(Q["b"])
            if s.sum() >= LEAST_RAYS:
                shared = Q["p"] * np.median(Q["b"][s]) / np.median(P["b"][s])
                break
        if shared:
            R = _period(P["spikes"], P["depth"], (shared * np.exp(-SAME_BAY), shared * np.exp(SAME_BAY)))
        if not R or R["chance"] > (MOST_THE_BAY if shared else MOST_ANY_BAY / len(sets)):
            continue
        P.update(R)
        got, off = _run(P, U, of, ray, curves, N, V, moved)
        if got is None:
            continue
        (P["run"], P["lean"]), (P["off"], P["lean_off"]) = got, off
        P["ribs"] = _family(P, ray, back, gray.shape[:2])
        if abs(P["run"]) > MOST_RUN or sum(r[1] for r in P["ribs"]) < FEWEST_FOUND:
            continue
        P["family"] = _called(got, off)
        done.append(P)
    if not done:
        return None
    pairs = []
    for i, P in enumerate(done):
        for j, Q in enumerate(done[i + 1:], i + 1):
            got = _between(P, Q, ray)
            if got and abs(got[0]) - CALL_CLEAR * got[1] >= FLAT_RUN and min(abs(P["lean"]), abs(Q["lean"])) >= FLAT_LEAN:
                pairs.append({"between": [i, j], "run": round(got[0], 2), "off": round(got[1], 2)})
                for R, S, d in ((P, Q, -got[0]), (Q, P, got[0])):       # were R flat, S would run d
                    if R["family"] == "undetermined" and abs(S["run"] - d) > CALL_CLEAR * np.hypot(S["off"], got[1]):
                        R["family"] = "3D"
    built, ribs = [], []
    for P in done:
        for pts, on, z in P["ribs"]:
            built.append({"pts": pts[::max(1, len(pts) // RULE_POINTS)].round(1).tolist(), "family": P["family"], "found": bool(on)})
            if on:
                ribs.append({"pts": built[-1]["pts"], "verdict": P["family"], "run": round(abs(P["run"]) * P["p"] / z, 3)})
    count = {v: sum(P["family"] == v for P in done) for v in ("flat", "3D", "undetermined")}
    kinds = ", ".join(f"{n} {v}" for v, n in count.items() if n)
    held = "" if moved is None else f" (\u00b1{moved[0]:.0f}, \u00b1{moved[1]:.0f})"
    says = (f"Read about the point {N[0]:.0f}, {N[1]:.0f}{held} with {len(lines)} lines through it. "
            f"{len(done)} {'family' if len(done) == 1 else 'families'} of rib at one bay, {done[0]['p'] / done[0]['depth'] * 100:.0f} % "
            f"of the nearest rib's depth: {kinds}. Their teeth lie on that bay by chance {max(P['chance'] for P in done) * 100:.2g} % of the time at worst.")
    said = lambda v, off: f"{v:.2f}" + (f" \u00b1 {off:.2f}" if np.isfinite(off) else "")
    says += " " + "; ".join(f"{P['family']}: runs {said(P['run'], P['off'])} of a bay and comes in {said(P['lean'], P['lean_off'])} for each unit it rises"
                            for P in done) + "."
    for q in pairs:
        says += f" Two of them lie on one section and run {abs(q['run']):.2f} \u00b1 {q['off']:.2f} of a bay apart, whatever the point, so they are not both flat."
    if count["undetermined"]:
        says += " Undetermined means its curves, or the point moved as far as its lines allow, could make it either."
    if moved is None:
        says += " Three lines or more are needed to say how far the point can move, so no family is called flat or 3D."
    elif V is None:
        says += " Uprights are needed to say how the planes across the vault lie, so no family is called flat or 3D by itself."
    return {"verdict": "repeats", "why": "", "says": says, "ribs": ribs, "point": [round(float(c), 1) for c in N],
            "lines": len(lines), "held_px": moved, "bay": round(done[0]["p"] / done[0]["depth"], 3),
            "chance": round(max(P["chance"] for P in done), 6),
            "families": [{"family": P["family"], "run": round(P["run"], 2), "off": None if not np.isfinite(P["off"]) else round(P["off"], 2),
                          "lean": round(P["lean"], 2), "lean_off": None if not np.isfinite(P["lean_off"]) else round(P["lean_off"], 2),
                          "p": round(P["p"], 4), "Z": round(P["Z"], 1), "chance": round(P["chance"], 6), "ribs": len(P["ribs"]),
                          "found": sum(r[1] for r in P["ribs"])} for P in done],
            "apart": pairs,
            "rule": {"verdict": "holds", "says": says, "ribs": built}}


def read(found, V, gray, lines, claimed=None):
    """What can be said of the vault in `found`, for the page: the rule of §53 where it holds,
    what repeats where it does not, and otherwise why neither, or None if nothing here looks
    like a vault. `lines` run to its point, and `claimed` is the perspective the page claims."""
    no = lambda why: {"verdict": "undetermined", "why": why, "ribs": []}
    if found["point"] is None and len(found["curves"]) < FEWEST_ASKED:
        return None
    if claimed == "five":
        return no("a vault is read by moving its ribs along straight rays from one point, "
                  "and in 5-point those rays bend")
    if found["point"] is None:
        return no("nothing says what point it runs to. Mark 3 lines running down it as a line family, "
                  "select that layer and Detect edges again")
    out = sort(found, V)
    if out["verdict"] == "sorted" and out["rule"]["verdict"] == "holds":
        return out
    more = repeats(found, V, gray, lines)
    if more:
        return more
    if out["verdict"] != "sorted":
        out["why"] += ", and nothing repeats along it at one bay"
    return out
