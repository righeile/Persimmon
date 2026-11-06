"""The nearest consistent perspective, and what it costs the drawing.

A hand-drawn scene never closes exactly. Three vanishing points marked off a
painting will not be a right-angled triple, the horizon will not be level when
the verticals are upright, and asking "is this consistent?" gets the answer no
every time, which helps nobody.

So ask the useful question instead: **what is the nearest perspective that IS
consistent, and how far does the drawing have to move to meet it?** One camera,
one orthogonal triple of vanishing points, chosen to minimise how much the
drawn lines must shift. Then every line carries the distance its own ends must
travel, in pixels, and the artist fixes the few that cost the most instead of
redrawing the composition.

The cost is measured where the artist sees it. Rotating a line about its own
midpoint to reach a vanishing point moves each end by (L/2)*sin(theta), so a
2 deg error on a 60 px mark is 1 px and nobody can see it, while the same 2 deg
on a 900 px mark is 16 px and it is the first thing you notice. Angle alone
ranks those the same and would send the artist to the wrong line.

Two numbers decide whether a disagreement is real at all:

  mark accuracy   hand marking measured ~9 px against the paint. A line that
                  needs to move less than that is not wrong; the mark is.
  drawn spread    how much the drawn lines disagree among THEMSELVES, before
                  any consistency constraint. If imposing a camera costs no
                  more than that, the camera is free.
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Measured median accuracy of hand marking against the paint -- but measured on
# ONE scan, so it is a fraction of the picture, not a count of pixels. All five
# testdata images happen to be 1280 on the long edge, so an absolute 9.0 passed
# every check here while under-flagging every larger scan: the same hand error
# that reads 9 px at 1280 reads 35 px on an A4 at 300 dpi, where a flat 9 px
# would call it exact. Scale it, and say what the scaling assumes.
MARK_PX = 9.0
MARK_REF_WH = (884.0, 1280.0)                 # the scan it was measured on
MARK_REF_DIAG = float(np.hypot(*MARK_REF_WH))  # ...1555.7 px corner to corner


def _mark_basis(diag):
    if diag is None:
        return (f"{MARK_PX:g} px -- no image size was given, so the figure "
                "measured on the reference scan is used unscaled")
    if abs(diag - MARK_REF_DIAG) < 1.0:
        return (f"{MARK_PX:g} px, measured by hand against the paint on a scan "
                "this size")
    return (f"{MARK_PX:g} px measured on a {MARK_REF_DIAG:.0f} px diagonal, "
            f"scaled to this picture's {diag:.0f} px diagonal")


def _mark_px(image_size):
    """What a hand mark is worth in THIS picture's pixels.

    Assumes the marking was done at a zoom that fits the picture on screen,
    which is what the marking tool does by default; mark at 400% and you do
    better than this, which only makes the verdict cautious, never generous.
    """
    if not image_size:
        return MARK_PX, None
    d = float(np.hypot(float(image_size[0]), float(image_size[1])))
    return MARK_PX * d / MARK_REF_DIAG, d


def _rodrigues(r):
    t = float(np.linalg.norm(r))
    if t < 1e-12:
        return np.eye(3)
    k = r / t
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * (Kx @ Kx)


# How many of the scene's three perpendicular directions converge inside the
# picture is not a matter of style. It is a statement about where the camera
# was pointed, and written that way the three cases are one model with one,
# two or three free angles:
#
#   three-point  the camera is pitched up or down, so none of the three world
#                axes lies in the picture plane and all three vanishing points
#                are finite. Yaw, pitch and roll all free.
#   two-point    the camera is level. The world's upright direction lies in
#                the picture plane, so uprights stay parallel on the page and
#                only the two horizontal directions converge. Pitch = 0.
#   one-point    the camera is level AND square to one wall. Two directions
#                lie in the picture plane -- horizontals stay horizontal,
#                uprights stay upright -- and only depth converges.
#                Pitch = 0 and yaw = 0; roll is whatever the paper was turned.
#
# Declaring the type is not a hint to the solver. It is the claim being
# checked: "how far is this drawing from the perspective I meant to draw".
MODES = ("one", "two", "three")

# What each type can be asked for, and what it cannot. In one-point
# perspective R reduces to a roll about the optical axis, which sends the
# world's depth axis onto the optical axis itself -- so the single vanishing
# point IS the principal point, and the focal length appears in no vanishing
# point at all. f is not ill-conditioned there, it is absent from the model:
# no number of marked lines can recover the lens from a one-point drawing.
_FREE_ANGLES = {"one": 1, "two": 2, "three": 3}

# A vanishing point is "at infinity" when its position carries no information,
# which is a distance relative to the picture and not a test on the
# homogeneous w. Reading w alone called a vanishing point 2e11 px away finite
# and let it calibrate the lens, which is how f = 1202604 px got printed for a
# 1500 px picture. FMAX is the same ceiling core/trim.py puts on a focal length
# read off a quadrilateral: past it, f is noise rather than a measurement.
FAR = 50.0        # x the picture diagonal
FMAX = 8.0        # x the picture diagonal


def _vp_xy(v, diag):
    """Where a vanishing point is, or None when it is effectively at infinity."""
    if abs(v[2]) <= 1e-12:
        return None
    p = np.asarray(v[:2], float) / v[2]
    return None if not np.isfinite(p).all() or np.hypot(*p) > FAR * diag else p


def _rot(yaw, pitch, roll):
    """Camera rotation as Rz(roll) @ Rx(pitch) @ Ry(yaw).

    This order is chosen so that the two constraints are just "set an angle to
    zero": pitch = 0 puts the world's upright direction in the picture plane
    (two-point), and yaw = 0 then puts a horizontal one there too (one-point).
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])
    Rx = np.array([[1.0, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]])
    return Rz @ Rx @ Ry


def _take_pp(params, i, pp_fixed):
    """Read the principal point out of a parameter vector, honouring pins.

    pp_fixed is None (both free), a pair of numbers (both pinned), or a pair
    with one None (that one free). The half-pinned case is not a convenience:
    two-point perspective genuinely cannot measure where pp sits ALONG the
    horizon -- slide it either way and the implied f changes to match, fitting
    the marks exactly at every position. Its height across the horizon is
    measurable. So x is pinned by assumption and y is measured.
    """
    if pp_fixed is None:
        return np.array([params[i], params[i + 1]]), i + 2
    fx, fy = pp_fixed
    if fx is None and fy is None:
        return np.array([params[i], params[i + 1]]), i + 2
    if fx is None:
        return np.array([params[i], float(fy)]), i + 1
    if fy is None:
        return np.array([float(fx), params[i]]), i + 1
    return np.array([float(fx), float(fy)]), i


def _vps_from(params, pp_fixed=None, mode="three"):
    """The three mutually perpendicular vanishing points a camera implies.

    Returns them in world-axis order: horizontal, upright, depth. For mode
    "one" the focal length is not a parameter and comes back as None, because
    it genuinely is not in the model -- see MODES above.
    """
    if mode != "three":
        n_ang = _FREE_ANGLES[mode]
        has_f = mode != "one"
        i = 0
        if has_f:
            logf = params[0]
            i = 1
        pp, i = _take_pp(params, i, pp_fixed)
        ang = list(params[i:i + n_ang])
        roll = ang[-1]
        yaw = ang[0] if n_ang >= 2 else 0.0
        pitch = ang[1] if n_ang >= 3 else 0.0
        # f cancels out of every vanishing point in one-point perspective, so
        # any value gives the same residual; 1.0 keeps the arithmetic tame.
        f = float(np.exp(np.clip(logf, -2.0, 14.0))) if has_f else 1.0
        K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
        return (K @ _rot(yaw, pitch, roll)).T, (f if has_f else None), pp
    logf = params[0]
    pp, i = _take_pp(params, 1, pp_fixed)
    r0, r1, r2 = params[i:i + 3]
    # A seed that walks off overflows exp and poisons the residual with nan.
    # It cannot win -- nan loses every comparison -- but clip it so the search
    # stays in a range where a camera means anything.
    f = float(np.exp(np.clip(logf, -2.0, 14.0)))
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    return (K @ _rodrigues(np.array([r0, r1, r2]))).T, f, pp


def _clip(segs, w, h):
    """Each drawn line cut down to the part that is inside the picture.

    A construction line dragged far past the canvas is still one line, but the
    distance its ENDS travel under a correction scales with how far it was
    dragged: on a real sheet one line ran 6176 px for 888 px of picture, and
    its "3.5 px move" was a measurement taken 2600 px off-canvas. Nothing out
    there is part of the composition, so nothing out there should be measured
    or weighted.

    The clipped segment lies on the same line, so every vanishing point is
    unchanged. What changes is that the pivot is now the middle of the visible
    stroke and the move is read at the picture's edge -- which is what "the
    least change to the composition" has to mean.

    Returns the clipped segments and the index of every line that missed the
    picture entirely.
    """
    out, missed = [], []
    for i, sg in enumerate(np.asarray(segs, float).reshape(-1, 4)):
        x0, y0, x1, y1 = sg
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        gone = False
        for pq, qq in ((-dx, x0), (dx, w - x0), (-dy, y0), (dy, h - y0)):
            if abs(pq) < 1e-12:
                if qq < 0:
                    gone = True
                    break
            else:
                r = qq / pq
                if pq < 0:
                    t0 = max(t0, r)
                else:
                    t1 = min(t1, r)
        if gone or t0 > t1:
            missed.append(i)
            continue
        out.append([x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy])
    return np.asarray(out, float).reshape(-1, 4), missed


def _seeds(segs, w, h):
    """Starting points taken from the drawing instead of from nowhere.

    A random small rotation is a hopeless seed. On a real two-point drawing the
    camera's rotation vector came out at |r| = 2.4 rad, which no sample from
    N(0, 0.4) ever reaches, and the solver slid off to f = 1e94 and 164 px of
    movement on a drawing that reconciles at 0.11 px.

    So start where the drawing already is: take each family's own vanishing
    point, map it into the camera frame at a trial f, and snap those directions
    to the nearest rotation. That is the camera the artist drew, to the extent
    one exists, and the fit only has to tidy it.
    """
    drawn = []
    for sg in segs:
        p1 = np.c_[sg[:, :2], np.ones(len(sg))]
        p2 = np.c_[sg[:, 2:], np.ones(len(sg))]
        drawn.append(np.linalg.svd(np.cross(p1, p2))[2][-1])
    cand = list(np.geomspace(0.4 * w, 6.0 * w, 9))
    if len(drawn) == 2:                       # the f that makes the two exactly
        a, b = [v[:2] / v[2] if abs(v[2]) > 1e-9 else None for v in drawn]
        if a is not None and b is not None:   # perpendicular at pp
            g = -((a[0] - w / 2) * (b[0] - w / 2) + (a[1] - h / 2) * (b[1] - h / 2))
            if g > 0:
                cand.insert(0, float(np.sqrt(g)))
    out = []
    for f in cand:
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        Ki = np.linalg.inv(K)
        c = [Ki @ v for v in drawn]
        c = [x / n for x, n in zip(c, [np.linalg.norm(x) for x in c]) if n > 1e-12]
        if len(c) < len(drawn):
            continue
        M = np.c_[tuple(c)] if len(c) == 3 else np.c_[c[0], c[1], np.cross(c[0], c[1])]
        # a vanishing point has no sign. Left-handed as they came, half the
        # time, the nearest rotation below is nowhere near them
        M[:, 2] *= np.sign(np.linalg.det(M)) or 1.0
        U, _, Vt = np.linalg.svd(M)
        R = U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt
        out.append((float(np.log(f)), Rotation.from_matrix(R).as_rotvec()))
    return out


def _move_px(seg, v):
    """Pixels each end of a segment actually travels when the segment is rotated
    about its own midpoint to meet the vanishing point `v` (homogeneous).

    The straight-line distance the end covers, L*sin(theta/2), not the
    linearised (L/2)*sin(theta) -- those agree only to third order, and the
    whole point of this number is that it is the distance the artist's pencil
    moves, so it should be that distance and not an approximation to it.
    """
    seg = np.asarray(seg, float).reshape(-1, 4)
    mx = (seg[:, 0] + seg[:, 2]) / 2
    my = (seg[:, 1] + seg[:, 3]) / 2
    d = np.stack([seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]], 1)
    L = np.hypot(d[:, 0], d[:, 1])
    if abs(v[2]) < 1e-12:
        r = np.broadcast_to(v[:2], d.shape).astype(float)
    else:
        r = np.stack([v[0] / v[2] - mx, v[1] / v[2] - my], 1)
    cross = np.abs(d[:, 0] * r[:, 1] - d[:, 1] * r[:, 0])
    dot = np.abs(d[:, 0] * r[:, 0] + d[:, 1] * r[:, 1])
    th = np.arctan2(cross, dot)              # undirected: a line has no arrow
    return L * np.sin(th / 2), np.degrees(th)


def drawn_spread(families):
    """How much each family's own lines disagree, before any camera is imposed.

    This is the floor. A consistent camera that costs no more than the lines
    already cost each other has taken nothing away from the drawing.
    """
    out = {}
    for name, seg in families.items():
        s = np.asarray(seg, float).reshape(-1, 4)
        if len(s) < 2:
            out[name] = None
            continue
        p1 = np.c_[s[:, :2], np.ones(len(s))]
        p2 = np.c_[s[:, 2:], np.ones(len(s))]
        H = np.cross(p1, p2)
        v = np.linalg.svd(H)[2][-1]          # the family's own best VP, unconstrained
        mv, _ = _move_px(s, v)
        out[name] = {"median_px": float(np.median(mv)), "worst_px": float(mv.max()),
                     "vp": (v[:2] / v[2]).tolist() if abs(v[2]) > 1e-12 else None}
    return out


def fit_level(families, *, image_size=None):
    """The same reconciliation for a level camera and TWO horizontal families.

    Most architectural painting is drawn from a level eye, and most people mark
    the two wall directions and not the uprights. That is enough: with no pitch
    or roll the horizon is horizontal, both vanishing points sit on it, and
    being perpendicular means

        (u_a - pp_x) (u_b - pp_x) = -f^2

    which is one equation short of free. Four unknowns -- f, the principal
    point, and the yaw -- against every line drawn.

    This is also the honest home for the "level" assumption. Imposing it buys a
    camera from two families, and the price is printed with the answer.
    """
    names = [n for n in families if len(np.asarray(families[n], float).reshape(-1, 4))]
    if len(names) != 2:
        return {"verdict": "undetermined",
                "why": (f"{len(names)} horizontal families; a level camera needs "
                        "exactly two, at right angles in the world"),
                "conditioning": {"families": len(names)}}
    w, h = image_size or (1000.0, 1000.0)
    mark, diag = _mark_px(image_size)
    segs, offpicture = [], {}
    for n in list(names):
        c, miss = _clip(families[n], w, h)
        if miss:
            offpicture[n] = miss
        segs.append(c)
    names = [n for n, c in zip(names, segs) if len(c)]
    segs = [c for c in segs if len(c)]
    if len(names) != 2:
        return {"verdict": "undetermined",
                "why": "a family's lines all fall outside the picture",
                "conditioning": {"off_picture": offpicture}}

    # What two families DETERMINE is where their two vanishing points sit, and
    # the height of the horizon. They do not determine f: with both VPs fixed,
    # (u_a - pp_x)(u_b - pp_x) = -f^2 trades the principal point against the
    # focal length along a whole one-parameter family, and fitting f directly
    # wanders down it (it returned 611 for a true 1100). So fit the three things
    # the lines constrain, then derive f from an assumed pp_x and report how
    # much that assumption is worth.
    pp_x = float(w / 2)

    def vps_of(params):
        ua, ub, ppy = params
        return np.array([ua, ppy, 1.0]), np.array([ub, ppy, 1.0])

    def resid(params):
        va, vb = vps_of(params)
        return np.concatenate([_move_px(segs[0], va)[0], _move_px(segs[1], vb)[0]])

    # start each VP where its own family already puts it
    p0 = []
    for sg in segs:
        p1 = np.c_[sg[:, :2], np.ones(len(sg))]
        p2 = np.c_[sg[:, 2:], np.ones(len(sg))]
        v = np.linalg.svd(np.cross(p1, p2))[2][-1]
        p0.append(float(v[0] / v[2]) if abs(v[2]) > 1e-9 else np.sign(v[0]) * 1e5)
    p0.append(float(h / 2))

    best = None
    for jig in (0.0, 0.1, -0.1):
        try:
            sol = least_squares(resid, [p0[0] * (1 + jig), p0[1] * (1 - jig), p0[2]],
                                method="lm", max_nfev=4000)
        except Exception:
            continue
        c = float(np.sqrt(np.mean(sol.fun ** 2)))
        if best is None or c < best[0]:
            best = (c, sol)
    if best is None:
        return {"verdict": "undetermined", "why": "the fit did not converge",
                "conditioning": {}}
    ua, ub, ppy = best[1].x
    va, vb = vps_of(best[1].x)
    pp = np.array([pp_x, float(ppy)])
    # The focal length is a SEPARATE question and may fail on its own. The
    # reconciled vanishing points, the horizon and every line's movement are
    # still good when it does, and throwing them away because a dependent
    # number failed would be the same mistake in the other direction.
    f2 = -(ua - pp_x) * (ub - pp_x)
    frange = [np.sqrt(g) for g in
              (-(ua - x) * (ub - x) for x in np.linspace(0.35 * w, 0.65 * w, 9))
              if g > 0]
    if f2 > 0:
        cam = {"focal_px": float(np.sqrt(f2)), "focal_refused": None}
    else:
        cam = {"focal_px": None, "focal_refused": (
            "both vanishing points land on the same side of the principal "
            "point, so these two directions cannot be at right angles under a "
            "level camera -- the scene was probably not shot level")}
    per, allmv = {}, []
    for n, s, v in zip(names, segs, (va, vb)):
        mv, ang = _move_px(s, v)
        allmv.append(mv)
        per[n] = {"move_px": mv.tolist(), "angle_deg": ang.tolist(),
                  "median_px": float(np.median(mv)), "worst_px": float(mv.max()),
                  "worst_line": int(np.argmax(mv)),
                  "vp": [float(v[0]), float(v[1])], "at_infinity": False}
    allmv = np.concatenate(allmv)
    free = drawn_spread(dict(zip(names, segs)))   # clipped, like the moves
    free_worst = max((d["worst_px"] for d in free.values() if d), default=0.0)
    cond = {"n_lines": int(len(allmv)), "assumed": "camera level (no pitch, no roll)",
            "median_move_px": float(np.median(allmv)),
            "worst_move_px": float(allmv.max()),
            "cost_rms_px": float(np.sqrt((allmv ** 2).mean())),
            "unconstrained_worst_px": float(free_worst),
            "price_of_consistency_px": float(allmv.max() - free_worst),
            "mark_accuracy_px": round(mark, 2),
            "mark_accuracy_basis": _mark_basis(diag), "off_picture": offpicture,
            "measured_on": "the part of each line inside the picture",
            "principal_point": pp.tolist(), **cam,
            "horizon_y": float(pp[1]), "per_family": per, "drawn_spread": free,
            "assumed_pp_x": pp_x, "vp_a": float(ua), "vp_b": float(ub),
            "focal_range_over_pp_x": ([round(min(frange), 1), round(max(frange), 1)]
                                      if frange else None),
            "note": ("two families fix the vanishing points and the horizon, not "
                     "the focal length; f here follows from putting the principal "
                     "point at the image centre")}
    verdict = ("already consistent" if float(allmv.max()) <= mark
               else "rectifiable")
    return {"value": (round(cam["focal_px"], 1) if cam["focal_px"] else None),
            "verdict": verdict, "conditioning": cond}


def _mean_angle(seg):
    """The direction a family's lines mostly run, in radians, undirected."""
    s = np.asarray(seg, float).reshape(-1, 4)
    d = np.stack([s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]], 1)
    # doubling the angle makes a line and its reverse agree before averaging
    a = 2 * np.arctan2(d[:, 1], d[:, 0])
    L = np.hypot(d[:, 0], d[:, 1])
    return float(np.arctan2((L * np.sin(a)).sum(), (L * np.cos(a)).sum()) / 2)


def _own_vp(seg):
    """A family's own unconstrained vanishing point, or None if parallel."""
    s = np.asarray(seg, float).reshape(-1, 4)
    if len(s) < 2:
        return None
    p1 = np.c_[s[:, :2], np.ones(len(s))]
    p2 = np.c_[s[:, 2:], np.ones(len(s))]
    v = np.linalg.svd(np.cross(p1, p2))[2][-1]
    return (v[:2] / v[2]) if abs(v[2]) > 1e-9 else None


def _starts(mode, segs, axes, w, h, free_pp, pp_fixed):
    """Where to start the search, per perspective type.

    Three-point keeps `_seeds`, which snaps the drawing's own vanishing points
    to the nearest rotation -- a random small rotation is a hopeless seed there
    and cost a wrong answer once already. The constrained types have one or two
    free angles instead of three, and both of those angles can be read almost
    directly off the marks, so they are seeded from the drawing and swept only
    where the drawing does not say.
    """
    if mode == "three":
        base = _seeds(segs, w, h)
        for f0 in (0.7 * w, 1.6 * w):          # a couple of blind ones as well
            base += [(np.log(f0), np.random.default_rng(k).normal(0, 0.4, 3))
                     for k in range(2)]
        return [([logf, w / 2, h / 2, *r] if free_pp else [logf, *r])
                for logf, r in base]

    at = {ax: i for i, ax in enumerate(axes)}
    # roll is what the paper was turned by, and the marks state it outright:
    # an upright family runs at 90 deg + roll, a horizontal one at roll
    rolls = [0.0]
    if 1 in at:
        rolls.insert(0, _mean_angle(segs[at[1]]) - np.pi / 2)
    if 0 in at:
        rolls.insert(0, _mean_angle(segs[at[0]]))
    rolls = [float(np.arctan2(np.sin(r), np.cos(r))) for r in rolls]

    pps = [(w / 2, h / 2)]
    if mode == "one" and 2 in at:
        v = _own_vp(segs[at[2]])               # in one-point the VP IS the pp
        if v is not None and np.isfinite(v).all():
            pps.insert(0, (float(v[0]), float(v[1])))

    out = []
    if mode == "one":
        for pp in (pps if free_pp else [None]):
            for r in rolls:
                out.append(([pp[0], pp[1], r] if free_pp else [r]))
        return out

    # Two-point, solved rather than swept. With no pitch both finite vanishing
    # points sit on the horizon through pp, at signed distances whose product
    # is -f^2. So fix pp's x, intersect the horizon with that x to get its y,
    # and f and yaw drop out in closed form. A blind sweep over f and yaw got
    # the same answers 80 starts and 20 seconds later.
    horiz = [at[a] for a in (0, 2) if a in at]
    for r in rolls[:2]:
        u = np.array([np.cos(r), np.sin(r)])
        if len(horiz) < 2:
            continue
        a, b = (_own_vp(segs[i]) for i in horiz)
        if a is None or b is None:
            continue
        px = float(pp_fixed[0]) if pp_fixed else w / 2.0
        d = b - a
        if abs(d[0]) < 1e-6:                   # horizon vertical at this roll
            continue
        pp = np.array([px, float(a[1] + d[1] * (px - a[0]) / d[0])])
        g = -float((a - pp) @ u) * float((b - pp) @ u)
        if not np.isfinite(g) or g <= 0:       # pp outside the two VPs
            continue
        f0 = float(np.sqrt(g))
        y0 = float(np.arctan2(float((b - pp) @ u), f0))
        p0 = [float(np.log(f0))]
        if free_pp:
            p0 += [pp[1]] if pp_fixed else [pp[0], pp[1]]
        out.append(p0 + [y0, r])

    # The closed form only solves at all when the family called upright really
    # is the upright one, so it doubles as the test of the axis assignment:
    # on three measured drawings it fired on the correct assignment and on no
    # other, answered in 1 ms, and the sweep below never beat it. So sweep
    # only where it said nothing.
    if out:
        return out
    for f0 in (0.8 * w, 2.0 * w):
        for y in np.radians([-40, -15, 15, 40]):
            for r in rolls[:2]:
                p0 = [float(np.log(f0))]
                if free_pp:
                    p0 += ([h / 2] if pp_fixed
                           else [w / 2, h / 2])
                out.append(p0 + [float(y), r])
    return out


def _assignments(mode, names, roles):
    """Which of the three world axes each family is claimed to run along.

    Axis 0 is a horizontal direction, 1 is the upright, 2 is depth. Most of
    the choice is free -- swapping two axes is just a different rotation, and
    the solver reaches it either way -- but the axis each MODE singles out is
    not free, so that one is either declared or tried.
    """
    n = len(names)
    roles = {k: v for k, v in (roles or {}).items() if k in names}
    want = {"vertical": 1, "upright": 1, "depth": 2}
    pinned = {names.index(k): want[v] for k, v in roles.items() if v in want}

    if mode == "three":
        return [list(range(n))]                 # any labelling, same rotation

    # For the constrained types the rotation is NOT free, so which axis each
    # family runs along changes the answer and every labelling has to be
    # tried. In particular the axis a mode singles out need not be marked at
    # all: two-point perspective is drawn by marking the two HORIZONTAL
    # families, and the uprights have no vanishing point to mark. Forcing one
    # family onto the upright axis gave 322 px of moves on testdata/4.jpg,
    # where both marked families are horizontal and neither is upright.
    out = []
    for axes in permutations((0, 1, 2), n):
        if any(axes[i] != a for i, a in pinned.items()):
            continue
        out.append(list(axes))
    return out or [list(range(n))]


def _pp_implausible(pp, f, w, h, diag):
    """Why a freely fitted principal point cannot be believed, or None.

    Three vanishing points locate the principal point only if their triangle
    is well shaped. A near-vertical family puts one of them tens of thousands
    of pixels away, the triangle degenerates to a sliver, and the solver can
    buy a LOWER residual by sliding pp onto one of the other two vanishing
    points -- where that family's constraint goes slack and f collapses with
    it. On testdata/4.jpg that returned f = 48 px with pp four image widths
    out, at 1.08 px of movement against the honest answer's 2.34, which is
    exactly how a wrong number gets believed.
    """
    if not (-w <= pp[0] <= 2 * w and -h <= pp[1] <= 2 * h):
        return (f"the principal point was fitted to ({pp[0]:.0f}, {pp[1]:.0f}), "
                f"outside a {w:.0f}x{h:.0f} picture -- the three vanishing "
                "points are too nearly collinear to place it, and a lower "
                "residual bought this way is overfitting, not a better camera")
    if f < 0.2 * diag:
        return (f"the fitted focal length, {f:.0f} px on a {diag:.0f} px "
                "diagonal, is shorter than any real lens -- the principal "
                "point ran onto a vanishing point and took f down with it")
    return None


def fit(families, *, image_size=None, pp_fixed=None, free_pp=True,
        mode="three", roles=None):
    """One camera and one orthogonal triple, as close to the drawing as possible.

    `families` maps a name to that family's drawn lines, [[x1,y1,x2,y2], ...].
    Two or three are used, and they are taken to be mutually perpendicular in
    the world -- which is the claim the artist is making by drawing a box of a
    room. One family is always consistent on its own, so there is nothing to
    reconcile and nothing to report.

    `mode` is the perspective type being claimed -- "one", "two" or "three"
    point -- and it is a constraint on the camera, not a hint to the solver:
    see MODES. Three-point leaves the camera free, which is what this did
    before the types existed, so that path is unchanged. Two-point forbids
    pitch, so uprights must come out parallel on the page. One-point forbids
    pitch and yaw, so horizontals must come out horizontal and uprights
    upright, and the only vanishing point there is sits at the principal
    point. The moves reported are then "how far from the perspective you said
    you were drawing", which is a different and more useful question than
    "how far from any perspective at all".

    `roles` optionally names which family runs along the axis its mode singles
    out -- {"family id": "vertical"} for two-point, {"family id": "depth"} for
    one-point. Without it every candidate is tried and the cheapest wins, and
    the one that won is reported as `axis_of`.

    Three families pin the focal length. TWO do not: the camera still has a
    free roll and a free pitch to spend, so the two drawn vanishing points can
    usually be made perpendicular at no cost at all, and f then depends on
    where the principal point is assumed to be. That range is reported beside
    f, and it is wide -- so with two families, trust the reconciliation and the
    moves, not the absolute 3-D numbers.

    Returns the camera, the reconciled vanishing points, and for every line the
    distance its ends must move to join them.
    """
    if mode not in MODES:
        return {"verdict": "undetermined",
                "why": ("the perspective type must be one of "
                        + ", ".join(f"{m}-point" for m in MODES)),
                "conditioning": {"mode": mode}}
    names = [n for n in families if len(np.asarray(families[n], float).reshape(-1, 4))]
    if len(names) not in (2, 3):
        return {"verdict": "undetermined",
                "why": (f"{len(names)} line families; two or three perpendicular "
                        "directions are needed -- one family is always consistent "
                        "on its own, so there is nothing to reconcile"),
                "conditioning": {"families": len(names), "mode": mode}}
    w, h = image_size or (1000.0, 1000.0)
    mark, diag = _mark_px(image_size)
    segs, offpicture = [], {}
    for n in list(names):
        c, miss = _clip(families[n], w, h)
        if miss:
            offpicture[n] = miss
        segs.append(c)
    names = [n for n, c in zip(names, segs) if len(c)]
    segs = [c for c in segs if len(c)]
    if len(names) not in (2, 3):
        return {"verdict": "undetermined",
                "why": "a family's lines all fall outside the picture",
                "conditioning": {"off_picture": offpicture, "mode": mode}}

    # Two families cannot locate the principal point: two perpendicular
    # directions give one equation and pp has two degrees of freedom, so the
    # six-parameter camera is rank-deficient and the solver walks off to
    # infinity (it reached f = 1.7e37 and 164 px moves on a drawing that
    # reconciles at 0.00 px). Pin pp and the parameterization is exactly
    # determined: four parameters against the four degrees of freedom two
    # perpendicular vanishing points have.
    #
    # One-point is the exception: there the principal point IS the vanishing
    # point, so pinning it to the frame centre would be assuming the answer.
    # Two-point needs only its lateral pin, below: with pp_x assumed, the two
    # horizontal vanishing points fix the roll (the line joining them), pp_y
    # (that line at the assumed x) and f -- four parameters against four, and
    # the horizon's HEIGHT is measured rather than assumed. Pinning pp whole
    # forced the horizon through the image centre and cost testdata/4.jpg 126
    # px of moves it does not owe.
    pinned_pp = False
    if len(names) == 2 and free_pp and mode == "three":
        free_pp = False
        pinned_pp = True
        if pp_fixed is None:
            pp_fixed = (w / 2, h / 2)

    # `pin` is what the camera is actually solved under: None where pp is
    # measured, a number where it is assumed.
    pin = None if free_pp else (pp_fixed or (w / 2, h / 2))
    lateral_pin = False
    if mode == "two" and pin is None:
        # Two-point cannot measure pp ALONG the horizon. Both finite vanishing
        # points lie on the horizon through pp at signed distances whose
        # product is -f^2, so sliding pp sideways only trades for a different
        # f and fits the marks exactly the whole way. Measured on a synthetic
        # camera: pp slid +-400 px gave f from 868 to 1254, every one of them
        # an exact fit, and the free solver duly returned pp = (234, 447) and
        # f = 1245 for a true (560, 470) and 1150. So x is assumed and y --
        # the horizon's height, which the two vanishing points DO fix -- is
        # measured.
        pin = ((w / 2) if pp_fixed is None else float(pp_fixed[0]), None)
        lateral_pin = True

    best = None
    for axes in _assignments(mode, names, roles):
        def resid(params, _ax=axes):
            V, _f, _pp = _vps_from(params, pin, mode)
            return np.concatenate([_move_px(s, V[a])[0] for s, a in zip(segs, _ax)])

        for p0 in _starts(mode, segs, axes, w, h,
                          pin is None or pin[1] is None, pin):
            try:
                # 4000 is what three-point has always used. The constrained
                # modes converge in a millisecond from a seed read off the
                # drawing, so the big budget only buys LM a long grind on the
                # axis assignments that are wrong anyway -- 5.6 s for one of
                # them, against 1 ms for the answer.
                sol = least_squares(resid, p0, method="lm",
                                    max_nfev=4000 if mode == "three" else 300)
            except Exception:
                continue
            c = float(np.sqrt(np.mean(sol.fun ** 2)))
            if best is None or c < best[0]:
                best = (c, sol, axes)
    if best is None:
        return {"verdict": "undetermined", "why": "the fit did not converge",
                "conditioning": {"mode": mode}}

    sol, axes = best[1], best[2]
    V, f, pp = _vps_from(sol.x, pin, mode)

    # A freely fitted principal point is only worth having when the vanishing
    # points can actually place it. When they cannot, pin it and say so, rather
    # than quote the camera the solver preferred. Only three-point can go wrong
    # this way: two-point already has its lateral pin (see above), and
    # one-point's pp is the measurement.
    if free_pp and mode == "three":
        bad = _pp_implausible(pp, float(f), w, h, diag if diag else float(np.hypot(w, h)))
        if bad:
            out = fit(families, image_size=image_size,
                      pp_fixed=pp_fixed or (w / 2, h / 2), free_pp=False, mode=mode,
                      roles=roles)
            if "conditioning" in out:
                out["conditioning"]["pp_refused"] = bad
                out["conditioning"]["pp_assumed"] = "the image centre"
            return out

    AXIS = ("horizontal", "upright", "depth")
    dg = diag if diag else float(np.hypot(w, h))
    per = {}
    allmv = []
    for n, s, a in zip(names, segs, axes):
        v = V[a]
        mv, ang = _move_px(s, v)
        allmv.append(mv)
        per[n] = {"move_px": mv.tolist(), "angle_deg": ang.tolist(),
                  "median_px": float(np.median(mv)), "worst_px": float(mv.max()),
                  "worst_line": int(np.argmax(mv)),
                  "axis": AXIS[a], "axis_index": int(a),
                  "vp": None if _vp_xy(v, dg) is None else _vp_xy(v, dg).tolist(),
                  "direction_deg": (None if _vp_xy(v, dg) is not None else
                                    round(float(np.degrees(np.arctan2(v[1], v[0])) % 180), 2)),
                  "at_infinity": _vp_xy(v, dg) is None}
    allmv = np.concatenate(allmv)
    free = drawn_spread(dict(zip(names, segs)))   # clipped, like the moves
    free_worst = max((d["worst_px"] for d in free.values() if d), default=0.0)

    cond = {"n_lines": int(len(allmv)),
            "median_move_px": float(np.median(allmv)),
            "worst_move_px": float(allmv.max()),
            "cost_rms_px": float(np.sqrt((allmv ** 2).mean())),
            "unconstrained_worst_px": float(free_worst),
            "price_of_consistency_px": float(allmv.max() - free_worst),
            "mark_accuracy_px": round(mark, 2),
            "mark_accuracy_basis": _mark_basis(diag),
            "focal_px": None if f is None else float(f),
            "principal_point": pp.tolist(),
            "mode": mode,
            "perspective": f"{mode}-point",
            "axis_of": {n: AXIS[a] for n, a in zip(names, axes)},
            "off_picture": offpicture,
            "measured_on": "the part of each line inside the picture",
            "per_family": per, "drawn_spread": free,
            "n_families": len(names),
            # Each mark has one angle to give and the camera has this many
            # numbers to fit them with. With none to spare the fit is exact by
            # construction, and 0 px of movement is arithmetic, not evidence.
            "parameters": int(len(sol.x)),
            "redundancy": int(len(allmv)) - int(len(sol.x))}
    if pinned_pp:
        cond["pp_assumed"] = "the image centre"

    # f lives only in the distances from pp to the FINITE vanishing points:
    # -f^2 = (a - pp).(b - pp) needs two of them, and a vanishing point that
    # sits at the principal point calibrates nothing, however many marks feed
    # it. This is how a too-general claim gets caught: fitting a true
    # one-point drawing as two- or three-point reconciles at 0.00 px, because
    # the specific model is nested in the general one, and the solver then
    # returns whatever f it happened to stop at -- 1781 and 3402 px on one
    # measured example whose true camera has no f at all. Geometry says so
    # outright, so say it instead of printing the number.
    usable = [(n, _vp_xy(V[a], dg)) for n, a in zip(names, axes)]
    usable = [(n, float(np.hypot(*(p - pp)))) for n, p in usable if p is not None]
    usable = [(n, d) for n, d in usable if max(mark, 1.0) < d < FAR * dg]
    if f is not None and (len(usable) < 2 or f > FMAX * dg):
        cond["focal_px"] = None
        where = (f"a focal length of {f:.0f} px on a {dg:.0f} px diagonal is "
                 "outside any lens that could have taken this picture"
                 if f > FMAX * dg else
                 f"only {len(usable)} of the {len(names)} families has a "
                 "vanishing point that is both inside the picture plane and "
                 "away from the centre of vision"
                 if usable else
                 "every family either runs parallel on the page or vanishes at "
                 "the centre of vision itself")
        cond["focal_refused"] = (
            f"the focal length cannot be read from this drawing: {where}, and "
            "f only enters the geometry through the distances from the centre "
            "of vision to two different vanishing points. The drawing does "
            f"reconcile as {mode}-point, but it would reconcile just as well "
            "for any lens, so the fitted value is the solver's starting guess "
            "rather than a measurement. A drawing like this is really "
            "one-point; check it as one-point, or mark the diagonals of floor "
            "or ceiling squares, whose vanishing point sits f away from the "
            "centre of vision along the horizon.")
        cond["focal_fitted_but_unmeasured"] = round(float(f), 1)
    elif f is not None:
        cond["focal_from"] = {n: round(d, 1) for n, d in usable}

    if lateral_pin:
        # f follows from where pp was assumed to sit along the horizon, so say
        # how much that assumption is worth by sliding it and watching f.
        u = np.array([np.cos(float(sol.x[-1])), np.sin(float(sol.x[-1]))])
        fin = [p for p in (_vp_xy(V[a], dg) for a in axes) if p is not None]
        fs = []
        if len(fin) >= 2:
            a0, b0 = fin[0], fin[1]
            for t in np.linspace(-0.15 * w, 0.15 * w, 7):
                g = -float((a0 - pp - t * u) @ u) * float((b0 - pp - t * u) @ u)
                if g > 0:
                    fs.append(np.sqrt(g))
        cond["pp_lateral_assumed"] = round(float(pin[0]), 1)
        cond["focal_range_over_pp"] = ([round(min(fs), 1), round(max(fs), 1)]
                                       if len(fs) > 1 else None)
        cond["focal_note"] = (
            "two-point perspective cannot measure where the centre of vision "
            f"sits along the horizon, so it was assumed at x = {pin[0]:.0f} "
            "and only its height was measured. Sliding that assumption across "
            "the middle 30% of the frame moves f over the range given beside "
            "it; the horizon and the reconciling moves do not depend on it.")

    if mode == "one":
        cond["focal_refused"] = (
            "one-point perspective puts the single vanishing point exactly at "
            "the principal point, so the focal length appears in no vanishing "
            "point at all and cannot be recovered here -- not for want of "
            "marks, but because it is absent from the geometry. The vanishing "
            "point itself IS the measurement: it is the viewer's centre of "
            "vision. To get f as well, mark the diagonals of squares on a "
            "floor or a tiled ceiling; their vanishing point sits f away from "
            "the centre of vision along the horizon.")
        cond["centre_of_vision"] = pp.tolist()

    # The camera as a rotation, for a program that has to be handed one: the
    # world's three directions in the camera frame, a row per axis in V's
    # order. Only under a measured lens -- K^-1 applied to a vanishing point
    # means nothing when f was the solver's starting guess.
    if f is not None:
        Kc = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
        A = np.linalg.solve(Kc, V.T).T
        # a roll needs no lens, so `stand` can use these when axes_cam is withheld
        cond["fitted_axes"] = (A / np.linalg.norm(A, axis=1, keepdims=True)).tolist()
        if cond["focal_px"] is not None:
            cond["axes_cam"] = cond["fitted_axes"]

    finite = [p for p in (_vp_xy(V[a], dg) for a in axes) if p is not None]
    # Not when the lateral pin is in force: that case already reported its own
    # range, and it is the honest one -- a level camera's principal point can
    # only slide ALONG the horizon, not over the middle of the frame in two
    # directions, so the 2-D sweep below would overstate it.
    if len(names) == 2 and len(finite) == 2 and not lateral_pin:
        # Two perpendicular directions leave f hostage to the principal point:
        # (va - pp).(vb - pp) + f^2 = 0 is one equation, and pp has two degrees
        # of freedom. Sweep pp over the middle 30% of the frame and print what
        # f does, so nobody quotes the single number as if it were measured.
        A = finite
        fs = [np.sqrt(g) for g in
              (-((A[0][0] - x) * (A[1][0] - x) + (A[0][1] - y) * (A[1][1] - y))
               for x in np.linspace(0.35 * w, 0.65 * w, 7)
               for y in np.linspace(0.35 * h, 0.65 * h, 7)) if g > 0]
        cond["focal_range_over_pp"] = ([round(min(fs), 1), round(max(fs), 1)]
                                       if fs else None)
        if mode == "three":
            cond["note"] = ("two families, so the camera still has a free roll "
                            "and pitch; f follows from where the principal "
                            "point is put and is not measured here")

    # The verdict is about the DRAWING, not the fit. If every line is already
    # within marking accuracy of a consistent perspective, there is nothing to
    # rectify and saying otherwise would send someone to redraw a correct line.
    value = None if f is None else round(float(f), 1)
    if float(allmv.max()) <= mark:
        if cond["redundancy"] <= 0:
            # Every line fits because none has anything to disagree with: the
            # camera has as many numbers as the marks have angles. That is the
            # arithmetic, not a finding about the drawing, so none is passed.
            return {"value": value, "verdict": "undetermined",
                    "why": (f"{cond['n_lines']} lines for the camera's "
                            f"{cond['parameters']} numbers: any such marks fit a "
                            "camera exactly, so this cannot say whether they are "
                            "consistent. Mark a third line in a family to test it"),
                    "conditioning": cond}
        return {"value": value, "verdict": "already consistent",
                "conditioning": cond}
    return {"value": value, "verdict": "rectifiable", "conditioning": cond}


def stand(families, *, image_size, roles=None, pp=None):
    """The camera a picture is stood up on, or a refusal.

    Two or three families are one camera: the flagged vertical and the next
    two in the order given. The centre of vision is taken at the middle of the
    frame unless that costs a line more than a mark, because on rooms with a
    known camera (PLAN.md section 51) the free fit took 2 to 8 s against
    0.03 s, and pinned the better lens. A crop is what breaks the pin, and
    then the free fit is worth its time. The third of a mark sits between the
    worst line on those rooms uncropped, 1.0 px, and cropped, 3.2 to 4.6. A
    picture already stood up says where its centre of vision went, as `pp`.

    Returns {fit, K, axes, up, pinned, measured}: `axes` are the room's
    directions in the camera, rows in the fit's order, and `up` the row that is
    upright: the flagged family's, else the one running nearest the picture's
    vertical. `pinned` says the centre of vision was assumed, and `measured`
    whether the lens was; a roll does not need it.
    """
    roles = roles or {}
    names = sorted(families, key=lambda n: roles.get(n) != "vertical")[:3]
    fam = {n: families[n] for n in names}
    r = fit(fam, image_size=image_size, free_pp=False, pp_fixed=pp)
    c = r.get("conditioning") or {}
    pinned = True
    if len(names) == 3 and c.get("worst_move_px", 0) > c.get("mark_accuracy_px", np.inf) / 3:
        r = fit(fam, image_size=image_size)
        c = r.get("conditioning") or {}
        pinned = "pp_assumed" in c
    if "fitted_axes" not in c:
        return {"fit": r, "K": None}
    f = c["focal_px"] or c["focal_fitted_but_unmeasured"]
    A = np.array(c["fitted_axes"])
    flagged = [n for n in names if roles.get(n) == "vertical"]
    up = c["per_family"][flagged[0]]["axis_index"] if flagged else int(np.argmax(np.abs(A[:, 1])))
    return {"fit": r, "axes": A, "up": up, "pinned": pinned,
            "measured": c["focal_px"] is not None,
            "K": np.array([[f, 0, c["principal_point"][0]],
                           [0, f, c["principal_point"][1]], [0, 0, 1.0]])}


SQUARE_DEG = 6.0      # three vanishing points a lens makes square to within this


def _horizon(a, b, up):
    """The image line through horizontal vanishing points, positive below it.

    Two give it outright; one and the uprights give it through that point,
    square to the uprights there. None when nothing places it.
    """
    a = np.asarray(a, float)
    l = np.cross(a, np.asarray(b, float)) if b is not None and abs(a[2] * b[2]) > 1e-12 else None
    if l is None and up is not None:
        v = a if abs(a[2]) > 1e-12 else b
        if v is None or abs(v[2]) < 1e-12:
            return None
        p, u = v[:2] / v[2], np.asarray(up, float)
        d = u[:2] if abs(u[2]) < 1e-12 else u[:2] / u[2] - p
        l = np.array([d[0], d[1], -(d @ p)])
    if l is None or np.hypot(*l[:2]) < 1e-12:
        return None
    return (l if l[1] > 0 else -l) / np.hypot(*l[:2])


def places(families, horizon):
    """Each family's place by where its line lies against the horizon: `floor`
    with nine parts of ten below it, `ceiling` above, `wall` across it. Only
    walls cross eye level, so a family held entirely on one side is lying on
    that surface and tells nothing of the room."""
    out = {}
    for n, q in families.items():
        q = np.asarray(q, float).reshape(-1, 4)
        ends = np.stack([q[:, :2], q[:, 2:]], 1)
        side = ends @ horizon[:2] + horizon[2]
        L = np.hypot(*(q[:, 2:] - q[:, :2]).T)
        below = L[(side > 0).all(1)].sum() / L.sum()
        above = L[(side < 0).all(1)].sum() / L.sum()
        out[n] = "floor" if below >= 0.9 else "ceiling" if above >= 0.9 else "wall"
    return out


def guess(families, vps, *, image_size, pp=None, along=None):
    """The simplest perspective detected families can be read as.

    Offered, like the detection under it, and used only when taken. Three
    families a lens makes square are read first: that needs no notion of which
    way is up, so a parquet heavier than the uprights cannot be taken for them.
    Of those, the three whose horizon the most line lies on, because a chance
    grouping meets off it alone. Against that horizon each family has a place,
    and the room's own horizontals are the heaviest square pair of walls, or
    else of floors. Then the cheapest type that moves no line more than a
    mark, because a picture is not 3-point for being able to be.

    `families` are lines by name and `vps` their homogeneous vanishing points.
    The centre of vision is taken at the middle of the frame, or at `pp`.
    `along` names a family known to be level and to run down the view, the axis
    of a vault: its ribs say so, and that outweighs however much else there is.
    """
    names = [n for n in families if len(families[n])]
    mark, diag = _mark_px(image_size)
    if len(names) < 2:
        return {"value": None, "verdict": "undetermined",
                "why": f"{len(names)} line families; two are needed to say anything",
                "conditioning": {"families": len(names)}}
    centre = np.array(pp if pp else (image_size[0] / 2, image_size[1] / 2), float)
    span = {n: np.asarray(families[n], float).reshape(-1, 4) for n in names}
    span = {n: q[:, 2:] - q[:, :2] for n, q in span.items()}
    mass = {n: float(np.hypot(*d.T).sum()) for n, d in span.items()}

    # the height of the centre of vision is left free: a level camera's sits
    # on the horizon, which a low one puts near the foot of the frame
    lens = np.geomspace(0.3, 4.0, 64)[:, None] * diag
    ys = np.array([centre[1]]) if pp else np.linspace(-0.25, 1.25, 31) * image_size[1]
    def square(group):
        """(degrees the worst pair misses square, the one running most upright
        in the picture, and the lens and height that came closest)."""
        v = np.array([vps[n] for n in group], float)
        d = np.empty((len(ys), lens.size, len(v), 3))
        d[..., 0] = v[:, 0] - centre[0] * v[:, 2]
        d[..., 1] = v[:, 1] - ys[:, None, None] * v[:, 2]
        d[..., 2] = lens * v[:, 2]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        c = (np.abs(d @ d.swapaxes(-1, -2)) - np.eye(len(v))).max((-1, -2))
        i, j = np.unravel_index(np.argmin(c), c.shape)
        return (float(np.degrees(np.arcsin(np.clip(c[i, j], 0, 1)))),
                d[i, j][int(np.argmax(np.abs(d[i, j][:, 1])))], (float(lens[j, 0]), float(ys[i])))

    # the real horizon passes through every horizontal the room has, and a
    # chance grouping of cornices with floor lines meets off it alone
    def held(group):
        """The families on the horizon `group` implies, and which of `group` is up."""
        up, at = square(group)[1:]
        v = np.array([vps[n] for n in names], float)
        d = np.c_[v[:, 0] - centre[0] * v[:, 2], v[:, 1] - at[1] * v[:, 2], at[0] * v[:, 2]]
        off = dict(zip(names, np.abs(d @ up) / np.linalg.norm(d, axis=1)))
        return [n for n in names if off[n] <= np.sin(np.radians(3.0))], max(group, key=off.get)

    triples = [(sum(mass[n] for n in held(g)[0]), sum(mass[n] for n in g), g)
               for g in combinations(names, 3) if square(g)[0] <= SQUARE_DEG]
    on = names
    if triples and along is None:
        group = max(triples)[2]
        on, up = held(group)
        picked = [n for n in group if n != up]
    else:
        steep = [n for n, d in span.items() if
                 np.median(np.degrees(np.arctan2(*np.abs(d).T))) < 30]
        up = max((n for n in steep if n != along), key=mass.get, default=None)
        rest = sorted((n for n in names if n != up), key=lambda n: (n != along, -mass[n]))
        picked = rest[:1 if up is not None else 2]

    other = vps[picked[1]] if len(picked) > 1 else None
    line = _horizon(vps[picked[0]], other, None if up is None else vps[up])
    place = places({n: families[n] for n in names}, line) if line is not None else {}
    place.update({n: "off" for n in place if n not in on})
    if up is not None:
        place[up] = "upright"
    # a pattern lies at any angle to the walls, so a wall pairs with a wall and
    # a floor with a floor: one of each proves nothing, and 45 deg apart they
    # are square under a lens two and a half times too long
    heavy = sorted(names, key=mass.get, reverse=True)
    walls = [n for n in heavy if place.get(n) in ("wall", "ceiling")]
    for kind in (walls, [n for n in heavy if place.get(n) == "floor"]):
        pairs = [(mass[a] + mass[b], a, b) for a, b in combinations(kind, 2)
                 if up is not None and along in (None, a, b) and square((up, a, b))[0] <= SQUARE_DEG]
        if pairs:
            picked = list(max(pairs)[1:])
            break
    fam = {n: families[n] for n in ([up] if up is not None else []) + picked}
    roles = {up: "vertical"} if up is not None else None

    # the type is the room's: a parquet is square under any turn of the camera
    # to the walls, so it can say what the camera is but not how the room lies
    room = picked if set(picked) <= set(walls) else walls[:1] or picked
    room = {n: families[n] for n in ([up] if up is not None else []) + room}
    costs, fits = {}, {}
    for mode in MODES:
        fits[mode] = fit(room, image_size=image_size, mode=mode, roles=roles,
                         free_pp=mode != "three", pp_fixed=pp)
        costs[mode] = (fits[mode].get("conditioning") or {}).get("worst_move_px")
    if not any(v is not None for v in costs.values()):
        return {"value": None, "verdict": "undetermined",
                "why": fits["three"].get("why") or "the families give no camera",
                "conditioning": {"families": len(names)}}
    near = [m for m in MODES if costs[m] is not None and costs[m] <= mark]
    # uprights and one horizontal fit 1-point for any vanishing point at all,
    # so with no other family to say so, it has to be in the frame, or the
    # family has to run parallel on the page
    if len(room) == 2:
        n = next(iter(set(room) - {up}))
        v = np.asarray(vps[n], float)
        d = v[:2] if abs(v[2]) < 1e-12 else v[:2] / v[2] - centre
        flat = _move_px(np.asarray(room[n], float).reshape(-1, 4), np.array([d[0], d[1], 0.0]))[0]
        inside = abs(v[2]) > 1e-12 and (0 <= v[:2] / v[2]).all() and (v[:2] / v[2] <= image_size).all()
        if not inside and flat.max() > mark and n != along:
            near = [m for m in near if m != "one"]
    mode = near[0] if near else min((m for m in MODES if costs[m] is not None), key=costs.get)
    c = (fit(fam, image_size=image_size, roles=roles, free_pp=False, pp_fixed=pp)
         if len(fam) != len(room) or set(fam) != set(room) else fits["three"]).get("conditioning") or {}
    cond = {"vertical": up, "horizontals": picked, "lines": {n: len(span[n]) for n in fam},
            # what to adopt: a floor's diagonals cannot be drawn in 1-point
            "take": list(room if mode == "one" else fam),
            "costs": {m: None if v is None else round(v, 2) for m, v in costs.items()},
            "mark_accuracy_px": round(mark, 2), "within_a_mark": bool(near),
            "square_deg": round(square(list(fam))[0], 2) if len(fam) == 3 else None,
            "pp_assumed": "at the middle of the frame" if pp is None else "where the turn left it",
            # uprights parallel: the lens and the tip are the frame's middle, assumed
            "focal_px": None if mode == "one" else c.get("focal_px"), "focal_refused": c.get("focal_refused"),
            "place": place, "horizon": None if line is None else line.tolist()}
    if "fitted_axes" in c:
        A = np.array(c["fitted_axes"])
        i = c["per_family"][up]["axis_index"] if up is not None else int(np.argmax(np.abs(A[:, 1])))
        u = -A[i] if A[i][1] > 0 else A[i]
        cond["roll_deg"] = round(float(np.degrees(np.arctan2(u[0], -u[1]))), 2)
        if mode != "one":
            cond["tilt_deg"] = round(float(np.degrees(np.arcsin(np.clip(u[2], -1, 1)))), 2)
    return {"value": mode, "verdict": f"{MODES.index(mode) + 1}-point", "conditioning": cond}


def against_vps(families, vps, *, image_size=None, focal_px=None,
                principal_point=None):
    """The drawing priced against a camera somebody else already solved.

    `vps` maps an axis label to that axis' vanishing point, homogeneous and in
    this picture's pixels -- camera_io.vps_h of a camera read from a Blender,
    fSpy or JSON file. Nothing is fitted: the camera is the claim, and the
    readout is what each line would have to do to agree with it. One family is
    enough here, which `fit` cannot say: the reference supplies the rest.

    The one thing decided is which axis each family belongs to, and the marks
    decide it -- the one-family-one-axis assignment with the least total
    movement. The runner-up is priced beside it, so a family that fits two
    axes about equally well says so instead of being quietly put on one.

    Returns the same {value, verdict, conditioning} as `fit`, per_family
    included, so a page that draws one draws the other.
    """
    labels = list(vps)
    V = [np.asarray(vps[a], float) for a in labels]
    names = [n for n in families if len(np.asarray(families[n], float).reshape(-1, 4))]
    if not names or len(names) > len(labels):
        return {"verdict": "undetermined",
                "why": (f"{len(names)} line families against a camera's "
                        f"{len(labels)} axes; between one and {len(labels)} "
                        "are needed, each along its own direction"),
                "conditioning": {"families": len(names), "mode": "reference"}}
    w, h = image_size or (1000.0, 1000.0)
    mark, diag = _mark_px(image_size)
    dg = diag if diag else float(np.hypot(w, h))
    segs, offpicture = [], {}
    for n in list(names):
        c, miss = _clip(families[n], w, h)
        if miss:
            offpicture[n] = miss
        segs.append(c)
    names = [n for n, c in zip(names, segs) if len(c)]
    segs = [c for c in segs if len(c)]
    if not names:
        return {"verdict": "undetermined",
                "why": "every marked line falls outside the picture",
                "conditioning": {"off_picture": offpicture, "mode": "reference"}}

    mv = [[_move_px(s, v) for v in V] for s in segs]
    sq = np.array([[float((m[0] ** 2).sum()) for m in row] for row in mv])
    n_lines = sum(len(s) for s in segs)

    def total(p):
        return float(sum(sq[i, a] for i, a in enumerate(p)))

    ranked = sorted(permutations(range(len(V)), len(names)), key=total)
    best, runner = ranked[0], ranked[1]
    margin = float(np.sqrt(total(runner) / n_lines) - np.sqrt(total(best) / n_lines))

    AX = labels
    per, allmv = {}, []
    for i, (n, a) in enumerate(zip(names, best)):
        m, ang = mv[i][a]
        allmv.append(m)
        v, p = V[a], _vp_xy(V[a], dg)
        per[n] = {"move_px": m.tolist(), "angle_deg": ang.tolist(),
                  "median_px": float(np.median(m)), "worst_px": float(m.max()),
                  "worst_line": int(np.argmax(m)),
                  "axis": AX[a], "axis_index": int(a),
                  "vp": None if p is None else p.tolist(),
                  "direction_deg": (None if p is not None else
                                    round(float(np.degrees(np.arctan2(v[1], v[0])) % 180), 2)),
                  "at_infinity": p is None,
                  # what this family would cost on each axis, so "which one"
                  # is a number the reader can see and not a label
                  "rms_on_axis_px": {AX[k]: round(float(np.sqrt((mv[i][k][0] ** 2).mean())), 2)
                                     for k in range(len(V))}}
    allmv = np.concatenate(allmv)
    free = drawn_spread(dict(zip(names, segs)))
    free_worst = max((d["worst_px"] for d in free.values() if d), default=0.0)
    cond = {"n_lines": int(len(allmv)),
            "median_move_px": float(np.median(allmv)),
            "worst_move_px": float(allmv.max()),
            "cost_rms_px": float(np.sqrt((allmv ** 2).mean())),
            "unconstrained_worst_px": float(free_worst),
            "price_of_consistency_px": float(allmv.max() - free_worst),
            "mark_accuracy_px": round(mark, 2),
            "mark_accuracy_basis": _mark_basis(diag),
            "focal_px": None if focal_px is None else float(focal_px),
            "principal_point": (None if principal_point is None
                                else [float(principal_point[0]), float(principal_point[1])]),
            "mode": "reference", "perspective": "the reference camera",
            "axis_of": {n: AX[a] for n, a in zip(names, best)},
            "off_picture": offpicture,
            "measured_on": "the part of each line inside the picture",
            "per_family": per, "drawn_spread": free, "n_families": len(names),
            "assignment": {"margin_px": round(margin, 2),
                           "ambiguous": bool(margin < mark),
                           "runner_up": {n: AX[a] for n, a in zip(names, runner)},
                           "basis": ("rms movement of the next-best way of putting "
                                     "the families on the axes, less this one's")}}
    value = None if focal_px is None else round(float(focal_px), 1)
    return {"value": value,
            "verdict": ("already consistent" if float(allmv.max()) <= mark
                        else "rectifiable"),
            "conditioning": cond}


def rectify(seg, v):
    """The one line, rotated about its own midpoint to meet `v`.

    The least change that fixes it: the midpoint does not move, so the line
    stays where it was placed in the composition and only its direction is
    corrected. Returns the corrected segment and how far each end travelled.
    """
    seg = np.asarray(seg, float).ravel()
    m = np.array([(seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2])
    half = float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]) / 2)
    if abs(v[2]) < 1e-12:
        d = np.asarray(v[:2], float)
    else:
        d = np.asarray(v[:2], float) / v[2] - m
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None, float("nan")
    d = d / n
    # keep the original end order, so the correction reads as a nudge not a flip
    if (seg[2] - seg[0]) * d[0] + (seg[3] - seg[1]) * d[1] < 0:
        d = -d
    a, b = m - half * d, m + half * d
    moved = float(max(np.linalg.norm(a - seg[:2]), np.linalg.norm(b - seg[2:])))
    return np.concatenate([a, b]), moved


def _demo():
    """A synthetic room, drawn by a shaky hand, reconciled."""
    rng = np.random.default_rng(0)
    f, pp = 1100.0, np.array([442.0, 640.0])
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    Rot = _rodrigues(np.array([0.05, 0.35, 0.01]))
    V = (K @ Rot).T
    fams, truth = {}, {}
    for i, name in enumerate(("depth", "frontal", "vertical")):
        v = V[i]
        segs = []
        for _ in range(7):
            m = rng.uniform([100, 100], [780, 1100])
            d = (v[:2] / v[2] - m) if abs(v[2]) > 1e-12 else v[:2]
            d = d / np.linalg.norm(d)
            L = rng.uniform(120, 700)
            segs.append([*(m - d * L / 2), *(m + d * L / 2)])
        truth[name] = np.array(segs)
        # a shaky hand: rotate each line a little about its midpoint
        shaky = []
        for s in segs:
            m = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
            half = np.hypot(s[2] - s[0], s[3] - s[1]) / 2
            d = np.array([s[2] - s[0], s[3] - s[1]]) / (2 * half)
            t = np.radians(rng.normal(0, 0.8))
            Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            d = Rm @ d
            shaky.append([*(m - d * half), *(m + d * half)])
        fams[name] = np.array(shaky)

    got = fit(fams, image_size=(884, 1280))
    assert got["verdict"] in ("rectifiable", "already consistent"), got
    c = got["conditioning"]
    assert abs(c["focal_px"] - f) / f < 0.25, f"f {c['focal_px']} vs {f}"
    # the reconciliation must not cost more than the shaky hand already did
    assert c["price_of_consistency_px"] < 8.0, c["price_of_consistency_px"]
    # a clean drawing must come back already consistent
    clean = fit(truth, image_size=(884, 1280))
    assert clean["verdict"] == "already consistent", clean["conditioning"]["worst_move_px"]
    assert clean["conditioning"]["worst_move_px"] < 1.0
    # ...but 2 lines a family is exactly what a camera's 6 numbers can take, so
    # it fits however the lines were drawn and must not be called consistent;
    # one more line in one family is the first that can disagree
    bare = fit({k: v[:2] for k, v in truth.items()}, image_size=(884, 1280))
    assert bare["verdict"] == "undetermined", bare["verdict"]
    assert bare["conditioning"]["redundancy"] == 0, bare["conditioning"]
    more = fit({k: v[:3 if k == "depth" else 2] for k, v in truth.items()},
               image_size=(884, 1280))
    assert more["verdict"] == "already consistent", more["conditioning"]
    assert more["conditioning"]["redundancy"] == 1
    # rectify puts a line exactly on its vanishing point and keeps the midpoint
    v = np.array([*c["per_family"]["depth"]["vp"], 1.0])
    s0 = fams["depth"][0]
    fixed, moved = rectify(s0, v)
    mv_before, _ = _move_px(s0, v)
    mv_after, _ = _move_px(fixed, v)
    assert mv_after[0] < 1e-6, mv_after
    assert abs(moved - mv_before[0]) < 1e-6, (moved, mv_before[0])
    m0 = np.array([(s0[0] + s0[2]) / 2, (s0[1] + s0[3]) / 2])
    m1 = np.array([(fixed[0] + fixed[2]) / 2, (fixed[1] + fixed[3]) / 2])
    assert np.linalg.norm(m0 - m1) < 1e-9, "the midpoint must not move"
    # Forcing a level camera on a scene that was not shot level costs real
    # pixels, and saying so is the point: this synthetic has 0.05 rad of pitch.
    tilted = fit_level({k: v for k, v in truth.items() if k != "vertical"},
                       image_size=(884, 1280))
    assert tilted["verdict"] == "rectifiable", tilted
    assert tilted["conditioning"]["worst_move_px"] > 20, tilted["conditioning"]
    # ...and here it costs so much that no level camera exists at all: both
    # vanishing points end up the same side of the centre, so f^2 < 0. The
    # reconciliation is still reported -- only f is refused.
    assert tilted["conditioning"]["focal_px"] is None, tilted["conditioning"]
    assert "same side" in tilted["conditioning"]["focal_refused"]

    # The configuration measured on testdata/4.jpg: two wall families plus a
    # near-vertical one whose vanishing point is 60 000 px above the picture.
    # That triangle is a sliver, and the solver buys a lower residual by
    # sliding pp out of the frame until a family's constraint goes slack and f
    # collapses with it. The free principal point has to be refused, not
    # reported -- the overfitted answer looks BETTER by every residual.
    def aim(vp, n, rng3):
        out = []
        for _ in range(n):
            m = rng3.uniform([100, 100], [780, 1100])
            d = np.array(vp, float) - m
            d = d / np.linalg.norm(d)
            t = np.radians(rng3.normal(0, 0.8))
            Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            d, half = Rm @ d, rng3.uniform(60, 220)
            out.append([*(m - d * half), *(m + d * half)])
        return np.array(out)

    rng3 = np.random.default_rng(7)
    sliver = {"right wall": aim((3364, 839), 12, rng3),
              "left wall": aim((-10700, 2330), 12, rng3),
              "vertical": aim((300, -60000), 24, rng3)}
    deg = fit(sliver, image_size=(884, 1280))
    dc = deg["conditioning"]
    assert dc.get("pp_refused"), ("a sliver triangle must not report a free pp",
                                  dc["principal_point"], dc["focal_px"])
    assert dc["principal_point"] == [442.0, 640.0], dc["principal_point"]
    assert 311 < dc["focal_px"] < 6000, dc["focal_px"]     # 0.2 * the diagonal

    # Scanning the same drawing larger must not change its verdict. Scale the
    # picture and every mark by k and every move scales by exactly k, so the
    # answer is only invariant if what a hand mark is worth scales too. This
    # fixture sits in the window where it did not: 5.9 px at a 1556 px
    # diagonal, 11.8 px at 3112 -- which a flat 9.0 px called "rectifiable",
    # condemning a drawing for nothing but being scanned at higher dpi.
    base = fit(fams, image_size=(884, 1280))
    assert base["verdict"] == "already consistent", base["conditioning"]
    for k in (2.0, 4.0, 0.5):
        big = fit({n: [[c * k for c in s] for s in segs] for n, segs in fams.items()},
                  image_size=(884 * k, 1280 * k))
        bc, bk = base["conditioning"], big["conditioning"]
        assert big["verdict"] == base["verdict"], (k, big["verdict"], bk)
        assert abs(bk["worst_move_px"] - k * bc["worst_move_px"]) < 0.05 * k, (k, bk)
        assert abs(bk["mark_accuracy_px"] - k * bc["mark_accuracy_px"]) < 1e-6, (k, bk)
    # and the figure says what it rests on, so nobody reads 36 px as measured
    assert "scaled to this picture" in big["conditioning"]["mark_accuracy_basis"]
    assert "a scan this size" in base["conditioning"]["mark_accuracy_basis"]
    assert "unscaled" in fit(fams)["conditioning"]["mark_accuracy_basis"]

    # A construction line dragged far past the canvas must read the same as the
    # same line stopped at the edge. Before clipping it read 7x larger, purely
    # because its ends were further out.
    stretched = {}
    for k, v in truth.items():
        if k == "vertical":
            continue
        q = np.asarray(v, float).copy()
        m = np.c_[(q[:, 0] + q[:, 2]) / 2, (q[:, 1] + q[:, 3]) / 2]
        stretched[k] = np.c_[m + (q[:, :2] - m) * 5.0, m + (q[:, 2:] - m) * 5.0]
    a = fit({k: v for k, v in truth.items() if k != "vertical"},
            image_size=(884, 1280))["conditioning"]
    b = fit(stretched, image_size=(884, 1280))["conditioning"]
    assert abs(a["worst_move_px"] - b["worst_move_px"]) < 0.5, \
        (a["worst_move_px"], b["worst_move_px"])

    # Two families, camera free: the drawn vanishing points can be made exactly
    # perpendicular at no cost, and the fit must FIND that. Seeded randomly it
    # used to slide to f = 1e94 and 164 px of movement instead.
    pair = fit({k: v for k, v in truth.items() if k != "vertical"},
               image_size=(884, 1280))
    assert pair["verdict"] == "already consistent", pair["conditioning"]
    assert pair["conditioning"]["worst_move_px"] < 1.0, pair["conditioning"]
    assert pair["conditioning"]["focal_range_over_pp"][0] < pair["value"] \
        < pair["conditioning"]["focal_range_over_pp"][1], pair["conditioning"]

    # ...and on a genuinely level camera it recovers f from two families alone.
    Rl = _rodrigues(np.array([0.0, 0.35, 0.0]))
    Vl = (K @ Rl).T
    lf = {}
    # world Y is the upright one and sits at infinity under a level camera, so
    # the two HORIZONTAL families are world X and world Z
    for i, name in zip((0, 2), ("depth", "frontal")):
        v = Vl[i]
        segs = []
        for _ in range(7):
            m = rng.uniform([100, 100], [780, 1100])
            dd = (v[:2] / v[2] - m) if abs(v[2]) > 1e-12 else v[:2]
            dd = dd / np.linalg.norm(dd)
            L = rng.uniform(120, 700)
            segs.append([*(m - dd * L / 2), *(m + dd * L / 2)])
        lf[name] = np.array(segs)
    lev = fit_level(lf, image_size=(884, 1280))
    assert lev["conditioning"]["focal_px"] is not None, lev["conditioning"]
    assert lev["verdict"] == "already consistent", lev["conditioning"]["worst_move_px"]
    assert lev["conditioning"]["worst_move_px"] < 2.0, lev["conditioning"]
    assert abs(lev["conditioning"]["focal_px"] - f) / f < 0.05, lev["conditioning"]["focal_px"]
    print("consensus.py self-check passed  "
          f"(f {c['focal_px']:.0f} vs {f:.0f} true, "
          f"worst move {c['worst_move_px']:.1f} px, "
          f"price of consistency {c['price_of_consistency_px']:.1f} px)")


if __name__ == "__main__":
    _demo()
