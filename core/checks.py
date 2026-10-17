"""Every readout, wrapped in the conditioning that says whether to believe it.

The rule the whole app is built on: a check returns `(value, conditioning,
verdict)` and the verdict may be `"undetermined"`. Nothing here returns a bare
number, because a bare number gets believed.

Two conditioning numbers do most of the work, and they catch different things:

  turning angle   how much of the curve was actually marked. Below ~120 deg the
                  conic fit is not merely noisy, it is often not an ellipse at
                  all (measured: 9 % non-ellipse at 90 deg, 94 % at 50 deg).
  trim spread     refit on the middle 80 % and 60 % of the marks and take the
                  range. A bootstrap on a dense smooth ridge reports +-0.02 on
                  ratios that trimming moves by 1.2 -- it sees mark noise and is
                  blind to the model being underdetermined (bug 5).

A third, `f_sensitivity`, re-runs anything needing the camera across the range
of focal lengths the vanishing points still allow. Which of the three dominates
is not fixed: on the 4.jpg window arcs f spans 45 % and moves the axis ratio by
0.010 while trimming moves it by 0.073, and on a depth readout it is the other
way round. So all three are computed and reported, never assumed.
"""
from __future__ import annotations

import numpy as np

from . import rhythm as R
from . import vp as V

# Measured on 200 trials per sweep, 60 marks, +-2 px, true axis ratio 1.25:
# turning 24-65 deg -> median ratio 9.8-34.7 (truth 1.25) and 67-94 % of fits
# are not even ellipses; 90 deg -> 1.473, still 18 % high; 113 deg -> 1.262,
# within 1 %. The plan guessed 60-90 deg; the measurement says 120.
TURN_MIN_DEG = 120.0


def readout(value, verdict, **conditioning):
    """One answer, its verdict and the numbers that justify the verdict."""
    return {"value": value, "verdict": verdict, "conditioning": conditioning}


def undetermined(why, **conditioning):
    return {"value": None, "verdict": "undetermined", "why": why,
            "conditioning": conditioning}


# --------------------------------------------------------------- arc geometry

def turning_deg(pts, k=None):
    """Tangent-direction change from one end of the marked arc to the other.

    Not the angular spread about the centroid, which reads 355-358 deg for arcs
    that turn 35-55 deg (bug 13). Not the conic's analytic tangents either:
    those are exact on clean marks (20.0, 45.0, 180.0 deg recovered to 0.1) and
    useless on real ones, because a shallow noisy arc fits a wildly wrong conic
    and its tangents come from that -- a true 45 deg arc read 167.9 deg.

    So: block chords. The marks are cut into B blocks, each block's chord gives
    the tangent at its own midpoint, and those directions are unwrapped before
    being differenced. A block chord is noise-averaged, and unwrapping is what
    lets the answer pass 180 deg -- taking the end-to-end angle directly folds
    a fully marked closed ellipse, the best input there is, to nearly 0 deg and
    refuses it.

    A block chord has the tangent of its midpoint, so B blocks span (B-1)/B of
    the arc. That bias is exact and is divided out.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    n = len(p)
    if n < 5:
        return float("nan")
    B = int(np.clip(n // 10, 3, 8)) if k is None else int(k)
    edge = np.linspace(0, n, B + 1).astype(int)
    seg = [(s, max(e - 1, s)) for s, e in zip(edge[:-1], edge[1:])]
    d = np.array([p[b] - p[a] for a, b in seg])
    keep = np.hypot(d[:, 0], d[:, 1]) > 1e-9
    d, seg = d[keep], [sg for sg, k_ in zip(seg, keep) if k_]
    if len(d) < 2:
        return float("nan")
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    # Exact debias: each chord carries the tangent of its own midpoint, so the
    # angle difference spans mid(last) - mid(first) of the n-1 intervals, not
    # all of them.
    mid0 = (seg[0][0] + seg[0][1]) / 2
    mid1 = (seg[-1][0] + seg[-1][1]) / 2
    if mid1 - mid0 < 1e-9:
        return float("nan")
    return float(abs(np.degrees(ang[-1] - ang[0])) * (n - 1) / (mid1 - mid0))


def trim_spread(fn, pts, keeps=(1.0, 0.8, 0.6)):
    """Re-run `fn` on the full marks, the middle 80 % and the middle 60 %.

    Returns (values, spread). This is the test that catches an underdetermined
    shape: trimming 20 % off a 27-55 deg arc moved an axis ratio from 1.076 to
    2.279, while the bootstrap on the same points said +-0.02.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    vals = []
    for keep in keeps:
        c = int(len(p) * (1 - keep) / 2)
        v = fn(p[c:len(p) - c] if c else p)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if len(vals) < 2:
        return vals, float("nan")
    return vals, float(max(vals) - min(vals))


def gaps(pts, factor=4.0):
    """Jumps in the marked sequence, in px. 311 of 1020 extracted 'arc' points
    were chandelier and cornice, and a 75 px gap plus a 282-point off-curve blob
    were both invisible in the residual (bug 12). Look before fitting."""
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 3:
        return []
    step = np.hypot(*np.diff(p, axis=0).T)
    med = float(np.median(step))
    return [{"after": int(i), "px": float(step[i])}
            for i in np.flatnonzero(step > max(factor * med, 10.0))]


def conic_beats_line(pts):
    """A conic must fit at least as well as a straight line: a line pair IS a
    conic, so losing to one is impossible and means the fit is broken. On raw
    pixels it happened (15.35 px vs 7.96 px) and that is what forced Hartley
    normalization. Kept as a permanent assertion (bug 1)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    c = p.mean(0)
    line_rms = float(np.sqrt((((p - c) @ np.linalg.svd(p - c)[2][1]) ** 2).mean()))
    return R.conic_resid(p), line_rms


def _conic_axes(pts):
    """The fitted conic's two semi-axes, in page pixels, shortest first.
    None when the fit is not an ellipse."""
    C = R.conic(pts)
    try:
        cen = np.linalg.solve(C[:2, :2], -C[:2, 2])
    except np.linalg.LinAlgError:
        return None
    k = float(cen @ C[:2, :2] @ cen - C[2, 2])
    if abs(k) < 1e-18:
        return None
    ev = np.linalg.eigvalsh(C[:2, :2] / k)
    if np.any(ev <= 0):
        return None
    return np.sort(1.0 / np.sqrt(ev))


def _axis_ratio(pts):
    """Axis ratio of the fitted conic, with no axis assignment -- `arch_shape`
    flips its assignment on near-circular conics, and a bare ratio has no axis
    to flip. Returns None when the fit is not an ellipse."""
    ax = _conic_axes(pts)
    return None if ax is None else float(ax[1] / ax[0])


def arc_quality(pts):
    """Everything that decides whether this arc can carry a shape readout."""
    p = np.asarray(pts, float).reshape(-1, 2)
    resid, line_rms = conic_beats_line(p)
    _, spread = trim_spread(_axis_ratio, p)
    return {"n": int(len(p)), "turning_deg": turning_deg(p), "conic_resid_px": resid,
            "line_resid_px": line_rms, "degenerate": bool(resid > line_rms),
            "trim_spread": spread, "gaps": gaps(p)}


# ------------------------------------------------------------- circle vs oval

# How far marking noise alone lifts the axis ratio of a true circle, in units
# of (residual px) / (semi-minor px * sqrt n). Measured over rings from 16 to
# 70 px across, 13 to 60 marks and 1 to 4 px of hand: the median lift is 2.6
# of these and the ninetieth percentile 5.
MARK_BIAS = 5.0

def circle(pts, K, vanish_line, *, K_alt=(), resolve=0.05):
    """Is the marked arc a circle on its own plane, or an oval?

    `vanish_line` must be the vanishing line of the plane the arc LIES IN. Arcs
    in a vertical wall measured against the horizon -- the vanishing line of
    HORIZONTAL planes -- produced plausible wrong numbers for a whole session
    (bug 3), so this is a required argument with no default and the caller has
    to have said which surface the feature is on.

    `K_alt` are other cameras the vanishing points still allow; the spread of
    the answer across them is the f-sensitivity. `resolve` is the departure
    from round the caller wants to be able to claim.
    """
    q = arc_quality(pts)
    if q["turning_deg"] < TURN_MIN_DEG:
        return undetermined(
            f"the arc turns {q['turning_deg']:.0f} deg; below {TURN_MIN_DEG:.0f} deg "
            "the fitted conic is often not an ellipse at all", **q)
    if q["degenerate"]:
        return undetermined("a straight line fits these marks better than a conic", **q)
    r = R.circle_check(pts, K, vanish_line)
    if r is None:
        return undetermined("the conic is not an ellipse on that plane", **q)
    ratio = float(r["ratio"])

    alts = [R.circle_check(pts, Ka, vanish_line) for Ka in K_alt]
    alts = [float(a["ratio"]) for a in alts if a is not None]
    f_sens = float(max(alts) - min(alts)) if len(alts) > 1 else 0.0

    _, trim = trim_spread(
        lambda s: (lambda x: x["ratio"] if x else None)(R.circle_check(s, K, vanish_line)),
        pts)
    unc = float(np.nanmax([trim, f_sens, 0.0]) / 2)
    q |= {"f_sensitivity": f_sens, "uncertainty": unc, "resolve": resolve}

    # Marking noise does not scatter the axis ratio, it BIASES it: every wobble
    # makes the fitted conic less round, never rounder, so a true circle marked
    # by hand reads above 1 and no tolerance centred on the reading can undo
    # that. How far above is measurable, and comes to about
    #     5 * (fit residual px) / (semi-minor px * sqrt n)
    # at the ninetieth percentile. When that is bigger than the departure the
    # caller asked to resolve, nothing here is an answer: a true circle and a
    # real oval read the same.
    n = int(q["n"])
    # The ring's narrow half-width on the page, taken from the marks and not
    # from the fitted conic: a big ellipse on a steeply slanted plane projects
    # to a conic that is no ellipse on the page at all, and has no semi-minor
    # there, while the marks always have a spread. For a ring sampled evenly,
    # that spread across the narrow way is b / sqrt 2.
    d = np.asarray(pts, float).reshape(-1, 2)
    narrow = float(np.linalg.svd(d - d.mean(0), compute_uv=False)[1]
                   / np.sqrt(n) * np.sqrt(2.0))
    # a conic has five degrees of freedom, so five marks fit exactly and say
    # nothing about the hand; 0.3 px keeps a traced curve from claiming none
    resid = float(q["conic_resid_px"]) * np.sqrt(n / max(n - 5, 1))
    floor = (MARK_BIAS * max(resid, 0.3) / (narrow * np.sqrt(n))
             if narrow > 1e-9 else np.inf)
    q |= {"mark_noise_floor": floor, "narrow_half_width_px": narrow}
    if floor > resolve:
        return undetermined(
            f"marks this shaky on a ring this size move the axis ratio by about "
            f"{floor:.2f} on their own, more than the {resolve:.2f} asked for: "
            f"mark more points round it, mark a bigger one, or ask for {floor:.2f}",
            ratio=ratio, **q)

    depart = abs(ratio - 1.0)
    tol = max(float(resolve), unc)
    q |= {"tolerance": tol}
    verdict = "circle" if depart <= tol else "oval"
    return readout(round(ratio, 4), verdict, long_axis=r["long_on_plane"].tolist(), **q)


# ------------------------------------------------------------------ the camera

def camera(vert_vp, vp_a, vp_b):
    """Focal length and principal point from three mutually orthogonal VPs.

    Reported with how hard each vanishing point pushes on f, because they do not
    push equally: on 4.jpg the depth VP across its whole plausible range moves f
    by 6 % and the frontal-wall VP moves it by 45 %. "f is not settled" is a
    statement about one vanishing point, not about all of them.
    """
    K = R.calibrate(vert_vp, vp_a, vp_b)
    if K is None:
        return undetermined("those three vanishing points are not a right-angled triple")
    return readout({"f": float(K[0, 0]), "pp": [float(K[0, 2]), float(K[1, 2])]},
                   "ok", K=K.tolist())


def f_range(vert_vp, vp_a, vp_b, *, a_range=None, b_range=None, n=5):
    """The cameras the vanishing points still allow, for an f-sensitivity run.

    Pass the interval each horizontal VP is known to within; returns the list of
    K to hand to `circle` as `K_alt`, and the span of f over them.
    """
    out = []
    cand_a = ([np.array([x, vp_a[1]]) for x in np.linspace(*a_range, n)]
              if a_range else [np.asarray(vp_a, float)])
    cand_b = ([np.array([x, vp_b[1]]) for x in np.linspace(*b_range, n)]
              if b_range else [np.asarray(vp_b, float)])
    for a in cand_a:
        for b in cand_b:
            K = R.calibrate(vert_vp, a, b)
            if K is not None:
                out.append(K)
    fs = [float(K[0, 0]) for K in out]
    return out, ((min(fs), max(fs)) if fs else None)


# ---------------------------------------------------------------- the horizon

def figures(pairs, a, b, *, resolve=0.03):
    """Where the horizon cuts each standing figure, as a fraction of its height.

    Figures of equal height on one ground plane all give the same fraction, and
    that is the entire check: no scale, no depth, no camera. The spread across
    the figures is both the result and its own conditioning.
    """
    fr = [R.figure_ratio(f, h, a, b) for f, h in pairs]
    fr = [float(x) for x in fr if np.isfinite(x)]
    if len(fr) < 2:
        return undetermined("two figures at least are needed to compare", n=len(fr))
    spread = float(max(fr) - min(fr))
    heights = [float(abs(np.asarray(h, float)[1] - np.asarray(f, float)[1]))
               for f, h in pairs]
    # A 9 px mark on the shortest figure is already this much of its height.
    floor = float(9.0 / min(heights)) if min(heights) > 0 else float("inf")
    cond = {"fractions": [round(x, 4) for x in fr], "spread": spread,
            "mark_noise_floor": floor, "heights_px": heights, "resolve": resolve}
    if spread <= max(floor, resolve):
        return readout(round(float(np.mean(fr)), 4), "consistent", **cond)
    return readout(round(float(np.mean(fr)), 4), "inconsistent",
                   worst=int(np.argmax(np.abs(np.array(fr) - np.mean(fr)))), **cond)


def shadow(triples, vert, a, b, *, mark_px=9.0):
    """Do these cast shadows come from one light?

    Three marks an object: its base where it meets the ground, its top, and
    the tip of its cast shadow. Nothing else is needed -- the light rays, the
    shadow lines on the ground and a check on the upright itself all come out
    of those three.

    Answered as a ladder of stricter lights, each priced in how far a marked
    tip sits off the two lines it must lie on -- px of mark movement, against
    what a hand mark is worth in this picture:

      the marks   both families fitted free. A miss already here means these
                  shadows share no single light at all, wherever it stands.
      a lamp      the light held over its own foot, collinear with the vertical
                  vanishing point. The least any real light can cost.
      the sun     the foot pushed onto the horizon as well.

    `value` names the strictest light that stays inside the floor. Which it is
    is itself a reading on the picture: a foot that will not go to the horizon
    says the light was a lamp in the room, and says where it stood.
    """
    T = [q for q in (triples or []) if len(q or []) == 3]
    if len(T) < 2:
        return undetermined(
            "two objects at least, each marked base, top and shadow tip: one "
            "shadow on its own agrees with every light there is", n=len(T))
    vert = np.asarray(vert, float)
    hz = None if a is None or b is None else V.horizon(a, b)
    got = R.shadow_light(T, vert, hz)

    # The uprights are marked here too, base to top, so whether they stand up
    # at all is free to ask -- and it comes first: a shadow read off something
    # that is not upright is being read against rays that do not belong to it.
    A = np.asarray(T, float).reshape(-1, 3, 2)
    ul = np.cross(vert, np.c_[A[:, 0], np.ones(len(A))])        # base to upright
    lean = (np.abs((ul * np.c_[A[:, 1], np.ones(len(A))]).sum(1))
            / np.maximum(np.hypot(ul[:, 0], ul[:, 1]), 1e-12))  # how far the top is off

    def px(key):
        v = got.get(key, {}).get("worst_px", np.inf)
        return v if np.isfinite(v) else np.inf

    free, lamp, sun = px("free"), px("lamp"), px("sun")
    floor = float(mark_px)
    sins = (got.get("sun") or got.get("lamp") or got["free"]).get("crossing_sin")
    cond = {"objects": len(T), "mark_accuracy_px": round(floor, 1),
            "marks_move_px": None if not np.isfinite(free) else round(free, 1),
            "lamp_move_px": None if not np.isfinite(lamp) else round(lamp, 1),
            "sun_move_px": None if not np.isfinite(sun) else round(sun, 1),
            "upright_move_px": round(float(max(lean)), 1),
            "ladder_why":
                "each of these is the farthest a marked shadow tip sits off the two "
                "lines it must lie on, for that light to be the one: the marks on "
                "their own first, then a light standing over its foot, then the sun "
                f"with its foot on the horizon. A hand mark here is worth {floor:.0f} px"}
    if sins and min(sins) < 0.15:
        cond["crossing_why"] = (
            f"object {int(np.argmin(sins)) + 1}'s ray and its shadow cross at only "
            f"{np.degrees(np.arcsin(min(sins))):.0f} deg, so where its tip lands is "
            "barely located at all -- the movement above still holds, the tip "
            "position below does not")
    if hz is None:
        cond["sun_why"] = ("no horizon here, so the sun cannot be told from a "
                           "lamp: what separates them is a foot infinitely far off")
    if max(lean) > floor:
        cond["upright_why"] = (
            f"one of these tops sits {max(lean):.0f} px off its own upright, more "
            "than a hand mark -- that shadow is being read against a ray that does "
            "not belong to it")

    def tips(key, drawable):
        """How far out the shadow tips are, and -- only where a light was
        actually found -- where that light puts them.

        The numbers go out either way: they are what the drawing would have to
        change, which is most worth knowing when the answer is no. The
        POSITIONS go out only with a light behind them, because they are what
        gets drawn, and a correction drawn from a refusal is a confident wrong
        answer at full opacity.
        """
        g = got.get(key) or {}
        if not g.get("tip_px"):
            return
        cond["tip_px"] = [round(float(q), 1) for q in g["tip_px"]]
        cond["tip_why"] = (
            f"the shadow tips are {max(cond['tip_px']):.0f} px out at worst"
            + (" under this light" if drawable else
               " under the light that fits them best, whether or not it can stand "
               "over its own foot")
            + " -- that is what the drawing would have to change, but it is the "
              "far end of a lever and no verdict is made on it")
        if drawable:
            cond["tip_at"] = [[round(float(q[0]), 1), round(float(q[1]), 1)]
                              for q in g["tip_at"]]

    def light(key, what):
        g = got[key]
        l, s = _pt(g["light"]), _pt(g["foot"])
        out = {"light": what, "light_vp": l, "foot": s}
        if l is None:
            out["light_note"] = ("the rays come out parallel on the page, so the "
                                 "light is infinitely far off to the side")
        tips(key, True)
        return out

    # Tried strictest first, and never gated on the free fit: a constrained
    # light can beat it on the worst line, because the free fit minimises the
    # algebraic residual over all of them and this is a maximum.
    if sun <= floor:
        cond["verdict_why"] = (
            f"sunlight: the foot of the light goes all the way to the horizon and "
            f"no tip is more than {sun:.1f} px off the lines it should sit on")
        return readout(light("sun", "the sun"), "consistent", **cond)
    if lamp <= floor:
        cond["verdict_why"] = (
            f"a light in the room, not the sun: held over its own foot it costs "
            f"{lamp:.1f} px, but driving that foot out to the horizon costs "
            + ("more than any light can" if not np.isfinite(sun) else f"{sun:.0f} px")
            + ". So the foot is where it says, and that is the lamp's own place "
              "on the ground")
        return readout(light("lamp", "a lamp, at that foot"), "consistent", **cond)
    if free <= floor:
        st = R.shadow_stray(T)
        cond["worst"] = (st[0] + 1 if st else
                         int(np.argmax((got.get("lamp") or {}).get("move_px")
                                       or [0.0])) + 1)
        cond["verdict_why"] = (
            f"the shadows agree with each other to {free:.1f} px, but with no light "
            f"that can stand over its own foot: the best one leaves object "
            f"{cond['worst']}'s tip {lamp:.0f} px out. Either these things are not "
            "upright -- the tops above say how far -- or the vertical vanishing "
            "point is not theirs")
        tips("free", False)
        return readout(None, "inconsistent", **cond)
    # Named leave one out: inside its own fit a blunder drags the light onto
    # itself and the loudest miss lands on an innocent neighbour -- here on
    # object 1 when object 3 was the one pushed.
    st = R.shadow_stray(T)
    cond["worst"] = (st[0] + 1 if st
                     else int(np.argmax(got["free"]["move_px"] or [0.0])) + 1)
    if st:
        cond["worst_why"] = (
            f"object {cond['worst']} is the one: left out of the fit, its own tip is "
            f"{st[1]:.0f} px off the light the others agree on")
    cond["verdict_why"] = (
        f"these shadows come from no one light: under the best one going, object "
        f"{cond['worst']}'s tip is still {free:.0f} px off the lines it should sit "
        f"on, past the {floor:.0f} px a hand mark is worth")
    tips("free", False)
    return readout(None, "inconsistent", **cond)


def _pt(p):
    """A homogeneous point as a page point, or None where it has none."""
    p = np.asarray(p, float)
    return None if abs(p[2]) < 1e-12 else [round(float(p[0] / p[2]), 1),
                                           round(float(p[1] / p[2]), 1)]


def spacing(posts, vp, a, b, px_per_mm, *, resolve_mm=2.0, mark_px=None):
    """Are these repeated uprights equally spaced in the world?

    The tolerance can never be finer than the hand that placed the marks. A
    dead-even six-post row marked to +-4 px reads 6.2 px out, so the 2 mm
    default alone called perfect geometry uneven whenever no real size had
    been declared and a millimetre was standing in for a pixel.
    """
    if len(posts) < 4:
        return undetermined("four posts at least: three always fit", n=len(posts))

    # The vanishing point of a line lies ON that line's image. So the chosen
    # VP has to sit along the row itself, and if it does not, the wrong family
    # was named and every number below is about some other direction. Measured
    # as a swing in pixels at the row's own ends, the way strays are.
    _mid, _e, crooked, off = R.row_frame(posts, vp)
    gate = float(mark_px) if mark_px else 9.0
    if crooked > gate:
        return undetermined(
            "these posts do not lie on one straight line in the world -- they "
            f"bend {crooked:.0f} px off the row, more than a hand mark "
            f"({gate:.0f} px). Even spacing is only defined along a straight run",
            bend_px=round(crooked, 1), mark_px=round(gate, 1), n=len(posts))
    if off > gate:
        return undetermined(
            "the vanishing point named is not the one this row runs to: it "
            f"sits {off:.0f} px off the row's own line, more than a hand mark "
            f"({gate:.0f} px). Pick the family whose lines run ALONG the row",
            vp_off_row_px=round(off, 1), mark_px=round(gate, 1), n=len(posts))

    r = R.spacing(posts, vp, a, b, px_per_mm)
    if r is None:
        return undetermined("a post sits on the vanishing point")
    # back into the order the posts were marked in. The fit works along the
    # row, which runs the other way whenever the far end was marked first, so
    # "worst_at 2" used to name the third post of a row marked right to left.
    err = np.empty(len(posts), float)
    err[np.asarray(r["order"], int)] = np.asarray(r["err_mm"], float)
    worst = float(np.max(np.abs(err)))
    floor = float(mark_px) / px_per_mm if mark_px else 0.0
    tol = max(resolve_mm, floor)
    cond = {"err_mm": [round(float(x), 3) for x in err],
            "ratio": [round(float(x), 4) for x in np.asarray(r["ratio"])],
            "worst_mm": worst, "resolve_mm": resolve_mm,
            "mark_noise_floor": floor, "tolerance_mm": tol, "n": len(posts)}
    if worst <= tol:
        return readout(round(worst, 3), "even", **cond)
    return readout(round(worst, 3), "uneven", worst_at=int(np.argmax(np.abs(err))), **cond)


def roof(slope_vp, vert_vp, rise_vp, *, resolve_px=15.0):
    """A tilted plane's slope VP, the vertical VP and the VP of the direction it
    rises along are collinear. No camera needed for the miss, only for degrees."""
    miss = float(R.tilted(slope_vp, vert_vp, rise_vp))
    if not np.isfinite(miss):
        return undetermined("the vertical and slope vanishing points coincide")
    cond = {"miss_px": miss, "resolve_px": resolve_px}
    return readout(round(miss, 2), "consistent" if miss <= resolve_px else "off", **cond)


def polygon(pts, ns=range(3, 13), idx=None):
    """Which regular polygon these marked corners are, with the whole ranking.

    Never just the winner: with part of the polygon marked a bigger n bends less
    over the same corners and wins on residual alone. Measured at +-3 px over
    200 trials, identification needs about n-1 corners -- a true octagon's five
    corners rank n=12 first and n=8 fifth.
    """
    p = np.asarray(pts, float).reshape(-1, 2)
    ranked = R.polygon_which(p, ns, idx)
    if not ranked:
        return undetermined("five corners at least: four fit anything", n=len(p))
    rows = [{"n": int(n), "rms_px": round(float(rms), 3), "rms_frac": round(float(fr), 5)}
            for n, rms, fr, _ in ranked]
    best, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    margin = float(second[1] / best[1]) if second and best[1] > 1e-9 else float("inf")
    need = int(best[0]) - 1
    cond = {"ranking": rows, "corners_marked": len(p), "corners_needed": need,
            "margin": margin}
    if len(p) < need:
        return undetermined(
            f"{len(p)} corners cannot identify a {best[0]}-gon; about {need} are needed",
            **cond)
    if margin < 2.0:
        return undetermined(
            f"n={best[0]} beats n={second[0]} by only {margin:.1f}x", **cond)
    return readout(int(best[0]), "identified", **cond)


# ------------------------------------------------- proportion, symmetry, stages
# core/figure.py is verified separately (`python core/figure.py`) and returns
# bare (value, conditioning, verdict) tuples. The geometry there is exact and is
# not reimplemented here; these wrappers only put it into the same vocabulary
# the rest of the app speaks, and attach the reason the refusal fired.

from . import figure as F                                             # noqa: E402


def _wrap(triple, reasons, value=lambda v: v):
    """A figure.py triple as a readout. `reasons` maps a conditioning test to
    the sentence shown when it is what refused."""
    val, cond, verdict = triple
    cond = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cond.items()}
    if verdict != "undetermined":
        return readout(value(val), verdict, **cond)
    why = next((msg for test, msg in reasons if test(cond)), "the input cannot support it")
    return undetermined(why, **cond)


def proportion(pts, v, span=(0, -1), unit=(0, 1), v_alt=()):
    """Heads tall, or any span measured in any unit, along one marked axis.

    Pixel heights are not proportion: a 7.5-head figure turned 72 deg reads 4.63
    heads off the page (bug 16). `v` is the vanishing point of the figure's OWN
    axis and is a user label, like a surface -- a standing figure uses the
    vertical one, a reclining figure does not.
    """
    return _wrap(F.axis_check(pts, v, span, unit, v_alt), [
        (lambda c: c["off_axis_frac"] > 0.05,
         "the marks are not on one straight axis; a bent pose is not an axis"),
        (lambda c: c["vp_spread"] > 0, "the axis's vanishing point is not pinned "
         "well enough: its uncertainty moves this answer more than 10 %"),
    ])


def symmetry(pairs, v_axis=None, v_pairs=None, floor=0.03):
    """Where the midline crosses each left/right pair, as a fraction of its width.

    0.5 is symmetric. Not perpendicular scatter about a fitted midline, which is
    blind to the displacement it is supposed to find -- 5.48 px of pushed
    shoulder showed up as 0.98 px of residual, and adding pairs made it worse
    (bug 17). Finds ONE odd pair; two at once rank the wrong one first.
    """
    return _wrap(F.midline(pairs, v_axis, v_pairs, floor), [
        (lambda c: c["worst_cross_angle_deg"] < 20.0,
         "the pairs run too nearly along the midline to cross it"),
        (lambda c: c["odd_margin"] < 1.5,
         "no pair stands out; two asymmetries at once can look like none"),
    ], value=lambda t: {"center": np.asarray(t[0]).tolist(),
                        "direction": np.asarray(t[1]).tolist(),
                        "asym_width_frac": np.asarray(t[2]).tolist()})


def drawing_deviation(src, dst, scale=None):
    """Per-point, out-of-sample deviation of a drawing from its reference.

    Leave-one-out, because the in-sample residual of a homography on few points
    is zero and proves nothing. Read only the worst point: leakage reached 58 %
    of the true error, so the second-worst is not a finding (bug 18).
    """
    return _wrap(F.deviation(src, dst, scale), [
        (lambda c: c["n_points"] < 6,
         "six points at least: below that the homography fits exactly and the "
         "residual says nothing"),
    ], value=lambda t: {"H": np.asarray(t[0]).tolist(),
                        "per_point_px": np.asarray(t[1]).tolist()})


def register_stages(src, dst):
    """The homography carrying one stage's marks onto the next photograph.

    Refuses below 8 points. At 4 the fit is exact and click noise passes
    straight through: half-pixel clicks cost 8 px on the worst transferred mark,
    against 1.27 px at 8 points.
    """
    s = np.asarray(src, float).reshape(-1, 2)
    if len(s) < 8:
        return undetermined(
            f"{len(s)} registration points; 8 or more are needed, because at 4 the "
            "homography is exact and click noise passes straight into every "
            "transferred mark", n_points=len(s))
    H = F.homography(s, np.asarray(dst, float).reshape(-1, 2))
    resid = np.linalg.norm(F.transfer(s, H) - np.asarray(dst, float).reshape(-1, 2), axis=1)
    return readout(H.tolist(), "ok", n_points=len(s),
                   fit_rms_px=float(np.sqrt((resid ** 2).mean())),
                   worst_px=float(resid.max()))


def correction(check_result, build):
    """A correction overlay, or nothing at all.

    No overlay may come out of an `undetermined` readout. A corrected line drawn
    from a vanishing point that is not pinned is a confident wrong answer
    rendered at full opacity, and that is the most expensive failure this
    project has had (bugs 3, 4, 9). Every emitter goes through here.
    """
    if check_result.get("verdict") == "undetermined":
        return {"overlay": None, "refused": check_result.get("why", "the check refused")}
    return {"overlay": build(check_result), "refused": None}


def horizon_consistency(vert_vp, vp_a, vp_b, width, pp=None):
    """Does the horizon's tilt agree with the vertical vanishing point?

    A square-pixel camera cannot have a tilted horizon and upright verticals at
    the same time: the roll implied by each must match. When they do not, there
    is no real focal length and `rhythm.calibrate` rightly returns nothing --
    the marks, not the solver, are what disagree. Reported in pixels of rise
    across the image, because that is what the user can go and re-mark.
    """
    a = np.asarray(vp_a, float)[:2]
    b = np.asarray(vp_b, float)[:2]
    v = np.asarray(vert_vp, float)[:2]
    if abs(b[0] - a[0]) < 1e-9:
        return undetermined("the two horizontal vanishing points share an x")
    slope = (b[1] - a[1]) / (b[0] - a[0])
    tilt = float(np.degrees(np.arctan(slope)))
    # The roll a vertical vanishing point implies is the direction from the
    # PRINCIPAL POINT to it. Reading it from y = 0 instead -- which is what
    # taking only the width allows -- invents disagreement: on an exactly
    # orthogonal triple with pp at (442, 640) the true roll is 2.0000 deg,
    # equal to the horizon's tilt to four places, and y = 0 made it 2.2030.
    # That spurious fifth of a degree crossed the 0.3 deg gate below on a
    # neighbouring camera and refused a triple that `rhythm.calibrate` solves
    # exactly. Where pp is not known, the image centre is the assumption the
    # rest of this app makes.
    c = (np.asarray(pp, float)[:2] if pp is not None
         else np.array([width / 2.0, 0.0]))
    dv = v - c
    roll = float(np.degrees(np.arctan2(dv[0], -dv[1]))) if abs(dv[1]) > 1e-9 else 0.0
    # A vertical vanishing point BELOW the picture -- which is simply a camera
    # pitched down -- points the other way along the same line, and read raw
    # that came back as -175.55 deg against a +4.25 deg horizon: a 179.8 deg
    # "disagreement" that refused a perfectly good camera. A direction is a
    # line, so fold it into the same (-90, 90] range the horizon tilt is in.
    roll = (roll + 90.0) % 180.0 - 90.0
    gap = abs((tilt - roll + 90.0) % 180.0 - 90.0)
    rise = float(abs(slope) * width)
    cond = {"horizon_tilt_deg": tilt, "vertical_roll_deg": roll,
            "disagreement_deg": gap,
            "roll_measured_from": [round(float(c[0]), 1), round(float(c[1]), 1)],
            "horizon_rise_px_over_image": rise}
    # 0.23 deg already halves f on 4.jpg's geometry; past ~0.3 there is no camera
    if gap > 0.3:
        return undetermined(
            f"the horizon tilts {tilt:+.2f} deg but the verticals say {roll:+.2f} deg; "
            "no square-pixel camera does both, so no focal length exists", **cond)
    return readout(round(gap, 3), "consistent", **cond)


def camera_level(vp_a, vp_b, pp_x, pp_y=None, *, a_range=None):
    """Focal length when the camera is level: the no-pitch case.

    `rhythm.calibrate` solves for f and both principal-point coordinates from
    three orthogonal vanishing points, and it is not ill-conditioned: on a clean
    near-level synthetic it still recovers f to 0.01. What it cannot do is
    invent a camera for three vanishing points that no camera could produce, and
    it signals that by returning nothing.

    That is what happens on 4.jpg's marked VPs, and the reason is worth keeping:
    f is hostage to the horizon's TILT, not to where the vanishing points sit
    along it. With the left VP near the principal point, the horizon rising over
    the page's width by

        0 px -> f = 1078;  5 px -> 932;  10 px -> 746;  15 px -> 476;  20 px -> none

    so a quarter of a degree of tilt -- fifteen pixels of vertical placement on
    a vanishing point thousands of pixels off the page -- is the difference
    between a camera and no camera. A focal length that "will not settle" is
    usually this, and `horizon_consistency` is the number to look at first.

    Assuming no pitch puts the principal point ON the horizon and fixes its x
    from the vertical VP, which leaves one equation and one unknown:
        f^2 = -(vA - pp) . (vB - pp)
    """
    a = np.asarray(vp_a, float)[:2]
    b = np.asarray(vp_b, float)[:2]
    if pp_y is None:                       # put it on the line joining the two VPs
        d = b - a
        pp_y = float(a[1] + (pp_x - a[0]) * d[1] / d[0]) if abs(d[0]) > 1e-9 else float(a[1])
    pp = np.array([float(pp_x), float(pp_y)])
    f2 = -float((a - pp) @ (b - pp))
    if f2 <= 0:
        return undetermined(
            "the two horizontal vanishing points are on the same side of the "
            "principal point, so they cannot be at right angles",
            pp=pp.tolist(), f2=f2)
    f = float(np.sqrt(f2))
    cond = {"pp": pp.tolist(), "assumed": "camera level (no pitch)"}
    if a_range:
        fs = []
        for x in np.linspace(*a_range, 7):
            g = -float((np.array([x, a[1]]) - pp) @ (b - pp))
            if g > 0:
                fs.append(np.sqrt(g))
        if fs:
            cond["f_range_over_vp_a"] = [round(min(fs), 1), round(max(fs), 1)]
    return readout(round(f, 1), "ok",
                   K=[[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]], **cond)


# --------------------------------------------------- real sizes in the scene

# A scene distance in millimetres is unreadable, so each declared unit has the
# unit its answer is said in. Imperial in, imperial out.
SCENE_UNIT = {"mm": ("m", 1000.0), "cm": ("m", 1000.0), "m": ("m", 1000.0),
              "in": ("ft", 304.8), "ft": ("ft", 304.8)}


def landmark(pairs, v, K, size_mm, *, mark_px, unit="cm", f_range=None,
             on_ground=True):
    """How far the viewer stands from something of known real size.

    The picture's own size (see app._px_per_mm) scales the PAPER. This scales
    the SCENE: a 180 cm figure in the drawing fixes the distance from the
    viewpoint to that figure, and -- if it stands on the ground the viewer
    stands on -- how high the eye is above that ground.

    The distance is proportional to the focal length and, near enough, to one
    over the figure's height on the page, so both of those uncertainties pass
    straight through. Measured on a synthetic camera, a 180 cm figure at the
    hand-marking accuracy of 9 px:

        625 px tall ->  +-3 %      240 px ->  +-8 %
         78 px tall -> +-25 %       28 px -> +-111 %

    so the answer is a range, never a number, and the last of those is refused
    rather than printed. f carries its own share on top: 1000-1250 px moved
    every one of those distances by +-11 %, independently of how well it was
    marked.
    """
    out_unit, per = SCENE_UNIT.get(unit, ("m", 1000.0))
    pr = [(np.asarray(a, float)[:2], np.asarray(b, float)[:2]) for a, b in pairs]
    if not pr:
        return undetermined("mark the base and the top of something whose real "
                            "size is known -- a standing figure, a door, a step")
    if not (size_mm and np.isfinite(size_mm) and size_mm > 0):
        return undetermined("say how big it really is, or there is nothing to "
                            "convert the picture into")
    if K is None:
        return undetermined("a real distance needs the camera, and the camera "
                            "is not determined here -- see the camera readout")

    K = np.asarray(K, float)
    fs = [float(K[0, 0])]
    if f_range and len(f_range) == 2 and all(np.isfinite(f_range)):
        fs = [float(min(f_range)), float(K[0, 0]), float(max(f_range))]

    def one(a, b, f):
        """Distance to the base, and the eye's height above its ground."""
        Kf = np.array([[f, 0, K[0, 2]], [0, f, K[1, 2]], [0, 0, 1.0]])
        # The vanishing point belongs to the camera it was measured with, so it
        # is taken as marked and only f varies underneath it.
        t = R.segment_depth(a, b, v, Kf)
        if t is None:
            return None
        d = t[0] * size_mm
        if not on_ground:
            return d, None
        # The base sits on the ground plane, whose normal is the segment's own
        # direction; the camera is at the origin, so its height above that
        # plane is how far the base point lies along that direction. True only
        # if this thing stands on the same ground the viewer does, which is why
        # the assumption has a name rather than hiding inside the distance.
        Ki = np.linalg.inv(Kf)
        vv = np.asarray(v, float).ravel()
        nrm = Ki @ (np.append(vv[:2], 1.0) if vv.size == 2 else vv)
        nrm = nrm / np.linalg.norm(nrm)
        ray = Ki @ np.append(a, 1.0)
        ray = ray / np.linalg.norm(ray)
        return d, abs(float(d * (ray @ nrm)))

    each, worst_rel = [], 0.0
    for i, (a, b) in enumerate(pr):
        h = float(np.hypot(*(b - a)))
        u = (b - a) / h if h > 1e-9 else np.zeros(2)
        # The answer is the camera as measured and the marks as placed. Every
        # other combination is there to say how far that answer could be out,
        # and must not be mistaken for it: reading the headline off the first
        # of them quietly reported the f = 1000 corner as the distance.
        nominal = one(a, b, float(K[0, 0]))
        if nominal is None:
            each.append({"n": i + 1, "height_px": round(h, 1), "distance": None,
                         "why": "this segment points along its own line of sight, "
                                "so it has no depth to give"})
            continue
        ds = [nominal[0]]
        for f in fs:
            for sa, sb in ((0.0, 0.0), (1.0, -1.0), (-1.0, 1.0)):
                got = one(a + sa * mark_px * u, b + sb * mark_px * u, f)
                if got is not None:
                    ds.append(got[0])
        lo, hi, mid = min(ds), max(ds), nominal[0]
        rel = max(hi - mid, mid - lo) / mid if mid > 0 else np.inf
        worst_rel = max(worst_rel, rel)
        each.append({
            "n": i + 1, "height_px": round(h, 1),
            "distance": round(mid / per, 3),
            "range": [round(lo / per, 3), round(hi / per, 3)],
            "plus_minus_pct": round(100 * rel, 1),
            "eye_height": None if nominal[1] is None else round(nominal[1] / per, 3)})

    good = [e for e in each if e.get("distance") is not None]
    cond = {"unit": out_unit, "real_size_mm": float(size_mm),
            "mark_accuracy_px": round(float(mark_px), 2),
            "focal_px": round(float(K[0, 0]), 1),
            "focal_range_px": None if len(fs) < 3 else [round(fs[0], 1), round(fs[-1], 1)],
            "per_landmark": each, "n": len(pr),
            "assumed": ("it stands upright on the same ground the viewer does"
                        if on_ground else "nothing about the ground")}
    if not good:
        return undetermined("none of these segments gives a depth: each points "
                            "along its own line of sight", **cond)
    # A range wider than the answer is not a measurement. At 28 px of figure
    # the honest span was 42 m to 198 m for a true 70 m, and printing "70 m"
    # beside it would be the confident wrong number this app exists to avoid.
    if worst_rel > 0.5:
        n = min(good, key=lambda e: e["height_px"])
        return undetermined(
            f"marked too small to measure: the shortest of these is "
            f"{n['height_px']:.0f} px tall against a hand mark worth "
            f"{mark_px:.0f} px, which puts its distance anywhere from "
            f"{n['range'][0]:.3g} to {n['range'][1]:.3g} {out_unit}. A range "
            "wider than the answer is not a distance -- mark something larger "
            "on the page, or nearer the viewer", **cond)
    return readout({"distance": [e["distance"] for e in good], "unit": out_unit},
                   "ok", **cond)
