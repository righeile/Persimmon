#!/usr/bin/env python3
"""Re-runs every verification the core modules were built against.

    .venv-apps/bin/python selfcheck.py

These are not unit tests in the usual sense: each one is a result that was
measured once, by hand, against synthetic geometry with known ground truth, and
then written into the plan as a number the app is entitled to rely on. If one
of them stops holding, something downstream is quietly wrong, so they run as
asserts and the app refuses to start without them.

Where a tolerance looks loose it is because the measurement said so -- the arch
depth resolves about 3 %, not 0.3 %, and pretending otherwise would be the same
mistake the conditioning rules exist to prevent.
"""
from __future__ import annotations

import sys

import numpy as np

from core import checks as C
from core import figure as F
from core import rectify as RC
from core import rhythm as R
from core import vp as V

OK, FAIL = [], []


def check(name, got, want=None, tol=None, pred=None):
    """One verified claim. Either `got` is within `tol` of `want`, or `pred` holds."""
    try:
        if pred is not None:
            good = bool(pred(got))
            shown = f"{got}"
        else:
            good = bool(np.shape(got) == np.shape(want)
                        and np.all(np.abs(np.asarray(got) - np.asarray(want)) <= tol))
            shown = (f"{got:.4f} vs {want:.4f} (tol {tol})"
                     if np.isscalar(got) else f"{got} vs {want}")
    except Exception as exc:                       # a check that cannot run has failed
        good, shown = False, f"raised {exc!r}"
    (OK if good else FAIL).append(name)
    print(f"  {'ok  ' if good else 'FAIL'}  {name:<46s} {shown}")
    return good


# --------------------------------------------------------------- the geometry

def camera(f=1000.0, pp=(40.0, -25.0), rx=0.17, ry=0.41, rz=0.06):
    """A camera with a known f and principal point, and a known orientation."""
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rot = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
           @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
           @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]]))
    return K, Rot


def vp_of(K, Rot, d):
    """The vanishing point of a world direction, as (x, y) on the page."""
    v = K @ Rot @ np.asarray(d, float)
    return v[:2] / v[2]


def project(K, Rot, t, pts_uv, e1, e2):
    """Points given in a plane's own (u, v) coordinates, onto the page."""
    X = (np.asarray(t, float)[None, :]
         + np.asarray(pts_uv, float)[:, :1] * (Rot @ e1)[None, :]
         + np.asarray(pts_uv, float)[:, 1:] * (Rot @ e2)[None, :])
    x = X @ K.T
    return x[:, :2] / x[:, 2:]


def vanish_line(K, Rot, normal):
    """The vanishing line of the plane with that world normal."""
    l = np.linalg.inv(K).T @ Rot @ np.asarray(normal, float)
    return l / np.linalg.norm(l)


def arcpts(sweep, ratio=1.0, n=90, r=1.0, start=0.0, center=(0.0, 0.0)):
    th = np.radians(np.linspace(start, start + sweep, n))
    return np.column_stack([center[0] + r * np.cos(th),
                            center[1] + r / ratio * np.sin(th)])


# ------------------------------------------------------------------ the checks

def conic_fits():
    print("\nconic fit  (plan: 0.0000 px exact; 1 px noise in -> 1.00 px out)")
    rng = np.random.default_rng(0)
    cases = {
        "exact circle": arcpts(360, 1.0, 200, r=180, center=(500, 400)),
        "offset circle": arcpts(360, 1.0, 200, r=90, center=(-300, 1200)),
        "90 deg arc": arcpts(90, 1.0, 120, r=240, center=(400, 400)),
        "tilted ellipse": arcpts(360, 1.0, 200, r=200, center=(500, 500))
                          @ np.array([[1, 0], [0, 0.6]]) @ np.array(
                              [[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]]),
        "42 deg shallow arc": arcpts(42, 1.0, 120, r=300, center=(400, 900)),
    }
    for name, p in cases.items():
        check(f"conic resid, {name}", R.conic_resid(p), 0.0, 1e-3)
    p = cases["exact circle"] + rng.normal(0, 1.0, cases["exact circle"].shape)
    check("conic resid, 1 px noise in", R.conic_resid(p), 1.00, 0.15)
    # bug 1: a line pair is a conic, so a conic cannot lose to a line.
    for name, p in cases.items():
        cr, lr = C.conic_beats_line(p)
        check(f"conic beats a line, {name}", (cr, lr), pred=lambda t: t[0] <= t[1] + 1e-6)


def calibration():
    print("\ncamera from 3 orthogonal VPs  (plan: f=1000.00 and centre (40,-25) exactly)")
    K, Rot = camera()
    v = [vp_of(K, Rot, d) for d in np.eye(3)]
    Kr = R.calibrate(*v)
    check("recovered f", float(Kr[0, 0]), 1000.0, 0.01)
    check("recovered centre", Kr[:2, 2], np.array([40.0, -25.0]), 0.01)
    # A near-level camera is NOT a problem for the free solve. That was briefly
    # believed and is wrong; recorded here so it is not believed again.
    K2, Rot2 = camera(f=1000.0, pp=(40.0, -25.0), rx=1e-5, ry=0.41, rz=0.0)
    va, vb, vc = (vp_of(K2, Rot2, d) for d in np.eye(3))
    check("near-level camera: free solve still exact", float(R.calibrate(vc, va, vb)[0, 0]),
          1000.0, 0.01)
    check("level camera: constrained solve agrees",
          C.camera_level(va, vb, pp_x=40.0, pp_y=-25.0)["value"], 1000.0, 1.0)
    # an infinite vanishing point must refuse, not raise an unconverged SVD
    check("infinite VP refuses instead of raising",
          R.calibrate(np.array([np.inf, np.inf]), va, vb), pred=lambda K: K is None)
    # What actually defeats 4.jpg: a tilted horizon against upright verticals.
    # No square-pixel camera does both, so there is no focal length to find.
    hc = C.horizon_consistency(np.array([287.0, -69267.0]), np.array([-60.1, 830.2]),
                               np.array([3634.8, 857.8]), width=884)
    check("tilted horizon vs upright verticals -> undetermined", hc["verdict"],
          pred=lambda v: v == "undetermined")
    check("  and no real f exists there",
          R.calibrate(np.array([287.0, -69267.0]), np.array([3634.8, 857.8]),
                      np.array([-60.1, 830.2])), pred=lambda K: K is None)
    check("  a level horizon on the same VPs does give one",
          C.camera_level(np.array([-60.1, 830.2]), np.array([3634.8, 830.2]),
                         pp_x=287.0)["value"], 1078.0, 3.0)

    # Two perpendicular directions and a pinned principal point: the only
    # camera a view along a nave or a street can give, because the second
    # horizontal family is not in the picture to be marked. f comes out of
    # the same equation, and what the pin costs is the thing to print.
    Kp, Rp = camera(f=1000.0, pp=(600.0, 600.0))
    vv, vz = vp_of(Kp, Rp, [0, 1., 0]), vp_of(Kp, Rp, [0, 0, 1.])
    check("upright + one horizontal, principal point pinned right",
          C.camera_level(vv, vz, pp_x=600.0, pp_y=600.0)["value"], 1000.0, 0.5)
    fs = [C.camera_level(vv, vz, pp_x=600.0 + dx, pp_y=600.0 + dy)["value"]
          for dx in (-60.0, 0.0, 60.0) for dy in (-60.0, 0.0, 60.0)]
    check("...and 5 % of the picture off it costs this much of f, %",
          100 * (max(fs) - min(fs)) / 1000.0, pred=lambda v: 20.0 < v < 45.0)
    check("...the true f inside the sweep", min(fs) <= 1000.0 <= max(fs),
          pred=lambda v: v is True)
    # What the pin costs is not a constant: f is near enough the product of
    # the two vanishing points' distances from it, so moving the pin by d
    # changes f by d(1/B - 1/A) -- B the NEARER vanishing point. A view along
    # a nave, with its uprights thousands of pixels off the page and its one
    # horizontal a thousand below, is the cheap end of that; a steep
    # three-point view with both close is the dear end, 31 % above.
    for A, B, xo, want in ((14000.0, 1000.0, 900.0, 6.0), (6000.0, 2000.0, 300.0, 2.2)):
        pa, pb = np.array([600.0, 600.0 - A]), np.array([600.0 + xo, 600.0 + B])
        f0 = C.camera_level(pa, pb, pp_x=600.0, pp_y=600.0)["value"]
        gs = [C.camera_level(pa, pb, pp_x=600.0 + dx, pp_y=600.0 + dy)["value"]
              for dx in (-60.0, 0.0, 60.0) for dy in (-60.0, 0.0, 60.0)]
        check(f"...{A:.0f} px up and {B:.0f} px down, 60 px of pin costs, %",
              100 * (max(gs) - min(gs)) / f0, want, 0.5)
        check("...which is the pin over the nearer vanishing point, %",
              100 * 60.0 * (1 / B - 1 / A), want, 0.6)
    # Two directions carry no redundancy, so nothing here can tell that they
    # are not square to each other -- and f pays for it steeply. Mark the
    # uprights and one horizontal of a nave and they ARE square; mark a
    # near-horizontal by mistake and the number is quietly wrong.
    def askew(deg):
        t = np.radians(deg)
        return C.camera_level(vv, vp_of(Kp, Rp, [0, np.sin(t), np.cos(t)]),
                              pp_x=600.0, pp_y=600.0)
    check("5 deg off square reads f as this, not 1000", askew(5.0)["value"],
          711.5, 1.0)
    check("...and nothing in the two marks can catch it",
          askew(5.0)["verdict"], pred=lambda v: v == "ok")
    check("20 deg off square does at least refuse", askew(20.0)["verdict"],
          pred=lambda v: v == "undetermined")
    # With one horizontal family the horizon is still known -- it is the
    # vanishing line of every plane square to the uprights -- and the checks
    # that need one (rows of posts, standing figures) refused on a nave until
    # they were given it.
    import app as A
    vx = vp_of(Kp, Rp, [1., 0, 0])
    doc = {"vps": {"up": list(vv), "nave": list(vz)}, "vertical_vp": "up"}
    ha, hb = A._horizon(doc, Kp)
    e = (hb - ha) / np.linalg.norm(hb - ha)
    d = vx - ha
    check("one horizontal family: the camera's horizon, px off the third VP",
          float(abs(e[0] * d[1] - e[1] * d[0])), 0.0, 1.0)
    check("...the named horizontal is on it too, px",
          float(abs(e[0] * (vz - ha)[1] - e[1] * (vz - ha)[0])), 0.0, 0.01)
    check("...and with no camera it still refuses",
          A._horizon(doc)[0], pred=lambda q: q is None)
    # The page sends the picture's size as image.w/h and never as width, so
    # a pin "at the middle of the picture" was landing 100 px off centre on a
    # 1200 px photograph and quietly moving f.
    pinned = A._camera({"perspective": "three", "vertical_vp": "up",
                        "image": {"w": 1200, "h": 900},
                        "vps": {"up": list(vv), "nave": list(vz)}})[0]
    check("the pin follows the picture, not a default 1000 px",
          pinned["conditioning"]["pp_assumed"], pred=lambda t: t == "600, 450")

def circle_vs_oval():
    print("\ncircle on a known plane  (plan: 1.00000 anywhere; 8 % oval -> 1.0800;"
          " noise floor 1.021)")
    K, Rot = camera()
    e1, e2, n = np.array([1., 0, 0]), np.array([0, 0, 1.]), np.array([0, 1., 0])
    l = vanish_line(K, Rot, n)
    t = Rot @ np.array([0, 6.0, 0])
    for where in ((0, 0), (2.0, 1.0), (-1.5, 2.5)):
        p = project(K, Rot, t, arcpts(360, 1.0, 200, r=1.0, center=where), e1, e2)
        check(f"true circle at {where}", R.circle_check(p, K, l)["ratio"], 1.0, 1e-4)
    p = project(K, Rot, t, arcpts(360, 1 / 1.08, 200, r=1.0), e1, e2)
    check("8 % oval", R.circle_check(p, K, l)["ratio"], 1.08, 2e-3)
    rng = np.random.default_rng(3)
    p = project(K, Rot, t, arcpts(360, 1.0, 200, r=1.0), e1, e2)
    got = [R.circle_check(p + rng.normal(0, 1.0, p.shape), K, l)["ratio"] for _ in range(40)]
    check("noise floor, 1 px marks", float(np.median(got)), 1.021, 0.03)


def ellipses():
    """A circle drawn into a plane, and the verdict on one that was marked.

    Two halves of the same tool. Drawing: given the plane and the camera, the
    ellipse a real circle there would cast. Measured by sending every drawn
    point back out to the plane along its own ray -- the scene's geometry, not
    the one the drawing used -- and asking how far each lands from the centre.
    Judging: whether marks round a rim are consistent with round, which has to
    be able to say YES. Comparing the departure to the uncertainty alone made
    "circle" all but unreachable: a true circle marked to +-1 px read 1.0072
    and was called an oval, and only an exact one ever came back round.
    """
    print("\nellipses  (plan: every drawn point lands one radius from the centre,"
          " to 1e-12; a hand-marked circle reads round)")
    K, Rot = camera()
    e1, e2, n = np.array([1., 0, 0]), np.array([0, 0, 1.]), np.array([0, 1., 0])
    l = vanish_line(K, Rot, n)
    t = Rot @ np.array([0, 6.0, 0])
    Ki, nc = np.linalg.inv(K), Rot @ n

    def out_to_plane(pts):
        """Page points back onto the scene's own plane, by ray intersection."""
        q = np.asarray(pts, float).reshape(-1, 2)
        d = (Ki @ np.column_stack([q, np.ones(len(q))]).T).T
        X = d * ((nc @ t) / (d @ nc))[:, None] - t
        return np.column_stack([X @ (Rot @ e1), X @ (Rot @ e2)])

    def dots(where, rad, ang=0.0, n_=20000):
        """What a user drags: the ring's own two ends. The drag's direction is
        the line between them, so the two ends settle on each other. Five
        passes converge; three leave 1 % of the span on a ring this skewed,
        which looked like a fault in the solver and was not.

        It settles on an axis, and on a nearly round ring that direction is
        barely determined -- the r 1.6 ring below lands 0.5 % out, which is a
        drag no circle exactly answers, so the solver returns the nearest one
        and the row below is loose by that much."""
        ring = project(K, Rot, t, arcpts(360, 1.0, n_, r=rad, center=where),
                       e1, e2)
        u = np.array([np.cos(ang), np.sin(ang)])
        for _ in range(8):
            k = ring @ u
            pa, pb = ring[k.argmin()], ring[k.argmax()]
            u = (pb - pa) / np.linalg.norm(pb - pa)
        return ring, pa, pb

    # The round trip that matters: take a real circle, drag its two ends, and
    # the same circle must come back. The marks are the ring's ENDS, not a
    # middle and a rim -- the image's middle is not the image of the middle,
    # so the old handle overran the point dragged to by up to 17 %, and on a
    # steeply foreshortened ring by 85 %.
    for where, rad, ang in (((0.0, 1.0), 1.0, 0.0), ((2.0, 1.0), 0.4, 0.0),
                            ((-1.5, 2.5), 1.6, 0.0), ((0.0, 1.0), 1.0, 0.7)):
        true_ring, pa, pb = dots(where, rad, ang)
        got = R.plane_circle(K, l, pa, pb, n=1440)
        P = out_to_plane(got)
        r = np.hypot(*(P - np.asarray(where)).T)
        check(f"dragging the ends of a circle at {where}, r {rad} gives it back",
              float(np.abs(r - rad).max()), 0.0, 2e-3 * rad)
        check("  and it reads back as round",
              R.circle_check(got, K, l)["ratio"], 1.0, 1e-9)
        u = (pb - pa) / np.linalg.norm(pb - pa)
        k = (got - pa) @ u
        check("  with its ends exactly at the dots",
              float(max(-k.min(), k.max() - np.linalg.norm(pb - pa))), 0.0, 0.15)

    # the ring is measured ON THE PLANE, not on the page: the same drag
    # further from the viewer must stand for a bigger circle, so an equal one
    # further off is drawn smaller
    near = R.plane_circle(K, l, *dots((0.0, 1.0), 1.0)[1:])
    far = R.plane_circle(K, l, *dots((0.0, 3.0), 1.0)[1:])
    check("an equal circle further off is drawn smaller",
          np.ptp(np.asarray(far)[:, 0]) < np.ptp(np.asarray(near)[:, 0]), pred=bool)

    # a circle that reaches the plane's vanishing line has no drawing at all
    y0 = -(l[0] * 540 + l[2]) / l[1]
    check("one that runs off the plane refuses",
          R.plane_circle(K, l, (340.0, y0), (740.0, y0)) is None, pred=bool)
    check("  and so does a drag that goes nowhere",
          R.plane_circle(K, l, (400.0, 900.0), (400.0, 900.0)) is None, pred=bool)

    # and the verdict, which is the half that was wrong
    rng = np.random.default_rng(11)
    exact = project(K, Rot, t, arcpts(360, 1.0, 200, r=1.0), e1, e2)
    for px in (1.0, 3.0, 6.0):
        got = [C.circle(exact + rng.normal(0, px, exact.shape), K, l)["verdict"]
               for _ in range(20)]
        check(f"a true circle marked to +-{px:.0f} px reads round",
              sum(v == "circle" for v in got), 20, 0)
    oval = project(K, Rot, t, arcpts(360, 1 / 1.30, 200, r=1.0), e1, e2)
    got = [C.circle(oval + rng.normal(0, 3.0, oval.shape), K, l)["verdict"]
           for _ in range(20)]
    check("a 30 % oval marked to +-3 px still reads oval",
          sum(v == "oval" for v in got), 20, 0)
    r3 = C.circle(exact + rng.normal(0, 3.0, exact.shape), K, l)
    check("  and the tolerance it was judged against is reported",
          r3["conditioning"].get("tolerance"), pred=lambda v: v is not None and v > 0)
    # Marking noise biases the ratio upward -- every wobble makes the conic
    # less round, never rounder -- so what the marks can resolve is a property
    # of the marks. A full rim at 200 marks can answer at 0.005; half a rim at
    # 20 cannot, and says what it could answer at instead of guessing.
    half = project(K, Rot, t, arcpts(180, 1.0, 20, r=1.0), e1, e2)
    fine = C.circle(exact + rng.normal(0, 3.0, exact.shape), K, l, resolve=0.005)
    rh = C.circle(half + rng.normal(0, 3.0, half.shape), K, l, resolve=0.005)
    check("a full rim answers at the fine tolerance asked for",
          fine["conditioning"]["tolerance"], 0.005, 1e-9)
    check("  half a rim, sparsely marked, refuses", rh["verdict"], pred=lambda v: v == "undetermined")
    check("  and names what it could resolve",
          rh["conditioning"]["mark_noise_floor"], pred=lambda v: v > 0.005)
    check("  which answering at does give a verdict",
          C.circle(half + rng.normal(0, 3.0, half.shape), K, l,
                   resolve=rh["conditioning"]["mark_noise_floor"] * 1.5)["verdict"],
          pred=lambda v: v in ("circle", "oval"))


def askew_table():
    """A table turned away from the wall, standing on the same floor.

    Its top is parallel to the floor, so it has the floor's vanishing line --
    the horizon -- whichever way it is turned. That is why a surface can be
    declared level and read against the horizon with no lines marked on the
    table at all. Measured here so the claim is not just an argument.
    """
    print("\nan askew table on the floor  (plan: its horizon is the room's, to 1e-12)")
    K, Rot = camera()
    th = np.radians(35.0)                       # the table, turned 35 deg
    room = (np.array([1., 0, 0]), np.array([0, 0, 1.]))
    table = (np.array([np.cos(th), 0, np.sin(th)]),
             np.array([-np.sin(th), 0, np.cos(th)]))

    def line_through(d1, d2):
        a, b = (np.append(vp_of(K, Rot, d), 1.0) for d in (d1, d2))
        l = np.cross(a, b)
        return l / np.linalg.norm(l)

    h_room, h_table = line_through(*room), line_through(*table)
    if np.dot(h_room, h_table) < 0:
        h_table = -h_table                      # a line has no sign
    check("the table's horizon is the room's", float(np.abs(h_room - h_table).max()),
          0.0, 1e-12)
    check("and both are the floor's vanishing line",
          float(np.abs(h_room - vanish_line(K, Rot, np.array([0, 1., 0]))).max()),
          0.0, 1e-12)

    # a circle lying on the table, read against the room's horizon
    t = Rot @ np.array([0, 6.0, 0])
    p = project(K, Rot, t, arcpts(360, 1.0, 200, r=1.0), *table)
    check("a round tabletop reads round on the room's horizon",
          R.circle_check(p, K, h_room)["ratio"], 1.0, 1e-4)

    # the contrast: a wall is not a floor, and the horizon says nothing there
    wall = vanish_line(K, Rot, np.array([0, 0, 1.]))
    check("the same marks on a wall's vanishing line are not even an ellipse",
          R.circle_check(p, K, wall), pred=lambda r: r is None)


def square_partner():
    """The fourth vanishing point, from the camera rather than from marks.

    Two horizontal families and the uprights are the camera. A third family is
    often an object standing askew to all of it, and then its OTHER horizontal
    is not free: two perpendicular directions have vanishing points conjugate
    in w = inv(K Kt), and a horizontal one is on the horizon as well, so one
    cross product picks it out. Marked instead, the same point is a reading on
    the camera -- a weak one, which is the second half of what is checked here.
    """
    print("\nthe square partner on the horizon  "
          "(plan: the camera says where the fourth vanishing point goes)")
    K, Rot = camera()
    hz = vanish_line(K, Rot, np.array([0, 1., 0]))        # y is up here
    worst, far = 0.0, 0.0
    for deg in (8, 17, 31, 45, 62, 80):
        th = np.radians(deg)
        d3 = np.array([np.cos(th), 0, np.sin(th)])
        d4 = np.array([-np.sin(th), 0, np.cos(th)])
        v3, v4 = vp_of(K, Rot, d3), vp_of(K, Rot, d4)
        got = R.square_partner(K, v3, hz)
        worst = max(worst, float(np.linalg.norm(got - v4)))
        far = max(far, float(abs(hz @ np.append(got, 1.0))))
    check("lands on the true perpendicular, 8 to 80 deg askew, px", worst, 0.0, 1e-8)
    check("and on the horizon, px", far, 0.0, 1e-9)

    # The upright direction is square to EVERY horizontal, so w v IS the
    # horizon and nothing is picked out. Refusing beats returning the garbage
    # that an almost-parallel intersection gives.
    check("the uprights have no one partner",
          R.square_partner(K, vp_of(K, Rot, np.array([0, 1., 0])), hz),
          pred=lambda q: q is None)

    # f is what this point rests on, and it rests on it badly: hundreds of
    # pixels of position for a tenth of f. What a line drawn to it suffers is
    # another matter, and that is the number worth printing.
    th = np.radians(31.0)
    v3 = vp_of(K, Rot, np.array([np.cos(th), 0, np.sin(th)]))
    base = R.square_partner(K, v3, hz)
    span, mid = 180.0, np.array([600.0, 500.0])
    moved, swing = [], []
    for s in (0.9, 1.1):
        Kf = K.copy()
        Kf[0, 0] = Kf[1, 1] = K[0, 0] * s
        hf = vanish_line(Kf, Rot, np.array([0, 1., 0]))
        q = R.square_partner(Kf, v3, hf)
        a, b = base - mid, q - mid
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        moved.append(float(np.linalg.norm(q - base)))
        swing.append(span * float(abs(a[0] * b[1] - a[1] * b[0])))
    check("f out by a tenth moves it this far, px", max(moved), pred=lambda x: x > 10)
    check("...which swings a 180 px mark only this far, px", max(swing),
          pred=lambda x: x < 3.0)
    check("so its position is this many times worse than its swing",
          max(moved) / max(swing), pred=lambda x: x > 10)

    # And the whole point of declaring it: a family that came out of the
    # camera must not then be evidence for the camera. Left in, an askew table
    # takes a room family's place -- the camera uses the first two horizontals
    # it finds -- and f comes back wrong, with a tick beside it.
    import app as A
    W, H = 1200.0, 900.0
    f0, pp = 1400.0, (600.0, 450.0)
    # this camera is one where the loose answer comes back CONFIDENT; on
    # others the non-orthogonal triple simply gives no camera at all, which
    # was never the dangerous case
    Kr, Rr = camera(f0, pp, -0.08, 0.46, 0.02)

    def pv(d):
        v = Kr @ Rr @ np.asarray(d, float)
        return (v[:2] / v[2]).tolist()

    doc = {"image": {"w": W, "h": H}, "width": W, "height": H,
           "perspective": "three", "vertical_vp": "up", "features": {},
           "vps": {"wall": pv([1, 0, 0]), "table": pv([np.cos(th), np.sin(th), 0]),
                   "up": pv([0, 0, 1.])}}
    with A.app.test_client() as cl:
        loose = cl.post("/api/check", json={"doc": doc}).get_json()["camera"]
        doc["square_of"] = {"table": "wall"}
        tight = cl.post("/api/check", json={"doc": doc}).get_json()["camera"]
        sq = cl.post("/api/square_vp",
                     json={"doc": doc, "of": "wall"}).get_json()
    def fof(got):
        """Three horizontals report f inside a camera, one reports f itself."""
        v = got["value"]
        return float(v["f"] if isinstance(v, dict) else v)

    check("an askew family left in the camera reads f as this, not 1400",
          fof(loose), pred=lambda x: abs(x - f0) > 300)
    check("declared square, the camera is the room's own f", fof(tight), f0, 1e-6)
    check("and the route gives the wall's partner, px off truth",
          float(np.linalg.norm(np.array(sq["value"])
                               - np.array(pv([0, 1., 0])))), 0.0, 1e-6)


def shadows():
    """Cast shadows, and the light that has to stand over its own foot.

    Three marks an object -- base, top, shadow tip -- and everything follows:
    the rays run to the light, the ground lines run to the light's own foot,
    and a light stands over its foot, so those two points and the vertical
    vanishing point are collinear. The sun is the case whose foot goes all the
    way to the horizon. What is checked here is that a true picture comes back
    exactly, that a lamp is not called the sun, that hand noise is survived and
    a blunder is not -- and the two measurements that decided HOW it is done.
    """
    print("\ncast shadows and the light over its foot  "
          "(plan: a light stands over its foot, so the marks force it)")
    # Metres, y up as everywhere else here, ground at y = 0. Five posts in
    # front of a camera at eye height, and the shadows land ON the page --
    # which is the whole point of the fixture: the first one put the tips tens
    # of thousands of px off the picture, and at that lever every currency
    # looks hopeless and nothing it measured was about drawings.
    f0, pp = 1400.0, (600.0, 450.0)
    K, Rot = camera(f0, pp, 0.09, 0.22, 0.02)
    eye = np.array([-0.6, -1.65, -9.0])
    UP = np.array([0.0, -1.0, 0.0])

    def img(X):
        X = np.atleast_2d(np.asarray(X, float))
        x = (Rot @ (X - eye).T).T @ K.T
        return x[:, :2] / x[:, 2:]

    def vanish(d):
        v = K @ Rot @ np.asarray(d, float)
        return v[:2] / v[2]

    vert = np.append(vanish(UP), 1.0)
    ha, hb = vanish([1, 0, 0.]), vanish([0, 0, 1.])
    rng = np.random.default_rng(3)
    base = np.column_stack([rng.uniform(-2.5, 2.0, 5), np.zeros(5),
                            rng.uniform(1.5, 7.0, 5)])
    hgt = rng.uniform(1.0, 2.4, 5)
    top = base + hgt[:, None] * UP

    def sun_tips(el_deg, az_deg):
        el, az = np.radians(el_deg), np.radians(az_deg)
        ray = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        return top + (hgt / np.sin(el))[:, None] * ray, ray

    def lamp_tips(P):
        """Where a lamp at world point `P` throws each top onto the ground."""
        P = np.asarray(P, float)
        k = P[1] / (P[1] - top[:, 1])
        return P[None, :] + k[:, None] * (top - P[None, :])

    def marks(tips, jitter=0.0, push=None, seed=7):
        b, tp, x = img(base), img(top), img(tips).copy()
        if push is not None:
            x[push[0]] = x[push[0]] + np.asarray(push[1], float)
        if jitter:
            g = np.random.default_rng(seed)
            b, tp, x = (q + g.normal(0, jitter, q.shape) for q in (b, tp, x))
        return [[b[i].tolist(), tp[i].tolist(), x[i].tolist()] for i in range(len(b))]

    def run(tr, **kw):
        return C.shadow(tr, vert, ha, hb, mark_px=9.0, **kw)

    tips, ray = sun_tips(35.0, 200.0)
    got = run(marks(tips))
    check("true sunlight is called sunlight", got["verdict"] + "/"
          + (got["value"] or {}).get("light", "-"), pred=lambda s: s == "consistent/the sun")
    check("and no tip is off the lines it sits on, px",
          got["conditioning"]["sun_move_px"], 0.0, 1e-6)
    check("the light comes back where the sun is, px",
          float(np.linalg.norm(np.array(got["value"]["light_vp"]) - vanish(ray))),
          0.0, 0.1)
    check("and its foot where the shadows run, px",
          float(np.linalg.norm(np.array(got["value"]["foot"])
                               - vanish(ray * np.array([1, 0, 1])))), 0.0, 0.1)
    # the readout rounds to 0.1 px, so the two exact claims are asked of the
    # geometry rather than of the printed answer
    hz = V.horizon(ha, hb)
    raw = R.shadow_light(np.asarray(marks(tips), float).reshape(-1, 3, 2),
                         vert, hz)["sun"]
    foot, lit = raw["foot"], raw["light"]
    check("that foot is ON the horizon, px",
          abs(float(hz @ foot / max(abs(foot[2]), 1e-12))), 0.0, 1e-6)
    j = np.cross(lit, foot)
    check("light, foot and the uprights are in one line, px",
          abs(float(j @ vert / max(float(np.hypot(j[0], j[1])), 1e-12))), 0.0, 1e-6)

    # A lamp in the room is the same construction with its foot left alone, and
    # it must not be read as the sun: the whole difference is whether that foot
    # will go to the horizon, and with shadows this short it is worth 126 px --
    # real, but only 14x the floor, not the thousands a long shadow gives.
    P = np.array([1.4, -3.4, 2.0])
    lamp = run(marks(lamp_tips(P)))
    check("a lamp is called a lamp", lamp["verdict"] + "/"
          + (lamp["value"] or {}).get("light", "-"),
          pred=lambda s: s == "consistent/a lamp, at that foot")
    check("the lamp itself comes back, px",
          float(np.linalg.norm(np.array(lamp["value"]["light_vp"]) - img(P).ravel())),
          0.0, 0.1)
    check("and the spot on the ground it stands on, px",
          float(np.linalg.norm(np.array(lamp["value"]["foot"])
                               - img(P * np.array([1, 0, 1])).ravel())), 0.0, 0.1)
    check("calling that lamp the sun would cost, px",
          lamp["conditioning"]["sun_move_px"], pred=lambda x: x > 100)

    # Hand noise is survived and a blunder is not, both at the same 9 px floor.
    for px, want in ((1.0, 4.0), (3.0, 9.0)):
        g = run(marks(tips, jitter=px))
        check(f"{px:.0f} px on every mark is still sunlight, px",
              g["conditioning"]["sun_move_px"],
              pred=lambda x, w=want, _v=g: x < w and _v["verdict"] == "consistent")
    across = marks(tips)
    k, bx = 2, np.asarray(across[2][2], float) - np.asarray(across[2][0], float)
    bx = bx / np.linalg.norm(bx)
    across[k][2] = (np.asarray(across[k][2], float)
                    + 40.0 * np.array([-bx[1], bx[0]])).tolist()
    bad = run(across)
    check("one tip pushed 40 px sideways is refused", bad["verdict"],
          pred=lambda s: s == "inconsistent")
    check("...naming the object", bad["conditioning"].get("worst"),
          pred=lambda q: q == 3)
    # and the two cheaper ways of naming it, which both point elsewhere -- the
    # reason shadow_stray fits on pairs and takes the least median
    A3 = np.asarray(across, float).reshape(-1, 3, 2)
    infit = R.shadow_light(A3, vert, hz)["free"]["move_px"]
    check("...which the worst miss in one fit gets wrong",
          int(np.argmax(infit)) + 1, pred=lambda q: q != 3)
    loo = []
    for i in range(len(A3)):
        kk = np.delete(np.arange(len(A3)), i)
        lv = V.fit(np.c_[A3[kk, 1], A3[kk, 2]])
        sv = V.fit(np.c_[A3[kk, 0], A3[kk, 2]])
        xi = np.append(A3[i, 2], 1.0)
        loo.append(max(abs(float(L @ xi)) / max(float(np.hypot(L[0], L[1])), 1e-12)
                       for L in (np.cross(lv, np.append(A3[i, 1], 1.0)),
                                 np.cross(sv, np.append(A3[i, 0], 1.0)))))
    check("...and leaving one out gets wrong too",
          int(np.argmax(loo)) + 1, pred=lambda q: q != 3)
    check("one object alone is undetermined", run(marks(tips)[:1])["verdict"],
          pred=lambda s: s == "undetermined")

    # The blind spot, measured and not papered over: a shadow pushed ALONG its
    # own direction is nearly the same picture as a higher sun, so the light
    # simply moves and takes most of it. On the longest shadow here, 40 px of
    # it reads as 8.5 px -- inside the hand's own slop. Which is the right
    # answer to the question this app asks: the least change that makes the
    # drawing consistent really is small there.
    along = marks(tips)
    d5 = np.asarray(along[4][2], float) - np.asarray(along[4][0], float)
    along[4][2] = (np.asarray(along[4][2], float)
                   + 40.0 * d5 / np.linalg.norm(d5)).tolist()
    ln = run(along)
    check("40 px ALONG the longest shadow is absorbed, px",
          ln["conditioning"]["sun_move_px"],
          pred=lambda x, _v=ln: x < 9.0 and _v["verdict"] == "consistent")

    # Why the movement and not the tip. The tip is where two lines cross, so
    # it carries 1/sin of their crossing angle; the hand's own 9 px lands 146
    # px out there, and a floor that size would pass every blunder above.
    noise = run(marks(tips, jitter=9.0))
    check("a 9 px mark is worth this much AT THE TIP, px",
          max(noise["conditioning"]["tip_px"]), pred=lambda x: x > 100)
    check("...but only this much as mark movement, px",
          noise["conditioning"]["marks_move_px"], pred=lambda x: x < 40)

    # Why the line is constructed and not scanned: one angle carries the whole
    # light, and it is read off points far from the page.
    T = np.asarray(marks(tips), float).reshape(-1, 3, 2)
    rays, grd = np.c_[T[:, 1], T[:, 2]], np.c_[T[:, 0], T[:, 2]]
    b3, t3 = np.c_[T[:, 0], np.ones(len(T))], np.c_[T[:, 1], np.ones(len(T))]
    xh = np.c_[T[:, 2], np.ones(len(T))]
    B = np.linalg.svd(vert[None, :] / np.linalg.norm(vert))[2][1:]

    def moved(l, s):
        o = [np.abs((L * xh).sum(1)) / np.maximum(np.hypot(L[:, 0], L[:, 1]), 1e-12)
             for L in (np.cross(l, t3), np.cross(s, b3))]
        return float(np.max(np.maximum(o[0], o[1])))

    grid = {}
    for n in (721, 7201):
        grid[n] = min(
            moved(V.fit_on_line(rays, np.array([np.cos(q), np.sin(q)]) @ B),
                  np.cross(np.array([np.cos(q), np.sin(q)]) @ B, hz))
            for q in np.linspace(0.0, np.pi, n, endpoint=False))
    check("a 721-step scan of that angle gets, px", grid[721], pred=lambda x: x > 50)
    check("a 7201-step scan gets, px", grid[7201], pred=lambda x: x > 10)
    check("the construction gets, px",
          R.shadow_light(T, vert, hz)["sun"]["worst_px"], pred=lambda x: x < 1e-6)


def horizon_families():
    """A third family of horizontals belongs on the horizon the first two give.

    Horizontals all share one horizon, whichever way they run, so a family
    added later is over-determined by the horizon as well as by its own lines.
    Holding the fit on that line is the smallest change that makes the drawing
    consistent -- which is the question this app asks -- and the payoff is
    measured here rather than assumed.
    """
    print("\na family held on the horizon  (plan: exactly on it; 253 -> 167 px to truth)")
    truth = np.array([8000.0, 2400.0])
    l = V.horizon([12000, 2400], [-500, 2400])          # the horizon, y = 2400
    mids = [(600, 3000), (1500, 3600), (900, 1800), (2200, 2800), (1200, 4200)]
    lens = [1400.0, 900.0, 1600.0, 700.0, 1100.0]

    def family(rng, sd_deg):
        """The same five lines, each marked sd_deg off true about its midpoint."""
        out = []
        for (mx, my), ln in zip(mids, lens):
            d = truth - np.array([mx, my])
            d /= np.linalg.norm(d)
            th = np.radians(rng.normal(0, sd_deg))
            d = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ d
            m = np.array([mx, my])
            out.append(np.r_[m - d * ln / 2, m + d * ln / 2])
        return np.array(out)

    s = family(np.random.default_rng(7), 0.7)
    v, vh = V.fit(s), V.fit_on_line(s, l)
    check("the held fit lands on the horizon", float(abs(l @ vh / np.linalg.norm(vh))),
          0.0, 1e-9)
    check("the free fit does not", float(abs(v[1] / v[2] - 2400.0)), 18.0, 0.5)

    # over 300 markings, not one, because one draw says nothing about a prior
    err = {"free": [], "held": []}
    for sd, want in ((0.7, (253, 167)), (2.0, (722, 469))):
        err["free"], err["held"] = [], []
        for k in range(300):
            t = family(np.random.default_rng(k), sd)
            a, b = V.fit(t), V.fit_on_line(t, l)
            err["free"].append(np.linalg.norm(a[:2] / a[2] - truth))
            err["held"].append(np.linalg.norm(b[:2] / b[2] - truth))
        check(f"{sd} deg marking, free fit misses by", float(np.median(err["free"])),
              float(want[0]), 2.0)
        check(f"{sd} deg marking, held on the horizon", float(np.median(err["held"])),
              float(want[1]), 2.0)

    # tidying onto the horizon: within the hand, and a fixed point
    from core import consensus as K
    mark = K._mark_px((3756, 5406))[0]
    out, moved, worst = V.tidy(s, V.fit(s), mark, line=l)
    check("every line is tidied onto the horizon", len(moved), 5, 0)
    check("the swing it applies", worst, 3.6, 0.1)
    check("which is inside a hand mark", (worst, mark), pred=lambda v: v[0] <= v[1])
    check("and is the displacement it really applied", worst,
          float(np.abs(out - s).max()), 0.05)
    w = V.fit(out)
    check("and the family's own fit now sits on it",
          float(abs(w[1] / w[2] - 2400.0)), 0.0, 1e-6)
    t = out.copy()
    for _ in range(5):
        t, _, _ = V.tidy(t, V.fit(t), mark, line=l)
    check("five more passes change nothing", float(np.abs(t - out).max()), 0.0, 1e-9)


def pdf_export():
    """The marks as a vector PDF: real geometry, real text, and no hidden layers.

    Checked by reading the PDF back, because "it opened" is not a measurement:
    an overlay rasterised into the page, or text turned into outlines, looks
    identical on screen and is useless in Illustrator.
    """
    print("\nvector PDF  (plan: stroked paths in the layer colours, text as text)")
    import re
    import zlib
    from core import export as EX

    doc = {
        "w": 1000, "h": 800,
        "horizon": [[4000.0, 400.0], [-3000.0, 400.0]],
        "layers": [
            {"kind": "family", "name": "a", "color": "#2ca8ff", "hidden": False,
             "lines": [[100, 300, 500, 340], [120, 600, 520, 610]],
             "pts": [], "far": [], "pairs": [], "vp": [4000.0, 400.0]},
            {"kind": "family", "name": "b", "color": "#ffd400", "hidden": True,
             "lines": [[100, 100, 400, 150]], "pts": [], "far": [], "pairs": [],
             "vp": None},
            {"kind": "surface", "name": "s", "color": "#ff6b35", "hidden": False,
             "lines": [], "pts": [], "far": [], "pairs": [], "vp": None,
             "quad": [[200, 500], [600, 520], [650, 700], [180, 680]]},
        ],
    }

    def content(raw):
        """The page's own drawing stream -- not the embedded font, which is
        bigger, and not the image, which does not inflate."""
        best = b""
        for m in re.finditer(rb"stream\r?\n", raw):
            a = m.end()
            try:
                st = zlib.decompress(raw[a:raw.find(b"endstream", a)])
            except zlib.error:
                continue
            if b" RG" in st and len(st) > len(best):
                best = st
        return best

    raw = EX.pdf(doc, None)
    check("it is a PDF", raw[:4], pred=lambda v: v == b"%PDF")
    page = content(raw)
    cols = {c.decode() for c in re.findall(rb"([\d.]+ [\d.]+ [\d.]+) RG", page)}
    blue = "0.1725490196 0.6588235294 1"
    yellow = "1 0.831372549 0"
    orange = "1 0.4196078431 0.2078431373"
    check("the visible family is stroked in its own colour", blue in cols,
          pred=lambda v: v is True)
    check("the surface too", orange in cols, pred=lambda v: v is True)
    check("the hidden layer is not drawn at all", yellow in cols,
          pred=lambda v: v is False)
    # A colour picked by hand goes out as picked -- the page lets a person
    # change a layer's colour (and the horizon's) because the palette's
    # yellow vanishes on cream paper, and a PDF that put it back would make
    # the choice a screen-only one.
    def strokes(pg):
        # the writer wraps long operator lines, so a triple can be split by a
        # newline: any whitespace, not one space
        return {tuple(round(float(x), 6) for x in c)
                for c in re.findall(rb"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+RG", pg)}

    def rgb(h):
        return tuple(round(int(h[i:i + 2], 16) / 255, 6) for i in (1, 3, 5))

    picked = strokes(content(EX.pdf(
        dict(doc, horizon_color="#101820",
             layers=[dict(doc["layers"][0], color="#7a1c20")]), None)))
    check("a layer's picked colour is the one stroked",
          rgb("#7a1c20") in picked, pred=lambda v: v is True)
    check("and the palette colour it replaced is gone",
          rgb("#2ca8ff") in picked, pred=lambda v: v is False)
    check("the horizon takes the colour it was given",
          rgb("#101820") in picked, pred=lambda v: v is True)
    check("the horizon is green when no colour is given",
          rgb("#00e676") in strokes(page), pred=lambda v: v is True)
    check("the lines are stroked paths, not an image",
          page.count(b" l\n"), pred=lambda v: v >= 3)
    check("and the rays to the vanishing point are dashed",
          len(re.findall(rb"\] [\d.]+ d", page)), pred=lambda v: v >= 3)
    check("the key is text, in the house face",
          sorted(set(re.findall(rb"/BaseFont\s*/[A-Z]+\+(\w+)", raw))),
          pred=lambda v: v == [b"ArialMT"])

    # an arch goes out as the Bezier path it is on the page, and a vault as
    # two of them with the slide drawn between
    bez = [[[200.0, 600.0], [200.0, 480.0], [500.0, 480.0], [500.0, 600.0]]]
    fz = [[[260.0, 580.0], [260.0, 480.0], [480.0, 480.0], [480.0, 580.0]]]
    ad = {"w": 1000, "h": 800, "horizon": None,
          "layers": [{"kind": "arch", "name": "v", "color": "#2ca8ff",
                      "hidden": False, "lines": [], "pts": [], "far": [],
                      "pairs": [], "vp": None, "arches": [[[200, 600], [500, 600]]],
                      "bez": [bez]}]}
    flat = content(EX.pdf(ad, None))
    ad["layers"][0]["bez"] = None                 # the polyline it replaced
    plain = content(EX.pdf(ad, None))
    ad["layers"][0]["bez"] = [bez]
    ad["layers"][0]["bez_far"] = [fz]
    ad["layers"][0]["vault"] = [[[350.0, 480.0], [370.0, 470.0]]]
    vault = content(EX.pdf(ad, None))
    check("the arch is a real Bezier in the PDF, not a polyline",
          flat.count(b" c\n") - plain.count(b" c\n"), 1, 0)
    check("a vault carries its far limb too",
          vault.count(b" c\n") - flat.count(b" c\n"), 1, 0)
    check("...with the ridge and the two springing joins between them",
          vault.count(b" l\n") - flat.count(b" l\n"), 3, 0)

    # the hidden layer again, this time shown: the only difference is its colour
    doc["layers"][1]["hidden"] = False
    shown = {c.decode() for c in
             re.findall(rb"([\d.]+ [\d.]+ [\d.]+) RG", content(EX.pdf(doc, None)))}
    check("showing it adds exactly that colour", sorted(shown - cols),
          pred=lambda v: v == [yellow])


def spaced_rows():
    """A row of posts, and what the tolerance has to be before it means anything.

    The maths: sending the vanishing point to infinity turns world-even into
    page-even, so u = 1/(t_vp - t) must come out as an arithmetic sequence.
    The trap is the tolerance. Hand marks scatter, and a 2 mm default standing
    in for 2 px called a dead-even row uneven.
    """
    print("\nspaced rows")
    f, cx, cy = 1800.0, 1200.0, 900.0
    proj = lambda P: np.array([cx + f * P[0] / P[2], cy + f * P[1] / P[2]])
    post = lambda k, extra=0.0: proj((3.0, 1.6, 6.0 + 2.0 * k + extra))
    pts = [post(k) for k in range(6)]
    vp = np.array([cx, cy])                       # the row's own vanishing point
    a, b = np.array([0.0, cy]), np.array([1.0, cy])

    r = C.spacing(pts, vp, a, b, 1.0)
    check("an even row reads even", r["value"], 0.0, 1e-6)
    check("  and says so", r["verdict"], pred=lambda v: v == "even")

    bad = list(pts)
    bad[3] = post(3, 0.20)                        # one post 20 cm out of step
    r = C.spacing(bad, vp, a, b, 1.0)
    check("20 cm out of step is caught", r["value"], 6.02, 0.05)
    check("  and named", r["conditioning"]["worst_at"], pred=lambda i: i == 3)

    # the same even row, marked by hand
    rng = np.random.default_rng(0)
    hand = [q + rng.normal(0, 4, 2) for q in pts]
    r = C.spacing(hand, vp, a, b, 1.0, resolve_mm=2.0)
    check("even row, +-4 px by hand, misses 2 mm", r["value"], 6.23, 0.05)
    check("  so the bare 2 mm default is wrong", r["verdict"],
          pred=lambda v: v == "uneven")
    r = C.spacing(hand, vp, a, b, 1.0, resolve_mm=2.0, mark_px=13.0)
    check("  the hand floor passes it", r["verdict"], pred=lambda v: v == "even")
    check("  and the floor is reported", r["conditioning"]["tolerance_mm"], 13.0, 1e-9)
    check("  without hiding what was asked for",
          r["conditioning"]["resolve_mm"], 2.0, 1e-9)

    # the floor is in millimetres, so it has to go through the scale
    r = C.spacing(hand, vp, a, b, 2.0, resolve_mm=0.1, mark_px=13.0)
    check("the floor scales with px per mm",
          r["conditioning"]["tolerance_mm"], 6.5, 1e-9)

    check("three posts refuse", C.spacing(pts[:3], vp, a, b, 1.0)["verdict"],
          pred=lambda v: v == "undetermined")

    # A vanishing point lies ON the image of every line running to it, so the
    # family named has to be the one the row itself runs along. Naming the
    # cross direction is the easy mistake, and it used to answer with a number.
    other = np.array([-5200.0, cy])                # the other horizontal family
    r = C.spacing(pts, other, a, b, 1.0, mark_px=13.0)
    check("the wrong family refuses", r["verdict"],
          pred=lambda v: v == "undetermined")
    check("  and says how far off the row",
          r["conditioning"]["vp_off_row_px"], pred=lambda x: x > 13.0)
    check("  the right one still answers",
          C.spacing(pts, vp, a, b, 1.0, mark_px=13.0)["verdict"],
          pred=lambda v: v == "even")
    check("  and hand scatter does not trip it",
          C.spacing(hand, vp, a, b, 1.0, mark_px=13.0)["verdict"],
          pred=lambda v: v == "even")

    bent = [q + np.array([0.0, 40.0 * (k % 2)]) for k, q in enumerate(pts)]
    check("a row that is not straight refuses",
          C.spacing(bent, vp, a, b, 1.0, mark_px=13.0)["verdict"],
          pred=lambda v: v == "undetermined")

    # tidying the row: the hand absorbed, the drawing left alone
    out, moved, worst, stray = R.tidy_row(hand, vp, 13.0)
    check("hand wobble is absorbed", len(moved), 6, 0)
    check("  the move applied stays under a mark", worst, pred=lambda x: x <= 13.0)
    # not dead zero: the tidy aims at a median fit and the check reads a
    # least-squares one, which on this row disagree by a tenth of a pixel
    check("  and the row comes out even",
          C.spacing(out, vp, a, b, 1.0, resolve_mm=0.01)["value"], 0.0, 0.2)
    again, moved2, worst2, _ = R.tidy_row(out, vp, 13.0)
    check("  tidying again moves nothing",
          float(np.max(np.abs(np.asarray(again) - np.asarray(out)))), 0.0, 1e-9)
    check("  so it is a fixed point", worst2, 0.0, 1e-9)

    # one post really out of place: that is the drawing talking
    drawn = [q.copy() for q in pts]
    drawn[4] = post(4, 0.9)
    out, moved, worst, stray = R.tidy_row(drawn, vp, 13.0)
    check("a post the others cannot predict is a stray", stray,
          pred=lambda s: s == [4])
    check("  and is left exactly as marked",
          float(np.hypot(*(np.asarray(out[4]) - np.asarray(drawn[4])))), 0.0, 1e-12)

    check("three posts tidy to nothing", R.tidy_row(pts[:3], vp, 13.0)[1],
          pred=lambda m: m == [])


def straightening():
    """Undoing a plane's foreshortening, measured on shapes with known truth.

    A drawing cannot be judged in perspective: a circle is an oval and a
    square a trapezium. The test is not that the warp looks right but that a
    shape of known shape comes back as itself.
    """
    print("\nstraightening")
    K, Rot = camera(f=1400.0, pp=(800.0, 600.0), rx=-0.22, ry=0.55, rz=0.0)
    t = np.array([0.0, 0.0, 8.0])
    proj = lambda P: (lambda q: q[:2] / q[2])(K @ (Rot @ np.asarray(P, float) + t))
    vpd = lambda d: (lambda q: q[:2] / q[2])(K @ (Rot @ np.asarray(d, float)))
    vx, vy = vpd([1, 0, 0]), vpd([0, 1, 0])

    sq = np.array([proj(P) for P in ([0, 0, 0], [1.2, 0, 0], [1.2, 1.2, 0], [0, 1.2, 0])])
    sides = lambda q: [float(np.hypot(*(q[(i + 1) % 4] - q[i]))) for i in range(4)]
    def corners(q):
        out = []
        for i in range(4):
            u, v = q[(i - 1) % 4] - q[i], q[(i + 1) % 4] - q[i]
            out.append(np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))))
        return out

    s = sides(sq)
    check("in the picture the square is not one",
          float(np.ptp(s) / np.mean(s)), pred=lambda x: x > 0.10)
    q = RC._apply(RC.homography(K, vx, vy, vp_up=vy), sq)
    s = sides(q)
    check("straightened, its sides are equal",
          float(np.ptp(s) / np.mean(s)), 0.0, 1e-9)
    check("  and its corners are square",
          float(np.max(np.abs(np.array(corners(q)) - 90.0))), 0.0, 1e-6)

    # the two directions a surface is spanned by need not be perpendicular
    va, vb = vpd([1, 0, 0]), vpd([0.6, 1, 0])
    q = RC._apply(RC.homography(K, va, vb, vp_up=vb), sq)
    s = sides(q)
    check("a surface spanned askew still gives true shape",
          float(np.ptp(s) / np.mean(s)), 0.0, 1e-9)
    check("  corners still square",
          float(np.max(np.abs(np.array(corners(q)) - 90.0))), 0.0, 1e-6)

    th = np.linspace(0, 2 * np.pi, 41)[:-1]
    ci = np.array([proj([2.4 + 0.5 * np.cos(a), 0.6 + 0.5 * np.sin(a), 0]) for a in th])
    r = np.hypot(*(ci - ci.mean(0)).T)
    check("in the picture the circle is an oval",
          float(np.ptp(r) / r.mean()), pred=lambda x: x > 0.05)
    q = RC._apply(RC.homography(K, vx, vy, vp_up=vy), ci)
    r = np.hypot(*(q - q.mean(0)).T)
    check("straightened, it is a circle", float(np.ptp(r) / r.mean()), 0.0, 1e-9)

    # a vanishing point names a direction but not which way along it
    check("upside down is detected",
          RC.upright_error(RC.homography(K, va, vb, vp_up=vb), sq),
          pred=lambda x: x is True)
    check("  and turning the up over fixes it",
          RC.upright_error(RC.homography(K, va, vb, vp_up=-np.asarray(vb)), sq),
          pred=lambda x: x is False)

    out, M = RC.straighten(np.zeros((1200, 1600, 3), np.uint8), K, sq, vx, vy, vp_up=vy)
    check("the rendered view is right way up", RC.upright_error(M, sq),
          pred=lambda x: x is False)
    check("  and fits the marks with a margin",
          float(np.ptp(RC._apply(M, sq)[:, 1]) / out.shape[0]), 0.89, 0.03)


def arches():
    print("\narch  (plan: rise/span 0.5000 +-0.010; crown 0.0000 +-0.006;"
          " springing 0.0000 +-0.004; depth 0.0000 px, 5.6 px under noise)")
    K, Rot = camera()
    e1, e2, n = np.array([1., 0, 0]), np.array([0, 0, 1.]), np.array([0, 1., 0])
    # Close enough that the arch spans ~700 px on the page: the crown and
    # springing tolerances below are fractions of span, so a 250 px arch would
    # fail them on mark noise alone and prove nothing.
    t = Rot @ np.array([0, 3.0, 0])
    vert, wall, norm = (vp_of(K, Rot, d) for d in (e2, e1, n))
    # 181 samples over 180 deg puts a mark exactly on the crown; with the crown
    # unsampled, `arch` reports the nearest mark and reads 0.0057 off centre.
    uv = arcpts(180, 1.0, 181, r=1.0)                 # a true semicircle on the wall
    near = project(K, Rot, t, uv, e1, e2)
    a = R.arch(near, K=K, vert_vp=vert, wall_vp=wall)
    check("rise/span, semicircle", a["rise_over_span"][0], 0.5, 1e-3)
    check("crown offset", a["crown_off_frac"][0], 0.0, 1e-3)
    check("springing level", a["springing_off_frac"][0], 0.0, 1e-3)

    rng = np.random.default_rng(5)
    got = [R.arch(near + rng.normal(0, 2.0, near.shape), K=K, vert_vp=vert, wall_vp=wall)
           for _ in range(40)]
    check("rise/span under +-2 px", float(np.std([g["rise_over_span"][0] for g in got])),
          0.0, 0.010)
    check("crown offset under +-2 px", float(np.std([g["crown_off_frac"][0] for g in got])),
          0.0, 0.006)
    check("springing under +-2 px", float(np.std([g["springing_off_frac"][0] for g in got])),
          0.0, 0.0045)

    far = project(K, Rot, t + (Rot @ n) * 0.9, uv, e1, e2)   # the same arch, slid back
    d = R.arch(near, far, K=K, vert_vp=vert, wall_vp=wall, normal_vp=norm)
    check("depth miss, exact", d["depth_rms_px"], 0.0, 1e-6)
    noisy = far + np.random.default_rng(7).normal(0, 2.0, far.shape)
    d2 = R.arch(near, noisy, K=K, vert_vp=vert, wall_vp=wall, normal_vp=norm)
    check("depth miss, +-2 px marks", d2["depth_rms_px"], 0.0, 6.0)
    # The far arc has to be the wrong SHAPE. A far arc scaled uniformly about
    # the normal axis is not wrong at all: 1.03 * (8.9 n + u e1 + v e2) is the
    # same ray as (8.641 n + u e1 + v e2), so it IS a correct arc at another
    # depth, and the homology rightly reports 0.0 px. Widening the span alone
    # is a shape no depth can explain.
    wrong = project(K, Rot, t + (Rot @ n) * 0.9, uv * np.array([1.03, 1.0]), e1, e2)
    d3 = R.arch(near, wrong, K=K, vert_vp=vert, wall_vp=wall, normal_vp=norm)
    check("depth resolves a 3 % too-wide far arc",
          (d3["depth_rms_px"], d2["depth_rms_px"]), pred=lambda t: t[0] > t[1])


def arch_build():
    print("\narch, drawn  (plan: the curve IS the projected arch, under 0.2 px;"
          " the crown over the middle of the WALL, not of the page)")
    # A wall squarely in front, six units off: the scene `arches()` uses puts
    # the wall's origin almost in the camera's own plane, which is harmless for
    # a projective check and useless here, where the page positions are the
    # thing being measured.
    K, Rot = camera()
    e1, e2 = np.array([1., 0, 0]), np.array([0, -1., 0])   # span, rise (page y is down)
    nrm = np.cross(e1, e2)
    t = np.array([0.0, 0.0, 6.0])
    vert, wall = vp_of(K, Rot, e2), vp_of(K, Rot, e1)
    l = vanish_line(K, Rot, nrm)

    def truth(span, rise, pointed, n=4001, u0=0.0):
        """The arch itself in the wall's own (u, v), as a dense polyline. Two
        circular arcs for a pointed one, one for a round one -- written out
        here from the textbook construction rather than from the code under
        test, which is the only way this row says anything."""
        s, h = span / 2.0, rise
        if pointed:
            tc, rho = (h * h - s * s) / (2 * s), (h * h + s * s) / (2 * s)
            out = []
            for sg in (+1, -1):
                c = np.array([sg * tc, 0.0])
                # atan2(-0.0, negative) is -pi, not +pi, so the span of a half
                # written straight from the endpoints can come out 303 deg the
                # wrong way round. Each half is under half a turn: say so.
                a0 = np.arctan2(0.0 - c[1], -sg * s - c[0])
                a1 = np.arctan2(h - c[1], 0.0 - c[0])
                while a1 - a0 > np.pi: a0 += 2 * np.pi
                while a0 - a1 > np.pi: a1 += 2 * np.pi
                th = np.linspace(a0, a1, n)
                out.append(np.column_stack([c[0] + rho * np.cos(th),
                                            c[1] + rho * np.sin(th)]))
            uv = np.vstack([out[0], out[1][::-1]])
        else:
            y, rho = (h * h - s * s) / (2 * h), (h * h + s * s) / (2 * h)
            thA = np.arctan2(0.0 - y, -s)
            if thA < 0: thA += 2 * np.pi        # v = y + rho sin th: the top is +pi/2
            th = np.linspace(thA, np.arctan2(-y, s), n)
            uv = np.column_stack([rho * np.cos(th), y + rho * np.sin(th)])
        return project(K, Rot, t, uv + np.array([u0, 0.0]), e1, e2)

    def apart(a, b):
        """Furthest either polyline strays from the other, in page pixels."""
        def one(p, q):
            seg = q[1:] - q[:-1]
            L2 = (seg * seg).sum(1); L2[L2 == 0] = 1e-18
            tt = np.clip(((p[:, None, :] - q[None, :-1, :]) * seg[None]).sum(2) / L2[None], 0, 1)
            foot = q[None, :-1, :] + tt[..., None] * seg[None]
            return float(np.linalg.norm(p[:, None, :] - foot, axis=2).min(1).max())
        return max(one(a, b), one(b, a))

    def feet(span, u0=0.0):
        return (project(K, Rot, t, [[u0 - span / 2, 0.0]], e1, e2)[0],
                project(K, Rot, t, [[u0 + span / 2, 0.0]], e1, e2)[0])

    span = 2.0
    A, B = feet(span)
    for name, mult, pointed in (("semicircle", 1.0, False), ("segmental", 0.45, False),
                                ("horseshoe", 1.3, False), ("equilateral", np.sqrt(3.0), True),
                                ("lancet", 2.4, True), ("blunt point", 1.15, True)):
        # the crown mark is deliberately sloppy -- off the middle and 6 % high --
        # because that is what a hand gives and the snap has to absorb it
        Cm = project(K, Rot, t, [[0.03 * span, mult * span / 2 * 1.06]], e1, e2)[0]
        g = R.plane_arch(K, l, A, B, Cm, pointed=pointed, span_vp=wall, n=400)
        check(f"{name} is the projected arch",
              apart(g["pts"], truth(span, g["rise_over_span"] * span, pointed)),
              0.0, 0.2)
    check("the feet stay on the two marks",
          max(np.linalg.norm(g["feet"][0] - A), np.linalg.norm(g["feet"][1] - B)), 0.0, 1e-6)

    # Snapping. Both are the point of the tool: the far foot is pulled onto a
    # line of the springing family, and the crown onto the upright over the
    # middle OF THE WALL.
    g = R.plane_arch(K, l, A, B + np.array([0.0, 25.0]), None, span_vp=wall)
    # onto the LINE, which is the claim -- not back to where the true foot was.
    # A mark 25 px off the line still says something about how far along it the
    # arch reaches, and that part of it is kept.
    la = np.cross(np.append(A, 1.0), np.append(wall, 1.0))
    check("a foot 25 px off the springing line is pulled onto it",
          float(abs(la @ np.append(g["feet"][1], 1.0)) / np.hypot(la[0], la[1])),
          0.0, 1e-6)
    g0 = R.plane_arch(K, l, A, B, None, span_vp=wall)
    mid = project(K, Rot, t, [[0.0, 0.0]], e1, e2)[0]        # the middle, on the wall
    up = np.append(vert, 1.0)
    lu = np.cross(np.append(mid, 1.0), up)
    check("the crown sits on the upright over that middle",
          float(abs(lu @ np.append(g0["crown"], 1.0)) / np.hypot(lu[0], lu[1])), 0.0, 1e-6)
    check("...which is not halfway between the feet on the page",
          float(np.linalg.norm((A + B) / 2 - mid) / np.linalg.norm(B - A)),
          pred=lambda d: d > 0.02)
    check("no crown mark means a semicircle", g0["rise_over_span"], 0.5, 1e-12)

    # Sharp or blunt is the whole difference between the two arches.
    turns = {}
    Cm = project(K, Rot, t, [[0.0, 0.75 * span]], e1, e2)[0]
    for p in (False, True):
        q = R.plane_arch(K, l, A, B, Cm, pointed=p, span_vp=wall, n=4001)["pts"]
        i = len(q) // 2
        v1, v2 = q[i] - q[i - 1], q[i + 1] - q[i]
        turns[p] = float(np.degrees(np.arccos(np.clip(
            v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
    check("a round arch is blunt at the top", turns[False], 0.0, 0.1)
    check("a pointed one is sharp there", turns[True], pred=lambda a: a > 15.0)

    # A single cubic over a whole semicircle is 1.8 % of the radius out, which
    # is why each half is its own curve. This is the measurement that choice
    # rests on, so it is a row and not a comment.
    def bez_err(deg):
        th = np.radians(deg)
        b = R._bez_arc(np.zeros(2), np.array([1.0, 0.0]),
                       np.array([np.cos(th), np.sin(th)]), 2001)
        return float(np.abs(np.hypot(*b.T) - 1).max())
    check("one cubic, a quarter turn", bez_err(90), 0.0, 3e-4)
    check("one cubic, the 105 deg a horseshoe's half needs", bez_err(105), 0.0, 8e-4)
    check("one cubic, a whole semicircle -- why it is halved", bez_err(180),
          pred=lambda e: e > 0.015)

    # Refusals.
    check("the two feet in one place", R.plane_arch(K, l, A, A, None, span_vp=wall),
          pred=lambda g: g is None)
    check("a crown on the springing line",
          R.plane_arch(K, l, A, B, (A + B) / 2, span_vp=wall), pred=lambda g: g is None)
    # Past the springing family's own vanishing point the wall has turned away;
    # there is no arch to draw there, only half of one behind the viewer.
    check("a span reaching the wall's vanishing line",
          R.plane_arch(K, l, A, A + (wall - A) * 1.02, None, span_vp=wall),
          pred=lambda g: g is None)


def polygons():
    print("\npolygon  (plan: true n beats runner-up 4-6x; needs about n-1 corners)")
    K, Rot = camera()
    e1, e2 = np.array([1., 0, 0]), np.array([0, 1., 0])
    t = Rot @ np.array([0, 0, -7.0])
    for n, want in ((5, 4.0), (6, 4.0), (8, 4.0)):
        th = 2 * np.pi * np.arange(n) / n
        p = project(K, Rot, t, np.column_stack([np.cos(th), np.sin(th)]), e1, e2)
        ranked = R.polygon_which(p)
        best, second = ranked[0], ranked[1]
        check(f"{n}-gon identified", best[0], pred=lambda g, n=n: g == n)
        check(f"{n}-gon margin over runner-up", second[1] / max(best[1], 1e-9),
              pred=lambda g, w=want: g >= w)
    # the stated limit: partial marking of a big polygon is genuinely ambiguous
    th = 2 * np.pi * np.arange(5) / 8
    p = project(K, Rot, t, np.column_stack([np.cos(th), np.sin(th)]), e1, e2)
    got = C.polygon(p, idx=range(5))
    check("5 corners of an octagon -> undetermined", got["verdict"],
          pred=lambda v: v == "undetermined")


def turning_and_threshold():
    print("\nturning angle and the shape gate  (measured this session)")
    for sweep in (20, 45, 60, 90, 120, 165, 180):
        p = arcpts(sweep, 1.0, 60, r=140)
        check(f"turning, clean {sweep:3d} deg", C.turning_deg(p), float(sweep), 2.5)
    rng = np.random.default_rng(11)
    for sweep in (30, 60, 90, 150):
        got = [C.turning_deg(arcpts(sweep, 1.0, 60, r=140) + rng.normal(0, 2, (60, 2)))
               for _ in range(60)]
        check(f"turning, +-2 px {sweep:3d} deg", float(np.median(got)), float(sweep), 12.0)
    # the threshold itself: below it the fit is often not even an ellipse
    bad = 0
    for _ in range(80):
        p = arcpts(60, 1.25, 60, r=140, start=rng.uniform(0, 360)) + rng.normal(0, 2, (60, 2))
        if C._axis_ratio(p) is None:
            bad += 1
    check("60 deg arcs mostly fail to fit an ellipse", bad / 80, pred=lambda r: r > 0.5)
    good = [C._axis_ratio(arcpts(150, 1.25, 60, r=140, start=rng.uniform(0, 360))
                          + rng.normal(0, 2, (60, 2))) for _ in range(80)]
    good = [g for g in good if g is not None]
    check("150 deg arcs recover a 1.25 ratio", float(np.median(good)), 1.25, 0.06)


def conditioning_refuses():
    print("\nthe design rule: ill-posed input must come back undetermined")
    K, Rot = camera()
    e1, e2, n = np.array([1., 0, 0]), np.array([0, 0, 1.]), np.array([0, 1., 0])
    l = vanish_line(K, Rot, n)
    t = Rot @ np.array([0, 6.0, 0])
    rng = np.random.default_rng(13)
    shallow = project(K, Rot, t, arcpts(50, 1.0, 60, r=1.0), e1, e2)
    shallow = shallow + rng.normal(0, 2.0, shallow.shape)
    check("50 deg arc -> undetermined", C.circle(shallow, K, l)["verdict"],
          pred=lambda v: v == "undetermined")
    full = project(K, Rot, t, arcpts(360, 1 / 1.25, 200, r=1.0), e1, e2)
    got = C.circle(full + rng.normal(0, 1.0, full.shape), K, l)
    check("full oval -> oval, not undetermined", got["verdict"], pred=lambda v: v == "oval")
    check("  and its value", got["value"], 1.25, 0.05)
    trueC = project(K, Rot, t, arcpts(360, 1.0, 200, r=1.0), e1, e2)
    check("full circle -> circle", C.circle(trueC, K, l)["verdict"],
          pred=lambda v: v == "circle")
    # a straight mark is not an arc
    line = np.column_stack([np.linspace(100, 700, 60), np.linspace(300, 500, 60)])
    check("a straight mark -> undetermined", C.circle(line, K, l)["verdict"],
          pred=lambda v: v == "undetermined")


def short_segment_guard():
    """Bug 9's guard, which has to hold in both axes and at any scan size."""
    print("\nthe bug-9 short-segment guard")
    w, h = 884, 1280                       # testdata/4.jpg
    # The real case that exposed it: detection's near-vertical family on 4.jpg,
    # 54 segments of median 58 px pointing at a vanishing point 59 000 px ABOVE
    # the picture. The old x-only test read x=323 as "inside" and said nothing.
    check("vertical family, short segments -> warned",
          V.short_for_vp(58.0, (323.0, -59397.0), w, h), pred=lambda v: v is True)
    check("  the x-only test it replaces saw nothing wrong",
          0 <= 323.0 <= 4000, pred=lambda v: v is True)
    # the other family on the same picture is nearer and must stay unflagged
    check("nearby vanishing point -> not warned",
          V.short_for_vp(52.0, (3364.0, 839.0), w, h), pred=lambda v: v is False)
    # long segments locate even a distant point
    check("long segments, distant vp -> not warned",
          V.short_for_vp(300.0, (323.0, -59397.0), w, h), pred=lambda v: v is False)
    # and the same picture scanned larger must get the same answer, or the
    # threshold is a pixel count again
    for k in (2.0, 4.0, 0.5):
        check(f"  same verdict at {k}x scan",
              V.short_for_vp(58.0 * k, (323.0 * k, -59397.0 * k), w * k, h * k),
              pred=lambda v: v is True)
    check("parallel family -> no distance to judge",
          V.short_for_vp(20.0, None, w, h), pred=lambda v: v is False)


def detection():
    """Detect edges: LSD's pieces joined, families by length, merges that agree."""
    print("\nedge detection")
    import json
    import re
    import app as A
    rng = np.random.default_rng(1)
    a, b = np.array([100.0, 300.0]), np.array([700.0, 330.0])
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    piece = lambda t0, t1, o=(0, 0): np.r_[a + t0 * u + o[0] * n, a + t1 * u + o[1] * n]
    # one drawn line in 12 pieces of 45 px, each end up to 0.3 px off it,
    # the way LSD leaves a pencil line
    s = np.array([piece(t, t + 45, rng.uniform(-0.3, 0.3, 2)) for t in np.arange(0, 600, 50.0)])
    j = V._join(s, 15)
    check("a line LSD broke in 12 is joined whole", len(j), 1, 0)
    check("  end to end, px", float(V._len(j).max()), 595.0, 1.0)
    check("the two edges of a stroke, 3.5 px apart, stay two",
          len(V._join(np.array([piece(0, 300), piece(300, 0, (3.5, 3.5))]), 15)), 2, 0)
    check("a gap wider than allowed stays open",
          len(V._join(np.array([piece(0, 200), piece(240, 440)]), 15)), 2, 0)
    # 40 pieces round an 8000 px radius: each turns 0.36 deg from the last,
    # well inside the 2 deg a join allows, and the arc is 62 px off straight.
    # A line within 1 px of it runs 358 px at most; with no straightness test
    # the joins ran to 545
    r = 8000.0
    P = lambda th: np.array([r * np.sin(th), r - r * np.cos(th)])
    arc = np.array([np.r_[P(t), P(t + 45 / r)] for t in np.arange(40) * 50 / r])
    check("an arc is not walked round into one line, px",
          float(V._len(V._join(arc, 15)).max()), pred=lambda L: L < 358)

    def toward(vp, k, x0, x1):
        m = np.c_[rng.uniform(x0, x1, k), rng.uniform(100, 1200, k)]
        d = np.asarray(vp, float) - m
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return np.c_[m - 40 * d, m + 40 * d]

    def two(A, B):
        s = np.r_[A, B]
        m = np.arange(len(s)) < len(A)
        return s, [(V.fit(s, np.flatnonzero(m)), m), (V.fit(s, np.flatnonzero(~m)), ~m)]
    size = (1280, 884)                     # testdata/4.jpg, and its families
    s, f = two(toward((405, -49893), 68, 50, 850), toward((-62, -4718), 23, 100, 400))
    check("curtain folds 15 deg off the verticals: kept apart", len(V.merge(s, f, size)), 2, 0)
    check("  the pooled median alone would have taken them",
          float(np.median(V.resid_deg(s, V.fit(s)))), pred=lambda m: m < 1.5)
    s, f = two(toward((4173, 951), 43, 50, 850), toward((4223, 786), 19, 50, 850))
    check("one family RANSAC split, 2.4 deg apart: pooled", len(V.merge(s, f, size)), 1, 0)

    def ray(K, v):
        v = np.linalg.solve(K, v)
        return v / np.linalg.norm(v)

    def off(K, f, v):
        """Degrees between an offered family and a vanishing point known otherwise."""
        h = np.r_[f["vp"], 1.0] if f["vp"] is not None else np.r_[f["direction"], 0.0]
        return float(np.degrees(np.arccos(min(1.0, abs(ray(K, h) @ ray(K, v))))))
    with A.app.test_client() as cl:
        def ask(name, **settings):
            return cl.post("/api/detect", json={"image": "testdata/" + name, **settings})
        fams = ask("shadowtest.png").get_json()["families"]
        Ks = np.array([[1400, 0, 600], [0, 1400, 450], [0, 0, 1.0]])
        marks = json.load(open(A.HERE / "testdata" / "shadowtest.marks.json"))
        for L in marks["layers"]:
            if L["kind"] == "family":
                # the uprights, ten pole edges, were never offered before
                check(f"shadowtest.png: {L['name']} offered, deg off",
                      min(off(Ks, f, V.fit(np.array(L["lines"], float))) for f in fams),
                      pred=lambda e: e < 0.6)
        sun = np.array([1498.4, 335.1, 1.0])          # the scene's sun, at its foot
        check("  its shadows, 0.58 diagonals of line, are not",
              min(off(Ks, f, sun) for f in fams), pred=lambda e: e > 2)
        fams = ask("4.jpg").get_json()["families"]
        K4 = np.array([[1250, 0, 442], [0, 1250, 640], [0, 0, 1.0]])
        right = np.array([3635, 858, 1.0])
        check("4.jpg: verticals, the curtain kept out, deg off",
              min(off(K4, f, np.array([287, -69267, 1.0])) for f in fams), pred=lambda e: e < 0.6)
        check("4.jpg: the right wall offered once, not as halves",
              sum(off(K4, f, right) < 5 for f in fams), 1, 0)
        # the page's detection settings, each of which has to reach the detector
        check("merge within 0 deg: the right wall comes back as its halves",
              sum(off(K4, f, right) < 5 for f in ask("4.jpg", merge_deg=0).get_json()["families"]), 2, 0)
        check("least line 0.5: shadowtest's shadows offered too, deg off",
              min(off(Ks, f, sun) for f in ask("shadowtest.png", floor=0.5).get_json()["families"]),
              pred=lambda e: e < 2)
        fams = ask("4.jpg", tol=0.5).get_json()["families"]
        check("spread 0.5 deg: 4.jpg's verticals from fewer lines, 68 at 1.5",
              min(fams, key=lambda f: off(K4, f, np.array([287, -69267, 1.0])))["n"], pred=lambda n: n < 50)
        check("join gaps 0: LSD's own 230 segments, as before joining",
              len(ask("shadowtest.png", join_pct=0).get_json()["segments"]), 230, 0)
        check("a setting that is not a number is refused",
              ask("4.jpg", tol="wide").status_code, 400, 0)
        # the page's presets, read from the page: the first is the defaults a
        # blank field falls back to, and each does what its name says
        page = (A.HERE / "static" / "index.html").read_text()
        box = page.split('id="detPreset"')[1].split("</select>")[0]
        presets = {name: dict(zip(("tol", "merge_deg", "floor", "join_pct"), map(float, v.split(","))))
                   for v, name in re.findall(r'<option value="([\d.,]+)"[^>]*>([^<]+)<', box)}
        check("preset freehand is the fields' own defaults", list(presets["freehand"].values()),
              [float(re.search(f'id="{q}"[^>]*value="([\\d.]+)"', page)[1])
               for q in ("detTol", "detMerge", "detFloor", "detJoin")], 0)
        fams = ask("shadowtest.png", **presets["ruled or rendered"]).get_json()["families"]
        check("preset ruled or rendered: shadowtest's families, worst deg off",
              max(min(off(Ks, f, V.fit(np.array(L["lines"], float))) for f in fams)
                  for L in marks["layers"] if L["kind"] == "family"), pred=lambda e: e < 0.6)
        fams = ask("shadowtest.png", **presets["sparse"]).get_json()["families"]
        check("preset sparse: shadowtest's shadows offered, deg off",
              min(off(Ks, f, sun) for f in fams), pred=lambda e: e < 2)


def fixture_4jpg():
    """The chandelier room, from the point sets kept in testdata/."""
    print("\ntestdata/4.jpg regression fixture")
    import pickle
    from pathlib import Path
    here = Path(__file__).resolve().parent / "testdata"
    if not (here / "arcs4.pkl").exists():
        print("  --    testdata missing, skipped")
        return
    arcs = [np.asarray(a, float) for a in pickle.load(open(here / "arcs4.pkl", "rb"))]
    check("four window arcs present", len(arcs), pred=lambda n: n == 4)
    turns = [C.turning_deg(a) for a in arcs]
    # The plan recorded 27-55 deg and concluded "shape undetermined for all four".
    # Re-measured from these point sets with the debiased estimator, and checked
    # by rendering the fits over the paint, they turn 138-171 deg.
    check("window arcs turn 130-175 deg", (min(turns), max(turns)),
          pred=lambda t: 130 <= t[0] and t[1] <= 180)
    left = np.load(here / "purple_left.npy")
    ridge = np.load(here / "vault_ridge.npy")
    check("left vault limb fits to 0.52 px", R.conic_resid(ridge), 0.52, 0.03)
    check("hand-marked left limb fits under 1 px", R.conic_resid(left),
          pred=lambda r: r < 1.0)
    # bug 10: the apex is an intersection, not a mark. The cornice line crosses
    # the left limb's arc at the recorded apex.
    cl = np.load(here / "cornice_line.npy")
    check("cornice line passes the apex y at x=412", (-cl[2] - cl[0] * 412) / cl[1],
          141.0, 1.0)
    # the right-hand mark is a straight edge, not an arc limb
    lwr = np.load(here / "purple_lwr.npy")
    cr, lr = C.conic_beats_line(lwr)
    check("purple_lwr is a straight edge, not a limb", (cr, lr),
          pred=lambda t: t[1] < t[0])
    # f, from the vanishing points as they currently stand
    Kr = R.calibrate(np.array([287.0, -69267.0]), np.array([3486.0, 844.0]),
                     np.array([-18.0, 844.0]))
    check("f from the current VPs", float(Kr[0, 0]), 988.0, 5.0)
    Ks, span = C.f_range(np.array([287.0, -69267.0]), np.array([3486.0, 844.0]),
                         np.array([-18.0, 844.0]), a_range=(3000, 6000))
    check("f spans 910-1320 over the right VP's range", span,
          pred=lambda s: s[0] < 920 and s[1] > 1300)


def figure_module():
    """core/figure.py carries its own verification; run it from here too, so one
    command covers every row of section 2."""
    print("\nproportion, symmetry and stage transfer  (core/figure.py)")
    from core import figure as F
    try:
        F._demo()
        check("figure.py self-check", "passed", pred=lambda v: v == "passed")
    except AssertionError as exc:
        check("figure.py self-check", f"assertion failed: {exc}", pred=lambda v: False)


def trim_module():
    """core/trim.py's own synthetic checks: aspect and the photographing
    camera recovered from a rectangle, and a square-on shot refused."""
    print("\nkeystone recovery  (core/trim.py)")
    from core import trim as T
    try:
        T._demo()
        check("trim.py self-check", "passed", pred=lambda v: v == "passed")
    except AssertionError as exc:
        check("trim.py self-check", f"assertion failed: {exc}", pred=lambda v: False)


def trim_on_photographs():
    """Finding the sheet on the real pictures -- and, four times out of five,
    correctly not finding one."""
    print("\nsheet detection on testdata")
    import cv2
    from pathlib import Path
    from core import trim as T
    here = Path(__file__).resolve().parent / "testdata"
    if not (here / "1.jpg").exists():
        print("  --    testdata missing, skipped")
        return

    # 1.jpg is a photograph of a drawing taped to a board. All four sheet
    # edges are in frame and all four are real: 7.2, 16.5, 20.9, 13.2 times
    # the picture's own texture.
    im = cv2.imread(str(here / "1.jpg"))
    q, why = T.find_quad(im)
    check("1.jpg sheet found", why, pred=lambda v: v is None)
    if q is not None:
        frac = cv2.contourArea(q.astype(np.float32)) / float(im.shape[0] * im.shape[1])
        check("1.jpg sheet covers ~75% of the frame", frac, 0.75, 0.04)
        ev = T.side_evidence(im, q)
        check("1.jpg all four sides are real edges", ev,
              pred=lambda e: all(v is not None and v >= T.EDGE for v in e))
        ow, oh, f, wy = T.shape_of(q, im.shape[1], im.shape[0])
        check("1.jpg keystone measurable", wy, pred=lambda v: v is None)
        check("1.jpg sheet is taller than wide", ow / oh, 0.756, 0.03)

    # 3, 4 and 5 are artwork already cropped to its own edges. There is no
    # sheet in them, and the whole point of the side test is that it says so:
    # without it, 4.jpg returned a confident quadrilateral spanning the
    # painting whose sides were 1.1 to 1.4 times the texture.
    for name in ("3.jpg", "4.jpg", "5.jpg"):
        im = cv2.imread(str(here / name))
        q, why = T.find_quad(im)
        check(f"{name} refuses to invent a sheet", q, pred=lambda v: v is None)
        check(f"{name} says why", why, pred=lambda v: isinstance(v, str) and len(v) > 20)

    # 2.jpg really is a photograph with a sheet in it, and Otsu still cannot
    # find it: the beige desk is almost as bright as the paper, so the bright
    # region runs to the frame. Refusing is right; claiming the frame is not.
    im = cv2.imread(str(here / "2.jpg"))
    q, why = T.find_quad(im)
    check("2.jpg does not return the frame as the sheet", q, pred=lambda v: v is None)
    check("2.jpg points the user at the manual corners", why,
          pred=lambda v: isinstance(v, str) and "yourself" in v)
    # ...and says it about the candidate it rejected, not about the user's
    # picture: the sheet in 2.jpg really is fully in frame.
    check("2.jpg does not claim the sheet is cut off", why,
          pred=lambda v: "edge of the sheet is inside" not in v)


def perspective_modes():
    """1-, 2- and 3-point are one camera model with one, two or three free
    angles, so the test is the one that matters: draw from a camera of a known
    type, then claim each type and see what the check says. The correct claim
    must reconcile at nothing, a tighter claim must cost real pixels, and a
    looser one must refuse the focal length it cannot have measured."""
    import numpy as np
    from core import consensus as K
    print("\n1-, 2- and 3-point perspective  (core/consensus.py)")
    W, H, PP, F = 1200.0, 900.0, np.array([560.0, 470.0]), 1150.0
    ANG = {"one": (0.0, 0.0, np.radians(7)),
           "two": (np.radians(33), 0.0, np.radians(4)),
           "three": (np.radians(28), np.radians(-11), np.radians(3))}

    def drawing(mode):
        y, p, r = ANG[mode]
        par = ([PP[0], PP[1], r] if mode == "one"
               else [np.log(F), PP[0], PP[1], y, r] if mode == "two"
               else [np.log(F), PP[0], PP[1], y, p, r])
        V, _, _ = K._vps_from(par, None, mode)
        rng = np.random.default_rng(7)
        fam = {}
        for i, n in enumerate(("horizontal", "upright", "depth")):
            v, segs = V[i], []
            for _ in range(6):
                a = np.array([rng.uniform(.15 * W, .85 * W), rng.uniform(.15 * H, .85 * H)])
                d = (v[:2] / v[2] - a) if abs(v[2]) > 1e-12 else v[:2]
                d = d / np.linalg.norm(d)
                segs.append([*a, *(a + rng.uniform(90, 220) * d)])
            fam[n] = segs
        return fam

    got = {}
    for true in K.MODES:
        fam = drawing(true)
        for claim in K.MODES:
            got[(true, claim)] = K.fit(fam, image_size=(W, H), mode=claim)

    for m in K.MODES:
        c = got[(m, m)]["conditioning"]
        check(f"a true {m}-point drawing reconciles as {m}-point",
              round(c["worst_move_px"], 3), pred=lambda v: v < 0.05)

    # The claim is what is being checked, so a tighter claim than the drawing
    # must cost pixels -- otherwise declaring the type would mean nothing.
    for true, claim in (("two", "one"), ("three", "one"), ("three", "two")):
        c = got[(true, claim)]["conditioning"]
        check(f"a {true}-point drawing does NOT pass as {claim}-point",
              round(c["worst_move_px"], 2), pred=lambda v: v > 5.0)

    # One-point measures the centre of vision and cannot measure the lens.
    c = got[("one", "one")]["conditioning"]
    check("one-point recovers the centre of vision",
          round(float(np.hypot(*(np.asarray(c["centre_of_vision"]) - PP))), 2),
          pred=lambda v: v < 1.0)
    check("one-point refuses the focal length", c["focal_px"], pred=lambda v: v is None)

    # The looser claims FIT a one-point drawing exactly -- the specific model is
    # nested in the general one -- so the only honest guard is on f, which the
    # drawing does not contain. It printed 1781 and 3402 px before this.
    for claim in ("two", "three"):
        c = got[("one", claim)]["conditioning"]
        check(f"a one-point drawing claimed as {claim}-point refuses f",
              [c["focal_px"], bool(c.get("focal_refused"))],
              pred=lambda v: v[0] is None and v[1])

    # Three-point, three families: both f and the principal point are measured.
    c = got[("three", "three")]["conditioning"]
    check("three-point recovers the focal length", round(c["focal_px"], 1),
          pred=lambda v: abs(v - F) < 25.0)
    check("three-point recovers the principal point",
          round(float(np.hypot(*(np.asarray(c["principal_point"]) - PP))), 1),
          pred=lambda v: v < 15.0)

    # Two-point cannot measure where the centre of vision sits ALONG the
    # horizon, only its height. Pinning x is the assumption; y is the answer.
    c = got[("two", "two")]["conditioning"]
    check("two-point measures the horizon's height", round(c["principal_point"][1], 1),
          pred=lambda v: abs(v - PP[1]) < 15.0)
    check("two-point says it assumed the lateral position",
          [c.get("pp_lateral_assumed"), bool(c.get("focal_note"))],
          pred=lambda v: v[0] is not None and v[1])
    rg = c.get("focal_range_over_pp")
    check("and the f range that assumption buys covers the true f",
          None if not rg else [round(rg[0]), round(rg[1])],
          pred=lambda v: v is not None and v[0] <= F <= v[1])

    # A bad perspective type must be refused, not quietly defaulted.
    check("an unknown perspective type is refused",
          K.fit(drawing("two"), image_size=(W, H), mode="four")["verdict"],
          pred=lambda v: v == "undetermined")


def real_size():
    """Pixels mean nothing at the easel. Given the picture's real size, every
    *_px number the app reports gets an mm twin -- so the test is that the
    conversion is right, that it follows the edge it was declared along, and
    that the two numbers that must NOT be converted are not."""
    import app as A
    print("\nreal-world size  (app.py)")
    W, H = 1200.0, 900.0

    check("297 mm wide on a 1200 px picture",
          round(A._px_per_mm({"value": 297, "unit": "mm", "along": "width"}, W, H), 4),
          pred=lambda v: abs(v - 1200 / 297) < 1e-4)
    check("the same size in cm, m, inches and feet all agree",
          [round(A._px_per_mm({"value": v, "unit": u, "along": "width"}, W, H), 6)
           for v, u in ((254.0, "mm"), (25.4, "cm"), (0.254, "m"),
                        (10.0, "in"), (10.0 / 12.0, "ft"))],
          pred=lambda v: max(v) - min(v) < 1e-6)
    # Declaring the short edge by mistake is the one error that silently scales
    # every answer, so height must be a different number, not a synonym.
    check("declaring the height uses the height",
          round(A._px_per_mm({"value": 297, "unit": "mm", "along": "height"}, W, H), 4),
          pred=lambda v: abs(v - 900 / 297) < 1e-4)
    for bad in (None, {}, {"value": 0, "unit": "mm"}, {"value": -5, "unit": "mm"},
                {"value": "abc", "unit": "mm"}, {"value": 10, "unit": "furlong"}):
        check(f"a size of {bad!r} gives no scale",
              A._px_per_mm(bad, W, H), pred=lambda v: v is None)

    ppm = 4.0        # 4 px per mm
    got = A._add_mm({"worst_move_px": 20.0, "move_px": [4.0, 8.0],
                     "focal_px": 1200.0, "photo_focal_px": 1600.0,
                     "n_lines": 6, "at_infinity": True, "name": "x",
                     "per_family": {"L1": {"worst_px": 10.0}}}, ppm)
    check("a pixel number gets its millimetre twin", got.get("worst_move_mm"),
          pred=lambda v: v == 5.0)
    check("so does a list of them", got.get("move_mm"), pred=lambda v: v == [1.0, 2.0])
    check("and one nested inside a family",
          got["per_family"]["L1"].get("worst_mm"), pred=lambda v: v == 2.5)
    # f in the picture's own pixels is the station point, not a lens.
    check("f becomes a viewing distance, not a focal length in mm",
          [got.get("viewing_distance_mm"), "focal_mm" in got],
          pred=lambda v: v == [300.0, False])
    # ...while the PHOTOGRAPHING camera's f lives in another image entirely.
    check("the photographing camera's f is left in pixels",
          [got.get("photo_focal_px"), "photo_focal_mm" in got],
          pred=lambda v: v == [1600.0, False])
    check("a count is not a length", "n_lines_mm" in got, pred=lambda v: v is False)
    check("a boolean is not a length", got.get("at_infinity"), pred=lambda v: v is True)
    check("the scale itself is reported", got.get("mm_per_px"), pred=lambda v: v == 0.25)
    check("wrapping twice changes nothing",
          A._add_mm(A._add_mm({"worst_move_px": 20.0}, ppm), ppm),
          pred=lambda v: v == {"worst_move_px": 20.0, "worst_move_mm": 5.0,
                               "mm_per_px": 0.25})
    check("no scale leaves the readout alone",
          A._add_mm({"worst_move_px": 20.0}, None), pred=lambda v: v == {"worst_move_px": 20.0})


def scene_distance():
    """A real size in the SCENE, turned into the viewer's distance from it.
    Drawn from a camera of known f, known height above the ground, with
    1.80 m figures standing on that ground at known depths -- so every number
    below has a truth to miss."""
    import numpy as np
    from core import checks as C
    print("\nreal sizes in the scene  (core/checks.py, core/rhythm.py)")
    f, pp, EYE, L = 1100.0, (442.0, 640.0), 1650.0, 1800.0
    K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    th, ro = np.radians(-9), np.radians(2)
    D = -np.array([np.sin(ro), np.cos(ro) * np.cos(th), np.sin(th)])
    D /= np.linalg.norm(D)
    proj = lambda P: ((K @ P)[:2] / (K @ P)[2]).tolist()

    def scene(depths, size=L):
        pairs, truth = [], []
        for Z in depths:
            x = 0.25 * Z
            y = (-EYE - (x * D[0] + Z * D[2])) / D[1]
            P1 = np.array([x, y, Z])
            pairs.append([proj(P1), proj(P1 + size * D)])
            truth.append(float(np.linalg.norm(P1)))
        return pairs, truth

    pairs, truth = scene((3000.0, 8000.0, 25000.0))
    r = C.landmark(pairs, K @ D, K, L, mark_px=9.0, unit="cm")
    check("three figures give three distances", r["verdict"], pred=lambda v: v == "ok")
    got = r["conditioning"]["per_landmark"]
    check("each distance matches the truth",
          [round(e["distance"] - t / 1000.0, 4) for e, t in zip(got, truth)],
          pred=lambda v: max(abs(x) for x in v) < 0.002)
    # The eye height is the payoff for an artist: it is the horizon, in metres.
    check("and the eye's height above the ground comes back",
          [e["eye_height"] for e in got],
          pred=lambda v: all(abs(x - 1.65) < 0.005 for x in v))
    # The range is the answer as much as the number: a figure marked small on
    # the page cannot place itself, and must say so rather than guess.
    check("a figure marked small is less certain than a large one",
          [e["plus_minus_pct"] for e in got],
          pred=lambda v: v[0] < v[1] < v[2] and v[0] < 5 and v[2] > 20)
    check("and every true distance is inside its own range",
          [bool(e["range"][0] <= t / 1000.0 <= e["range"][1])
           for e, t in zip(got, truth)], pred=lambda v: all(v))
    # f is not exact either, and the distance is proportional to it.
    wide = C.landmark(pairs, K @ D, K, L, mark_px=9.0, unit="cm",
                      f_range=[1000.0, 1250.0])["conditioning"]["per_landmark"]
    check("an uncertain focal length widens every range",
          [round(w["plus_minus_pct"] - e["plus_minus_pct"], 1)
           for w, e in zip(wide, got)], pred=lambda v: all(x > 5 for x in v))
    check("but does not move the answer",
          [round(w["distance"] - e["distance"], 6) for w, e in zip(wide, got)],
          pred=lambda v: all(x == 0 for x in v))

    tiny, _ = scene((70000.0,))
    check("a figure too small to measure is refused, not guessed",
          C.landmark(tiny, K @ D, K, L, mark_px=9.0, unit="cm")["verdict"],
          pred=lambda v: v == "undetermined")

    # Imperial in, imperial out -- answering a 5.9 ft figure in metres is a
    # unit conversion the person asking did not want.
    ft, tft = scene((8000.0,), size=5.9 * 304.8)
    rf = C.landmark(ft, K @ D, K, 5.9 * 304.8, mark_px=9.0, unit="ft")
    check("a size given in feet is answered in feet",
          [rf["conditioning"]["unit"],
           round(rf["conditioning"]["per_landmark"][0]["distance"] - tft[0] / 304.8, 2)],
          pred=lambda v: v[0] == "ft" and abs(v[1]) < 0.05)

    # A direction is a line, and the uprights of a level camera have no
    # vanishing point on the page at all -- that is a point at infinity, not a
    # missing input.
    flat = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
    up = np.array([0.0, -1.0, 0.0])
    P1 = np.array([300.0, EYE, 6000.0])
    seg = [[proj(P1), proj(P1 + L * up)]]
    r2 = C.landmark(seg, np.array([0.0, -1.0, 0.0]), flat, L, mark_px=9.0, unit="cm")
    check("uprights parallel on the page still give a distance",
          [r2["verdict"],
           round(r2["conditioning"]["per_landmark"][0]["distance"]
                 - float(np.linalg.norm(P1)) / 1000.0, 3)],
          pred=lambda v: v[0] == "ok" and abs(v[1]) < 0.002)

    for why, args in (("without the camera", dict(K=None)),
                      ("without a real size", dict(size_mm=0.0)),
                      ("with nothing marked", dict(pairs=[]))):
        kw = dict(pairs=pairs, v=K @ D, K=K, size_mm=L, mark_px=9.0, unit="cm")
        kw.update(args)
        check(f"no distance {why}", C.landmark(**kw)["verdict"],
              pred=lambda v: v == "undetermined")


def stray_lines():
    """A line that does not belong to the family it was put in.

    Built on the user's own marks, where ten blue lines converge and one of
    them does not -- the case that was being reported as consistent."""
    import json
    from pathlib import Path
    from core import consensus as K

    MARK = K._mark_px((3756, 5406))[0]

    def fam(lines, mark=None):
        s = np.asarray(lines, float)
        return V.strays(s, V.fit(s), MARK if mark is None else mark)

    s = np.asarray(json.loads((Path(__file__).with_name("testdata")
                               / "stray_family.json").read_text())["lines"], float)
    move, bad = fam(s)
    check("a hand mark on this picture is worth 38 px", MARK, want=38.1, tol=0.1)
    check("the stray line is found", bad, pred=lambda v: v == [5])
    check("and it is the one asking for 46 px", move[5], want=45.7, tol=0.5)
    check("the rest stay under 28 px",
          float(np.delete(move, 5).max()), pred=lambda v: v < 28)
    # Degrees alone would also condemn line 9: 2.44 deg, but on a 409 px
    # segment that is 9 px of swing, well inside the hand that drew it.
    check("a short line off by 2.4 deg is not a stray",
          (round(float(move[8]), 1), 8 in bad), pred=lambda v: v[0] < 9 and not v[1])

    tidy = np.delete(s, 5, axis=0)
    check("a family with no stray flags none", fam(tidy)[1], pred=lambda v: v == [])

    def twist(t, i, deg):
        t = t.copy()
        m = np.array([(t[i, 0] + t[i, 2]) / 2, (t[i, 1] + t[i, 3]) / 2])
        a = np.radians(deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        t[i, :2] = m + R @ (t[i, :2] - m)
        t[i, 2:] = m + R @ (t[i, 2:] - m)
        return t

    # The whole point of leaving one out: the two longest lines capture a
    # length-weighted fit, so in-fit they scored 3.2 and 3.7 times the median
    # and went unflagged. Lengths 2548 and 2278 px, indices 0 and 5 of `tidy`.
    for i in (0, 5):
        check(f"a 4 deg twist on the longest lines is caught (line {i + 1})",
              fam(twist(tidy, i, 4.0))[1], pred=lambda v, i=i: v == [i])

    # ...and the floor the hand-mark condition sets. A 4 deg twist on a
    # 1030 px line is 5.1 times the median, but only 38 px of swing on a
    # picture where the hand itself is worth 38, so it is not called.
    check("a twist the hand could account for is not called",
          fam(twist(tidy, 8, -4.0))[1], pred=lambda v: v == [])
    check("...until the hand is steady enough to tell it apart",
          fam(twist(tidy, 8, -4.0), mark=3.8)[1], pred=lambda v: v == [8])

    check("two lines never give a stray", fam(s[:2])[1], pred=lambda v: v == [])

    # Absorbing the hand's wobble: only what is within a hand mark moves, and
    # the stray is left exactly where it was drawn.
    out, moved, worst = V.tidy(s, V.fit(s), MARK)
    check("the tidy absorbs nine of ten lines", len(moved), want=9, tol=0)
    check("and never touches the stray", 5 in moved, pred=lambda v: v is False)
    # The swing reported is the one applied. It used to be the leave-one-out
    # figure that CHOSE the line, which is a different number -- the page said
    # 27.9 px for a tidy that moved nothing further than 10.4.
    check("the swing it applies", round(worst, 1), 10.4, 0.05)
    check("no line moves further than a hand mark",
          (round(worst, 1), float(np.abs(out - s).max())),
          pred=lambda v: v[0] <= MARK and v[1] <= MARK)
    check("and the leave-one-out figure that chose them was larger",
          round(float(V.strays(s, V.fit(s), MARK)[0][moved].max()), 1), 27.9, 0.05)
    check("the family it leaves is tighter",
          round(float(np.delete(V.strays(out, V.fit(out), MARK)[0], 5).max()), 1),
          pred=lambda v: v < 14)
    check("the stray survives the tidy",
          V.strays(out, V.fit(out), MARK)[1], pred=lambda v: v == [5])

    # Tidying is a fixed point, which it only is because the target is the
    # point the NON-strays agree on. Against the whole family's own fit, five
    # passes walked the vanishing point 174 px and laundered the stray into
    # agreement.
    cur = s.copy()
    for _ in range(5):
        cur = V.tidy(cur, V.fit(cur), MARK)[0]
    v5 = V.fit(cur)
    check("five tidies land where one does",
          float(np.hypot(*(v5[:2] / v5[2] - V.fit(out)[:2] / V.fit(out)[2]))),
          pred=lambda v: v < 1.0)
    check("and the stray is still a stray",
          V.strays(cur, v5, MARK)[1], pred=lambda v: v == [5])
    check("two lines are never tidied", V.tidy(s[:2], V.fit(s[:2]), MARK)[1],
          pred=lambda v: v == [])


def ellipse_plane_default():
    """Which plane a feature is read in when none was named.

    An ellipse falls back to the ground, because it is DRAWN: the ring lands
    on the picture and a circle taken as flat when it was meant to stand in a
    wall is wrong to look at. An arc only reports a number, so it still has to
    say which plane it is in -- that is bug 3, and it stays refused.

    The plane that FACES the viewer was tried as the default on 2026-10-10 and
    is wrong. The deciding measurement is the last block here: a ring the user
    drew by hand round a stool's base, which the ground reproduces and the
    facing plane misses by nearly half.
    """
    print("\nthe plane a feature is read in  (plan: an ellipse defaults to the "
          "ground, an arc never does)")
    import app as A

    K, Rot = camera()
    along, across = np.array([1., 0, 0]), np.array([0, 0, 1.])
    up = np.array([0, 1., 0])
    doc = {"vps": {"a": vp_of(K, Rot, along).tolist(),
                   "b": vp_of(K, Rot, across).tolist(),
                   "v": vp_of(K, Rot, up).tolist()},
           "horizon_vps": ["a", "b"], "vertical_vp": "v",
           "surfaces": {"W": {"name": "the wall", "vps": ["a", "v"]}}}

    # Two vanishing lines are compared where it matters -- in pixels, at the
    # corners of the picture. The dot product of the normalized coefficient
    # vectors is no test at all: any two lines passing far from the origin are
    # both nearly (0, 0, 1), and the floor and a wall agreed to 1e-5 that way
    # while lying thousands of pixels apart.
    corners = np.array([[0., 0, 1], [1600, 0, 1], [0, 1200, 1], [1600, 1200, 1]])

    def px(l):
        return corners @ (np.asarray(l) / np.hypot(l[0], l[1]))

    def apart(l_a, l_b):
        d = np.abs(np.abs(px(l_a)) - np.abs(px(l_b)))
        return float(np.max(d))

    def shape(ring):
        """A ring's axis ratio, and the heading of its long axis."""
        q = ring - ring.mean(0)
        ev, V = np.linalg.eigh(q.T @ q / len(q))
        return float(np.sqrt(ev[1] / ev[0])), V[:, 1]

    ground, wall = vanish_line(K, Rot, up), vanish_line(K, Rot, across)
    l, why = A._vanish_line(doc, None, ground=True)
    check("no surface named gives the ground's vanishing line, in px",
          apart(l, ground), 0.0, 1e-9)
    check("and refuses nothing", why, pred=lambda w: w is None)

    l2, why2 = A._vanish_line(doc, None)
    check("the same without the ground default is refused",
          (l2, why2), pred=lambda r: r[0] is None and "no plane is assumed" in r[1])

    l3, _ = A._vanish_line(doc, "W", ground=True)
    check("a named surface still wins over the ground, in px",
          apart(l3, wall), 0.0, 1e-9)
    check("and the two planes are nowhere near each other",
          apart(l3, ground), pred=lambda v: v > 500.0)

    # the horizon needs two horizontal families; one is not enough, and the
    # refusal has to name the ground rather than a surface nobody named
    thin = dict(doc, horizon_vps=["a"])
    l4, why4 = A._vanish_line(thin, None, ground=True)
    check("one horizontal family cannot give the ground its line",
          (l4, why4), pred=lambda r: r[0] is None and r[1].startswith("the ground's"))

    # ---- the ring that decided it -------------------------------------------
    #
    # Drawn by hand in Illustrator over a Persimmon export, round the
    # base of a stool standing on the floor: an axis-aligned ellipse 27.03 x
    # 13.44 pt, centre (330.97, 511.74) on a 500 x 720 pt page, horizon 321.3.
    # A ground circle IS that ring. A circle square to the viewer is half as
    # flat, which is the whole argument in one number.
    Wp, Hp, y_h = 500.244, 720.0, 321.3
    cen, semi, drew = np.array([330.97, 511.74]), 27.029, 27.029 / 13.443
    hor = np.array([0.0, 1.0, -y_h]); hor /= np.hypot(hor[0], hor[1])

    def at(f):
        Kp = np.array([[f, 0, Wp / 2], [0, f, Hp / 2], [0, 0, 1.]])
        return R.plane_circle(Kp, hor, cen - [semi, 0], cen + [semi, 0], n=512)

    best = min(np.arange(200., 900., 0.5), key=lambda f: abs(shape(at(f))[0] - drew))
    got = at(best)
    check("the user's ring is a GROUND circle, 2.011 to 1",
          shape(got)[0], drew, 2e-3)
    check("...under an ordinary lens, not a freak one",
          float(2 * np.degrees(np.arctan(Wp / 2 / best))), pred=lambda v: 40. < v < 100.)
    check("...landing where it was drawn, in pt",
          float(np.linalg.norm(got.mean(0) - cen)), 0.0, 1.0)
    # a circle square to the LINE OF SIGHT was tried as the default and is
    # half as flat as the drawn ring. That is the whole argument, in one number.
    Kp = np.array([[best, 0, Wp / 2], [0, best, Hp / 2], [0, 0, 1.]])
    w = np.linalg.solve(Kp @ Kp.T, np.append(cen, 1.0))
    face = R.plane_circle(Kp, w / np.linalg.norm(w),
                          cen - [semi, 0], cen + [semi, 0], n=512)
    check("read as square to the LINE OF SIGHT it is nowhere near",
          shape(face)[0], pred=lambda v: v < 1.3)


def ellipse_drawn_square_on():
    """The free ring is drawn SQUARE ON, so the two dots are its long diameter.

    The user's own ranking: "the most important point is to draw the circle
    with the two points ON its longer diameter". Only the square-on
    construction does it. The circle this picture's camera casts leans away
    from the centre of vision, and its long diameter misses the dots by 4 to
    16 px -- the dots are not even on it.
    """
    print("\nthe free ring, drawn square on  (plan: the two dots are the ends "
          "of its long diameter, in any camera)")
    import app as A

    def ends_along(r, a, b):
        """How far the ring reaches past the two dots along the drag, and back
        from them -- 0 and 0 when the dots ARE the long diameter's ends."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = (b - a); half = np.linalg.norm(u) / 2.0; u = u / (2 * half)
        t = (r - (a + b) / 2.0) @ u
        return float(abs(np.max(t) - half)), float(abs(np.min(t) + half))

    def across(r, a, b):
        """Its half-width the other way: this must be the SHORTER one."""
        u = (np.asarray(b, float) - a); u = u / np.linalg.norm(u)
        return float(np.max(np.abs((r - (np.asarray(a, float) + b) / 2.0)
                                   @ [-u[1], u[0]])))

    def ratio(r):
        q = r - r.mean(0); ev = np.linalg.eigvalsh(q.T @ q / len(q))
        return float(np.sqrt(ev[1] / ev[0]))

    def axis_miss(r, a, b):
        m = r.mean(0); q = r - m
        _, V = np.linalg.eigh(q.T @ q / len(q)); t = q @ V[:, 1]
        e0, e1 = r[int(np.argmin(t))], r[int(np.argmax(t))]
        return max(min(np.linalg.norm(e0 - a), np.linalg.norm(e1 - a)),
                   min(np.linalg.norm(e0 - b), np.linalg.norm(e1 - b)))

    # a camera with roll AND pitch, because moving the principal point sideways
    # passes a level camera and fails this one: it flattened a ring reading
    # 2.26 to 1.06, having quietly asked for a different plane
    K, Rot = camera()
    l = vanish_line(K, Rot, np.array([0, 1., 0]))
    worst_true = 0.0
    for c in ([830., 820.], [300., 700.], [1200., 950.], [40., -25.]):
        c = np.array(c, float); a, b = c - [130., 0], c + [130., 0]
        r = R.square_on_circle(K, l, a, b, n=8192)
        over, under = ends_along(r, a, b)
        # exact in the ellipse; what is measured here is a polyline, and 8192
        # points round a 260 px ring is where the sampling stops showing
        check(f"({c[0]:.0f},{c[1]:.0f}): the ring ends at the dots, px",
              max(over, under), 0.0, 2e-3)
        check(f"({c[0]:.0f},{c[1]:.0f}): and that is its LONG way, not its short",
              across(r, a, b), pred=lambda v, h=130.: v <= h + 1e-9)
        worst_true = max(worst_true, axis_miss(R.plane_circle(K, l, a, b, n=2048), a, b))
    check("the circle the camera casts misses them by this much, px",
          worst_true, pred=lambda v: v > 4.0)

    # ...and the dots are not even on that one: pinning a circle to two page
    # points is four equations in three unknowns, consistent only by symmetry
    c = np.array([1200., 950.]); a, b = c - [130., 0], c + [130., 0]
    leaning = R.plane_circle(K, l, a, b, n=2048)
    check("...and the dots are not even ON it, px",
          max(float(np.min(np.linalg.norm(leaning - a, axis=1))),
              float(np.min(np.linalg.norm(leaning - b, axis=1)))),
          pred=lambda v: v > 1.0)

    # it is a real foreshortening, not a drawn oval: on the centre of vision,
    # where the picture plane adds no stretch of its own, the two agree
    K2 = np.array([[357., 0, 250.1], [0, 357., 360.], [0, 0, 1.]])
    l2 = np.array([0., 1., -321.3])
    pp = K2[:2, 2]; a, b = pp - [60., 0], pp + [60., 0]
    sq = R.square_on_circle(K2, l2, a, b, n=4096)
    tr = R.plane_circle(K2, l2, a, b, n=4096)
    check("on the centre of vision it IS the circle the camera casts",
          ratio(sq), ratio(tr), 1e-9)
    check("...the same curve, in px",
          float(np.linalg.norm(sq[:, None] - tr[None], axis=2).min(1).max()), 0.0, 1e-9)
    # and at every size: a cosine-squashed oval would drift here, by 1.3 % at
    # the middle width and more at the wide one
    for h in (20., 120.):
        aa, bb = pp - [h, 0], pp + [h, 0]
        check(f"...at half-width {h:.0f} too",
              ratio(R.square_on_circle(K2, l2, aa, bb, n=4096)),
              ratio(R.plane_circle(K2, l2, aa, bb, n=4096)), 1e-9)

    # away from it the picture plane stretches further than the eye sees, which
    # is why both are drawn rather than one
    c = np.array([1200., 950.]); a, b = c - [130., 0], c + [130., 0]
    check("far off axis the two part company, as they should",
          ratio(R.plane_circle(K, l, a, b, n=512)) / ratio(R.square_on_circle(K, l, a, b, n=512)),
          pred=lambda v: v > 1.5)

    # edge on at the horizon there is no ring to draw
    hz_pt = np.array([600., -(l[0] * 600 + l[2]) / l[1]])
    check("a ring on the horizon is refused",
          R.square_on_circle(K, l, hz_pt - [130., 0], hz_pt + [130., 0]),
          pred=lambda r: r is None)

    # the endpoint hands the page both curves, and only for a free ring
    doc = {"vps": {"a": vp_of(K, Rot, [1., 0, 0]).tolist(),
                   "b": vp_of(K, Rot, [0, 0, 1.]).tolist(),
                   "v": vp_of(K, Rot, [0, 1., 0]).tolist()},
           "horizon_vps": ["a", "b"], "vertical_vp": "v",
           "surfaces": {"W": {"name": "the wall", "vps": ["a", "v"]}}}
    seg = [700., 820., 960., 820.]
    with A.app.test_client() as cl:
        free = cl.post("/api/ellipse", json={
            "doc": dict(doc, features={"E": {"kind": "ellipse", "ring_from": seg}}),
            "feature": "E"}).get_json()
        wall = cl.post("/api/ellipse", json={
            "doc": dict(doc, features={"E": {"kind": "ellipse", "ring_from": seg,
                                             "surface": "W"}}), "feature": "E"}).get_json()
    check("a free ring comes back with both curves",
          [bool(x) for x in (free or {}).get("rings_true") or []],
          pred=lambda k: k == [True])
    check("one in a named surface comes back with one",
          [bool(x) for x in (wall or {}).get("rings_true") or []],
          pred=lambda k: k == [False])
    e = free["ends"][0]        # the drag, levelled: those are the dots now
    over, under = ends_along(np.asarray(free["rings"][0]), e[:2], e[2:])
    check("and the one to draw against ends at the dots, px",
          max(over, under), 0.0, 0.05)


def arch_one_bezier():
    """A round arch is ONE Bezier curve, and what that costs.

    "make the round arch a single bezier curve". A cubic is not a circle, so
    the curve is a drawing of the arch and not the arch; the number travels
    with it.
    """
    print("\nthe arch as a Bezier path  (plan: one curve round, one a side "
          "pointed, and the error said out loud)")
    import app as A
    K, Rot = camera()
    vert = vp_of(K, Rot, [0, 1., 0]); wall = vp_of(K, Rot, [1., 0, 0])
    l = np.cross(np.append(vert, 1.0), np.append(wall, 1.0))
    H = R.plane_frame(K, l); Hi = np.linalg.inv(H)

    def back(q):
        z = Hi @ [q[0], q[1], 1.0]; return z[:2] / z[2]

    def fwd(q):
        z = H @ [q[0], q[1], 1.0]; return z[:2] / z[2]

    feet = [(400., 900.), (600., 900.)]
    rd = R.plane_arch(K, l, *feet, None, span_vp=wall)
    pt = R.plane_arch(K, l, *feet, None, pointed=True, span_vp=wall)

    # the arch as it REALLY images, built from the circle itself. `pts` is a
    # 191-point polyline and measuring against it adds half a sample spacing
    # -- 1.4 px read as 3.6, which is the whole quantity under test
    A_, B_ = back(rd["feet"][0]), back(rd["feet"][1])
    M = (A_ + B_) / 2.0; sp = float(np.hypot(*(B_ - A_))) / 2.0
    d = (B_ - A_) / (2 * sp); up = np.array([-d[1], d[0]])
    if np.linalg.norm(fwd(M + sp * up) - rd["crown"]) > \
       np.linalg.norm(fwd(M - sp * up) - rd["crown"]):
        up = -up
    th = np.linspace(0.0, np.pi, 4001)       # from foot A, the way the curve runs
    true = np.array([fwd(M + sp * (np.cos(t) * -d + np.sin(t) * up)) for t in th])

    def off(bez):
        return R.curve_off(bez, true)
    check("a round arch is one curve", len(rd["bez"]), 1, 0)
    check("a pointed one is a curve a side", len(pt["bez"]), 2, 0)
    check("the curve starts and ends at the springing, px",
          max(float(np.linalg.norm(np.array(rd["bez"][0][0]) - rd["feet"][0])),
              float(np.linalg.norm(np.array(rd["bez"][0][3]) - rd["feet"][1]))),
          0.0, 1e-9)
    check("the pointed one's two meet at the point, px",
          float(np.linalg.norm(np.array(pt["bez"][0][3]) - pt["bez"][1][0])), 0.0, 1e-9)

    # it leaves the springing the way the real arch does, which is the one
    # thing the fit is not allowed to trade away
    u = np.array(rd["bez"][0][1]) - rd["bez"][0][0]; u /= np.linalg.norm(u)
    v = true[1] - true[0]; v /= np.linalg.norm(v)
    check("...along the arch's own tangent there, deg",
          float(abs(np.degrees(np.arctan2(abs(u[0]*v[1] - u[1]*v[0]), u @ v)))),
          0.0, 0.5)

    # the price, re-measured here rather than taken from the endpoint
    # off the SNAPPED feet: the far one is pulled onto the springing family,
    # so the mark is not where the arch stands
    rise = float(np.linalg.norm(rd["crown"]
                                - (rd["feet"][0] + rd["feet"][1]) / 2.0))
    check("one cubic over a round arch, % of rise",
          100 * off(rd["bez"]) / rise, pred=lambda v: 1.5 < v < 2.5)
    check("a curve a side over a pointed one, % of rise",
          100 * off(pt["bez"]) / rise, pred=lambda v: v < 1.0)
    check("and that is the number it reports, px",
          rd["bez_off"], off(rd["bez"]), 0.05)
    # what it replaced: the projected arc, sampled. Twenty times closer, and
    # it had no handles to take hold of -- that is the trade, in one row.
    check("the polyline it replaced was 20x closer, px",
          float(np.linalg.norm(np.asarray(rd["pts"])[:, None] - true[None],
                               axis=2).min(1).max()),
          pred=lambda v: v < off(rd["bez"]) / 10)

    # fitting ON THE PAGE is the whole reason it is only 2 %: the obvious
    # construction -- a cubic drawn in the arch's own plane, control points
    # mapped through -- is further out, and gets worse with foreshortening
    k = 4.0 / 3.0 * sp
    classic = [[fwd(q).tolist() for q in (A_, A_ + k * up, B_ + k * up, B_)]]
    check("the classic 4/3 construction, mapped, is worse",
          off(classic) / off(rd["bez"]), pred=lambda v: v > 1.4)

    check("a flat arc has no curve to fit",
          R.fit_cubic(np.zeros((20, 2)), [1., 0], [-1., 0]), pred=lambda v: v is None)

    # and every arch in a family carries its own path and its own number
    doc = {"vps": {"a": wall.tolist(), "b": vp_of(K, Rot, [0, 0, 1.]).tolist(),
                   "v": vert.tolist()},
           "horizon_vps": ["a", "b"], "vertical_vp": "v",
           "surfaces": {"W": {"name": "the wall", "vps": ["a", "v"]}}}
    ms = [[400., 900., 600., 900.], [700., 880., 860., 880.],
          [920., 866., 1050., 866.]]
    with A.app.test_client() as cl:
        r = cl.post("/api/arch", json={
            "doc": dict(doc, features={"A": {"kind": "arch", "arch_from": ms,
                                             "surface": "W", "vertical_vp": "v",
                                             "wall_vp": "a"}}),
            "feature": "A"}).get_json()
    check("three arches, three paths",
          [len(b) for b in r["bez"]], pred=lambda v: v == [1, 1, 1])
    check("...each with its own number, px",
          r["bez_off"], pred=lambda v: len(v) == 3 and all(0 < x < 6 for x in v))


def arch_point_on_the_curve():
    """One more point ON the curve, and what it buys.

    Two feet and a top leave the arch nothing to decide: the centre is forced
    onto the springing line or onto the crown's upright, and which of those it
    is was a toggle. Real arches are not all built that way -- a drop or a
    stilted arch raises its centres -- so the toggle picks the wrong circle and
    the only way back was to drag the Bezier handles, which is drawing, not
    measuring. Three points name one circle.
    """
    print("\na point on the curve  (plan: three points name one circle, so the "
          "round/pointed toggle has nothing left to decide)")
    f = 1400.0
    K = np.array([[f, 0, 600.], [0, f, 600.], [0, 0, 1.]])
    yaw = np.radians(34.0)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1., 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rot = Ry @ np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    eye = np.array([-1.5, -9.0, 1.7])

    def page(P):
        x = ((np.asarray(P, float) - eye) @ Rot.T) @ K.T
        return x[:, :2] / x[:, 2:]

    def wall(u, v):                      # the arch's plane, u across and v up
        return np.column_stack([u, np.zeros_like(u), np.asarray(v, float) + 4.0])

    s, h = 2.0, 3.4
    A2, T2 = np.array([-s, 0.]), np.array([0., h])
    mid = (A2 + T2) / 2.0
    nrm = np.array([-(T2 - A2)[1], (T2 - A2)[0]])
    nrm = nrm / np.linalg.norm(nrm)
    # the app's own centre, and it lies on the bisector of foot-to-apex like
    # every other centre that reaches both
    forced = np.array([(h * h - s * s) / (2 * s), 0.])
    k0 = float((forced - mid) @ nrm)

    def true_arch(c, n=400):
        """the arch REALLY built about c: left arc, then its mirror"""
        def arc(c, p0, p1):
            c = np.asarray(c, float)
            a0 = np.arctan2(*(np.asarray(p0, float) - c)[::-1])
            a1 = np.arctan2(*(np.asarray(p1, float) - c)[::-1])
            a1 += 2 * np.pi * (-1 if a1 - a0 > np.pi else 1 if a1 - a0 < -np.pi else 0)
            t = np.linspace(a0, a1, n)
            r = float(np.hypot(*(np.asarray(p0, float) - c)))
            return c + r * np.column_stack([np.cos(t), np.sin(t)])
        return np.vstack([arc(c, A2, T2), arc([-c[0], c[1]], T2, [s, 0.])[1:]])

    def miss(a, b):
        return float(np.linalg.norm(np.asarray(a)[:, None]
                                    - np.asarray(b)[None], axis=2).min(1).max())

    vert = page(np.array([[0., 0., 1e6]]))[0]
    rows = []
    for up_m in (0.0, 0.45, 0.90):
        P2 = true_arch(mid + (k0 + up_m) * nrm)
        im = page(wall(P2[:, 0], P2[:, 1]))
        A_, B_ = im[0], im[-1]
        top = page(wall([0.], [h]))[0]
        span = R.span_vp_of(K, vert, A_, B_)
        l = np.cross(np.append(vert, 1.0), np.append(span, 1.0))
        was = R.plane_arch(K, l, A_, B_, top, pointed=True, span_vp=span, n=240)
        mark = im[len(im) // 4]                   # one point, on the left limb
        now = R.plane_arch(K, l, A_, B_, top, pointed=True, span_vp=span, n=240,
                           through=mark)
        rows.append((up_m, miss(was["pts"], im), miss(now["pts"], im), now, mark))
        check("centres raised %.2f m: forced %5.1f px, with the mark %4.1f px"
              % (up_m, rows[-1][1], rows[-1][2]), rows[-1][2],
              pred=lambda v: v < 2.0)
    # the arch the app forces is one of the family, so nothing is lost there
    check("the forced arch is still itself, px", rows[0][1], pred=lambda v: v < 2.0)
    check("a raised centre the toggle cannot reach, px", rows[2][1],
          pred=lambda v: v > 40.0)
    check("...and the mark brings it back, x better",
          rows[2][1] / rows[2][2], pred=lambda v: v > 20.0)
    check("the curve goes THROUGH the mark, px",
          float(np.min(np.linalg.norm(rows[2][3]["pts"] - rows[2][4], axis=1))),
          0.0, 0.05)
    check("a cubic a quarter, so four", len(rows[2][3]["bez"]), 4, 0)

    # the toggle is not overruled, it is answered: both give the same circle
    P2 = true_arch(mid + k0 * nrm)
    im = page(wall(P2[:, 0], P2[:, 1]))
    A_, B_, top = im[0], im[-1], page(wall([0.], [h]))[0]
    span = R.span_vp_of(K, vert, A_, B_)
    l = np.cross(np.append(vert, 1.0), np.append(span, 1.0))
    kw = dict(span_vp=span, n=240, through=im[len(im) // 4])
    gp = R.plane_arch(K, l, A_, B_, top, pointed=True, **kw)
    gr = R.plane_arch(K, l, A_, B_, top, pointed=False, **kw)
    check("round or pointed, with the mark, px apart", miss(gp["pts"], gr["pts"]),
          0.0, 1e-6)
    # and the page puts the mark down on the curve it has drawn, so it costs
    # nothing to add one: what moves is the cubic's own error, not the arch
    for po in (True, False):
        g0 = R.plane_arch(K, l, A_, B_, top, pointed=po, span_vp=span, n=240)
        g1 = R.plane_arch(K, l, A_, B_, top, pointed=po, span_vp=span, n=240,
                          through=g0["pts"][len(g0["pts"]) // 4])
        check("%s: a mark put on its own curve moves it, px"
              % ("pointed" if po else "round  "), miss(g0["pts"], g1["pts"]),
              pred=lambda v: v < 2.0)
    # marked on the far limb it is the same mark: an arch is symmetric there
    ln = R.plane_arch(K, l, A_, B_, top, span_vp=span, n=240, through=im[200])
    rt = R.plane_arch(K, l, A_, B_, top, span_vp=span, n=240,
                      through=im[len(im) - 1 - 200])
    check("marked on either limb, px apart", miss(ln["pts"], rt["pts"]), 0.0, 0.01)
    # a roof has no circle to name
    st = R.plane_arch(K, l, A_, B_, top, span_vp=span, n=240, shape="straight")
    st2 = R.plane_arch(K, l, A_, B_, top, span_vp=span, n=240, shape="straight",
                       through=im[200])
    check("a roof ignores it, px", miss(st["pts"], st2["pts"]), 0.0, 1e-9)


def arch_handle_pairs():
    """One limb's handle moves the other, across the WALL and not the page.

    "make it so that adjusting one limb automatically adjusts the other
    symmetrically but in perspective". The arch is symmetric about its own
    upright and on the page it is not, because the near limb images wider --
    so whatever the hand does has to cross the plane. It crosses it carried,
    not reflected: "the plane of the arch should bend itself". The arch's
    own mirror is still here, because it is what says the fitted cubic is
    deliberately NOT symmetric, which is why an offset travels and a handle
    does not.
    """
    print("\nhandles in pairs  (plan: the move crosses the arch's own plane)")
    K, Rot = camera()
    vert = vp_of(K, Rot, [0, 1., 0]); wall = vp_of(K, Rot, [1., 0, 0])
    l = np.cross(np.append(vert, 1.0), np.append(wall, 1.0))
    H = R.plane_frame(K, l); Hi = np.linalg.inv(H)

    def back(q):
        z = Hi @ [q[0], q[1], 1.0]; return z[:2] / z[2]

    def fwd(q):
        z = H @ [q[0], q[1], 1.0]; return z[:2] / z[2]

    # a steep enough view that the two limbs are plainly unequal on the page
    ft = ((200., 1000.), (1150., 840.))
    rd = R.plane_arch(K, l, *ft, None, span_vp=wall)
    pt = R.plane_arch(K, l, *ft, None, pointed=True, span_vp=wall)
    mir = np.array(rd["mirror"])

    def mp(q):
        z = mir @ [float(q[0]), float(q[1]), 1.0]; return z[:2] / z[2]

    fa, fb, cr = rd["feet"][0], rd["feet"][1], rd["crown"]
    check("the mirror swaps the two feet, px",
          float(np.linalg.norm(mp(fa) - fb)), 0.0, 1e-7)
    check("...and holds the crown still, px",
          float(np.linalg.norm(mp(cr) - cr)), 0.0, 1e-7)
    check("...and twice over is where you started, px",
          float(np.linalg.norm(mp(mp(fa)) - fa)), 0.0, 1e-7)

    # the arch as it really images, to measure symmetry against
    A_, B_ = back(fa), back(fb)
    M = (A_ + B_) / 2.0; sp = float(np.hypot(*(B_ - A_))) / 2.0
    d = (B_ - A_) / (2 * sp); up = np.array([-d[1], d[0]])
    if np.linalg.norm(fwd(M + sp * up) - cr) > np.linalg.norm(fwd(M - sp * up) - cr):
        up = -up
    th = np.linspace(0.0, np.pi, 2001)
    true = np.array([fwd(M + sp * (np.cos(t) * -d + np.sin(t) * up)) for t in th])

    def onto(q):
        """How far a set of page points lands from the arch itself, px."""
        q = np.asarray(q, float)
        return float(np.linalg.norm(q[:, None] - true[None], axis=2).min(1).max())

    check("it maps the arch onto itself, px", onto([mp(q) for q in true[::8]]),
          0.0, 1e-6)

    # the page's own mirror is a different map, and this is the drawing that
    # shows it: reflect across the perpendicular bisector of the two feet
    u = (fb - fa) / np.linalg.norm(fb - fa)
    mid = (fa + fb) / 2.0

    def flat(q):
        return mid + (np.asarray(q, float) - mid) - 2 * ((q - mid) @ u) * u

    check("mirroring on the PAGE instead leaves the arch, px",
          onto([flat(q) for q in true[::8]]), pred=lambda v: v > 10)

    # The fitted curve is NOT wall-symmetric, and that is not a bug: the
    # best drawing of the arch on the page is a different curve from the
    # best wall-symmetric one, and forcing the symmetry costs real accuracy.
    # So the fit is left where the camera put it and the HAND'S MOVE is what
    # gets mirrored -- which also means nothing jumps when a handle is first
    # touched.
    P = np.array(rd["bez"][0])
    check("the best page cubic is not wall-symmetric, px",
          float(np.linalg.norm(mp(P[1]) - P[2])), pred=lambda v: v > 5)
    check("...so mirroring the handle itself would jerk it on first touch, px",
          float(np.linalg.norm(mp(P[1]) - P[2])), pred=lambda v: v > 5)

    def err(k):
        u_ = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0])
        Q = np.array([P[0], P[0] + k * u_, mp(P[0] + k * u_), P[3]])
        return R.curve_off([Q.tolist()], true)
    k0 = float(np.linalg.norm(P[1] - P[0]))
    ks = np.linspace(0.5 * k0, 1.6 * k0, 120)
    check("...and forcing it on would cost accuracy, x",
          min(err(k) for k in ks) / R.curve_off([P.tolist()], true),
          pred=lambda v: v > 1.2)

    # so: the hand moves one handle and the other follows BY THE IMAGE OF
    # THAT MOVE, carried across the arch's plane -- taken off the page where
    # the hand is and put back on where the other handle stands. Not
    # reflected: "I need a different kind of symmetry. the plane of the arch
    # should bend itself." The plane still has to be crossed, though, and the
    # page's own offset, which would move the far handle by exactly as much,
    # is not the same thing.
    Q_ = [np.array(b) for b in pt["bez"]]
    check("pointed: the mirror pairs the two anchors at the point, px",
          float(np.linalg.norm(mp(Q_[0][3]) - Q_[1][0])), 0.0, 1e-7)
    check("pointed: and the springing anchors across, px",
          float(np.linalg.norm(mp(Q_[0][0]) - Q_[1][3])), 0.0, 1e-7)

    # which limb is which: the foot NEARER the wall's vanishing point on the
    # page is the one standing further off in the world
    far_i = 0 if np.linalg.norm(fa - wall) < np.linalg.norm(fb - wall) else 1
    h_far, h_near = (P[1], P[2]) if far_i == 0 else (P[2], P[1])
    pull = np.array([26.0, -34.0])

    def carry(pl, src, dst, mv):
        """the hand's move at `src`, carried across the plane to `dst`"""
        Hp, Hp_i = (np.asarray(x, float) for x in pl)

        def f(m, q):
            z = m @ [float(q[0]), float(q[1]), 1.0]
            return z[:2] / z[2]

        return f(Hp, f(Hp_i, dst) + (f(Hp_i, src + mv) - f(Hp_i, src))) - dst

    def moved(src, dst):
        return (float(np.linalg.norm(carry(rd["plane"], src, dst, pull)))
                / float(np.linalg.norm(pull)))
    check("a pull on the near limb moves the far one by less",
          moved(h_near, h_far), pred=lambda v: 0.5 < v < 0.9)
    check("...and a pull on the far limb moves the near one by more",
          moved(h_far, h_near), pred=lambda v: v > 1.2)
    check("the page's own offset, which moves it by exactly as much, is a "
          "different answer, %",
          100.0 * abs(1.0 - moved(h_near, h_far)), pred=lambda v: v > 10)

    # And the plane has to be the one the arch is in NOW. The page used to
    # map the move once, at drag time, and store the far handle's offset: a
    # move made while the arch stood upright then kept the WALL'S answer when
    # the arch was tipped off the wall, and the far handle landed further
    # from its place than the whole move was long. On the page it looked like
    # the back limb leaning the wrong way.
    span = R.span_vp_of(K, vert, *ft)
    tip_l, tip_deg = R.lean_line(K, vert, span, *ft, (520., 240.))
    tip = R.plane_arch(K, tip_l, *ft, (520., 240.), pointed=True, span_vp=span)
    hand, far_h = np.array(tip["bez"][0])[1], np.array(tip["bez"][1])[2]
    check("tipped off the wall, deg", tip_deg, pred=lambda v: v > 30)
    check("...the wall's plane, kept, puts the far handle further off than "
          "the move is long, px",
          float(np.linalg.norm(carry(rd["plane"], hand, far_h, pull)
                               - carry(tip["plane"], hand, far_h, pull))),
          pred=lambda v: v > float(np.linalg.norm(pull)))


def arch_rib_pair_handles():
    """A vault rib bends as one piece: the plane bends, it is not reflected.

    "the farther side of the rib still moves outward, away from the closer
    side. it should move in the same direction, because it's a rib!" --
    "I need a different kind of symmetry. the plane of the arch should bend
    itself."

    Two springings a bay apart with the boss between them stand 1.4 spans
    tall in their own leaning plane (span the 7 m bay, rise the boss's
    distance from the middle of it, out over the nave and up). Reflecting
    the hand's move in that plane is the rigid arch: both limbs bulge out
    together, and the far HANDLE goes 0.5x to 6.9x the hand's move, usually
    the other way across the page. Carrying the move instead bends the
    plane: the far handle follows the hand within a few degrees at 0.56x
    from the near limb and 1.75x from the far one, and the two limbs swing
    the same way -- which is what a bend pinned at the feet and the crown
    does, one way to the top and the other way past it.
    """
    print("\na vault rib pair  (plan: the plane bends, carrying the move)")
    f = 1200.0
    K = np.array([[f, 0, 600.], [0, f, 600.], [0, 0, 1.0]])
    pit = np.radians(12.0)                       # tipped up at the vault
    Rot = (np.array([[1., 0, 0], [0, np.cos(pit), -np.sin(pit)],
                     [0, np.sin(pit), np.cos(pit)]])
           @ np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]]))
    eye = np.array([6.0, -22.0, 1.6])            # the near end of a 12 m nave

    def page(P):
        X = (np.asarray(P, float) - eye) @ Rot.T
        x = X @ K.T
        return x[:, :2] / x[:, 2:]

    def ap(m, q):
        z = np.asarray(m, float) @ np.array([q[0], q[1], 1.0])
        return z[:2] / z[2]

    vv = K @ Rot @ np.array([0., 0., 1.])
    vert = vv[:2] / vv[2]
    t = np.linspace(0.0, 1.0, 301)
    gain, ang, bent, mirrored_gain, same_side = [], [], [], [], []
    for boss, label in ((3.5, "boss 3.5 m"), (8.0, "boss 8 m"), (12.0, "boss 12 m")):
        pA, pB = page([[0., 0., 7.]])[0], page([[0., 7., 7.]])[0]
        pC = page([[6.0, 3.5, 7.0 + boss]])[0]
        span = R.span_vp_of(K, vert, pA, pB)
        l, deg = R.lean_line(K, vert, span, pA, pB, pC)
        got = R.plane_arch(K, l, pA, pB, pC, pointed=True, span_vp=span, n=400)
        segs = [np.array(b, float) for b in got["bez"]]
        Hp, Hp_i = (np.array(x, float) for x in got["plane"])
        mir = np.array(got["mirror"], float)
        n = 2 * len(segs)

        def out_of(s):
            """which way that limb bulges, on the page"""
            P = segs[s]
            q = R.bez_at(P, t)
            ch = P[3] - P[0]
            u = np.array([-ch[1], ch[0]]) / np.linalg.norm(ch)
            return u if u @ (q[len(q) // 2] - (P[0] + P[3]) / 2) > 0 else -u

        for j in range(n):
            s, p = j // 2, n - 1 - j
            c, cf = segs[s][1 + (j % 2)], segs[p // 2][1 + (p % 2)]
            hand = 28.0 * out_of(s)              # push this limb fuller
            moved = ap(Hp, ap(Hp_i, cf) + (ap(Hp_i, c + hand) - ap(Hp_i, c))) - cf
            gain.append(float(np.linalg.norm(moved)) / 28.0)
            ang.append(float(np.degrees(np.arccos(np.clip(
                moved @ hand / (np.linalg.norm(moved) * 28.0), -1.0, 1.0)))))
            bent.append(float(moved @ out_of(p // 2)))
            mv = ap(mir, c + hand) - ap(mir, c)
            mirrored_gain.append(float(np.linalg.norm(mv)) / 28.0)
            same_side.append(float(mv @ out_of(p // 2)))
        check(f"{label}: rise/span {got['rise_over_span']:.2f}, lean "
              f"{deg:.0f} deg, the far handle follows at",
              f"{min(gain[-4:]):.2f} to {max(gain[-4:]):.2f} x",
              pred=lambda v: 0.5 < min(gain[-4:]) and max(gain[-4:]) < 2.0)

    check("the partner follows the hand's own direction, deg",
          max(ang), pred=lambda v: v < 10.0)
    check("...at between, x", f"{min(gain):.2f} and {max(gain):.2f}",
          pred=lambda v: 0.5 < min(gain) and max(gain) < 2.0)
    check("...so the far limb bends the other way, px",
          max(bent), pred=lambda v: v < -10.0)

    # what reflecting it in the plane does instead, which is the rigid arch
    # and is what the page was showing
    check("reflected instead: both limbs bulge out together, px",
          min(same_side), pred=lambda v: v > 1.0)
    check("...and the far handle leaps, x",
          f"{min(mirrored_gain):.2f} to {max(mirrored_gain):.2f} x",
          pred=lambda v: max(mirrored_gain) > 4.0)


def camera_unpinned_by_marks():
    """A tidied family must not certify a camera it never pinned down.

    The Worcester rib is the case: ten uprights down a nave, each swung at most
    1.6 px onto their common point, reported that point to +-2e-7 px and a
    focal length of 14379 to +-3 %. Nothing was wrong with the camera -- it was
    never measured. The marks allow 843 .. 26554 px, four cameras out of nine
    across their spread do not exist at all, and the same three arch marks read
    a rib's rise over span anywhere from 0.14 to 5.5.
    """
    print("\nthe camera a tidied family does and does not pin")
    import app as A
    im = [1200, 1200]

    def fam(vp, along, c0, c1, jit, seed, axis=1):
        """Segments aimed at `vp`, nudged by a hand worth `jit` px.

        `axis` is the coordinate c0 .. c1 runs in: down the page for uprights,
        across it for a family whose point is off to the side.
        """
        g = np.random.default_rng(seed)
        v, out = np.asarray(vp, float), []
        for q in along:
            a = np.array([q, c0] if axis else [c0, q], float)
            d = v - a
            b = a + d * ((c1 - c0) / d[axis])
            out.append(np.concatenate([a, b]) + g.normal(0, jit, 4))
        return [w.tolist() for w in out]

    # ten uprights down a nave: parallel to within the hand, which is exactly
    # when their point is unplaced and f is whatever it landed on
    # exactly converging, because that is what a tidied family IS: every line
    # was swung onto the one point, so resampling them can only return it
    nave = fam([4006., -207088.], np.linspace(260, 980, 10), 1050., 250., 0.0, 1)
    wall = fam([242., 1590.], np.linspace(300, 900, 4), 300., 560., 0.0, 2)
    tid = {"n": 10, "worst": 1.6}
    with A.app.test_client() as cl:
        loose = cl.post("/api/fit_vp", json={"lines": nave, "image": im,
                                             "tidied": tid}).get_json()
        kept = cl.post("/api/fit_vp", json={"lines": nave, "image": im}).get_json()
        wl = cl.post("/api/fit_vp", json={"lines": wall, "image": im,
                                          "tidied": {"n": 4, "worst": 4.5}}).get_json()
    spread = lambda r: (r["conditioning"]["bootstrap_y"][1]
                        - r["conditioning"]["bootstrap_y"][0])
    check("tidied, the spread carries the 1.6 px back, px", spread(loose),
          pred=lambda s: s > 1e4)
    check("resampling those lines alone says, px", spread(kept),
          pred=lambda s: s < 1.0)

    doc = {"image": {"w": im[0], "h": im[1]}, "w": im[0], "h": im[1],
           "perspective": "three",
           "vertical_vp": "v", "surfaces": {},
           "vps": {"v": loose["value"], "h": wl["value"]},
           "vp_ranges": {"v": [loose["conditioning"]["bootstrap_x"],
                               loose["conditioning"]["bootstrap_y"]],
                         "h": [wl["conditioning"]["bootstrap_x"],
                               wl["conditioning"]["bootstrap_y"]]},
           "features": {"A": {"kind": "arch", "name": "a rib", "lean": True,
                              "shape": "pointed", "vertical_vp": "v",
                              "wall_vp": "h",
                              "arch_from": [[767.6, 802.4, 863.1, 659.3,
                                             461.6, 235.8]]}}}
    with A.app.test_client() as cl:
        ck = cl.post("/api/check", json={"doc": doc}).get_json()
        ar = cl.post("/api/arch", json={"doc": doc, "feature": "A"}).get_json()
    cam = ck["camera"]["conditioning"]
    check("cameras across that spread that do not exist",
          cam.get("focal_marks_refused") or "",
          pred=lambda s: "no camera at all" in s)
    check("f over the pp pin, as it always said, factor",
          cam["focal_range_over_pp"][1] / cam["focal_range_over_pp"][0],
          pred=lambda k: k < 1.2)
    check("and the camera panel says so itself, not OK",
          [ck["camera"]["verdict"], ck["camera"]["value"],
           cam.get("fitted_focal_px") is not None],
          pred=lambda q: q[0] == "undetermined" and q[1] is None and q[2])
    check("f over the marks' own spread, factor",
          cam["focal_range_over_marks"][1] / cam["focal_range_over_marks"][0],
          pred=lambda k: k > 3)
    rng, one = ar["rise_over_span_range"][0], ar["rise_over_span"][0]
    check("the rib's rise/span, as drawn", one, pred=lambda v: 0.1 < v < 1.0)
    check("and over the cameras the marks allow", rng,
          pred=lambda q: q[0] <= one <= q[1] and q[1] > 2 * q[0])

    # and the other way: a family that DOES place its point keeps its camera,
    # or the jitter above would cry wolf on every tidied drawing
    K, Rot = camera(f=1100.0, pp=(600.0, 600.0))
    vv, vh = vp_of(K, Rot, [0, 1., 0]), vp_of(K, Rot, [1., 0, 0])
    good = {"v": fam(vv, np.linspace(300, 900, 8), 1000., 300., 0.0, 3),
            "h": fam(vh, np.linspace(250, 950, 8), 250., 1000., 0.0, 4, axis=0)}
    with A.app.test_client() as cl:
        fits = {k: cl.post("/api/fit_vp", json={"lines": s, "image": im,
                                                "tidied": {"n": 8, "worst": 1.6}}
                           ).get_json() for k, s in good.items()}
        doc2 = dict(doc, vps={k: r["value"] for k, r in fits.items()},
                    vp_ranges={k: [r["conditioning"]["bootstrap_x"],
                                   r["conditioning"]["bootstrap_y"]]
                               for k, r in fits.items()})
        ck2 = cl.post("/api/check", json={"doc": doc2}).get_json()
    c2 = ck2["camera"]["conditioning"]
    check("a family that does place its point: f, px", ck2["camera"]["value"],
          1100.0, 120.0)
    check("and f over ITS spread, factor",
          c2["focal_range_over_marks"][1] / c2["focal_range_over_marks"][0],
          pred=lambda k: k < 1.5)
    check("and its camera panel stays an answer",
          ck2["camera"]["verdict"], pred=lambda v: v == "ok")
    check("so nothing is called unpinned there",
          sorted(k for k in c2 if "unpinned" in k or "marks_refused" in k),
          pred=lambda k: k == [])


def two_point_never_looked_at_its_verticals():
    """Two-point assumes the camera is level. When uprights ARE marked, that
    assumption is measurable -- and it was never measured.

    The nave photograph is the case: longitudinal, vertical and transverse
    families, all three confidently placed, that no camera can hold at once.
    Three-point said so ("the horizon tilts -6.07 deg but the verticals say
    +0.94 deg"); two-point threw the vertical family away and answered
    f = 1538.2, and the rib read through it leaned 39 deg and stood 2.1 spans
    tall. The level camera still goes back as K, because the arch is drawn
    from it and a drawing is a guide, not a claim.
    """
    print("\nthe camera two-point does not check  (plan: a level camera keeps "
          "its uprights square to its horizon)")
    import app as A
    V = {"long": [242.0384055078994, 1589.632428492438],
         "vert": [4006.252857611392, -207087.59724245578],
         "cross": [7136.100062906801, 856.06943735577]}
    doc = {"image": {"w": 1200, "h": 1200}, "perspective": "two", "surfaces": {},
           "vertical_vp": "vert", "vps": V, "vp_ranges": {},
           "features": {"A": {"kind": "arch", "name": "a rib", "lean": True,
                              "shape": "pointed", "vertical_vp": "vert",
                              "wall_vp": "long", "normal_vp": "cross",
                              "arch_from": [[683.5158817086531, 947.0700985761217,
                                             751.9715224534502, 836.1719605695508,
                                             405.5859802847755, 458.2968236582694]]}}}
    with A.app.test_client() as cl:
        two = cl.post("/api/check", json={"doc": doc}).get_json()["camera"]
        three = cl.post("/api/check", json={"doc": dict(doc, perspective="three")}
                        ).get_json()["camera"]
        ar = cl.post("/api/arch", json={"doc": doc, "feature": "A"}).get_json()
    cons = two["conditioning"]["horizon_consistency"]
    check("the three families disagree by, deg", cons["disagreement_deg"], 7.01, 0.02)
    check("...which is this much horizon across the picture, px",
          cons["horizon_rise_px_over_image"], 127.7, 0.5)
    check("two-point used to answer, px", two["conditioning"]["fitted_focal_px"],
          1538.2, 0.1)
    check("...and now refuses, as three-point already did",
          [two["verdict"], two["value"], three["verdict"]],
          pred=lambda q: q == ["undetermined", None, "undetermined"])
    check("but the arch still draws from the level camera",
          [ar.get("lean_deg", [None])[0], ar.get("rise_over_span", [None])[0]],
          pred=lambda q: q[0] is not None and abs(q[0] - 38.96) < 0.05
          and abs(q[1] - 2.104) < 0.002)

    # and the other way: a level camera's own three points must stay an answer,
    # or every two-point drawing with uprights marked is refused
    f, pp, roll = 1100.0, np.array([600.0, 600.0]), np.radians(3.0)
    u = np.array([np.cos(roll), np.sin(roll)])          # along the horizon
    good = {"a": (pp + 900 * u).tolist(),               # p * q = -f^2 : at right angles
            "b": (pp - (f * f / 900.0) * u).tolist(),
            "v": (pp + 1e5 * np.array([-u[1], u[0]])).tolist()}
    lvl = dict(doc, vps=good, vertical_vp="v", features={})
    bare = dict(lvl, vps={k: q for k, q in good.items() if k != "v"},
                vertical_vp=None)
    with A.app.test_client() as cl:
        ok = cl.post("/api/check", json={"doc": lvl}).get_json()["camera"]
        no_v = cl.post("/api/check", json={"doc": bare}).get_json()["camera"]
    check("a level camera's own uprights: f, px", ok["value"], f, 1.0)
    check("...the same f the two horizontals alone gave", ok["value"], no_v["value"], 1e-9)
    check("...and nothing is called inconsistent",
          [ok["verdict"], ok["conditioning"]["horizon_consistency"]["disagreement_deg"]],
          pred=lambda q: q[0] == "ok" and q[1] < 0.01)


def arch_lean_keeps_the_crown():
    """One crown mark is two numbers, and the lean takes both.

    Named a depth vanishing point as well, the arch used to refuse outright --
    "one crown mark cannot say both how far back the vault goes and how far
    the arch leans" -- which left the drawing with nothing on it. The lean
    wins now and the depth point is dropped with a note, because a vault is
    the thing that can be left out and still leave an arch.

    Also the rib that configuration is actually for: two springings a bay
    apart on one wall and the boss between them, which is the pair of
    half-diagonals seen as one arch leaning in over the nave. Measured
    against a real circular arch in a real plane under a known camera.
    """
    print("\narches that lean, and the depth point they cannot also feed")
    import app as A
    V = {"long": [242.0384055078994, 1589.632428492438],
         "vert": [4006.252857611392, -207087.59724245578],
         "cross": [7136.100062906801, 856.06943735577]}
    feat = {"kind": "arch", "name": "a rib", "shape": "pointed", "lean": True,
            "vertical_vp": "vert", "wall_vp": "long", "normal_vp": "cross",
            "arch_from": [[683.5158817086531, 947.0700985761217,
                           751.9715224534502, 836.1719605695508,
                           405.5859802847755, 458.2968236582694]]}
    doc = {"image": {"w": 1200, "h": 1200}, "perspective": "two", "surfaces": {},
           "vertical_vp": "vert", "vps": V, "vp_ranges": {}, "features": {"A": feat}}
    with A.app.test_client() as cl:
        lean = cl.post("/api/arch", json={"doc": doc, "feature": "A"}).get_json()
        flat = cl.post("/api/arch", json={
            "doc": dict(doc, features={"A": dict(feat, lean=False)}),
            "feature": "A"}).get_json()
    check("leaning with a depth point named, the arch is still drawn",
          len(lean.get("arches") or []), pred=lambda v: v == 1)
    check("...the depth point is dropped, and says so",
          [lean["vault"][0], (lean.get("depth_unused") or "")],
          pred=lambda q: q[0] is None and "not read while this arch leans" in q[1])
    check("...and with the lean off the same mark slides it back",
          [flat["vault"][0] is not None, flat.get("depth_unused")],
          pred=lambda q: q[0] and q[1] is None)

    # the rib itself: a 12 m nave, a 7 m bay, the boss 8 m above the springing
    K = np.array([[1200.0, 0, 600.0], [0, 1200.0, 600.0], [0, 0, 1.0]])
    pitch = np.radians(12.0)
    Rot = (np.array([[1., 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                     [0, np.sin(pitch), np.cos(pitch)]])
           @ np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]]))
    eye = np.array([6.0, -22.0, 1.6])

    def page(P):
        x = ((np.asarray(P, float) - eye) @ Rot.T) @ K.T
        return x[:, :2] / x[:, 2:]

    W, Lb, Hb, zs = 12.0, 7.0, 8.0, 12.0
    ftA, ftB = np.array([0.0, 0.0, zs]), np.array([0.0, Lb, zs])
    boss = np.array([W / 2, Lb / 2, zs + Hb])
    d3 = (ftB - ftA) / np.linalg.norm(ftB - ftA)
    up3 = boss - (ftA + ftB) / 2
    r, h = np.linalg.norm(ftB - ftA) / 2, float(np.linalg.norm(up3))
    up3 = up3 / h
    yc = (h * h - r * r) / (2 * h)
    rho = h - yc
    a0, a1 = np.arctan2(-yc, -r), np.arctan2(-yc, r)
    # the long way, through the top: past a semicircle the two end angles are
    # on the same side and a plain linspace sweeps the minor arc underneath
    ps = a0 - np.linspace(0.0, (a0 - a1) % (2 * np.pi), 2001)
    arc = page((ftA + ftB) / 2 + np.outer(rho * np.cos(ps), d3)
               + np.outer(yc + rho * np.sin(ps), up3))
    pA, pB, pC = page([ftA])[0], page([ftB])[0], page([boss])[0]
    vert = (K @ Rot @ [0, 0, 1.0])[:2] / (K @ Rot @ [0, 0, 1.0])[2]
    span = R.span_vp_of(K, vert, pA, pB)
    l, deg = R.lean_line(K, vert, span, pA, pB, pC)
    rib = R.plane_arch(K, l, pA, pB, pC, span_vp=span, n=2048)
    q = np.asarray(rib["pts"], float)
    check("the springing line is %.0f px on the page" % np.hypot(*(pB - pA)),
          float(np.hypot(*(pB - pA))), pred=lambda v: v < 210)
    check("the lean it leans, deg", deg, np.degrees(np.arctan2(W / 2, Hb)), 1e-6)
    check("the rise it stands, over span", rib["rise_over_span"], h / (2 * r), 1e-9)
    check("and the rib lands on the real rib, px",
          float(np.linalg.norm(q[:, None] - arc[None], axis=2).min(1).max()),
          0.0, 3.0)

    # what that last 2 px is: the PLANE cubic, not the camera. A cubic holds a
    # circle while the sweep is small, and a round arch's sweep grows with its
    # rise -- a pointed one's shrinks, which is why the rib above is drawn
    # through the shape that suffers.
    def cubic_off(rs, pointed):
        s, A2 = 1.0, np.array([-1.0, 0.0])
        hh = 2 * s * rs
        ap = np.array([0.0, hh])
        c = (np.array([(hh * hh - s * s) / (2 * s), 0.0]) if pointed
             else np.array([0.0, (hh * hh - s * s) / (2 * hh)]))
        P = R._bez_ctrl(c, A2, ap)
        rr = float(np.linalg.norm(A2 - c))
        w = R.bez_at(P, np.linspace(0, 1, 4001))
        return 100.0 * float(np.abs(np.linalg.norm(w - c, axis=1) - rr).max()) / rr
    check("a round arch at rise/span 0.50, cubic off by % of radius",
          cubic_off(0.50, False), 0.027, 0.002)
    check("...at 0.75, a real horseshoe", cubic_off(0.75, False), 0.105, 0.005)
    check("...at 1.43, past anything built", cubic_off(1.43, False),
          pred=lambda v: v < 0.5)
    check("a pointed arch only gets tighter as it rises",
          cubic_off(2.10, True), pred=lambda v: v < 0.001)


def arch_vault():
    """The crown slides along the depth lines and the arch becomes a vault.

    "when a flat arch is built, it's point should be able to slide along the
    'depth' lines to build a 'vault'". Where the crown sits says two things
    at once -- the rise, read where the depth line through it crosses the
    upright over the middle of the span, and the depth, read off what is
    left. The slide is a homology, which is what the arch CHECK already
    measures a vault by, so the thing built and the thing checked are one.
    """
    print("\nan arch slid back into a vault  (plan: the crown off the wall "
          "says the depth, and the check agrees)")
    K, Rot = camera()
    vert = vp_of(K, Rot, [0, 1., 0]); wall = vp_of(K, Rot, [1., 0, 0])
    deep = vp_of(K, Rot, [0, 0, 1.])
    l = np.cross(np.append(vert, 1.0), np.append(wall, 1.0))
    feet = ((330., 940.), (690., 910.))

    flat = R.plane_arch(K, l, *feet, None, span_vp=wall)
    check("a crown on the upright is a flat arch", flat["mu"], 0.0, 0.0)
    check("...and it has no far limb", "far_pts" in flat, pred=lambda v: v is False)

    # slide that crown a quarter of the way to the depth vanishing point
    C = flat["crown"]
    C2 = C + 0.25 * (np.asarray(deep, float) - C)
    v = R.plane_arch(K, l, *feet, C2, span_vp=wall, depth_vp=deep)
    check("slid along the depth line, it is a vault", v["mu"],
          pred=lambda m: abs(m) > 1e-9)
    check("the crown stays where it was put, px",
          float(np.linalg.norm(np.asarray(v["far_crown"]) - C2)), 0.0, 1e-7)
    check("...and the arch it springs from is unchanged",
          v["rise_over_span"], flat["rise_over_span"], 1e-9)
    check("...with its own crown back on the wall, px",
          float(np.linalg.norm(v["crown"] - C)), 0.0, 1e-7)

    def off_line(a, b):
        """How far b sits off the depth line through a, px."""
        a = np.asarray(a, float); b = np.asarray(b, float)
        u = np.asarray(deep, float) - a
        u = u / np.linalg.norm(u)
        w = b - a
        return float(abs(w[0] * u[1] - w[1] * u[0]))
    near, far = np.asarray(v["pts"]), np.asarray(v["far_pts"])
    check("every point of the far limb sits on its own depth line, px",
          max(off_line(a, b) for a, b in zip(near[::20], far[::20])), 0.0, 1e-7)
    check("...the two feet included, px",
          max(off_line(a, b) for a, b in zip(v["feet"], v["far_feet"])),
          0.0, 1e-7)
    check("the far limb is smaller, as it stands further back",
          float(np.linalg.norm(far[0] - far[-1])
                / np.linalg.norm(near[0] - near[-1])),
          pred=lambda r: 0.5 < r < 0.999)

    # and the app's own vault CHECK, fed the vault the app just built
    got = R.arch(near, far, K=K, vert_vp=vert, wall_vp=wall, normal_vp=deep)
    check("the vault check reads it as one slide, rms px",
          got["depth_rms_px"], 0.0, 1e-6)
    check("...at the depth it was built at", got["depth"] / v["mu"],
          1.0, 1e-6)
    check("...and both limbs the same arch", abs(got["rise_over_span"][0]
                                                 - got["rise_over_span"][1]),
          pred=lambda d: d < 0.02)

    # what the DRAWN far limb costs: the near cubic put through the homology,
    # which keeps the vault one object but is no longer a cubic's best fit
    Hm = np.array(v["homology"])
    fb = [[R._apply(Hm, c) for c in P] for P in v["bez"]]
    check("the far limb drawn from the near one costs no more than the near",
          R.curve_off(fb, far) / v["bez_off"], pred=lambda x: x < 1.2)


def arch_gothic():
    """A rib that crosses the bay, and a rib that leans off the wall.

    "it still stays flat. what I mean is sliding it off the surface of it's
    supports, like in gothic arches" -- both ways, as asked. The springing
    line can be the arch's own plane, which is how a diagonal rib gets a
    diagonal plane instead of being flattened onto the wall; and the arch can
    tip about that line, which is how a rib leaves its respond and goes over
    the bay. Everything here is measured against a real circular arch standing
    in a real plane under a known camera, so "it looks right" never comes
    into it.
    """
    print("\narches that leave the wall  (plan: the feet give the plane, and "
          "the crown can tip it off the supports)")
    K, Rot = camera()
    vert = vp_of(K, Rot, [0, 1., 0])
    wall = vp_of(K, Rot, [1., 0, 0])
    t = np.array([-1.1, -1.6, 9.0])              # foot A, in camera coords
    r, h = 1.0, 0.84                             # half-span and rise, in the plane
    d = np.array([np.cos(np.radians(55.)), 0.0, np.sin(np.radians(55.))])
    LEAN = 28.0                                  # how far it tips off the wall

    def scene(deg):
        """A real arch leaning `deg` off the wall on that springing line: its
        two feet, its top, and the arc itself, as the camera sees them."""
        nrm = np.cross(d, [0.0, 1.0, 0.0])
        nrm /= np.linalg.norm(nrm)
        up = (np.cos(np.radians(deg)) * np.array([0.0, 1.0, 0.0])
              + np.sin(np.radians(deg)) * nrm)
        yc = (h * h - r * r) / (2 * h)               # the circle's centre, up
        rho = h - yc
        ps = np.linspace(np.arctan2(-yc, -r), np.arctan2(-yc, r), 1201)
        uv = np.column_stack([r + rho * np.cos(ps), yc + rho * np.sin(ps)])
        page = project(K, Rot, t, np.vstack([[[0, 0], [2 * r, 0], [r, h]], uv]),
                       d, up)
        return page[0], page[1], page[2], page[3:]

    def off(built, arc):
        """The furthest the built arch strays from the real one, px."""
        q = np.asarray(built, float)
        return float(np.linalg.norm(q[:, None] - arc[None], axis=2).min(1).max())

    def upright(vp):
        return np.cross(np.append(vert, 1.0), np.append(vp, 1.0))

    # (a) no wall named: the arch stands in the vertical plane through its feet
    A, B, Cn, arc = scene(0.0)
    span = R.span_vp_of(K, vert, A, B)
    check("the springing line's own vanishing point, px off true",
          float(np.linalg.norm(span - vp_of(K, Rot, d))), 0.0, 1e-6)
    rib = R.plane_arch(K, upright(span), A, B, Cn, span_vp=span)
    check("a rib built on it lies on the real rib, px",
          off(rib["pts"], arc), 0.0, 0.2)       # the plane cubic's own 0.12 px
    check("...at the rise it really has", rib["rise_over_span"], h / (2 * r),
          1e-9)
    wal = R.plane_arch(K, upright(wall), A, B, Cn, span_vp=wall)
    check("...landing on both feet and the top it was given, px",
          max(float(np.linalg.norm(np.asarray(rib["feet"][1]) - B)),
              float(np.linalg.norm(rib["crown"] - Cn))), 0.0, 1e-9)
    check("the wall's plane slides the far foot off its mark, px",
          float(np.linalg.norm(np.asarray(wal["feet"][1]) - B)),
          pred=lambda v: v > 2.0)
    check("...and the top too, px",
          float(np.linalg.norm(wal["crown"] - Cn)), pred=lambda v: v > 5.0)
    hz = np.linalg.inv(K @ K.T) @ np.append(vert, 1.0)
    lvl = np.array([-hz[1], hz[0]])
    check("a springing line level with the horizon has no vanishing point",
          R.span_vp_of(K, vert, (300.0, 700.0), (300.0, 700.0) + 400 * lvl
                       / np.linalg.norm(lvl)),
          pred=lambda v: v is None)

    # (b) the feet stay put and the whole arch tips about them
    A2, B2, Cn2, arc2 = scene(LEAN)
    check("leaning, it springs from the same two feet, px",
          max(float(np.linalg.norm(A - A2)), float(np.linalg.norm(B - B2))),
          0.0, 1e-9)
    span2 = R.span_vp_of(K, vert, A2, B2)
    lean, deg = R.lean_line(K, vert, span2, A2, B2, Cn2)
    check("the lean one crown mark says, deg", deg, LEAN, 1e-6)
    tip = R.plane_arch(K, lean, A2, B2, Cn2, span_vp=span2)
    check("...and the arch it tips into is the real one, px",
          off(tip["pts"], arc2), 0.0, 0.2)
    check("...the same arch as before, only leaning", tip["rise_over_span"],
          h / (2 * r), 1e-9)
    check("...with its top exactly on the mark, px",
          float(np.linalg.norm(tip["crown"] - Cn2)), 0.0, 1e-6)
    flat = R.plane_arch(K, upright(span2), A2, B2, Cn2, span_vp=span2)
    check("built flat, the top is not where it was put, px",
          float(np.linalg.norm(flat["crown"] - Cn2)), pred=lambda v: v > 5.0)
    check("...and it reads the rise wrong", flat["rise_over_span"],
          pred=lambda v: abs(v - h / (2 * r)) > 0.05)
    check("a top over the middle leans nothing, deg",
          R.lean_line(K, vert, span, A, B, Cn)[1], 0.0, 1e-6)
    same = R.plane_arch(K, R.lean_line(K, vert, span, A, B, Cn)[0], A, B, Cn,
                        span_vp=span)
    check("...and tips into the plane it was already in, px",
          float(np.abs(np.asarray(same["pts"]) - rib["pts"]).max()), 0.0, 1e-9)
    mid = F.harmonic_mid(A2, B2, span2)
    check("a top on the springing line says nothing about the lean",
          R.lean_line(K, vert, span2, A2, B2,
                      mid + 0.3 * (np.asarray(span2) - mid))[0],
          pred=lambda v: v is None)
    back = mid + (mid - Cn2)                     # tipped the other way instead
    check("leaning it the other way is a different arch, px",
          off(R.plane_arch(K, R.lean_line(K, vert, span2, A2, B2, back)[0],
                           A2, B2, back, span_vp=span2)["pts"], arc2),
          pred=lambda v: v > 20.0)


def arch_straight_roof():
    """Straight sides make the arch a roof, and the slope is read off it.

    "roof slope should go to arch: 'arches and roofs', by making the sides
    straight". A gable is two straight limbs on the wall, and slid back along
    the depth lines it is a pitched roof. Then each slope and its twin on the
    far gable are ONE direction in the world, so they meet at the slope's
    vanishing point -- the roof check's first input, which until now had to be
    named by hand.
    """
    print("\narches and roofs  (plan: straight sides, a pitch, and the slope "
          "vanishing point off the drawing)")
    K, Rot = camera()
    vert = vp_of(K, Rot, [0, 1., 0]); wall = vp_of(K, Rot, [1., 0, 0])
    deep = vp_of(K, Rot, [0, 0, 1.])
    l = np.cross(np.append(vert, 1.0), np.append(wall, 1.0))
    feet = ((330., 940.), (690., 910.))

    def off_chord(q):
        """The worst departure of a run of points from its own chord, px."""
        q = np.asarray(q, float)
        u = q[-1] - q[0]
        u = u / np.linalg.norm(u)
        w = q - q[0]
        return float(np.max(np.abs(w[:, 0] * u[1] - w[:, 1] * u[0])))

    g = R.plane_arch(K, l, *feet, None, span_vp=wall, shape="straight")
    rd = R.plane_arch(K, l, *feet, None, span_vp=wall)
    n = len(g["pts"]) // 2
    check("a straight side images straight, px",
          max(off_chord(g["pts"][:n + 1]), off_chord(g["pts"][n:])), 0.0, 1e-9)
    check("...where a round arch's side is not, px",
          off_chord(rd["pts"][:n + 1]), pred=lambda x: x > 20.0)
    check("so the path drawn IS the roof, px off it", g["bez_off"], 0.0, 0.0)
    check("...where one cubic over a round arch costs something, px",
          rd["bez_off"], pred=lambda x: x > 0.5)
    check("every handle lies on the slope itself, px",
          max(off_chord(P) for P in g["bez"]), 0.0, 1e-9)
    check("no top marked stands it at 45 deg",
          float(np.degrees(np.arctan(2 * g["rise_over_span"]))), 45.0, 1e-9)
    check("a gable on the wall has no slope to measure", "slope_vps" in g,
          pred=lambda v: v is False)

    # slide that crown back: the depth line through it passes through the
    # gable's own crown, so the pitch is untouched and only the depth changes
    Cr = g["crown"]
    back = lambda t: Cr + t * (np.asarray(deep, float) - Cr)
    v = R.plane_arch(K, l, *feet, back(0.3), span_vp=wall, depth_vp=deep,
                     shape="straight")
    check("slid back, it is a roof with two slopes", len(v["slope_vps"]), 2, 0)
    check("...at the pitch the gable had", v["rise_over_span"],
          g["rise_over_span"], 1e-9)
    check("...the two sides meeting at different points, px apart",
          float(np.linalg.norm(np.subtract(*v["slope_vps"]))),
          pred=lambda d: d > 50.0)

    # the truth, from the camera and not from the builder: a 45 deg roof's two
    # slopes run along X+Y and X-Y in the world
    want = [vp_of(K, Rot, [1., 1., 0]), vp_of(K, Rot, [1., -1., 0])]
    check("they meet where the camera puts that 45 deg direction, px",
          max(min(float(np.linalg.norm(np.subtract(sv, w))) for w in want)
              for sv in v["slope_vps"]), 0.0, 1e-6)
    far = R.plane_arch(K, l, *feet, back(0.6), span_vp=wall, depth_vp=deep,
                       shape="straight")
    check("...and how far back the roof goes does not move them, px",
          float(np.max(np.abs(np.subtract(far["slope_vps"], v["slope_vps"])))),
          0.0, 1e-6)

    # which is exactly what the roof check measures: the slope's vanishing
    # point, the vertical one and the one it rises along are collinear
    miss = [float(R.tilted(sv, vert, wall)) for sv in v["slope_vps"]]
    check("the roof check reads the built slope as true, px", max(miss), 0.0, 1e-6)
    check("...and says so", [C.roof(sv, vert, wall)["verdict"]
                             for sv in v["slope_vps"]],
          pred=lambda z: z == ["consistent", "consistent"])
    check("a slope is not a horizontal: px off the horizon",
          min(float(R.tilted(wall, deep, sv)) for sv in v["slope_vps"]),
          pred=lambda x: x > 200.0)

    # steeper means nearer the vertical vanishing point, all the way to a wall
    def climb(sv):
        """Where the slope VP sits along wall -> vertical, 0 flat, 1 upright."""
        a, b = np.asarray(wall, float), np.asarray(vert, float)
        return float((np.subtract(sv, a) @ (b - a)) / ((b - a) @ (b - a)))
    steep = R.plane_arch(K, l, *feet, back(0.3) + 0.25 *
                         (np.asarray(vert, float) - back(0.3)),
                         span_vp=wall, depth_vp=deep, shape="straight")
    check("a steeper roof is steeper", steep["rise_over_span"],
          pred=lambda r: r > g["rise_over_span"] + 0.1)
    check("...and its slopes sit nearer the vertical VP",
          min(climb(sv) for sv in steep["slope_vps"]) -
          max(climb(sv) for sv in v["slope_vps"]), pred=lambda d: d > 0.05)


def ellipse_family_level():
    """A family of rings on one plane, each on a diameter level with the horizon.

    "ellipses on the horizontal planes should have strictly horizontal
    diameters". They do, and not by convention: the diameter parallel to the
    picture plane has both ends at the SAME DEPTH, so they recede by the same
    amount and the chord cannot tilt. Its image is parallel to the plane's
    vanishing line exactly. Nothing else about the ellipse is level -- the
    camera's own ring leans a long way -- so the drag is levelled for the hand.
    """
    print("\na family of ground rings  (plan: one plane, one camera, every "
          "diameter level with the horizon)")
    import app as A
    K, Rot = camera()
    l = vanish_line(K, Rot, np.array([0, 1., 0]))
    hz = np.array([-l[1], l[0]]) / np.hypot(l[0], l[1])

    def lean(seg):
        """Degrees between a drag and the plane's vanishing line."""
        u = np.array([seg[2] - seg[0], seg[3] - seg[1]], float)
        u = u / np.linalg.norm(u)
        a = abs(np.degrees(np.arctan2(abs(u[0] * hz[1] - u[1] * hz[0]), u @ hz)))
        return min(a, 180.0 - a)

    def depth(p):
        """Camera-frame Z of the ground point imaging at p -- the eye at (0,1,0)
        looking at the plane Y = 0, the plane this vanishing line belongs to."""
        d = np.linalg.inv(K) @ np.append(np.asarray(p, float), 1.0)
        return float((d * (-1.0 / (Rot.T @ d)[1]))[2])

    a, b = np.array([700., 980.]), np.array([980., 900.])
    got = R.horizon_diameter(a, b, l)
    check("a slanted drag comes back level with the horizon, deg",
          lean([*got[0], *got[1]]), 0.0, 1e-9)
    check("...and that is WHY: both ends at one depth",
          depth(got[0]) / depth(got[1]), 1.0, 1e-9)
    check("...which the drag as made was not", depth(a) / depth(b),
          pred=lambda v: abs(v - 1) > 0.05)
    check("...turned about its middle, so the ring stays put, px",
          float(np.linalg.norm((got[0] + got[1]) / 2 - (a + b) / 2)), 0.0, 1e-9)
    check("...and keeps its length, px",
          float(np.linalg.norm(got[1] - got[0]) - np.linalg.norm(b - a)), 0.0, 1e-9)
    check("...it really was slanted to start with, deg",
          lean([*a, *b]), pred=lambda v: v > 10)
    check("a drag the length of a point is refused",
          R.horizon_diameter(a, a, l), pred=lambda v: v is None)

    # one call, one plane, one camera: the family is consistent because it was
    # built that way, not because anything reconciled it afterwards
    doc = {"vps": {"a": vp_of(K, Rot, [1., 0, 0]).tolist(),
                   "b": vp_of(K, Rot, [0, 0, 1.]).tolist(),
                   "v": vp_of(K, Rot, [0, 1., 0]).tolist()},
           "horizon_vps": ["a", "b"], "vertical_vp": "v", "surfaces": {}}
    segs = [[700., 980., 980., 900.], [300., 1100., 520., 1100.],
            [1150., 860., 1290., 855.]]
    with A.app.test_client() as cl:
        r = cl.post("/api/ellipse", json={
            "doc": dict(doc, features={"E": {"kind": "ellipse", "ring_from": segs}}),
            "feature": "E"}).get_json()
    check("three drags, three rings", len(r["rings"] or []), 3, 0)
    check("every one levelled, worst deg", max(lean(e) for e in r["ends"]), 0.0, 1e-9)
    # ...and each foreshortened by where it sits: rounder further from the
    # horizon, which is the whole point of building them on one plane
    def flat(q, e):
        q = np.asarray(q, float)
        return (float(np.ptp(q @ np.array([l[0], l[1]]) / np.hypot(l[0], l[1])))
                / float(np.ptp(q @ hz)),
                abs(l @ np.append(q.mean(0), 1.0)) / np.hypot(l[0], l[1]))
    got = sorted(flat(q, e) for q, e in zip(r["rings"], r["ends"]))
    check("rounder the further from the horizon",
          [round(d, 1) for _, d in got], pred=lambda d: d == sorted(d))
    check("...and it is a real spread, not a rounding", got[-1][0] - got[0][0],
          pred=lambda v: v > 0.2)

    # the arch is a family the same way: several on one wall, one camera
    wdoc = dict(doc, surfaces={"W": {"name": "the wall", "vps": ["a", "v"]}})
    ms = [[400., 900., 600., 900.], [700., 880., 860., 880.],
          [920., 866., 1050., 866.]]
    with A.app.test_client() as cl:
        r = cl.post("/api/arch", json={
            "doc": dict(wdoc, features={"A": {"kind": "arch", "arch_from": ms,
                                              "surface": "W", "vertical_vp": "v",
                                              "wall_vp": "a"}}),
            "feature": "A"}).get_json()
    check("three spans, three arches", len(((r or {}).get("arches")) or []), 3, 0)
    check("...every one a half-round, rise over span",
          r["rise_over_span"], pred=lambda v: all(abs(x - 0.5) < 1e-6 for x in v))

def circle_in_a_square():
    """The circle inscribed in a square two families of lines make, and the
    line that closes such a square.

    "auto ellipse/circle insertion into a square shape formed by two line
    families" and "snap a second line of a family to a square if there are two
    lines of another family overlapping". Two vanishing points name two
    directions without saying the angle between them, so a parallelogram is as
    good as a square to them: the CAMERA is what says "square", and where it
    cannot the answer is undetermined. The circle itself needs no camera -- four
    corners fix one homography and the circle goes along with it -- so it is
    drawn either way and the reading beside it says whether it was earned.
    """
    print("\na circle in a square, and the line that closes one  (plan: ring on "
          "the projected circle to 0.05 px; snap onto the true side to 0.05 px; "
          "refused when the focal slack is worth more than a hand mark)")
    import copy
    import app as A

    W, H = 1400.0, 1000.0
    th = 0.3
    e1 = np.array([np.cos(th), 0, np.sin(th)])        # the tiles' two directions,
    e2 = np.array([-np.sin(th), 0, np.cos(th)])       # square to each other

    def scene(depth=9.0, ry=0.46):
        """A floor of unit tiles `depth` in front of a camera of known f: the
        camera, the floor's own (u, v) onto the page, and the document the page
        would send. A has lines at v = 0, 1 and 3; B at u = 0 and 1."""
        K, Rot = camera(1400.0, (W / 2, H / 2), -0.08, ry, 0.02)
        O = Rot @ np.array([-1.0, -1.6, depth])

        def P(uv):
            return project(K, Rot, O, np.atleast_2d(uv), e1, e2)

        def seg(u0, v0, u1, v1):
            a, b = P([[u0, v0], [u1, v1]])
            return [float(a[0]), float(a[1]), float(b[0]), float(b[1])]

        def pv(d):
            return vp_of(K, Rot, d).tolist()

        doc = {"image": {"w": W, "h": H}, "width": W, "height": H,
               "perspective": "three", "vertical_vp": "up", "features": {},
               "horizon_vps": ["A", "B"],
               "vps": {"A": pv(e1), "B": pv(e2), "up": pv([0, 1., 0])},
               "vp_names": {"A": "tiles along", "B": "tiles across", "up": "uprights"},
               "family_lines": {"A": [seg(-0.4, v, 1.6, v) for v in (0, 1, 3)],
                                "B": [seg(u, -0.4, u, 3.4) for u in (0, 1)]}}
        return K, P, seg, pv, doc

    def area(q):
        q = np.asarray(q, float)
        return abs(float(np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                                - np.roll(q[:, 0], -1) * q[:, 1]))) / 2.0

    K, P, seg, pv, doc = scene()
    va, vb = np.array(doc["vps"]["A"]), np.array(doc["vps"]["B"])
    unit = P([[0, 0], [1, 0], [1, 1], [0, 1]])

    # --- the construction: one cross product after another, nothing to converge
    a = R.seg_line(seg(-0.4, 0, 1.6, 0), va)
    bi, bj = (R.seg_line(seg(u, -0.4, u, 3.4), vb) for u in (0, 1))
    both = R.square_completions(K, va, vb, a, bi, bj)
    check("a square closes two ways, one on each side of the line",
          sum(g is not None for g in both), 2, 0)
    for s in (1, -1):
        truth = P([[0, 0], [1, 0], [1, s], [0, s]])
        line, q = min((g for g in both if g is not None),
                      key=lambda g: float(np.abs(g[1] - truth).max()))
        check(f"...the side at v = {s:+d}: its corners, px worst",
              float(np.abs(q - truth).max()), 0.0, 1e-6)
        check("...and the line through them, px worst",
              max(abs(float(line @ np.append(P([[u, s]])[0], 1.0)))
                  for u in (-0.3, 0.2, 0.9, 1.4)), 0.0, 1e-6)

    # --- the circle: the image of the inscribed one, with no camera in it
    ring = R.cell_ring(unit, 96)
    t = np.linspace(0, 2 * np.pi, 8192, endpoint=False)
    dense = P(np.column_stack([0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)]))
    check("the ring lies on the projected circle, px worst",
          float(np.sqrt(((ring[:, None] - dense[None]) ** 2).sum(-1)).min(1).max()),
          0.0, 0.05)
    mids = P([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]])
    check("...touching each side at the image of its middle, px",
          max(float(np.sqrt(((ring - m) ** 2).sum(1)).min()) for m in mids),
          0.0, 1e-6)
    # ...which is NOT the middle of the side on the page: a near, oblique tile
    # puts the two several pixels apart, and a ring drawn to the page's
    # midpoints would be wrong by that much
    near = scene(depth=3.0)[1]([[0, 0], [1, 0], [1, 1], [0, 1]])
    nmid = scene(depth=3.0)[1]([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]])
    check("...not the page's midpoint of the side, px (near tile, side "
          f"{np.linalg.norm(near[1] - near[0]):.0f} px)",
          float(np.linalg.norm((near + np.roll(near, -1, 0)) / 2 - nmid, axis=1).min()),
          pred=lambda v: v > 3.0)

    # --- what the cell is, read through the camera
    rd = R.cell_reading(K, unit)
    check("a square reads 1 : 1", rd[0], 1.0, 1e-9)
    check("...at 90 deg between its sides", rd[1], 90.0, 1e-6)
    rect = P([[0, 0], [1, 0], [1, 1.5], [0, 1.5]])
    check("a 1 x 1.5 rectangle reads 1 : 1.5", 1.0 / R.cell_reading(K, rect)[0], 1.5, 1e-9)
    c70, s70 = np.cos(np.radians(70)), np.sin(np.radians(70))
    rd = R.cell_reading(K, P([[0, 0], [1, 0], [1 + c70, s70], [c70, s70]]))
    check("a rhombus of 70 deg reads 70 deg", rd[1], 70.0, 1e-6)
    check("...its sides equal", rd[0], 1.0, 1e-9)
    Kx = np.array(K, float); Kx[0, 0] = Kx[1, 1] = 1.1 * K[0, 0]
    r10 = R.cell_reading(Kx, unit)[0]
    check("a focal length a tenth out reads the same square, long : short",
          max(r10, 1.0 / r10), pred=lambda v: 1.05 < v < 1.15)

    # --- which cell a click is in
    sa, sb = doc["family_lines"]["A"], doc["family_lines"]["B"]
    got = R.square_cell(sa, sb, va, vb, P([[0.5, 0.5]])[0])
    check("a click in the first tile finds its four corners, px worst",
          max(float(np.sqrt(((got - c) ** 2).sum(1)).min()) for c in unit),
          0.0, 1e-6)
    check("...in round order, so the same area", area(got), area(unit), 1e-6)
    long = P([[0, 1], [1, 1], [1, 3], [0, 3]])
    got = R.square_cell(sa, sb, va, vb, P([[0.5, 2.0]])[0])
    check("a click past a line finds the NEAREST two, px worst",
          max(float(np.sqrt(((got - c) ** 2).sum(1)).min()) for c in long),
          0.0, 1e-6)
    for name, p in (("below the last line", [0.5, -2.0]),
                    ("left of the first", [-0.2, 0.5]),
                    ("right of the last", [1.5, 0.5])):
        check(f"a click {name} is no cell", R.square_cell(sa, sb, va, vb, P([p])[0]),
              pred=lambda c: c is None)
    check("lines that cross before they close one are no cell",
          R.square_cell([[0., 0., 100., 0.], [0., 100., 100., 100.]],
                        [[40., 0., 60., 100.], [60., 0., 40., 100.]],
                        np.array([1e6, 0.]), np.array([50., 50.]), np.array([50., 25.])),
          pred=lambda c: c is None)

    # --- the snap, through the route the page calls
    def snap(d, line):
        with A.app.test_client() as cl:
            return cl.post("/api/square_snap", json={
                "doc": d, "into": "A", "line": [float(x) for x in line]}).get_json()

    true_line = seg(0, 1, 1, 1)
    L = R.seg_line(true_line)
    n = L[:2]
    one = copy.deepcopy(doc)
    one["family_lines"]["A"] = [seg(-0.4, 0, 1.6, 0)]      # the first is free

    def off_truth(r):
        return max(abs(float(L @ np.append(p, 1.0)))
                   for p in np.array(r["value"], float).reshape(2, 2))

    r = snap(one, true_line)
    check("a line already on the square is left there", r["verdict"], pred=lambda v: v == "ok")
    check("...moved by, px", r["conditioning"]["moved_px"], 0.0, 0.05)
    check("...between lines 1 and 2 of the other family",
          r["conditioning"]["between_lines"], pred=lambda s: s.startswith("lines 1 and 2 of"))
    check("...one line's slack in f is worth, px (a hand mark is 10)",
          r["conditioning"]["f_moves_it_px"], pred=lambda v: 0.0 < v < 10.0)
    check("...and it says the span was assumed, not measured",
          r["conditioning"]["focal_span_assumed"], pred=lambda v: v is True)
    hand = np.array(true_line, float) + np.array([*(6 * n), *(-5 * n)])
    r = snap(one, hand)
    check("a hand line 6 px one end, 5 px the other, snaps on", r["verdict"],
          pred=lambda v: v == "ok")
    check("...onto the true side, px worst", off_truth(r), 0.0, 0.05)
    check("...having moved it, px", r["conditioning"]["moved_px"], pred=lambda v: 5.5 < v < 6.5)
    r = snap(one, np.array(true_line, float) + np.array([*(40 * n), *(40 * n)]))
    check("a line 40 px away is somewhere else on purpose: left",
          r["verdict"], pred=lambda v: v == "undetermined")
    check("...and the answer says how far the nearest square was, px",
          (r.get("conditioning") or {}).get("nearest_px"), 40.0, 0.5)

    # a family with no vanishing point of its own is given the camera's
    der = copy.deepcopy(one)
    del der["vps"]["A"]
    der["vps"]["C"] = pv(e1)
    der["vp_names"]["C"] = "the wall"
    der["family_lines"]["C"] = [seg(0, 5, 2, 5), seg(0, 6, 2, 6)]
    der["horizon_vps"] = ["B", "C"]
    r = snap(der, true_line)
    check("one line and no point: the camera puts the other family's partner there",
          r["verdict"], pred=lambda v: v == "ok")
    check("...onto the same true side, px worst", off_truth(r), 0.0, 0.05)
    check("...and says it derived the point",
          r["conditioning"]["vp_of_this_family"], pred=lambda s: s.startswith("derived"))

    # not at right angles in the world: a square is not what they make
    ax = np.array([np.cos(th + np.radians(30)), 0, np.sin(th + np.radians(30))])
    sk = copy.deepcopy(one)
    sk["vps"]["C"], sk["vp_names"]["C"] = sk["vps"]["A"], "the wall"
    sk["vps"]["A"], sk["horizon_vps"] = pv(ax), ["B", "C"]
    r = snap(sk, true_line)
    check("a family 30 deg off square to the other is refused",
          r["verdict"], pred=lambda v: v == "undetermined")
    check("...naming the angle it found, 60 deg", r.get("why") or "",
          pred=lambda s: "60 deg" in s and "not 90" in s)

    bare = copy.deepcopy(one)
    bare["vertical_vp"] = None
    bare["vps"].pop("up")
    r = snap(bare, true_line)
    check("no camera, no square", (r["verdict"], "needs the camera" in (r.get("why") or "")),
          pred=lambda t: t == ("undetermined", True))
    empty = copy.deepcopy(one)
    empty["family_lines"]["A"] = []
    check("no line of this family yet: the first is free",
          snap(empty, true_line)["verdict"], pred=lambda v: v == "undetermined")
    lone = copy.deepcopy(one)
    lone["family_lines"]["B"] = lone["family_lines"]["B"][:1]
    check("one line of the other family is not two",
          snap(lone, true_line)["verdict"], pred=lambda v: v == "undetermined")

    # the camera's own slack: this is the answer's conditioning, not an extra
    Kn, Pn, segn, _, near_doc = scene(depth=3.0)
    near_doc["family_lines"]["A"] = [segn(-0.4, 0, 1.6, 0)]
    r = snap(near_doc, segn(0, 1, 1, 1))
    c = r.get("conditioning") or {}
    check("a near tile's side moves more than a hand mark with f: refused",
          (r["verdict"], c.get("f_moves_it_px", 0) > c.get("mark_accuracy_px", 1e9)),
          pred=lambda t: t == ("undetermined", True))
    check("...and the reason says the span was assumed",
          r.get("why") or "", pred=lambda s: "a tenth either side" in s)
    Kn, Pn, segn, _, obl = scene(depth=3.0, ry=0.9)
    obl["family_lines"]["A"] = [segn(-0.4, 0, 1.6, 0)]
    r = snap(obl, segn(0, 1, 1, 1))
    check("a square that is gone at one end of the focal span: refused",
          (r["verdict"], "at one end of the focal lengths" in (r.get("why") or "")),
          pred=lambda t: t == ("undetermined", True))

    # --- the circle and its reading, through the route
    def cells(d, pts, resolve=None, **kw):
        d = copy.deepcopy(d)
        d["features"] = {"E": dict({"kind": "ellipse", "cells": pts}, **kw)}
        if resolve:
            d["features"]["E"]["resolve"] = resolve
        with A.app.test_client() as cl:
            return cl.post("/api/square_cell", json={"doc": d, "feature": "E"}).get_json()

    r = cells(doc, [P([[0.5, 0.5]])[0].tolist(), P([[0.5, 2.0]])[0].tolist(), [10., 10.]])
    sq, rc, none = r["cells"]
    check("three clicks, three answers, the third none", len(r["cells"]), 3, 0)
    check("the circle the route sends is the projected one, px worst",
          float(np.sqrt(((np.array(sq["ring"])[:, None] - dense[None]) ** 2).sum(-1))
                .min(1).max()), 0.0, 0.05)
    check("a click outside every cell says so, with no ring",
          ("ring" in none, "no cell here" in none.get("why", "")),
          pred=lambda t: t == (False, True))
    rd = sq["reading"]
    check("a true square, at a 5 % resolve, with f only assumed: not settled",
          rd["verdict"], pred=lambda v: v == "undetermined")
    check("...because the focal slack alone reads it up to long : short",
          rd["conditioning"]["aspect_range"][1], pred=lambda v: 1.05 < v < 1.15)
    check("...at the fitted focal length it is exactly square",
          rd["conditioning"]["aspect"], 1.0, 1e-6)
    check("...and says the span was assumed", rd["conditioning"]["focal_span_assumed"],
          pred=lambda v: v is True)
    rd12 = cells(doc, [P([[0.5, 0.5]])[0].tolist()], resolve=0.12)["cells"][0]["reading"]
    check("the same square at a 12 % resolve is ok", rd12["verdict"],
          pred=lambda v: v == "ok")
    rd = rc["reading"]
    check("a 1 x 2 cell is off, 2 : 1", (rd["verdict"], round(rd["value"], 3)),
          pred=lambda t: t == ("off", 2.0))
    check("...and no focal length in the span makes it square",
          rd["conditioning"]["aspect_range"][0], pred=lambda v: v > 1.5)
    check("...longer along the family it runs along", rd["conditioning"]["longer_along"],
          pred=lambda s: s == "tiles across")

    # no camera: the circle is still the inscribed one, and says it is unearned
    nocam = copy.deepcopy(doc)
    nocam["vertical_vp"] = None
    nocam["vps"].pop("up")
    nc = cells(nocam, [P([[0.5, 0.5]])[0].tolist()])["cells"][0]
    check("with no camera the ring is still drawn", len(nc.get("ring") or []), 96, 0)
    check("...and its reading is undetermined, saying why",
          (nc["reading"]["verdict"], "without a camera" in nc["reading"]["why"]),
          pred=lambda t: t == ("undetermined", True))
    few = copy.deepcopy(doc)
    few["family_lines"]["B"] = few["family_lines"]["B"][:1]
    check("a family with one line cannot make a cell: refused, not guessed",
          cells(few, [P([[0.5, 0.5]])[0].tolist()]).get("error") is not None, pred=lambda v: v)

    # --- the Check button: round by construction is not a measurement of the drawing
    dd = copy.deepcopy(doc)
    dd["features"] = {"E": {"kind": "ellipse", "cells": [P([[0.5, 0.5]])[0].tolist()]}}
    with A.app.test_client() as cl:
        ck = cl.post("/api/check", json={"doc": dd}).get_json()["results"]["E"]
    check("circles drawn in squares are not checked against themselves",
          (ck["verdict"], "round by construction" in ck.get("why", "")),
          pred=lambda t: t == ("undetermined", True))
    # ...but a rim marked round what was DRAWN is measured as ever
    tt = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    dd["features"]["E"]["pts"] = P(np.column_stack(
        [0.5 + 0.5 * np.cos(tt), 0.5 + 0.5 * np.sin(tt)])).tolist()
    with A.app.test_client() as cl:
        ck = cl.post("/api/check", json={"doc": dd}).get_json()["results"]["E"]
    check("a rim marked round the drawing is measured, not skipped",
          ck["verdict"], pred=lambda v: v in ("circle", "oval"))


def top_view():
    """The drawing seen from above, read back off the vanishing points.

    "if vanishing points and horizon are established, it should translate onto
    a top view in a small window in the bottom right". The plan is the camera
    seen from above, so each number in it is held against the world the camera
    was built in: the angle each family makes with the line of sight, how far
    back the eye stands, how much of the view the picture spans, where each
    vanishing point sits on the picture plane. And it refuses where the camera
    does, or draws a guide and says so where the camera is only a guide.
    """
    print("\nthe top view  (plan: directions to 1e-9 deg and distances to 1e-6 px "
          "against the world; refused where the camera is)")
    import app as A

    W, H = 1400.0, 1000.0
    up_w = np.array([0, -1.0, 0])                     # y runs down the page
    dirs = {"A": 0.0, "B": np.pi / 2, "T": 0.6}       # a room, and a table askew

    def world(K, Rot):
        """The plan straight from the world: each family's angle off the line
        of sight on the ground, toward the picture's right; the eye's distance
        from the picture plane; and the angle the horizon's stretch across the
        picture subtends."""
        z = Rot.T @ np.array([0, 0, 1.0])
        fwd = z - (z @ up_w) * up_w
        cos_pitch = float(np.linalg.norm(fwd))
        fwd /= cos_pitch
        x = Rot.T @ np.array([1.0, 0, 0])
        right = x - (x @ up_w) * up_w - (x @ fwd) * fwd
        right /= np.linalg.norm(right)

        def ang(d):
            return float(np.degrees(np.arctan2(d @ right, d @ fwd)))
        out = {}
        for k, a in dirs.items():
            d = np.array([np.sin(a), 0, np.cos(a)])
            out[k] = ang(d * np.sign((Rot @ d)[2]))   # the end in front of the eye
        l = vanish_line(K, Rot, up_w)
        rays = [Rot.T @ np.linalg.solve(K, [x, -(l[0] * x + l[2]) / l[1], 1.0])
                for x in (0.0, W)]
        fov = abs(ang(rays[1]) - ang(rays[0]))
        cv = vp_of(K, Rot, fwd)                       # the centre of vision
        along = np.array([l[1], -l[0]]) * np.sign(l[1])  # the horizon, rightwards
        return out, K[0, 0] / cos_pitch, fov, cv, along

    scenes = {"level, centred": camera(1000.0, (W / 2, H / 2), 0.0, 0.35, 0.0),
              "pitched, rolled, off-centre": camera(1250.0, (760.0, 455.0), -0.21, 0.52, 0.07)}
    for label, (Kt, Rot) in scenes.items():
        vps = {k: vp_of(Kt, Rot, [np.sin(a), 0, np.cos(a)]) for k, a in dirs.items()}
        u = Kt @ Rot @ up_w
        if abs(u[2]) > 1e-9:                          # a level camera's are parallel
            vps["up"] = u[:2] / u[2]
        # a slope climbing over the table's run: its vanishing point is off the
        # horizon, and from above it runs where the run does
        vps["S"] = vp_of(Kt, Rot, [np.sin(0.6), -0.5, np.cos(0.6)])
        p = RC.plan(Kt, vps["A"], vps["B"], vps, (W, H))
        want, dist, fov, cv, along = world(Kt, Rot)
        print(f"  -- {label}")
        check("each family's angle off the line of sight, deg worst",
              max(abs(float(np.degrees(np.arctan2(*p["families"][k]["dir"]))) - want[k])
                  for k in dirs), 0.0, 1e-9)
        check("...with the picture's right on the plan's right",
              [int(np.sign(p["families"][k]["dir"][0])) for k in dirs],
              [int(np.sign(want[k])) for k in dirs], 0)
        q = RC.plan(Kt, vps["B"], vps["A"], vps, (W, H))
        check("...whichever way round the horizon's two points come",
              max(float(np.abs(np.subtract(q["families"][k]["dir"], p["families"][k]["dir"])).max())
                  for k in dirs), 0.0, 1e-12)
        check("the eye stands f / cos(pitch) back from the picture plane, px",
              p["distance_px"], dist, 1e-6)
        check("the picture spans the angle its horizon subtends, deg", p["fov_deg"], fov, 1e-9)
        al = along / np.linalg.norm(along)
        check("each vanishing point sits on the picture plane as far along as on "
              "the page's horizon, px worst",
              max(abs(p["families"][k]["at"][0] - float((vps[k] - cv) @ al))
                  + abs(p["families"][k]["at"][1] - dist) for k in dirs), 0.0, 1e-6)
        fs = p["families"]
        check("a slope, from above, runs where its own level run does, deg / px",
              max(abs(float(np.degrees(np.arctan2(*fs["S"]["dir"]) - np.arctan2(*fs["T"]["dir"])))),
                  float(np.abs(np.subtract(fs["S"]["at"], fs["T"]["at"])).max())), 0.0, 1e-6)
        if "up" in vps:
            check("the uprights are a point from above, not a direction",
                  "up" in p["families"], pred=lambda v: v is False)
    Kl, Rl = scenes["level, centred"]
    p = RC.plan(Kl, vp_of(Kl, Rl, [0, 0, 1.0]), vp_of(Kl, Rl, [1.0, 0, 0.3]), {}, (W, H))
    check("level and centred: the picture spans 2 atan(w / 2f), deg", p["fov_deg"],
          float(np.degrees(2 * np.arctan(W / 2 / 1000.0))), 1e-9)

    # --- the page's way in: the document it sends, and what comes back
    Kt, Rot = scenes["pitched, rolled, off-centre"]
    pv = {k: vp_of(Kt, Rot, [np.sin(a), 0, np.cos(a)]).tolist() for k, a in dirs.items()}
    pv["up"] = vp_of(Kt, Rot, up_w).tolist()
    doc = {"image": {"w": W, "h": H}, "width": W, "height": H, "perspective": "three",
           "vertical_vp": "up", "horizon_vps": ["A", "B"], "vps": pv, "features": {}}
    want, dist, fov, _, _ = world(Kt, Rot)
    with A.app.test_client() as cl:
        def ask(d):
            return cl.post("/api/plan", json={"doc": d}).get_json()
        r = ask(doc)
        check("three vanishing points: a plan, measured", r["verdict"], "ok",
              pred=lambda v: v == "ok")
        check("...the eye as far back as in the world, px", r["value"]["distance_px"], dist, 1e-6)
        check("...the field of view inside the range the focal slack allows",
              r["conditioning"]["fov_deg_range"],
              pred=lambda g: g[0] < r["value"]["fov_deg"] < g[1])
        check("...and that slack said to be assumed: the marks gave none",
              r["conditioning"]["focal_span_assumed"], True, pred=lambda v: v is True)
        check("...in pixels only while the picture's real size is not given",
              [k for k in r["value"] if k.endswith("_mm")], pred=lambda v: v == [])
        cm35 = {"value": 35, "unit": "cm", "along": "width"}   # 1400 px: 4 px to the mm
        rs = ask(dict(doc, scale=cm35))
        check("given it, the eye's distance in mm too, mm", rs["value"].get("distance_mm"),
              dist / 4, 1e-3)
        check("...with its range, and the scale the plan is drawn to",
              (rs["conditioning"].get("distance_range_mm"), rs["value"].get("mm_per_px")),
              pred=lambda v: v[0] == [round(x / 4, 3) for x in rs["conditioning"]["distance_range_px"]]
              and v[1] == 0.25)
        r = ask(dict(doc, perspective="one"))
        check("one-point: refused, no focal length to stand the eye at",
              (r["verdict"], r["why"]), pred=lambda v: v[0] == "undetermined"
              and "focal length" in v[1])
        nov = {k: v for k, v in doc.items() if k != "vertical_vp"}
        r = ask(nov)
        check("three-point with no family ticked upright: refused, and says so",
              (r["verdict"], r["why"]), pred=lambda v: v[0] == "undetermined"
              and "vertical" in v[1])
        lv = {k: vp_of(Kl, Rl, [np.sin(a), 0, np.cos(a)]).tolist() for k, a in dirs.items()}
        two = dict(doc, perspective="two", vps=lv)
        two.pop("vertical_vp")
        r = ask(two)
        check("two-point, level: the eye f back, px", r["value"]["distance_px"], 1000.0, 1e-6)
        check("...with the assumed principal point said", r["conditioning"],
              pred=lambda c: "principal point" in c.get("camera_assumed", ""))
        r = ask(dict(doc, perspective="two", scale=cm35))
        check("two-point under a pitched camera's uprights: drawn, as a guide, to scale",
              (r["verdict"], (r["value"] or {}).get("mm_per_px"), r.get("why", "")),
              pred=lambda v: v[0] == "undetermined" and v[1] == 0.25
              and v[2].startswith("drawn from a camera the marks do not measure"))


def camera_in_out():
    """A camera in, and a camera out.

    Another program's solved camera is read into this picture's pixels and the
    marks are priced against it; the camera the marks agree on is written back
    as a Blender script. Every convention between the two -- the lens on the
    LARGER side, the shift, the flipped axes -- is a place to be wrong by a
    mirror, so they are checked three ways: algebraically (always), against
    the fSpy fixture whose own vanishing points were solved by another program
    (always), and by running the exported script in a real Blender (when one is
    installed) and reading the camera back out of the .blend it saves.
    """
    print("\ncamera in and out  (fSpy / Blender / JSON, priced in px of movement)")
    import json
    import subprocess
    import tempfile
    from pathlib import Path
    import app as A
    from core import camera_io as CIO
    from core import consensus as KC

    here = Path(__file__).resolve().parent / "testdata"
    rng = np.random.default_rng(11)

    # ---- the conversions, with no program in the loop
    worst_f = worst_pp = worst_ax = 0.0
    for W, H, f, pp in ((1200, 900, 1400.0, (655.0, 430.0)),
                        (884, 1280, 1100.0, (470.0, 700.0)),      # portrait: the trap
                        (1000, 1000, 900.0, (380.0, 590.0))):     # square, big shift
        Rcv = KC._rodrigues(np.array([0.3, -0.5, 0.2]))
        cam = {"f": f, "pp": list(pp), "size": [W, H]}
        n = CIO.blender_numbers(cam, Rcv)
        back = CIO.from_blender(
            {"name": "c", "type": "PERSP", "lens": n["lens"], "sensor_width": 36.0,
             "sensor_height": 24.0, "sensor_fit": "AUTO", "shift_x": n["shift_x"],
             "shift_y": n["shift_y"], "matrix_world": n["matrix_world"]},
            [W, H, 100, 1, 1])
        cmp_ = CIO.compare(back, f, pp, [Rcv[:, i] for i in range(3)])
        worst_f = max(worst_f, abs(back["f"] - f))
        worst_pp = max(worst_pp, abs(back["pp"][0] - pp[0]), abs(back["pp"][1] - pp[1]))
        worst_ax = max(worst_ax, cmp_["rotation_deg"])
    check("a camera out then in again: f, px", worst_f, 0.0, 1e-9)
    check("...principal point, px", worst_pp, 0.0, 1e-9)
    check("...rotation, deg", worst_ax, 0.0, 1e-6)
    # the lens is laid along the picture's LARGER side: on a portrait picture
    # lens * W / sensor is the wrong answer and it is 31 % off
    check("a portrait picture's lens uses its height, not its width",
          CIO.blender_numbers({"f": 1100.0, "pp": [442.0, 640.0], "size": [884, 1280]},
                              np.eye(3))["lens"], 1100.0 * 36.0 / 1280.0, 1e-9)

    # ---- the fSpy fixture: a camera another program solved
    fs = here / "fspy_untitled.fspy"
    if not fs.exists():
        print("  --    testdata/fspy_untitled.fspy missing, skipped")
        return
    cam = CIO.read(fs)["camera"]
    check("fSpy: focal length, px", cam["f"], 1054.167, 0.01)
    check("fSpy: principal point, px", cam["pp"], [640.0, 480.0], 1e-6)
    check("fSpy: its matrix and its own vanishing points agree, deg",
          cam["source"]["vp_vs_matrix_deg"], 0.0, 0.01)
    check("fSpy: the lens as a photographer says it, mm on 36",
          CIO.describe(cam)["lens_mm_on_36"], 29.648, 0.01)
    check("fSpy: up is read off the picture, and it is Z", cam["up"], pred=lambda u: u == "Z")
    hz = CIO.describe(cam)["horizon"]
    check("fSpy: the horizon sits below the centre (the camera looks up)",
          -hz[2] / hz[1], pred=lambda y: y > 480)

    # ---- priced against the marks that were made IN fSpy
    marks = json.loads((here / "fspy_untitled.marks.json").read_text())
    fam = {L["name"]: L["lines"] for L in marks["layers"]}
    key = {"first": "first VP, as marked in fSpy", "second": "second VP, as marked in fSpy",
           "vert": "verticals, as marked in fSpy"}
    with A.app.test_client() as cl:
        r = cl.post("/api/camera_in", json={"path": "testdata/fspy_untitled.fspy",
                                            "image_size": [1280, 960]}).get_json()
        ref = r["camera"]
        out = cl.post("/api/consensus", json={"families": fam, "image_size": [1280, 960],
                                              "reference": ref}).get_json()
        one = cl.post("/api/consensus", json={"families": {"v": fam[key["vert"]]},
                                              "image_size": [1280, 960],
                                              "reference": ref}).get_json()
        bad = cl.post("/api/camera_in", json={"path": "testdata/fspy_untitled.fspy",
                                              "image_size": [884, 1280]})
        img = cl.post("/api/camera_in", json={"path": "testdata/4.jpg"})
        mk = cl.post("/api/camera_in", json={"path": "testdata/4.marks.json"})
    rc = out["reference"]["conditioning"]
    per = rc["per_family"]
    check("the two families that DEFINE the camera cost nothing, px",
          max(per[key["first"]]["worst_px"], per[key["second"]]["worst_px"]), 0.0, 1e-6)
    check("...the verticals marked beside them do not agree with it, px",
          per[key["vert"]]["worst_px"], 29.6, 0.5)
    check("...which is more than a hand mark is worth", rc["worst_move_px"],
          pred=lambda v: v > rc["mark_accuracy_px"])
    check("...and the verdict says so", out["reference"]["verdict"],
          pred=lambda v: v == "rectifiable")
    check("each family went to the axis its own lines define",
          [rc["axis_of"][key[k]] for k in ("first", "second", "vert")],
          pred=lambda v: v == ["X", "Y", "Z"])
    check("...by a margin far past a hand mark, px", rc["assignment"]["margin_px"],
          pred=lambda v: v > 10 * rc["mark_accuracy_px"])
    check("one family is enough against a reference",
          one["reference"]["conditioning"]["axis_of"]["v"], pred=lambda a: a == "Z")
    check("...and without a reference it is still refused",
          A.app.test_client().post("/api/consensus", json={
              "families": {"v": fam[key["vert"]]}, "image_size": [1280, 960]}).status_code,
          400, 0)
    check("the marks say so from their own side too: price of the reference, px",
          rc["price_of_reference_px"], 29.6, 0.5)
    check("every one of those exactly-determined fits says it has no redundancy",
          out["free"]["conditioning"]["redundancy"], 0, 0)
    check("...and so passes no verdict: 6 lines fit 6 numbers whatever they are",
          out["free"]["verdict"], pred=lambda v: v == "undetermined")
    check("...and says what would test it", out["free"].get("why", ""),
          pred=lambda w: "6 lines for the camera's 6 numbers" in w and "third line" in w)
    with A.app.test_client() as cl:
        cw = cl.post("/api/camera_out", json={
            "families": fam, "image_size": [1280, 960], "roles": {key["vert"]: "vertical"},
            "picture": "testdata/fspy_untitled.jpg"})
    cwd = cw.get_json()
    check("such a camera is still written, because it is the one those lines define",
          cw.status_code, 200, 0)
    check("...with the header saying it was not tested",
          "NOT TESTED" in cwd["script"].split("import bpy")[0], pred=bool)
    check("...and the reference card knows its baseline is vacuous",
          rc["own_redundancy"], 0, 0)
    with A.app.test_client() as cl:
        cn = cl.post("/api/camera_out", json={
            "families": {"L3": fam[key["vert"]], "L1": fam[key["first"]],
                         "L2": fam[key["second"]]}, "image_size": [1280, 960],
            "roles": {"L3": "vertical"}, "names": {"L3": "the uprights"}}).get_json()
    check("a layer is called by its own name in the header, not its id",
          any("'the uprights'" in n for n in cn["conditioning"]["notes"]), pred=bool)
    check("a camera of another shape is refused, in words", bad.get_json().get("error", ""),
          pred=lambda e: "aspect" in e and "same shape" in e)
    check("a picture is not a camera file", img.status_code, 400, 0)
    check("a marks file is not a camera", mk.get_json().get("error", ""),
          pred=lambda e: "neither a Blender camera" in e)

    # ---- plain JSON, both forms
    own = {"f": 1100.0, "pp": [442.0, 640.0], "size": [884, 1280],
           "axes": {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}}
    pj = CIO.from_json(own)
    check("a Persimmon camera in JSON reads back as it was", [pj["f"], *pj["pp"]],
          [1100.0, 442.0, 640.0], 1e-9)
    # 50 mm across the LONG side of a portrait frame: 800 x 1200 -> 1200 / 36 px per mm
    bj = CIO.from_json({"lens": 50.0, "matrix_world": np.eye(4).tolist(),
                        "resolution": [800, 1200]})
    check("a Blender camera in JSON: the lens is laid along the long side, px", bj["f"],
          50.0 * 1200 / 36.0, 1e-9)
    check("...and its principal point is the middle", bj["pp"], [400.0, 600.0], 1e-9)
    try:
        CIO.from_json({"hello": 1})
        refused = False
    except ValueError:
        refused = True
    check("JSON that is no camera is refused, in words", refused, pred=bool)

    # ---- the pricing is what it says: a camera that IS the drawing's costs hand-shake
    f0, pp0, size = 1100.0, np.array([442.0, 640.0]), (884, 1280)
    K0 = np.array([[f0, 0, pp0[0]], [0, f0, pp0[1]], [0, 0, 1.0]])
    R0 = KC._rodrigues(np.array([0.30, 0.45, 0.02]))
    V0 = (K0 @ R0).T

    def drawn(v, n=9, shake=0.3):
        o = []
        for _ in range(n):
            m = rng.uniform([100, 100], [780, 1100])
            d = (v[:2] / v[2] - m) if abs(v[2]) > 1e-12 else v[:2]
            d = d / np.linalg.norm(d)
            L = rng.uniform(150, 600)
            t = np.radians(rng.normal(0, shake))
            d = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ d
            o.append([*(m - d * L / 2), *(m + d * L / 2)])
        return o

    fams = {"a": drawn(V0[0]), "up": drawn(V0[1]), "b": drawn(V0[2])}
    camd = {"f": f0, "pp": pp0.tolist(), "size": list(size),
            "axes": {a: R0[:, i].tolist() for i, a in enumerate("XYZ")}, "up": None,
            "source": {}}
    with A.app.test_client() as cl:
        r1 = cl.post("/api/consensus", json={"families": fams, "image_size": size,
                                             "reference": camd}).get_json()
        # the same camera with its axes listed in another order: the marks, not
        # the labels, decide which line goes where
        sh = dict(camd, axes={"X": camd["axes"]["Z"], "Y": camd["axes"]["X"],
                              "Z": camd["axes"]["Y"]})
        r2 = cl.post("/api/consensus", json={"families": fams, "image_size": size,
                                             "reference": sh}).get_json()
        # eight degrees of yaw off
        Rw = KC._rodrigues(np.array([0.0, np.radians(8), 0.0])) @ R0
        wr = dict(camd, axes={a: Rw[:, i].tolist() for i, a in enumerate("XYZ")})
        r3 = cl.post("/api/consensus", json={"families": fams, "image_size": size,
                                             "reference": wr}).get_json()
    c1, c2, c3 = (x["reference"]["conditioning"] for x in (r1, r2, r3))
    check("the camera the drawing came from: worst move, px", c1["worst_move_px"], 0.0, 8.0)
    check("...already consistent", r1["reference"]["verdict"],
          pred=lambda v: v == "already consistent")
    check("...and the free fit has marks to spare", r1["free"]["conditioning"]["redundancy"],
          pred=lambda v: v > 0)
    check("each family went to the axis it was drawn toward",
          [c1["axis_of"][k] for k in ("a", "up", "b")], pred=lambda v: v == ["X", "Y", "Z"])
    check("relabelled axes, same price", c2["worst_move_px"], c1["worst_move_px"], 1e-6)
    check("...and every family follows its own vanishing point to its new label",
          [c2["axis_of"][k] for k in ("a", "up", "b")], pred=lambda v: v == ["Y", "Z", "X"])
    check("a camera 8 deg of yaw out costs real pixels, worst px", c3["worst_move_px"],
          pred=lambda v: v > 3 * c1["mark_accuracy_px"])

    # a level camera's uprights are parallel on the page: they must be CORRECTED
    # to parallel, which used to be drawn nowhere (a vanishing point at infinity
    # has no coordinates to rectify toward)
    Rl = KC._rodrigues(np.array([0.0, 0.4, 0.0]))
    Vl = (K0 @ Rl).T
    lf = {"a": drawn(Vl[0], shake=1.0), "v": drawn(Vl[1], shake=1.0)}
    lcam = dict(camd, axes={a: Rl[:, i].tolist() for i, a in enumerate("XYZ")})
    with A.app.test_client() as cl:
        rl = cl.post("/api/consensus", json={"families": lf, "image_size": size,
                                             "reference": lcam}).get_json()
    pv = rl["reference"]["conditioning"]["per_family"]["v"]
    ang = [np.degrees(np.arctan2(q[3] - q[1], q[2] - q[0])) % 180.0 for q in pv["corrected"]]
    check("a level camera: uprights are at infinity", pv["at_infinity"], pred=bool)
    check("...and their corrections all run one way, deg spread", float(np.ptp(ang)), 0.0, 1e-6)
    check("...the way the camera says, deg", float(np.mean(ang)), pv["direction_deg"], 1e-3)

    # ---- the camera out, from marks, and back through a real Blender
    body = {"families": fams, "image_size": list(size), "mode": "three",
            "roles": {"up": "vertical"}, "picture": "testdata/4.jpg", "yaw": 0}
    with A.app.test_client() as cl:
        o = cl.post("/api/camera_out", json=body)
        o1 = cl.post("/api/camera_out", json=dict(body, mode="one"))
        o2 = cl.post("/api/camera_out", json=dict(body, families={"a": fams["a"]}))
        o3 = cl.post("/api/camera_out", json=dict(body, roles={}))
        o4 = cl.post("/api/camera_out", json=dict(body, yaw=1))
    od = o.get_json()
    check("camera out: written", o.status_code, 200, 0)
    check("...the lens, mm on 36 (true 1100 px on a 1280 side = 30.9)", od["value"],
          1100.0 * 36 / 1280, 0.6)
    check("...says what it rests on", od["conditioning"]["notes"],
          pred=lambda n: any("lines marked" in x for x in n)
          and any("flagged vertical" in x for x in n))
    check("...a picture is attached", od["conditioning"]["picture_attached"], pred=bool)
    check("one-point has no lens, so it writes none", o1.status_code, 422, 0)
    check("...and says why", o1.get_json()["error"], pred=lambda e: "focal length" in e)
    check("one family is not a camera", o2.status_code, 422, 0)
    check("no vertical flagged: the prior is used, and says so",
          o3.get_json()["conditioning"]["up_from"], pred=lambda u: "nothing was flagged" in u)

    exe = CIO.blender_exe()
    if exe is None:
        print("  --    no Blender found; the export is not run through a real one")
        return
    # What a pinhole says those world points do, from the same fit the route ran
    # (so this is Blender's lens/shift arithmetic against the plain formula, with
    # nothing of this program's inverse conversion in between)
    WORLD_PTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.4, -0.9, 0.7],
                          [-2.0, 1.5, 3.0]])
    fr = KC.fit({k: np.asarray(v, float) for k, v in fams.items()}, image_size=size,
                mode="three", roles={"up": "vertical"})["conditioning"]
    Rw = CIO.world_frame(fr["axes_cam"], fr["per_family"]["up"]["axis_index"], 0)
    Kp = np.array([[fr["focal_px"], 0, fr["principal_point"][0]],
                   [0, fr["focal_px"], fr["principal_point"][1]], [0, 0, 1.0]])
    pc = (Rw @ WORLD_PTS.T).T + np.array([0.0, 0.0, 10.0])   # 10 units in front
    uv = (Kp @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    probe = r"""
import bpy, json
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
sc = bpy.context.scene
W, H = sc.render.resolution_x, sc.render.resolution_y
def px(p):
    v = world_to_camera_view(sc, sc.camera, Vector(p))
    return [v.x * W, (1 - v.y) * H, v.z]
print("@@PROBE@@" + json.dumps({"o": px((0, 0, 0)), "z": px((0, 0, 1)), "x": px((1, 0, 0)),
                                "pts": [px(p) for p in PTS],
                                "bg": len(sc.camera.data.background_images)}))
bpy.ops.wm.save_as_mainfile(filepath=SAVE_AS)
"""
    with tempfile.TemporaryDirectory() as d:
        got = {}
        for tag, resp in (("front", od), ("turned", o4.get_json())):
            sp, pp_, bl = (Path(d) / f"{tag}.py", Path(d) / f"{tag}_probe.py",
                           Path(d) / f"{tag}.blend")
            sp.write_text(resp["script"])
            pp_.write_text(f"SAVE_AS = {str(bl)!r}\nPTS = {WORLD_PTS.tolist()!r}\n" + probe)
            so, se, code = CIO.run_blender(["--python", str(sp), "--python", str(pp_)])
            line = next((l for l in so.splitlines() if l.startswith("@@PROBE@@")), None)
            got[tag] = (json.loads(line[9:]) if line else None,
                        CIO.read_blend(bl)[0][1] if bl.exists() else None)
    pr, back = got["front"]
    check("Blender ran the script and saved", back, pred=lambda b: isinstance(b, dict))
    cd = od["conditioning"]
    check("...f read back out of the .blend, px", back["f"], cd["focal_px"], 1e-3)
    check("...principal point, px", back["pp"], cd["principal_point"], 1e-3)
    check("...the world's origin lands on the principal point, px", pr["o"][:2],
          cd["principal_point"], 1e-3)
    check("...+Z is up the picture (a smaller y), px above the origin", pr["o"][1] - pr["z"][1],
          pred=lambda v: v > 20)
    check("...+X runs to the right, px", pr["x"][0] - pr["o"][0], pred=lambda v: v > 20)
    check("...six world points, px worst against the plain pinhole formula",
          float(np.abs(np.array(pr["pts"])[:, :2] - uv).max()), 0.0, 0.01)
    check("...the picture is behind it", pr["bg"], 1, 0)
    # the axes it wrote ARE the ones the marks agreed on
    check("...the three axes are the drawing's, deg",
          CIO.compare(back, f0, pp0, [R0[:, i] for i in range(3)])["rotation_deg"],
          0.0, 1.0)
    pt, _ = got["turned"]
    check("a quarter turn moves where +X lands, px",
          float(np.hypot(pt["x"][0] - pr["x"][0], pt["x"][1] - pr["x"][1])),
          pred=lambda v: v > 20)
    check("...and leaves up alone, px", pt["z"][:2], pr["z"][:2], 1e-3)


def five_point():
    """5-point perspective, checked as the stereographic projection it is.

    Compass-drawn 5-point puts every straight edge on a circle through two
    opposite points of the circle 90 deg out, which is what the stereographic
    projection does exactly. So a scene is built in it, marked as runs of
    points, and the camera has to come back; then the refusals that PLAN.md
    sec 50 measured, and the guard that stops linear perspective calling such
    a drawing consistent.
    """
    print("\n5-point  (the sphere's camera off runs of points; the curved-line guard)")
    import app as A
    from core import curvi as CV
    from scipy.spatial.transform import Rotation

    W, H, c0, f0 = 1200, 900, np.array([610.0, 440.0]), 220.0
    R0 = Rotation.from_rotvec([0.15, -0.2, 0.05]).as_matrix()   # rows: the world's axes

    def scene(nl=3, k=4, sd=0.0, seed=0, nfam=3, f=f0):
        """{family: runs of k points along nl straight edges}, as drawn in 5-point."""
        rng = np.random.default_rng(seed)
        out = {}
        for a in range(nfam):
            runs = []
            while len(runs) < nl:
                X = rng.uniform(-3, 3, 3)
                Q = X + np.linspace(-12, 12, 2000)[:, None] * R0[a]
                D = Q / np.linalg.norm(Q, axis=1, keepdims=True)
                p = CV.project(D, c0, f)
                i = np.flatnonzero((D[:, 2] > 0) & (p[:, 0] > 0) & (p[:, 0] < W)
                                   & (p[:, 1] > 0) & (p[:, 1] < H))
                if len(i) < 20 or np.any(np.diff(i) > 1):
                    continue
                s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p[i], axis=0).T))])
                if s[-1] < 0.15 * np.hypot(W, H):
                    continue
                j = np.interp(np.linspace(0.05, 0.95, k) * s[-1], s, np.arange(len(i)))
                runs.append((p[i][np.round(j).astype(int)] + rng.normal(0, sd, (k, 2))).tolist())
            out[f"F{a}"] = runs
        return out

    def turn(c):
        A_ = np.asarray(c["axes_cam"])
        return max(float(np.degrees(np.arccos(min(1.0, abs(A_[a] @ R0[a])))))
                   for a in range(len(c["per_family"])))

    D = np.random.default_rng(0).normal(size=(50, 3))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    D[:, 2] = np.abs(D[:, 2])
    check("the projection and its inverse undo each other",
          float(np.abs(CV.rays(CV.project(D, c0, f0), c0, f0) - D).max()), 0.0, 1e-9)

    exact = CV.fit(scene(), image_size=(W, H))
    ce = exact["conditioning"]
    check("exact marks: the scale, %", 100 * abs(ce["focal_px"] / f0 - 1), 0.0, 0.1)
    check("...the centre, px", float(np.hypot(*(np.asarray(ce["principal_point"]) - c0))),
          0.0, 0.5)
    check("...the three directions, deg", turn(ce), 0.0, 0.05)
    check("...and nothing to move", exact["verdict"], pred=lambda v: v == "already consistent")
    noisy = CV.fit(scene(sd=3.0, seed=1), image_size=(W, H))
    cn = noisy["conditioning"]
    check("marked by hand (3 px): the scale within 3 %",
          100 * abs(cn["focal_px"] / f0 - 1), pred=lambda v: v < 3)
    check("...the directions within 1.5 deg", turn(cn), pred=lambda v: v < 1.5)
    check("...and still consistent", noisy["verdict"], pred=lambda v: v == "already consistent")

    off = scene(sd=1.0, seed=1)
    off["F1"][2] = (np.asarray(off["F1"][2]) + [0.0, 30.0]).tolist()
    moved = CV.fit(off, image_size=(W, H))
    pm = moved["conditioning"]["per_family"]["F1"]
    check("a line drawn 30 px off its arc has to move", pm["worst_px"],
          pred=lambda v: v > moved["conditioning"]["mark_accuracy_px"])
    check("...and it is that line", pm["worst_line"], 2, 0)
    check("...so the drawing is rectifiable", moved["verdict"],
          pred=lambda v: v == "rectifiable")

    ends = CV.fit({k: [[r[0], r[-1]] for r in v] for k, v in scene().items()},
                  image_size=(W, H))
    check("lines marked by their two ends show no bend: refused", ends["verdict"],
          pred=lambda v: v == "undetermined")
    sparse = CV.fit(scene(nl=2, k=3, sd=3.0, nfam=2, seed=4), image_size=(W, H))
    check("two lines of three points a family: no camera quoted",
          sparse["conditioning"].get("focal_px"), pred=lambda v: v is None)

    curv = scene(sd=1.0, seed=2)
    rng = np.random.default_rng(5)
    lin = {}
    for name, v in (("a", (2400.0, 420.0)), ("b", (-1800.0, 450.0))):
        lin[name] = []
        for _ in range(3):
            p = rng.uniform([200, 150], [1000, 750])
            lin[name].append([(p + t * (np.asarray(v) - p)).tolist() for t in (0.0, 0.1, 0.2)])
    with A.app.test_client() as cl:
        def post(url, **kw):
            return cl.post(url, json=dict(families={}, image_size=[W, H], **kw))
        three = post("/api/consensus", curves=curv, mode="three").get_json()["free"]
        five = post("/api/consensus", curves=curv, mode="five").get_json()["free"]
        out3 = post("/api/camera_out", curves=curv, mode="three")
        out5 = post("/api/camera_out", curves=curv, mode="five")
        straight = post("/api/consensus", curves=lin, mode="three").get_json()["free"]
        vp = cl.post("/api/fit_vp", json={"lines": [], "curves": lin["a"],
                                          "image": [W, H]}).get_json()
        chk = cl.post("/api/check", json={"doc": {"perspective": "five", "features": {
            "r": {"kind": "posts", "pts": [[0, 0], [10, 10], [20, 20]]}}}}).get_json()
    check("a 5-point drawing checked as 3-point is refused, not called consistent",
          three["verdict"], pred=lambda v: v == "undetermined")
    check("...naming the lines that bend",
          sum(map(len, three["conditioning"]["curved_lines"].values())), pred=lambda v: v >= 3)
    check("...and no camera is written for it", out3.status_code, 422, 0)
    check("checked as 5-point it is consistent", five["verdict"],
          pred=lambda v: v == "already consistent")
    check("...with each line's arc and its whole line to draw",
          [len(five["conditioning"]["per_family"]["F0"][k]) for k in ("corrected", "guides")],
          [3, 3], 0)
    check("no 5-point camera is written yet", out5.status_code, 422, 0)
    check("straight lines marked by three points are not refused",
          "curved_lines" in straight["conditioning"], pred=lambda v: v is False)
    check("...and each counts as a line toward its vanishing point",
          vp["conditioning"]["n_lines"], 3, 0)
    check("the straight-line checks refuse in 5-point", chk["results"]["r"]["verdict"],
          pred=lambda v: v == "undetermined")


def main():
    print(__doc__.splitlines()[0])
    conic_fits()
    calibration()
    circle_vs_oval()
    ellipses()
    askew_table()
    square_partner()
    ellipse_plane_default()
    ellipse_drawn_square_on()
    ellipse_family_level()
    circle_in_a_square()
    top_view()
    arch_one_bezier()
    arch_point_on_the_curve()
    arch_handle_pairs()
    camera_unpinned_by_marks()
    two_point_never_looked_at_its_verticals()
    arch_lean_keeps_the_crown()
    arch_rib_pair_handles()
    arch_vault()
    arch_gothic()
    arch_straight_roof()
    shadows()
    horizon_families()
    pdf_export()
    spaced_rows()
    straightening()
    arches()
    arch_build()
    polygons()
    turning_and_threshold()
    conditioning_refuses()
    short_segment_guard()
    detection()
    figure_module()
    trim_module()
    trim_on_photographs()
    perspective_modes()
    real_size()
    scene_distance()
    stray_lines()
    camera_in_out()
    five_point()
    fixture_4jpg()
    print(f"\n{len(OK)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
