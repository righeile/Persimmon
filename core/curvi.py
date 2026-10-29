"""5-point perspective, checked as the projection it is.

5-point perspective draws a straight edge as an arc through two opposite
points of a circle, and an edge coming straight at the viewer as a straight
line to the circle's centre. That construction is exactly the stereographic
projection of the view sphere: it keeps circles circles, and every straight
line in the scene lands on a circle through two opposite points of the circle
90 deg out from the centre. So a compass-drawn 5-point picture approximates
nothing. It has a camera, and it is checked the way a linear drawing is
checked against a pinhole: the nearest consistent 5-point picture, and how far
each marked line has to move to meet it.

The camera is a centre c, a scale f and three perpendicular directions. A
direction d = (x, y, z) in the camera's frame (x right, y down, z ahead) lands
at c + 2 f (x, y) / (1 + z): straight ahead is the centre, 90 deg out is the
circle of radius 2 f, and straight behind is infinitely far away.

A marked line is a run of points. The rays to them lie on one plane through
the eye, and the planes of one family all contain the family's direction, so
each line costs one number (which plane) and the camera six. A line's move is
how far its farthest point is from the curve its plane draws.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from core.consensus import FAR, FMAX, _mark_basis, _mark_px

# The refusal rules PLAN.md §47.3 measured: of 600 fits, the 20 that came out
# more than 10 % off in f were all caught by a standard error over 10 % of f,
# by f at its floor, or by no redundancy -- and only two good fits were.
# The rules on the turn and on spare points are §50's: of 300 fits, the wrong
# answers that went unflagged fell from 9 to 2.
SE_MAX = 0.10
SE_TURN = 2.0       # deg, on the turn: past it the vanishing points are guesses
SPARE = 4           # points to spare before a standard error means anything
F_FLOOR = 0.02      # x the picture diagonal: a circle that small holds no drawing


def project(D, c, f):
    """Unit directions in the camera's frame, (n, 3), to points on the page."""
    D = np.atleast_2d(D)
    return np.asarray(c, float) + 2.0 * f * D[:, :2] / np.maximum(1.0 + D[:, 2:], 1e-12)


def rays(P, c, f):
    """Points on the page, (n, 2), to unit directions from the eye."""
    q = (np.atleast_2d(np.asarray(P, float)) - c) / (2.0 * f)
    s = (q ** 2).sum(1, keepdims=True)
    return np.hstack([2.0 * q, 1.0 - s]) / (1.0 + s)


def chord(pts):
    """A run of points as the straight line nearest it, and how far it bends.

    The bend is how far the farthest point lies from that line: about the
    least move that would make the mark straight. The line runs between the feet of the
    first and last points, so it covers what was marked.
    """
    P = np.asarray(pts, float)
    m = P.mean(0)
    _, v = np.linalg.eigh((P - m).T @ (P - m))
    t = (P - m) @ v[:, 1]
    return (np.concatenate([m + t[0] * v[:, 1], m + t[-1] * v[:, 1]]),
            float(np.abs((P - m) @ v[:, 0]).max()))


def _plane(r):
    """The unit normal of the plane through the eye nearest some rays."""
    n = np.cross(r[0], r[1]) if len(r) == 2 else np.linalg.eigh(r.T @ r)[1][:, 0]
    return n / np.linalg.norm(n)


def _directions(fams, c, f):
    """Each family's direction off its own marks at a trial lens, made square."""
    D = []
    for fam in fams:
        N = np.array([_plane(rays(P, c, f)) for P in fam])
        D.append(np.linalg.eigh(N.T @ N)[1][:, 0])
    D = np.array(D)
    if len(D) == 2:
        D = np.vstack([D, np.cross(D[0], D[1])])
    U, _, Vt = np.linalg.svd(D)
    A = U @ Vt
    if np.linalg.det(A) < 0:
        A[2] = -A[2]          # a direction and its opposite are one family
    return A


def _frames(A, refs):
    """Two unit vectors square to each direction, moving with it."""
    E1 = np.cross(A, refs)
    E1 /= np.linalg.norm(E1, axis=1, keepdims=True)
    return E1, np.cross(A, E1)


def _unpack(x, pin, A0):
    c, x = (x[:2], x[2:]) if pin is None else (pin, x)
    A = A0 @ Rotation.from_rotvec(x[1:4]).as_matrix().T
    return c, float(np.exp(x[0])), A, x[4:]


def _normals(A, refs, fam_of, psi):
    E1, E2 = _frames(A[:len(refs)], refs)
    return np.cos(psi)[:, None] * E1[fam_of] + np.sin(psi)[:, None] * E2[fam_of]


def _resid(x, P, idx, fam_of, pin, A0, refs):
    """Every marked point's distance from its line's curve, in page pixels.

    The angle off the plane times the projection's own scale there, 2f/(1+z)
    px a radian: stereographic is conformal, so that is one number however
    the curve runs through the point.
    """
    c, f, A, psi = _unpack(x, pin, A0)
    r = rays(P, c, f)
    s = (r * _normals(A, refs, fam_of, psi)[idx]).sum(1)
    return 2.0 * f / np.maximum(1.0 + r[:, 2], 1e-9) * np.arcsin(np.clip(s, -1.0, 1.0))


def _start(fams, c0, f0, pin):
    A0 = _directions(fams, c0, f0)
    refs = np.eye(3)[np.argmin(np.abs(A0[:len(fams)]), axis=1)]
    E1, E2 = _frames(A0[:len(fams)], refs)
    psi = [np.arctan2(n @ E2[a], n @ E1[a])
           for a, fam in enumerate(fams) for n in (_plane(rays(P, c0, f0)) for P in fam)]
    x0 = np.concatenate([[] if pin is not None else c0, [np.log(f0)], np.zeros(3), psi])
    return x0, A0, refs


def _arc(n, ra, rb, c, f, k=24):
    """The curve a plane draws, between the feet of two rays on it."""
    a = ra - (ra @ n) * n
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    q = rb - (rb @ n) * n
    t = np.linspace(0.0, np.arctan2(q @ b, q @ a), k)
    return project(np.cos(t)[:, None] * a + np.sin(t)[:, None] * b, c, f)


def _guide(n, d, r, c, f, k=64):
    """The whole line a plane draws: from one vanishing point to the other,
    through the side the marks are on, and not round the back."""
    m = np.cross(n, d)
    if (r @ m).sum() < 0:
        m = -m
    t = np.linspace(0.0, np.pi, k)
    D = np.cos(t)[:, None] * d + np.sin(t)[:, None] * m
    return project(D[D[:, 2] > -0.9], c, f)


def _great_circle(u, c, f, k=181):
    """The great circle square to u, drawn from behind round to behind."""
    e1 = np.cross(u, np.eye(3)[np.argmin(np.abs(u))])
    e1 /= np.linalg.norm(e1)
    t = np.linspace(0.0, 2 * np.pi, k)[:-1]
    D = np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * np.cross(u, e1)
    D = np.roll(D, -int(np.argmin(D[:, 2])), axis=0)
    return project(D[D[:, 2] > -0.9], c, f)


def _vp(d, c, f, diag):
    p = project(d, c, f)[0]
    return None if not np.isfinite(p).all() or np.hypot(*(p - c)) > FAR * diag else p


def _r(a):
    return np.round(np.asarray(a, float), 1).tolist()


def fit(families, *, image_size, roles=None, pp_fixed=None):
    """The nearest consistent 5-point picture, and what reaching it costs.

    `families` maps a name to its marks, each a run of two or more points
    [[x, y], ...] along one straight edge of the scene. Two or three families,
    taken to be perpendicular in the world, as for linear perspective; two
    already fix the centre, the scale and the turn, because in 5-point every
    family vanishes at two finite points. `roles` may name the upright family
    ({"id": "vertical"}), which only decides which great circle is drawn as
    the horizon: the camera is free to point anywhere, so no labelling of the
    families costs more than another.

    The centre is measured, since a drawing's circle is where the artist put
    it. When the marks cannot place it, it is pinned at the picture's centre
    and the answer says so.
    """
    w, h = float(image_size[0]), float(image_size[1])
    mark, diag = _mark_px(image_size)
    names = [n for n in families if len(families[n])]
    if len(names) not in (2, 3):
        return {"verdict": "undetermined",
                "why": (f"{len(names)} line families; 5-point needs two or three "
                        "perpendicular directions, as linear perspective does"),
                "conditioning": {"families": len(names), "mode": "five"}}
    fams = [[np.asarray(P, float).reshape(-1, 2) for P in families[n]] for n in names]
    # A line marked by its two ends shows no bend, and the bend is what 5-point
    # is read from: measured, families of two-point marks fit a camera 40 deg
    # and 50 % wrong as tightly as the right one.
    thin = [n for n, fam in zip(names, fams) if sum(len(P) >= 3 for P in fam) < 2]
    if thin:
        return {"verdict": "undetermined",
                "why": (f"{len(thin)} of the {len(names)} families "
                        f"{'has' if len(thin) == 1 else 'have'} fewer than two lines "
                        "marked with three or more points. 5-point is read from how "
                        "the lines bend, and a line marked by its two ends shows "
                        "none: click along each line and press Enter to end it"),
                "conditioning": {"thin_families": thin, "mode": "five",
                                 "perspective": "five-point"}}
    marks = [P for fam in fams for P in fam]
    fam_of = np.array([a for a, fam in enumerate(fams) for _ in fam])
    idx = np.concatenate([np.full(len(P), j) for j, P in enumerate(marks)])
    P = np.vstack(marks)
    pin = None if pp_fixed is None else np.asarray(pp_fixed, float)
    c0 = np.array([w / 2, h / 2]) if pin is None else pin

    # The scale is unknown and the fit is not convex in it, so start from a
    # spread of circles, from a fifth of the picture across to four times it.
    got = []
    for s in (0.1, 0.17, 0.28, 0.45, 0.75, 1.25, 2.0):
        try:
            x0, A0, refs = _start(fams, c0, s * min(w, h), pin)
            sol = least_squares(_resid, x0, args=(P, idx, fam_of, pin, A0, refs),
                                method="trf", x_scale="jac", max_nfev=400)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if np.isfinite(sol.cost):
            got.append((sol, A0, refs))
    if not got:
        return {"verdict": "undetermined", "why": "the fit did not converge",
                "conditioning": {"mode": "five"}}
    got.sort(key=lambda g: g[0].cost)
    sol, A0, refs = got[0]
    c, f, A, psi = _unpack(sol.x, pin, A0)
    # The standard errors below are the fit's curvature where it stopped, and
    # say nothing of a second camera elsewhere that fits as well. The starts
    # do: one that lands on a different lens or turn, for about the same
    # residual, means the marks cannot choose between them.
    rms = np.sqrt(2 * sol.cost / len(P))
    rival = None
    for g in got[1:]:
        _c, _f, _A, _ = _unpack(g[0].x, pin, g[1])
        turn = float(np.degrees(np.arccos(np.clip(np.abs((A * _A).sum(1)), 0, 1)).max()))
        if (np.sqrt(2 * g[0].cost / len(P)) <= max(1.25 * rms, rms + 0.1 * mark)
                and F_FLOOR * diag < _f < FMAX * diag
                and (abs(_f / f - 1) > SE_MAX or turn > 5.0)):
            rival = (_f, turn)
            break
    res, npar = sol.fun, len(sol.x)
    red = len(P) - npar

    # standard errors from the fit's own curvature and scatter
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * (res @ res) / max(red, 1)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(npar, np.inf)
    if red <= 0:
        se = np.full(npar, np.inf)
    k = 0 if pin is not None else 2
    se_f = f * float(se[k])
    se_c = None if pin is not None else float(max(se[0], se[1]))
    se_rot = float(np.degrees(se[k + 1:k + 4].max()))

    # A centre the marks cannot place is pinned, and said to be -- the free
    # one looks better by every residual, which is why it has to be refused
    # out loud rather than quoted.
    if pin is None:
        bad = None
        if not np.isfinite(se_c) or se_c > 0.05 * diag:
            bad = ("the marks place the centre only to "
                   + (f"+-{se_c:.0f} px" if np.isfinite(se_c) else "no precision at all"))
        elif not (-0.25 * w <= c[0] <= 1.25 * w and -0.25 * h <= c[1] <= 1.25 * h):
            bad = f"the fit put the centre at ({c[0]:.0f}, {c[1]:.0f}), off the picture"
        if bad:
            out = fit(families, image_size=image_size, roles=roles, pp_fixed=(w / 2, h / 2))
            if "conditioning" in out:
                out["conditioning"]["pp_refused"] = bad
                out["conditioning"]["pp_assumed"] = "the picture's centre"
            return out

    N = _normals(A, refs, fam_of, psi)
    R = rays(P, c, f)
    moves = np.array([np.abs(res[idx == j]).max() for j in range(len(marks))])
    per, start = {}, 0
    for a, (name, fam) in enumerate(zip(names, fams)):
        j = np.arange(start, start + len(fam))
        start += len(fam)
        d = A[a]
        front, back = (d, -d) if d[2] >= 0 else (-d, d)
        vf, vb = _vp(front, c, f, diag), _vp(back, c, f, diag)
        mv = moves[j]
        per[name] = {
            "move_px": mv.tolist(), "median_px": float(np.median(mv)),
            "worst_px": float(mv.max()), "worst_line": int(np.argmax(mv)),
            "vp": None if vf is None else vf.tolist(),
            "vp_back": None if vb is None else vb.tolist(),
            "direction_cam": d.tolist(),
            # the curve each line is moved onto, over the stretch marked, and
            # the whole line it belongs to, vanishing point to vanishing point
            "corrected": [_r(_arc(N[i], R[idx == i][0], R[idx == i][-1], c, f)) for i in j],
            "guides": [_r(_guide(N[i], d, R[idx == i], c, f)) for i in j]}

    vert = [names.index(n) for n, v in (roles or {}).items()
            if v == "vertical" and n in names]
    up = A[vert[0]] if vert else np.cross(A[0], A[1])
    # through the centre, edge to edge: past 180 deg the angle BETWEEN two rays
    # folds back, and a 5-point picture is often wider than that
    th = np.degrees(np.arccos(rays([[0, c[1]], [w, c[1]], [c[0], 0], [c[0], h]], c, f)[:, 2]))
    field = [float(th[0] + th[1]), float(th[2] + th[3])]
    cond = {"n_lines": len(marks), "n_points": int(len(P)),
            "median_move_px": float(np.median(moves)), "worst_move_px": float(moves.max()),
            "cost_rms_px": float(np.sqrt(np.mean(res ** 2))),
            "mark_accuracy_px": round(mark, 2), "mark_accuracy_basis": _mark_basis(diag),
            "focal_px": f, "focal_se_px": round(se_f, 1) if np.isfinite(se_f) else None,
            "principal_point": c.tolist(),
            "centre_se_px": None if se_c is None or not np.isfinite(se_c) else round(se_c, 1),
            "rim_radius_px": 2 * f, "field_deg": [round(x, 1) for x in field],
            "axes_se_deg": round(se_rot, 2) if np.isfinite(se_rot) else None,
            "lens": "stereographic", "mode": "five", "perspective": "five-point",
            "horizon": _r(_great_circle(up / np.linalg.norm(up), c, f)),
            "horizon_from": ("square to the upright family" if vert else
                             "the first two families, taken as the horizontals"),
            "axes_cam": A.tolist(), "per_family": per, "n_families": len(names),
            "measured_on": ("every marked point, as its distance from the curve "
                            "its line is moved onto"),
            "parameters": npar, "redundancy": int(red)}
    if pin is not None:
        cond["pp_assumed"] = "as given"
    note = ("5-point is read as the stereographic projection, which is what a "
            "compass draws: every straight edge an arc through two opposite points "
            "of the circle 90 deg out from the centre")

    why = None
    if rival:
        cond["ambiguous"] = (
            f"a second camera fits the marks about as well: its circle is "
            f"{2 * rival[0]:.0f} px in radius against {2 * f:.0f}, turned "
            f"{rival[1]:.0f} deg from this one")
        why = cond["ambiguous"] + ", so the marks do not decide between them"
    elif f > FMAX * diag:
        why = (f"the lines hardly curve: as 5-point their circle would be "
               f"{4 * f:.0f} px across on a {diag:.0f} px picture, and past "
               f"{FMAX:g} diagonals a scale is noise rather than a measurement. "
               "Drawn like this the picture is linear perspective; check it as "
               "1-, 2- or 3-point")
    elif f < F_FLOOR * diag:
        why = (f"the fit shrank the circle to {4 * f:.0f} px across, smaller than "
               "any drawing is built on, which is the fit failing and not a lens")
    elif red < SPARE:
        why = (f"the marks have {max(red, 0)} points to spare, and below {SPARE} "
               "the fit's own error bars are not to be trusted")
    elif not np.isfinite(se_f) or se_f > SE_MAX * f:
        why = (f"the marks place it only to +-{100 * se_f / f:.0f} %, and past "
               f"{100 * SE_MAX:.0f} % it is not quoted")
    elif se_rot > SE_TURN:
        why = (f"the marks turn the camera only to +-{se_rot:.1f} deg, and past "
               f"{SE_TURN:g} deg it is not quoted")
    if why:
        # one camera: the scale and the turn are fitted together, and a turn
        # that is not known leaves the scale that went with it no better
        if not (rival or f > FMAX * diag or f < F_FLOOR * diag):
            why += (". Mark more points along the lines, or lines further out "
                    "towards the circle, where 5-point bends most")
        cond["axes_refused"] = cond["focal_refused"] = why
        cond["focal_fitted_but_unmeasured"] = round(f, 1)
        # nothing derived from the circle is quoted when the circle is not
        cond.update(focal_px=None, rim_radius_px=None, field_deg=None)
        cond["focal_note"] = note + "."
    else:
        cond["focal_note"] = (
            note + f", here {2 * f:.0f} px in radius about ({c[0]:.0f}, {c[1]:.0f}). "
            f"Through the centre the picture spans {field[0]:.0f} deg across and "
            f"{field[1]:.0f} deg down.")
    value = None if cond["focal_px"] is None else round(f, 1)

    if red <= 0:
        return {"value": value, "verdict": "undetermined",
                "why": (f"{len(P)} marked points for the {npar} numbers of the "
                        "camera and its lines: any such marks fit exactly, so this "
                        "cannot say whether they are consistent. Mark a third point "
                        "along a line, or another line"),
                "conditioning": cond}
    return {"value": value,
            "verdict": "already consistent" if moves.max() <= mark else "rectifiable",
            "conditioning": cond}
