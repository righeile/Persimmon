"""A repeat on a plane, in perspective.

An even repeat needs no camera. The plane's vanishing line and one repeat fix
the whole lattice: the repeat's two sides run to the points where they meet
that line, and their far corners set the scale along each.

Finding one needs no camera either. Take the vanishing line to infinity and
parallel lines on the plane come out parallel on the page, at their true even
steps, so a surface says which plane it is by being even on that one.
"""
from __future__ import annotations

from math import ceil, floor

import numpy as np
from scipy.optimize import least_squares

from . import checks as C
from .rectify import _apply

MOTIFS = ("grid", "checker", "brick", "herringbone", "mirrored herringbone")
MAX_TILES = 6000         # past this the repeat is too small for its surface
SMALLEST_PX = 2.0        # a repeat this small on the page is grey, not a pattern


def lattice(l, o, p, q):
    """H taking the plane's unit square to the picture, with corners o, p, q.

    (0, 0) goes to o, (1, 0) to p, (0, 1) to q, and the plane's horizon to `l`.
    None when a side lies along that line.
    """
    l = np.asarray(l, float)
    o, p, q = (np.append(np.asarray(x, float), 1.0) for x in (o, p, q))
    cols = []
    for x in (p, q):
        v = np.cross(np.cross(o, x), l)
        a, b = np.cross(x, v), np.cross(x, o)       # x cross (t v + o) = 0
        i = int(np.argmax(np.abs(a)))
        if abs(a[i]) < 1e-9:
            return None
        cols.append(-b[i] / a[i] * v)
    H = np.c_[cols[0], cols[1], o]
    return H if abs(np.linalg.det(H)) > 1e-9 else None


def steps(motif, ratio=4.0):
    """The pattern's own two translations, as columns, in plane units.

    A grid repeats on the unit square. A brick repeat is one brick long and
    two courses tall. A herringbone repeat is one plank, long by wide, and
    `ratio` is how many widths long it is. Its stair climbs one way or the
    other, and that is not the same floor seen from another corner.
    """
    k = float(ratio)
    return {"brick": np.array([[1.0, 0.5], [0.0, 0.5]]),
            "herringbone": np.array([[1 / k, 1.0], [1.0, -k]]),
            "mirrored herringbone": np.array([[1 / k, 1.0], [-1.0, k]])}.get(motif, np.eye(2))


def check(l, marks, *, motif="grid", ratio=4.0, mark=9.0):
    """How far marked corners are from one true, even repeat on the plane.

    The first three marks are the repeat and the rest are corners of it further
    off. Each is read as the nearest corner of the lattice, nearest first,
    so that a long run is never a whole repeat out; then all the marks pull on
    the fit together, with the vanishing line held. The answer is the furthest
    any mark is from it.
    """
    m = np.asarray(marks, float).reshape(-1, 2)
    if len(m) < 3:
        return C.undetermined("a repeat is three marks: a corner, and the next one along each side",
                              marks=len(m))
    if len(m) == 3:
        return C.undetermined("three marks make the repeat and leave nothing to test it: "
                              "mark corners further along the pattern", marks=3)
    order = np.r_[0, 1, 2, 3 + np.argsort(np.hypot(*(m[3:] - m[0]).T))]
    m = m[order]
    T, Ti = (lambda t: (t, np.linalg.inv(t)))(steps(motif, ratio))

    def on(x):
        return lattice(l, *x.reshape(3, 2))

    def miss(x, st):
        H = on(x)
        return np.full(2 * len(st), 1e6) if H is None else (_apply(H, st) - m[:len(st)]).ravel()

    def nearest(H, pts):
        return (T @ np.round(Ti @ _apply(np.linalg.inv(H), pts).T)).T

    x = m[:3].ravel()
    if on(x) is None:
        return C.undetermined("the repeat lies along the plane's vanishing line", marks=len(m))
    st = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    for i in range(3, len(m)):
        st = np.vstack([st, nearest(on(x), m[i:i + 1])])
        if i & (i - 1) == 0 or i == len(m) - 1:     # a refit is the slow part
            x = least_squares(miss, x, args=(st,)).x
    for _ in range(3):                              # a closer fit can re-read a mark
        read = np.vstack([st[:3], nearest(on(x), m[3:])])
        if np.array_equal(read, st):
            break
        st, x = read, least_squares(miss, x, args=(read,)).x
    H = on(x)
    d = np.hypot(*(_apply(H, st) - m).T)
    worst = float(d.max())
    span = np.ptp(np.round(Ti @ st[np.r_[0, 3:len(st)]].T), axis=1)    # past the repeat itself
    out = C.readout(round(worst, 2), "even" if worst <= mark else "drifts",
                    marks=len(m), spare=2 * len(m) - 6, spanned=span.tolist(),
                    median_px=round(float(np.median(d)), 2),
                    mark_accuracy_px=round(float(mark), 2),
                    H=H.tolist(), moves=np.c_[m, _apply(H, st)].tolist())
    if not span.all():
        out["conditioning"]["note"] = ("the marks run along one direction of the repeat, "
                                       "so the other is not tested")
    return out


def _rect(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)


def _tiles(motif, k, lo, hi):
    """(corners, shaded) of every tile over the box, in plane units."""
    if motif == "mirrored herringbone":
        for q, s in _tiles("herringbone", k, (lo[0], 1 - hi[1]), (hi[0], 1 - lo[1])):
            yield q * (1, -1) + (0, 1), s
    elif motif == "herringbone":
        # in plank widths, where the two directions have one length
        c = np.array([[k * lo[0], lo[1]], [k * lo[0], hi[1]],
                      [k * hi[0], lo[1]], [k * hi[0], hi[1]]])
        a, b = (c[:, 0] + c[:, 1]) / 2, (c[:, 0] - c[:, 1]) / (2 * k)
        for i in range(floor(a.min()) - 1, ceil(a.max()) + 2):
            for j in range(floor(b.min()) - 1, ceil(b.max()) + 2):
                u, v = i + k * j, i - k * j
                yield _rect(u / k, v, 1.0, 1.0), False
                yield _rect(u / k + 1, v + 1 - k, 1 / k, k), False
    elif motif == "brick":
        for j in range(floor(2 * lo[1]), ceil(2 * hi[1])):
            x = 0.5 * (j % 2)
            for i in range(floor(lo[0] - x), ceil(hi[0] - x)):
                yield _rect(i + x, j / 2, 1.0, 0.5), False
    else:
        for i in range(floor(lo[0]), ceil(hi[0])):
            for j in range(floor(lo[1]), ceil(hi[1])):
                yield _rect(i, j, 1.0, 1.0), motif == "checker" and (i + j) % 2 == 0


def _cut(poly, e):
    """What of a convex polygon has e @ (x, y, 1) >= 0."""
    h = poly @ e[:2] + e[2]
    keep = []
    for i, a in enumerate(poly):
        j = (i + 1) % len(poly)
        if h[i] >= 0:
            keep.append(a)
        if h[i] * h[j] < 0:
            keep.append(a + (poly[j] - a) * h[i] / (h[i] - h[j]))
    return np.array(keep).reshape(-1, 2)


def _clip(poly, edges):
    for e in edges:
        poly = _cut(poly, e)
        if len(poly) < 3:
            break
    return poly


def tiles(H, motif="grid", ratio=4.0, region=None, repeats=8):
    """The pattern on the picture: {"polys": ..., "shaded": ...}, or {"why": ...}.

    It stops where the repeat falls below a couple of pixels, and it covers
    `region` in the picture, or `repeats` each way from the repeat marked.
    """
    H = np.asarray(H, float)
    Hi = np.linalg.inv(H)
    far = np.sqrt(max(np.hypot(*(_apply(H, _rect(0, 0, 1, 1))[[1, 3]] - H[:2, 2] / H[2, 2]).T).min(), 1e-3)
                  / SMALLEST_PX)
    band = np.array([H[2] - (0, 0, 0.02), (0, 0, far) - H[2]])     # the camera's side, and the far edge
    if region is not None:
        r = _cut(np.asarray(region, float), Hi[2] - (0, 0, 1 / far))
        if len(r) < 3:
            return {"why": "the region lies beyond the plane's vanishing line"}
        r = _apply(Hi, r)
    else:
        r = _rect(-repeats, -repeats, 2 * repeats + 1, 2 * repeats + 1)
    r = _clip(r, band)
    if len(r) < 3:
        return {"why": "nothing of the plane is in front of the camera there"}
    if r[:, 0] @ np.roll(r[:, 1], -1) < r[:, 1] @ np.roll(r[:, 0], -1):
        r = r[::-1]
    edges = [np.array([a[1] - b[1], b[0] - a[0], a[0] * b[1] - a[1] * b[0]])
             for a, b in zip(r, np.roll(r, -1, axis=0))]
    k = float(ratio)
    lo, hi = r.min(0), r.max(0)
    w, h = hi - lo
    many = (k * w + h) ** 2 / (2 * k) if "herringbone" in motif else w * h * (2 if motif == "brick" else 1)
    if many > MAX_TILES:
        return {"why": f"about {int(many)} repeats would fit there; mark a larger repeat or a smaller surface"}
    polys, shaded = [], []
    for q, s in _tiles(motif, k, lo, hi):
        q = _clip(q, edges)
        if len(q) >= 3:
            polys.append(_apply(H, q).round(2).tolist())
            shaded.append(s)
    return {"polys": polys, "shaded": shaded}


def _flat(pts, l, c):
    """Picture points with the line `l` sent to infinity, the scale at `c` kept."""
    pts = np.asarray(pts, float)
    return (pts - c) * ((l[:2] @ c + l[2]) / (pts @ l[:2] + l[2]))[:, None]


def _back(flat, l, c):
    """The picture points `_flat` came from."""
    w = l[:2] @ c + l[2]
    return c + flat * (w / (w - flat @ l[:2]))[:, None]


def _along(v, l, c):
    """Which way lines running to `v` go in the flat picture, at `c`."""
    t = v[:2] - v[2] * c
    a, b = _flat(np.array([c, c + 40 * t / max(np.hypot(*t), 1e-12)]), l, c)
    return b - a


def comb(segs, vp, l, least=6.0):
    """Even steps among lines running to `vp`, on the plane with vanishing line `l`.

    {"step", "teeth", "lines", "strength"}, or None. Steps are in pixels at
    the middle of the lines. Read as a periodogram, so a line that is missing
    costs nothing. Each line counts once however long it is and in however many
    pieces, or a few near lines are all the evidence. Toward the vanishing line
    a step is lost in the rounding, so that half is left out.
    """
    q = np.asarray(segs, float).reshape(-1, 4)
    mid = (q[:, :2] + q[:, 2:]) / 2
    l, v = np.asarray(l, float), np.asarray(vp, float)
    w = np.abs(mid @ l[:2] + l[2])
    near = np.flatnonzero(w >= np.median(w) / 2)
    if len(near) < 6:
        return None
    q, mid = q[near], mid[near]
    c = np.average(mid, axis=0, weights=np.hypot(*(q[:, 2:] - q[:, :2]).T))
    d = _along(v, l, c)
    x = _flat(mid, l, c) @ (np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-12))
    x = x - x.min()
    span = float(x.max())
    if span < 5 * least:
        return None
    one = np.unique(np.round(x / 2) * 2)            # a line, by where it lies
    f = np.arange(5 / span, 1 / least, 1 / (4 * span))
    z = np.exp(2j * np.pi * np.outer(one, f)).sum(0)
    r = np.abs(z) / len(one)
    hit = np.flatnonzero(r >= max(0.6, 3.0 / np.sqrt(len(one))))
    if not len(hit):
        return None
    i = int(hit[0])                                 # the widest step that holds
    while i + 1 < len(r) and r[i + 1] >= r[i]:
        i += 1
    on = np.abs((x * f[i] - np.angle(z[i]) / (2 * np.pi) + 0.5) % 1 - 0.5) <= 0.25
    teeth = len(np.unique(np.round(x[on] * f[i])))
    if teeth < 6:
        return None
    return {"step": round(float(1 / f[i]), 2), "teeth": teeth, "lines": near[on],
            "strength": round(float(r[i]), 2)}


def _sides(q, l):
    """The lines wholly on each side of the line `l`, as masks: (above zero, below)."""
    a, b = q[:, :2] @ l[:2] + l[2], q[:, 2:] @ l[:2] + l[2]
    return (a > 0) & (b > 0), (a < 0) & (b < 0)


def _unit(l):
    return np.asarray(l, float) / max(np.hypot(*np.asarray(l, float)[:2]), 1e-12)


def _span(pts, v0, v1):
    """The diagonal of the surface between v0 and v1 that holds `pts`.

    All but the furthest half: toward the vanishing line a step on the page is
    a mile on the plane, and one line up there would be most of the surface.
    """
    l = _unit(np.cross(v0, v1))
    pts = np.asarray(pts, float)
    w = np.abs(pts @ l[:2] + l[2])
    pts = pts[w >= np.median(w) / 2]
    c = pts.mean(0)
    B = np.c_[_along(v0, l, c), _along(v1, l, c)]
    if abs(l[:2] @ c + l[2]) < 1e-9 or abs(np.linalg.det(B)) < 1e-9:
        return None
    ab = np.linalg.solve(B, _flat(pts, l, c).T)
    return _back((B @ np.c_[ab.min(1), ab.max(1)]).T, l, c).ravel().tolist()


def found(families, vps, guess, image_size, pp=None):
    """Patterned surfaces among detected families, one entry a surface.

    Lines wholly on the floor or ceiling and not the room's are a pattern
    however they fall, and so is whatever is square to them there: real planks
    leave too few edges to show a step. For the rest evenness decides: each
    horizontal family is tried on the floor's plane and on the wall it could
    lie in, and the uprights on each wall, which finds tiles that share the
    room's own vanishing points. `guess` is `consensus.guess`, for the horizon.
    """
    c = guess.get("conditioning") or {}
    hz, up, place = c.get("horizon"), c.get("vertical"), c.get("place") or {}
    if hz is None:
        return []
    hz = np.asarray(hz, float)
    room = [n for n in c.get("horizontals") or [] if place.get(n) != "floor"]
    fam = {n: np.asarray(q, float).reshape(-1, 4) for n, q in families.items()}
    vp = {n: np.asarray(v, float) for n, v in vps.items()}
    level = [n for n in fam if place.get(n) in ("wall", "ceiling", "floor")]

    best = {}                       # family: (surface, its lines there, comb)
    def attempt(n, l, plus, minus, plane):
        for i, (side, name) in enumerate(zip(_sides(fam[n], l), (plus, minus))):
            r = comb(fam[n][side], vp[n], l) if name and side.sum() >= 6 else None
            if r and (n not in best or r["strength"] > best[n][2]["strength"]):
                best[n] = ((name, plane, i), np.flatnonzero(side)[r["lines"]], r)
    for n in level:
        # a cornice's tiers run as evenly as tiles, but they are not on a ceiling
        attempt(n, hz, "floor", "ceiling" if place[n] == "ceiling" else None, ())
        if up is not None:
            l = _unit(np.cross(vp[n], vp[up]))
            attempt(n, l, "wall", "wall", (n, up))
            attempt(up, l, "wall", "wall", (n, up))

    f = c.get("focal_px")
    centre = np.array(pp if pp else (image_size[0] / 2, image_size[1] / 2), float)
    Ki = None if not f else np.linalg.inv(np.array([[f, 0, centre[0]], [0, f, centre[1]], [0, 0, 1.0]]))
    def apart(a, b):
        d, e = Ki @ vp[a], Ki @ vp[b]
        return float(np.degrees(np.arccos(min(abs(d @ e) / np.linalg.norm(d) / np.linalg.norm(e), 1.0))))

    for n in level:
        if n not in best and n not in room and place[n] in ("floor", "ceiling"):
            i = 0 if place[n] == "floor" else 1
            best[n] = ((place[n], (), i), np.flatnonzero(_sides(fam[n], hz)[i]), None)
    if Ki is not None:
        for n in level:
            on = [k for k in best if k in level and best[k][0][1] == () and k not in room]
            if n not in best and n not in room and any(abs(apart(n, k) - 90) <= 6 for k in on):
                key = best[on[0]][0]
                best[n] = (key, np.flatnonzero(_sides(fam[n], hz)[key[2]]), None)

    groups = {}
    for n, (key, lines, r) in best.items():
        if len(lines):
            groups.setdefault(key, []).append((n, lines, r))
    out = []
    for (name, plane, _), rows in groups.items():
        names = [n for n, _, _ in rows]
        pts = np.vstack([fam[n][m].reshape(-1, 2) for n, m, _ in rows])
        flat = [n for n in names if n != up]
        # the two directions that span it: its own, or one of them and any
        # other on the same vanishing line
        pair = list(plane) or (flat + [n for n in level if n not in flat])[:2]
        if len(pair) < 2:
            continue
        ends = _span(pts, vp[pair[0]], vp[pair[1]])
        if ends is None:
            continue
        row = {"plane": name, "families": names, "lines": {n: m.tolist() for n, m, _ in rows},
               "n": int(sum(len(m) for _, m, _ in rows)),
               "step_px": [r["step"] for _, _, r in rows if r],
               "vps": [(vp[n][:2] / vp[n][2] if abs(vp[n][2]) > 1e-12 else vp[n][:2] * 1e7).tolist()
                       for n in pair],
               "patch": ends}
        if name != "wall" and Ki is not None and flat:
            if up is not None:
                row["off_deg"] = round(max(abs(90 - apart(n, up)) for n in flat), 2)
            if room:
                row["to_room_deg"] = round(min(apart(n, m) for n in flat for m in room), 1)
        out.append(row)
    return out
