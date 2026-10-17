"""Trim a photograph down to the artwork, and take the camera back out of it.

A flat scan needs none of this. A photograph does: the drawing is taped to a
board on a desk, and the picture also contains the board, the tape, the desk,
a cable and whatever else was lying there. Two separate things go wrong.

The cheap one is pollution. The line detector is perfectly happy to fit a
vanishing-point family to the edge of the desk.

The expensive one is the keystone. Photographing a flat sheet from off-axis
applies a homography to it, and that homography composes with the perspective
the artist drew. Lines stay lines and families stay concurrent, so nothing
looks broken -- but every vanishing point has moved, the horizon has tilted,
and the camera the app then recovers is the artist's camera times the
photographer's. Asking "is this horizon level" of such an image answers a
question about how the photograph was taken.

So the sheet's own quadrilateral is found, and the image is mapped back to the
rectangle it must have been. The target rectangle's shape is not assumed: it
is recovered from the quadrilateral, which also yields the focal length of the
camera that took the photograph. When the sheet's edges come out parallel
there was no keystone to undo, that recovery is ill-posed, and saying so is
better than inventing an aspect ratio -- the trim degenerates to a plain crop,
which is what a scan wanted anyway.
"""
from __future__ import annotations

import cv2
import numpy as np

# Below this the quadrilateral is a parallelogram as far as the pixels can
# tell, the photographing camera is not recoverable from it, and the trim is
# a crop. Measured as the fractional convergence of each pair of opposite
# edges; 1e-3 is about a quarter of a pixel of keystone across 1280 px.
FLAT = 1e-3


def _order(pts):
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, float).reshape(4, 2)
    c = p.mean(0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    # y points down, so increasing angle runs left, up, right, down: already
    # clockwise on screen. Reversing it here to "make it clockwise" was the
    # first version, and it silently handed back TL, BL, BR, TR -- which swaps
    # width for height everywhere downstream.
    p = p[np.argsort(ang)]
    k = int(np.argmin(p[:, 0] + p[:, 1]))        # start at the top-left corner
    return np.roll(p, -k, axis=0)


def shape_of(quad, w, h):
    """The rectangle this quadrilateral is a picture of.

    Returns ``(out_w, out_h, focal_px, why_refused)``. ``focal_px`` is the
    camera that took the PHOTOGRAPH, not anything to do with the drawing; it
    is a by-product, and it is None when the keystone is too slight to measure
    it -- in which case the output rectangle is just the quadrilateral's own
    mean side lengths and the warp is very nearly a crop.

    The construction is the standard one for a single view of a rectangle: the
    two vanishing points of the rectangle's own edge pairs are orthogonal
    directions, which is one constraint each on the camera, and with square
    pixels and the principal point at the image centre that is enough for both
    the focal length and the aspect ratio.
    """
    q = _order(quad)
    m = np.c_[q, np.ones(4)]                     # TL, TR, BR, BL homogeneous
    m1, m2, m4, m3 = m[0], m[1], m[2], m[3]      # the derivation's lettering:
    #                                              m1 TL, m2 TR, m3 BL, m4 BR
    d2 = np.dot(np.cross(m2, m4), m3)
    d3 = np.dot(np.cross(m3, m4), m2)
    if abs(d2) < 1e-12 or abs(d3) < 1e-12:
        return _fallback(q, "the corners are degenerate")
    k2 = np.dot(np.cross(m1, m4), m3) / d2
    k3 = np.dot(np.cross(m1, m4), m2) / d3

    # k2 and k3 are 1 exactly when that pair of opposite edges is parallel in
    # the image. Each pair contributes one vanishing point, and the two are
    # orthogonal directions, which is what pins the camera down:
    # (v1 - pp).(v2 - pp) + f^2 = 0. Send one of those points to infinity and
    # the constraint degenerates into a statement about where the OTHER one
    # lies -- f drops out of it entirely and is not recoverable.
    #
    # So EITHER pair being parallel is disqualifying, not both. Requiring both
    # was the first version, and on a sheet photographed nearly square-on
    # (one pair parallel to 4 parts in 10 000, the other converging) it
    # returned f = 41 716 px for a picture 1600 px corner to corner.
    near = [abs(k2 - 1.0) < FLAT, abs(k3 - 1.0) < FLAT]
    if all(near):
        return _fallback(q, "the sheet's opposite edges are parallel in the "
                            "picture, so it was shot square-on and there is no "
                            "keystone to undo -- trimming is a plain crop")
    if any(near):
        return _fallback(q, "one pair of the sheet's edges is parallel in the "
                            "picture and the other converges, which fixes "
                            "nothing about the camera -- the shape was taken "
                            "from the corners instead, and the keystone, if "
                            "any, is too slight to undo reliably")

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    u0, v0 = w / 2.0, h / 2.0
    den = n2[2] * n3[2]
    if abs(den) < 1e-12:
        return _fallback(q, "only one pair of edges converges, which fixes the "
                            "aspect ratio but not the camera -- trimmed as a crop")
    f2 = -((n2[0] - n2[2] * u0) * (n3[0] - n3[2] * u0) +
           (n2[1] - n2[2] * v0) * (n3[1] - n3[2] * v0)) / den
    if not np.isfinite(f2) or f2 <= 0:
        # The same near-parallelogram as the two refusals above, reached from
        # the other side: the orthogonality constraint wants a negative f^2.
        # This is what a sheet photographed almost square-on does, and it is
        # the common case for a careful photograph.
        return _fallback(q, "these corners are too close to a parallelogram "
                            "for the camera to be recovered -- the sheet was "
                            "shot very nearly square-on, so the shape was "
                            "taken from the corners and there is little or no "
                            "keystone to undo")
    f = float(np.sqrt(f2))
    diag = float(np.hypot(w, h))
    # A focal length this long means the projection is all but orthographic
    # and f is being read out of rounding noise in the corner positions. The
    # real pictures here sit at 0.6 to 1.7 diagonals; 8 leaves room for a long
    # lens and still refuses the 26 diagonals that a near-parallelogram gave.
    if f > 8.0 * diag:
        return _fallback(q, f"the corners are so nearly a parallelogram that "
                            f"the camera comes out at f = {f:.0f} px on a "
                            f"{diag:.0f} px diagonal, which is noise rather "
                            f"than a measurement -- the shape was taken from "
                            f"the corners instead")

    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1.0]])
    Ai = np.linalg.inv(A)
    B = Ai.T @ Ai
    num, den2 = float(n2 @ B @ n2), float(n3 @ B @ n3)
    if den2 <= 0 or num <= 0:
        return _fallback(q, "the aspect ratio came out non-positive")
    ratio = float(np.sqrt(num / den2))           # width / height

    # Size the output so nothing is squeezed: keep the longer measured side.
    side_w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    side_h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    if ratio >= side_w / max(side_h, 1e-9):
        ow, oh = side_w, side_w / ratio
    else:
        ow, oh = side_h * ratio, side_h
    return int(round(ow)), int(round(oh)), f, None


def _fallback(q, why):
    """No recoverable camera: take the rectangle straight off the corners."""
    ow = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    oh = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return int(round(ow)), int(round(oh)), None, why


def warp(img, quad, out_wh):
    """Map the quadrilateral onto a clean rectangle of out_wh."""
    q = _order(quad).astype(np.float32)
    ow, oh = int(out_wh[0]), int(out_wh[1])
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    H = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, H, (ow, oh), flags=cv2.INTER_CUBIC)


def side_evidence(img, quad):
    """How much each side of the quad looks like a real edge.

    Returns four numbers: the median brightness step across the side, divided
    by the median step the picture's own texture produces over the same
    distance. A side that lies along the image border has no outside to
    compare against and comes back None.

    This is the whole difference between finding the sheet and inventing it.
    Measured on the five test pictures, the one quadrilateral that really was
    a sheet scored 7.2, 16.5, 20.9 and 13.2; every quadrilateral that was a
    bright patch of paint inside an already-cropped artwork scored 0.8 to 1.5.
    Nothing landed in between.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 \
        else img.astype(np.float32)
    H, W = g.shape
    d = max(4.0, 0.012 * float(np.hypot(W, H)))
    di = int(d)
    texture = max(float(np.median(np.abs(g[:, di:] - g[:, :-di]))), 0.1)
    q = _order(quad)
    mid = q.mean(0)
    out = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        n = np.array([-(b - a)[1], (b - a)[0]], float)
        nl = np.linalg.norm(n)
        if nl < 1e-9:
            out.append(None)
            continue
        n /= nl
        if np.dot(n, mid - (a + b) / 2) > 0:     # make n point out of the quad
            n = -n
        t = np.linspace(0.12, 0.88, 40)[:, None]
        ins, outs = [], []
        for pt in a + t * (b - a):
            pi, po = pt - n * d, pt + n * d
            if not (0 <= po[0] < W and 0 <= po[1] < H
                    and 0 <= pi[0] < W and 0 <= pi[1] < H):
                continue                          # this stretch is off-picture
            ins.append(g[int(pi[1]), int(pi[0])])
            outs.append(g[int(po[1]), int(po[0])])
        out.append(None if len(ins) < 10 else
                   float(np.median(np.abs(np.array(ins) - np.array(outs)))) / texture)
    return out


# A side counts as a real edge above this many times the picture's own
# texture. The gap it sits in is enormous (1.5 to 7.2), so the exact value
# does not matter; what matters is that all four sides have to clear it.
EDGE = 3.0


def find_quad(img, min_area=0.12):
    """The sheet's four corners, or None and the reason there are none.

    The sheet is found as the large bright region -- paper photographed on a
    desk is lighter than most of what surrounds it -- and then checked, which
    is the part that matters. Without the check this returns a confident
    quadrilateral for an artwork that was already cropped, because there is
    always some bright patch of paint whose outline approximates to four
    corners. On testdata/4.jpg that was a shape running corner to corner
    across the painting, and it came back indistinguishable from the one real
    answer in the set.

    Everything runs on an 800 px copy; the corners are scaled back up.
    """
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape[:2]
    s = 800.0 / max(H, W)
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    k = 1.0 / s if s < 1 else 1.0

    blur = cv2.GaussianBlur(small, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # close the tape, clips and shadow gaps that break the sheet's outline
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page = small.shape[0] * small.shape[1]
    best, best_a = None, 0.0
    for c in cs:
        a = cv2.contourArea(c)
        if a < min_area * page or a <= best_a:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
            ap = cv2.approxPolyDP(c, eps * peri, True)
            if len(ap) == 4 and cv2.isContourConvex(ap):
                best, best_a = ap.reshape(4, 2).astype(float), a
                break
        else:
            # not a quadrilateral at any tolerance: fall back to its
            # minimum-area rectangle, which is what a sheet with a corner
            # hidden under a clip needs
            best, best_a = cv2.boxPoints(cv2.minAreaRect(c)).astype(float), a
    if best is None:
        return None, ("nothing in this picture looks like a sheet with a border "
                      "around it -- place the corners yourself if it needs trimming")

    quad = _order(best * k)
    ev = side_evidence(img, quad)
    off = [i for i, v in enumerate(ev) if v is None]
    weak = [i for i, v in enumerate(ev) if v is not None and v < EDGE]
    names = ["top", "right", "bottom", "left"]
    if off and not weak:
        # Say what was rejected, not what the picture contains. The first
        # wording announced that only one edge of the SHEET was in frame,
        # which on testdata/2.jpg was simply false -- the sheet is fully
        # visible there; it was the bright region that ran to the frame,
        # because the desk is nearly as light as the paper.
        got = ", ".join(names[i] for i in range(4) if i not in off)
        return None, (f"the bright region found here runs off the edge of the "
                      f"picture -- only its {got} side could be checked, and "
                      "four corners cannot be placed on it. If there is a "
                      "sheet, drag the corners onto it yourself; they may sit "
                      "outside the picture if it continues past the frame")
    if weak:
        v = max(x for x in ev if x is not None)
        return None, (f"no sheet edge was found: the best candidate's strongest "
                      f"side is only {v:.1f} times the picture's own texture, "
                      "which is what an already-cropped artwork looks like -- "
                      "nothing to trim, or place the corners yourself")

    frac = cv2.contourArea(quad.astype(np.float32)) / float(W * H)
    if frac > 0.995:
        return None, "the sheet already fills the picture"
    return quad, None


def _demo():
    """Project known rectangles with known cameras and recover them."""
    # a 1.5:1 sheet, photographed from off to one side
    for true_ratio, rvec, f_true in ((1.5, (0.30, 0.22, 0.05), 1400.0),
                                     (0.70, (-0.25, 0.35, 0.0), 900.0),
                                     (1.0, (0.18, -0.30, 0.1), 2000.0)):
        W, H = 1600, 1200
        rect = np.array([[-true_ratio, -1.0, 0], [true_ratio, -1.0, 0],
                         [true_ratio, 1.0, 0], [-true_ratio, 1.0, 0]]) * 100
        R = cv2.Rodrigues(np.array(rvec, float))[0]
        t = np.array([0, 0, 900.0])
        K = np.array([[f_true, 0, W / 2], [0, f_true, H / 2], [0, 0, 1.0]])
        p = (K @ (R @ rect.T + t[:, None])).T
        quad = p[:, :2] / p[:, 2:]
        ow, oh, f, why = shape_of(quad, W, H)
        assert why is None, why
        assert abs(f - f_true) / f_true < 0.02, (f, f_true)
        assert abs((ow / oh) - true_ratio) / true_ratio < 0.02, (ow / oh, true_ratio)

    # Shot square-on: the quadrilateral is a rectangle, the camera is NOT
    # recoverable from it, and the honest answer is to say so rather than
    # return whatever a near-singular division produced.
    flat = np.array([[100.0, 80.0], [900.0, 80.0], [900.0, 620.0], [100.0, 620.0]])
    ow, oh, f, why = shape_of(flat, 1000, 700)
    assert f is None and "square-on" in why, (f, why)
    assert (ow, oh) == (800, 540), (ow, oh)

    # ...and a rotated rectangle is still a rectangle: no keystone either way
    th = np.radians(20.0)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    turned = (flat - flat.mean(0)) @ Rm.T + flat.mean(0)
    ow2, oh2, f2, why2 = shape_of(turned, 1000, 700)
    assert f2 is None, f2
    assert abs(ow2 - 800) <= 1 and abs(oh2 - 540) <= 1, (ow2, oh2)

    # One pair parallel, the other converging: f is NOT determined, however
    # confidently the algebra produces a number. These are the corners of the
    # sheet in testdata/2.jpg, read off the photograph by hand.
    half = np.array([[34.0, 54.0], [1229.0, 62.0], [1224.0, 904.0], [79.0, 896.0]])
    ow3, oh3, f3, why3 = shape_of(half, 1280, 960)
    assert f3 is None, f3
    assert "fixes nothing about the camera" in why3, why3
    # the fallback averages each pair of opposite sides, so 1195 and 1145 wide
    assert (ow3, oh3) == (1170, 843), (ow3, oh3)

    # ...and the same refusal reached the other way, through an absurd f
    almost = np.array([[100.0, 80.0], [900.0, 80.3], [900.4, 620.0], [100.0, 620.0]])
    f4 = shape_of(almost, 1000, 700)[2]
    assert f4 is None, f4

    # the warp puts the corners exactly on the output rectangle
    img = np.zeros((700, 1000, 3), np.uint8)
    out = warp(img, flat, (800, 540))
    assert out.shape[:2] == (540, 800), out.shape

    # corner ordering survives any input order
    import itertools
    base = _order(flat)
    for perm in itertools.permutations(range(4)):
        assert np.allclose(_order(flat[list(perm)]), base), perm

    print("trim.py self-check passed  (aspect and photo focal length recovered "
          "to 2%, square-on and single-pair keystones refused)")


if __name__ == "__main__":
    _demo()
