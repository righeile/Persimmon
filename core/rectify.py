"""Looking at a marked shape as if the plane it lies in were facing us.

A drawing cannot be judged in perspective: a circle is an oval, a square is a
trapezium, and the eye has no way to allow for it. Undo the plane's own
foreshortening and the shape is there to be looked at.

Nothing here assumes the surface is rectangular. The plane is given by the two
vanishing points that span it, which is a statement about two world directions
and says nothing about the shape of anything lying in it.
"""
from __future__ import annotations

import cv2
import numpy as np

MAX_SIDE = 1600          # the straightened view, at most this many pixels
MARGIN = 0.06            # a little context around the marks


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def frame(K, vp_a, vp_b, vp_up=None):
    """The rotation that turns the camera to face the plane square on.

    `vp_a` and `vp_b` are the vanishing points of two directions lying in the
    plane; they need not be perpendicular, and the frame is made orthonormal
    around the plane's own normal, so the result is true shape either way.
    `vp_up` says which of them should stand upright in the view.
    """
    Ki = np.linalg.inv(np.asarray(K, float))
    d = [_unit(Ki @ np.append(np.asarray(v, float)[:2], 1.0))
         for v in (vp_a, vp_b)]
    n = _unit(np.cross(d[0], d[1]))
    if not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-9:
        return None
    up = d[0] if vp_up is None else _unit(
        Ki @ np.append(np.asarray(vp_up, float)[:2], 1.0))
    e2 = _unit(up - (up @ n) * n)                 # in the plane, pointing up
    if np.linalg.norm(e2) < 1e-9:
        return None
    e1 = _unit(np.cross(e2, n))
    return np.stack([e1, e2, n])                  # rows: a proper rotation


def homography(K, vp_a, vp_b, vp_up=None):
    """Image to a face-on view of the plane. Scale and position are arbitrary
    here; `fit` pins them to the marks."""
    R = frame(K, vp_a, vp_b, vp_up)
    if R is None:
        return None
    K = np.asarray(K, float)
    return K @ R @ np.linalg.inv(K)


def _apply(H, pts):
    p = np.c_[np.asarray(pts, float).reshape(-1, 2), np.ones(len(pts))]
    q = (H @ p.T).T
    w = q[:, 2:3]
    return q[:, :2] / np.where(np.abs(w) < 1e-12, np.nan, w)


def fit(H, pts, max_side=MAX_SIDE):
    """H with a scale and shift that put `pts` in a picture of their own.

    The straightened marks keep roughly the size they had, so the view is
    comparable with the picture it came from rather than arbitrarily blown up.
    Returns (H, width, height) or None if the marks land behind the camera.
    """
    q = _apply(H, pts)
    if not np.all(np.isfinite(q)):
        return None
    lo, hi = q.min(0), q.max(0)
    span = np.maximum(hi - lo, 1e-6)
    pad = MARGIN * float(span.max())
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    # keep the marks' own scale, but never render more than MAX_SIDE
    s = min(1.0, max_side / float(span.max()))
    w = int(round(span[0] * s))
    h = int(round(span[1] * s))
    if w < 8 or h < 8:
        return None
    T = np.array([[s, 0, -lo[0] * s], [0, s, -lo[1] * s], [0, 0, 1.0]])
    return T @ H, w, h


def straighten(image, K, pts, vp_a, vp_b, vp_up=None, outline=True):
    """The marked region, seen as if its plane faced us. Returns (bgr, H).

    The marks are drawn back on so what is being looked at is not in doubt --
    straightening a region and then wondering which blob was the marked one
    wastes the whole point of looking.
    """
    R = frame(K, vp_a, vp_b, vp_up)
    if R is None:
        return None, None
    K = np.asarray(K, float)
    Ki = np.linalg.inv(K)
    H = K @ R @ Ki
    if upright_error(H, pts):
        # a vanishing point names a direction, not which way along it, so the
        # view can come out upside down. Half a turn in the plane fixes it and
        # keeps the rotation proper.
        H = K @ np.diag([-1.0, -1.0, 1.0]) @ R @ Ki
    got = fit(H, pts)
    if got is None:
        return None, None
    M, w, h = got
    out = cv2.warpPerspective(np.asarray(image), M, (w, h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(20, 20, 20))
    if outline:
        q = _apply(M, pts).round().astype(np.int32)
        cv2.polylines(out, [q], True, (90, 230, 120), 2, cv2.LINE_AA)
    return out, M


def upright_error(H, pts):
    """Whether the view came out upside down relative to the marks.

    A vanishing point names a direction but not which way along it, so the
    'up' handed to `frame` may be down. Compare the marks' own top-to-bottom
    order before and after; if it reversed, the frame needs turning over.
    """
    q = _apply(H, pts)
    if not np.all(np.isfinite(q)):
        return False
    p = np.asarray(pts, float).reshape(-1, 2)
    a, b = int(np.argmin(p[:, 1])), int(np.argmax(p[:, 1]))
    return bool(q[a, 1] > q[b, 1])
