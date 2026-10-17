#!/usr/bin/env python3
"""Persimmon -- linear perspective, checked against the marks.

    python app.py              # then open http://127.0.0.1:8830

Semi-automatic by design. Automatic extraction failed repeatedly on real
paintings -- occlusion, watercolor texture, low contrast -- while hand marking
measured ~9 px against the paint, which is ample for every check here. So the
page is a marking tool first and the detector is an accelerator the user can
accept, reject or ignore.

The page holds the document and the server holds nothing between requests. Every
check travels with the marks it is about, so a result and the marks it came from
cannot disagree.

Nothing here answers without saying how well it knows. Every readout leaves as
{value, verdict, conditioning} and the verdict may be "undetermined"; the page
renders that as loudly as a number. Four results had to be withdrawn in this
project after being reported with confidence they had not earned.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

from core import camera_io as CIO
from core import checks as C
from core import consensus as K
# `K` is the camera matrix inside _run_one, which shadows the module there, so
# the one place the scaled mark accuracy lives needs a name of its own. A flat
# 9 px under-flags every scan bigger than the one it was measured on.
from core.consensus import _mark_px as _mark_accuracy
from core import export as EX
from core import rectify as RC
from core import rhythm as R
from core import trim as TR
from core import vp as V
from labkit import browse as LKB

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.json.sort_keys = False
app.register_blueprint(LKB.flask_blueprint())

OPENS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")
MAX_SIDE = 4000


def _err(msg, code=400):
    return jsonify({"error": msg}), code


def _body():
    return request.get_json(force=True, silent=True) or {}


def _pts(x):
    a = np.asarray(x, float).reshape(-1, 2)
    if not np.isfinite(a).all():
        raise ValueError("a mark has a non-finite coordinate")
    return a


def _xy(x):
    a = np.asarray(x, float).ravel()
    if a.size == 3 and abs(a[2]) > 1e-12:
        a = a[:2] / a[2]
    return a[:2]


# ------------------------------------------------------------- real-world size

# A pixel means nothing to an artist standing at an easel. Given the picture's
# real size, every deviation this app prints can be said in the units the
# correction will be made in. One declaration serves the whole document: the
# posts check already took a px_per_mm by hand, and now inherits this instead.
TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}

# f in the picture's own pixels is NOT a camera lens. Converted to the
# picture's real units it is the distance the eye belongs at for the drawing
# to look right -- the station point. Calling it a focal length in mm would
# invite reading it as the lens of whatever photographed the sheet, which is a
# different number living in a different image (see trim.photo_focal_px, which
# is deliberately left in pixels here for exactly that reason).
_RENAME_MM = {"focal_px": "viewing_distance_mm"}
# mm_per_px is this function's own output and ends in _px by accident of
# naming, so wrapping a readout twice would mint "mm_per_mm".
_NO_MM = ("photo_focal_px", "mm_per_px")


def _px_per_mm(scale, w, h):
    """Pixels per millimetre from a declared real size, or None.

    `scale` is {"value", "unit", "along"} -- the size of the picture AS IT IS
    NOW, which is the trimmed picture when a trim is in force. That is why it
    is resolved here against the live width and height on every request rather
    than stored as a number: trim the photograph to the sheet and the same
    declaration keeps meaning the same edge.
    """
    if not scale:
        return None
    try:
        v = float(scale.get("value"))
    except (TypeError, ValueError):
        return None
    f = TO_MM.get(scale.get("unit") or "mm")
    span = float(h if scale.get("along") == "height" else w)
    if not f or not np.isfinite(v) or v <= 0 or span <= 0:
        return None
    return span / (v * f)


def _add_mm(obj, ppm):
    """Put a millimetre twin beside every pixel number in a readout.

    Applied to the whole response rather than to the handful of fields anyone
    remembered: every check in this app already names its lengths `*_px`, so
    one pass covers the ones written since as well.
    """
    if ppm is None or not np.isfinite(ppm) or ppm <= 0:
        return obj
    if isinstance(obj, list):
        return [_add_mm(x, ppm) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for k, v in obj.items():
        out[k] = _add_mm(v, ppm)
        if not (isinstance(k, str) and k.endswith("_px")) or k in _NO_MM:
            continue
        name = _RENAME_MM.get(k, k[:-3] + "_mm")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[name] = round(float(v) / ppm, 3)
        elif isinstance(v, list) and v and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            out[name] = [round(float(x) / ppm, 3) for x in v]
    if "mm_per_px" not in out and any(k.endswith("_mm") for k in out):
        out["mm_per_px"] = round(1.0 / ppm, 5)
    return out


def _resolve(path):
    """An image path the user chose, kept inside somewhere sensible."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = HERE / p
    p = p.resolve()
    if p.suffix.lower() not in OPENS:
        raise ValueError(f"{p.suffix} is not an image this opens")
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    return p


CAMERA_EXTS = (".blend", ".fspy", ".json")


def _camera_file(path):
    """A camera file the user chose: .blend, .fspy or .json and nothing else."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = HERE / p
    p = p.resolve()
    if p.suffix.lower() not in CAMERA_EXTS:
        raise ValueError(f"{p.suffix or 'that'} is not a camera file this reads: "
                         + ", ".join(CAMERA_EXTS))
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    return p


# --------------------------------------------------------------------- pages

def _trim(spec):
    """A trim as the page sends it, or None. Raises ValueError if malformed."""
    if spec in (None, "", "null"):
        return None
    if isinstance(spec, str):
        spec = json.loads(spec)
    if not spec or not spec.get("corners"):
        return None
    q = _pts(spec["corners"])
    if len(q) != 4:
        raise ValueError("a trim needs exactly four corners")
    out = spec.get("out")
    return {"corners": q, "out": (None if not out else
                                  (int(out[0]), int(out[1])))}


def _load(path, spec):
    """The picture the user is actually looking at.

    A photograph of a drawing is not the drawing: it carries the desk, the
    tape and the board, and -- the part that matters -- a keystone, because
    shooting a flat sheet off-axis composes a homography onto the perspective
    the artist drew. Every vanishing point moves and the horizon tilts, so a
    horizon read off the untrimmed photograph is partly a statement about
    where the photographer stood.

    So the trim travels with the request and is applied here, once, for every
    route that touches pixels. Nothing is written to disk: the user's original
    file is never modified, and there is no derived file to fall out of step
    with the marks.
    """
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError("that file did not decode as an image")
    t = _trim(spec)
    if t is None:
        return im, None
    h, w = im.shape[:2]
    ow, oh, f, why = TR.shape_of(t["corners"], w, h)
    if t["out"]:
        ow, oh = t["out"]
    if not (0 < ow <= MAX_SIDE and 0 < oh <= MAX_SIDE):
        raise ValueError(f"a trim to {ow}x{oh} is not a usable size")
    return TR.warp(im, t["corners"], (ow, oh)), {
        "out": [int(ow), int(oh)], "photo_focal_px": None if f is None else round(f, 1),
        "keystone": why or ("The keystone has been undone: the sheet's "
                            "opposite edges converged in the photograph, so "
                            "it was shot off-axis."),
    }


def _png(im):
    ok, buf = cv2.imencode(".png", im)
    if not ok:
        raise ValueError("the trimmed picture would not encode")
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "app": "Persimmon"})


@app.get("/api/testdata")
def testdata():
    d = HERE / "testdata"
    out = []
    for p in sorted(d.glob("*")):
        if p.suffix.lower() in OPENS:
            im = cv2.imread(str(p))
            if im is not None:
                out.append({"path": str(p.relative_to(HERE)), "name": p.name,
                            "w": int(im.shape[1]), "h": int(im.shape[0]),
                            "marks": (p.parent / (p.stem + ".marks.json")).is_file()})
    return jsonify(out)


@app.get("/api/marks")
def marks():
    """A marks file sitting beside its picture.

    Only `<picture>.marks.json` next to an openable image, so this cannot be
    talked into serving anything else.
    """
    try:
        img = _resolve(request.args.get("path", ""))
    except ValueError as exc:
        return _err(str(exc), 404)
    p = img.parent / (img.stem + ".marks.json")
    if not p.is_file():
        return _err("no marks saved beside that picture", 404)
    return send_file(p, mimetype="application/json")


@app.get("/api/image")
def image():
    """The picture, trimmed if the page asked for it.

    Untrimmed it sends the original file untouched. Trimmed it sends PNG of
    the warp -- never JPEG, because this image is what every mark is placed
    against and re-compressing it would move the paint under the marks.
    """
    try:
        p = _resolve(request.args.get("path", ""))
    except ValueError as exc:
        return _err(str(exc), 404)
    spec = request.args.get("trim")
    if not spec or spec == "null":
        return send_file(p)
    try:
        im, _ = _load(p, spec)
        return _png(im)
    except (ValueError, json.JSONDecodeError) as exc:
        return _err(str(exc))


@app.get("/api/imageinfo")
def imageinfo():
    try:
        p = _resolve(request.args.get("path", ""))
    except ValueError as exc:
        return _err(str(exc), 404)
    try:
        im, tinfo = _load(p, request.args.get("trim"))
    except (ValueError, json.JSONDecodeError) as exc:
        return _err(str(exc))
    return jsonify({"path": str(p), "w": int(im.shape[1]), "h": int(im.shape[0]),
                    "trim": tinfo,
                    "marks": (p.parent / (p.stem + ".marks.json")).is_file()})


# ------------------------------------------------------------- the accelerator

@app.post("/api/detect")
def detect():
    """Line segments and the vanishing-point families they fall into.

    An accelerator, never the backbone: the user accepts, rejects and merges
    what comes back. Families are merged (`vp.merge`) before they are offered,
    because sequential RANSAC splits one wide family in two and offering eight
    where four are real wastes the user's attention on pairs 2-3 deg apart.
    """
    b = _body()
    try:
        p = _resolve(b.get("image", ""))
    except ValueError as exc:
        return _err(str(exc), 404)
    min_len = float(b.get("min_len", 40))
    try:
        im, _ = _load(p, b.get("trim"))
    except (ValueError, json.JSONDecodeError) as exc:
        return _err(str(exc))
    # on the trimmed picture, or the desk and the drafting board get their own
    # vanishing-point families and they are the straightest lines in the frame
    s, gray = V.lines(im, min_len=min_len)
    ih, iw = gray.shape[:2]
    if not len(s):
        return jsonify({"segments": [], "families": [],
                        "note": "no line segments found at this minimum length"})
    fams = V.vps(s, n=int(b.get("n", 6)), tol=float(b.get("tol", 1.5)))
    merged = V.merge(s, fams, tol=float(b.get("tol", 1.5)))
    out = []
    for v, m in merged:
        pt = (v[:2] / v[2]).tolist() if abs(v[2]) > 1e-12 else None
        seg = s[m]
        L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        out.append({"vp": pt, "at_infinity": pt is None,
                    "direction": v[:2].tolist() if pt is None else None,
                    "n": int(m.sum()),
                    "segments": seg.tolist(),
                    "median_len_px": float(np.median(L)),
                    "longest_px": float(L.max()),
                    # bug 9, and it has to hold in both axes and at any scan
                    # size -- see V.short_for_vp
                    "short_segment_warning": V.short_for_vp(
                        float(np.median(L)), pt, iw, ih)})
    return jsonify({"segments": s.tolist(), "families": out,
                    "merged_from": len(fams)})


def _homography(spec, w, h):
    """Original-picture coordinates -> trimmed coordinates. None means none."""
    t = _trim(spec)
    if t is None:
        return None
    ow, oh, _f, _why = TR.shape_of(t["corners"], w, h)
    if t["out"]:
        ow, oh = t["out"]
    q = TR._order(t["corners"]).astype(np.float32)
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    return cv2.getPerspectiveTransform(q, dst)


@app.post("/api/retrim")
def retrim():
    """Marks from one trim's coordinates into another's.

    Marks are stored against the picture the user was looking at when
    placing them, so changing the trim afterwards would silently move every
    one of them off the paint. Rather than forbid that, or quietly corrupt
    the document, the marks are carried through: out of the old trim, back
    into the original photograph, and into the new one.
    """
    b = _body()
    try:
        p = _resolve(b.get("image", ""))
    except ValueError as exc:
        return _err(str(exc), 404)
    im = cv2.imread(str(p))
    if im is None:
        return _err("that file did not decode as an image")
    h, w = im.shape[:2]
    try:
        A = _homography(b.get("from"), w, h)     # old trim, or None
        Bm = _homography(b.get("to"), w, h)      # new trim, or None
        pts = _pts(b.get("points") or [])
    except (ValueError, json.JSONDecodeError) as exc:
        return _err(str(exc))
    if not len(pts):
        return jsonify({"points": []})
    if A is not None:
        pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2),
                                       np.linalg.inv(A)).reshape(-1, 2)
    if Bm is not None:
        pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Bm).reshape(-1, 2)
    return jsonify({"points": pts.tolist()})


@app.post("/api/findsheet")
def findsheet():
    """Where the sheet is in a photograph -- or why it cannot be said.

    It refuses far more often than it answers, and that is the point. Asked
    for a sheet in an artwork that was already cropped, the brightness search
    always finds some patch of paint whose outline has four corners; on
    testdata/4.jpg it returned a quadrilateral spanning the painting. What
    separates the two is whether the sides are real edges, measured against
    the picture's own texture: the one true sheet in the test set scored 7 to
    21 times texture on all four sides, every invented one scored 0.8 to 1.5.
    """
    b = _body()
    try:
        p = _resolve(b.get("image", ""))
    except ValueError as exc:
        return _err(str(exc), 404)
    im = cv2.imread(str(p))
    if im is None:
        return _err("that file did not decode as an image")
    h, w = im.shape[:2]
    q, why = TR.find_quad(im)
    if q is None:
        return jsonify({"corners": None, "note": why,
                        "suggest": _inset(w, h)})
    ow, oh, f, flat = TR.shape_of(q, w, h)
    return jsonify({
        "corners": q.tolist(), "out": [ow, oh],
        "photo_focal_px": None if f is None else round(f, 1),
        "edge_strength": [None if v is None else round(v, 1)
                          for v in TR.side_evidence(im, q)],
        "note": flat or (f"the sheet is keystoned, so it was shot off-axis; "
                         f"undoing it gives a {ow}x{oh} picture, and the "
                         f"camera that took the photograph had f = {f:.0f} px"),
    })


def _inset(w, h):
    """Somewhere to start dragging from when nothing was found."""
    mx, my = 0.08 * w, 0.08 * h
    return [[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]]


def _horizon_line(b):
    """The horizon a family was told to sit on, as a homogeneous line.

    Two vanishing points, and never this family's own: the line has to be
    outside information or holding the fit on it would mean nothing.

    One horizontal family and the uprights are outside information too, once
    there is a camera: the horizon is then inv(K Kt) v. The page used to need
    two other horizontal families and so could not hold the third family of a
    view down a nave on a horizon the check path already knew.
    """
    h = b.get("horizon")
    if not h or len(h) != 2:
        doc = b.get("doc")
        if doc:
            _cam, Kc = _camera(doc)
            if Kc is not None:
                a, q = _horizon(doc, Kc)
                if a is not None:
                    return V.horizon(a, q)
        return None
    try:
        return V.horizon(h[0], h[1])
    except (TypeError, ValueError, IndexError):
        return None


@app.post("/api/fit_vp")
def fit_vp():
    """The vanishing point of the lines the user drew, with its own spread.

    Weighted by length, and reported with a bootstrap interval, because a
    vanishing point quoted without one has already caused a wrong answer to be
    preferred over a right one.
    """
    b = _body()
    try:
        s = np.asarray(b.get("lines") or [], float).reshape(-1, 4)
    except ValueError:
        return _err("lines must be [[x1,y1,x2,y2], ...]")
    if len(s) < 2:
        return _err("two lines at least")
    L = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])

    def solve(idx):
        return V.fit(s, idx)

    v = solve(np.arange(len(s)))
    # A bootstrap over TWO lines is meaningless: half the resamples draw the
    # same line twice and return a degenerate point, which came out of the page
    # as "x within -0 to 3686". Two lines define their vanishing point exactly
    # and say nothing about how well it is known; three is where that question
    # starts having an answer.
    mark = _mark_accuracy(b.get("image"))[0]
    tid = b.get("tidied") or None
    # A tidied family agrees by construction: every line in it was swung onto
    # the fitted point, so resampling THOSE lines returns the same point to
    # 1e-7 px and certifies a camera the marks never pinned down. The draws put
    # back what the tidy took out -- its own worst swing, which is a floor on
    # the disagreement that was there. Measured on the Worcester rib: ten
    # uprights down a nave, tidied by at most 1.6 px, gave f = 14379 with the
    # reported spread +-3 %; honestly the marks allow 7600 .. 32600 and a fifth
    # of the draws give no camera at all, and the same three arch marks read a
    # rise over span of 0.26 at one end of that range and 2.4 at the other.
    jit = float(tid.get("worst") or 0.0) if tid else 0.0
    boot = []
    if len(s) >= 3:
        rng = np.random.default_rng(0)
        for _ in range(200):
            k = rng.integers(0, len(s), len(s))
            if len(np.unique(k)) < 2:
                continue
            w = V.fit(s[k] + rng.normal(0.0, jit, (len(k), 4)) if jit else s[k])
            if abs(w[2]) > 1e-12:
                boot.append(w[:2] / w[2])
    pt = (v[:2] / v[2]).tolist() if abs(v[2]) > 1e-12 else None
    move, stray = V.strays(s, v, mark)
    cond = {"n_lines": len(s), "median_len_px": float(np.median(L)),
            "longest_px": float(L.max()),
            "resid_deg": V.resid_deg(s, v).tolist(),
            "move_px": [round(float(x), 1) for x in move],
            "mark_accuracy_px": round(float(mark), 1)}
    if tid:
        cond["tidied_lines"] = int(tid.get("n") or 0)
        cond["tidied_why"] = (
            f"{tid['n']} of {len(s)} lines were swung onto this point, each by "
            f"at most {float(tid.get('worst') or 0):.1f} px -- within what a "
            f"hand mark is worth here, so it was marking and not drawing. "
            f"Those lines now agree by construction, so the spread below is "
            f"measured with {float(tid.get('worst') or 0):.1f} px of marking "
            "noise put back on every end -- resampling lines that were swung "
            "onto one point would otherwise report that point as certain")
    if stray:
        rest = np.delete(move, stray)
        cond["stray_lines"] = [i + 1 for i in stray]
        cond["stray_why"] = (
            f"line{'s' if len(stray) > 1 else ''} "
            f"{', '.join(str(i + 1) for i in stray)} of {len(s)}, marked white "
            f"on the picture, must swing "
            f"{', '.join(f'{move[i]:.0f}' for i in stray)} px to meet the point "
            f"the others agree on, where the rest need {rest.max():.0f} px or "
            f"less and a hand mark here is worth {mark:.0f} px. Either it "
            "belongs to another family or it is misdrawn -- and the point "
            "above was fitted with it included")
    # Held on the horizon, the point ON the horizon is the answer -- that is
    # what the declaration means -- and the free fit becomes conditioning. The
    # page used to draw the free one with the held one in the small print,
    # which put the cross visibly off the horizon it was said to be on.
    l = _horizon_line(b)
    held = None
    if l is not None and pt is not None:
        vh = V.fit_on_line(s, l)
        if abs(vh[2]) > 1e-12:
            held = (vh[:2] / vh[2]).tolist()
            hmove = 0.5 * L * np.tan(np.radians(V.resid_deg(s, vh)))
            worst = float(hmove.max())
            cond["free_point"] = [round(float(x), 1) for x in pt]
            cond["horizon_move_px"] = [round(float(x), 1) for x in hmove]
            gap = float(np.hypot(held[0] - pt[0], held[1] - pt[1]))
            cond["horizon_why"] = (
                "every family of horizontals shares one horizon, so this point "
                f"is held on the one the other families give. Left free the "
                f"lines put it at {pt[0]:.0f}, {pt[1]:.0f}, {gap:.0f} px away; "
                f"on the horizon they swing at most {worst:.1f} px to meet the "
                "point above "
                + ("-- within what a hand mark is worth here "
                   f"({mark:.0f} px), so Tidy can absorb it"
                   if worst <= mark else
                   f"-- beyond the {mark:.0f} px a hand mark is worth here, so "
                   "some of it is the drawing and Tidy will leave that part "
                   "alone"))
    if len(s) < 3 and held is None:
        cond["spread_unknown"] = ("two lines meet at exactly one point, so their "
                                  "residual is zero by construction and carries no "
                                  "information; a third line is what measures this one")
    elif len(s) < 3:
        cond["spread_unknown"] = ("two lines meet at exactly one point, so they "
                                  "cannot measure each other -- but the horizon can "
                                  "measure them, and the swing below is that "
                                  "measurement")
    if boot and pt:
        B = np.array(boot)
        cond["bootstrap_x"] = [float(np.percentile(B[:, 0], 5)),
                               float(np.percentile(B[:, 0], 95))]
        cond["bootstrap_y"] = [float(np.percentile(B[:, 1], 5)),
                               float(np.percentile(B[:, 1], 95))]
    if pt is None:
        return jsonify(C.readout(None, "at_infinity",
                                 direction=v[:2].tolist(), **cond))
    if np.median(L) < 60:
        cond["short_segments"] = True
    return jsonify(C.readout(held or pt, "check these" if stray else "ok", **cond))


@app.post("/api/square_vp")
def square_vp():
    """The fourth vanishing point: the one square to a marked horizontal.

    Two horizontal families and the uprights are the camera. A third family is
    often not part of that at all but an object standing askew to it -- a
    table turned off the wall, an archway in a wall that is not the room's --
    and then its OTHER horizontal direction is not free: it is square to the
    one that was marked, and the camera plus the horizon say where it goes.
    One cross product, `R.square_partner`.

    Derived on every call and never stored. The point moves when the camera
    moves, and a stored one would quietly go on answering for the camera it
    was made in.
    """
    b = _body()
    doc = b.get("doc") or {}
    of = b.get("of")
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    if of not in vps:
        return _err("name the horizontal family this one is square to")
    cam, Kc = _camera(doc)
    if Kc is None:
        return jsonify(C.undetermined(
            "the direction square to a horizontal is wherever the camera puts "
            "it, and this drawing has no camera: " + (cam.get("why") or "")))
    a, q = _horizon(doc, Kc)
    if a is None:
        return jsonify(C.undetermined(q))
    l = V.horizon(a, q)
    v = R.square_partner(Kc, vps[of], l)
    name = (doc.get("vp_names") or {}).get(of, of)
    if v is None:
        return jsonify(C.undetermined(
            f"nothing square to {name} is picked out here: either it is the "
            "upright direction, which every horizontal is square to, or its "
            "partner runs parallel to the picture and has no vanishing point "
            "on the page"))
    cond = {"square_of": of, "square_of_name": name,
            "square_why": f"square to {name} in the world, and on the horizon "
                          "-- the camera, not the marks, is what puts it here"}
    if cam.get("verdict") == "undetermined":
        cond["camera_undetermined"] = cam.get("why")

    # What the camera's own slack costs it. In pixels of POSITION this point is
    # hopeless -- f out by a twentieth moves it hundreds of pixels -- and in
    # pixels of SWING, which is what a line drawn to it actually suffers, the
    # same slack is worth under one. So both go out: it is a point to draw to,
    # never a position to quote.
    lo, hi, assumed = _focal_span(cam, Kc)
    cond["focal_span_px"] = [round(lo, 1), round(hi, 1)]
    cond["focal_span_assumed"] = assumed
    alt = []
    for fx in (lo, hi):
        Kx = np.array(Kc, float)
        Kx[0, 0] = Kx[1, 1] = fx
        u = R.square_partner(Kx, vps[of], V.horizon(*_horizon(doc, Kx)))
        if u is not None:
            alt.append(u)
    try:
        s = np.asarray(b.get("lines") or [], float).reshape(-1, 4)
    except ValueError:
        s = np.zeros((0, 4))
    # A derived family usually has no marks at all, and then there is no
    # length on the page to say what a swing is worth. Borrow the partner's:
    # same object, same hand. Only the sweep uses it -- the gap below is this
    # family's own lines against the derived point, and the partner's lines
    # run to the partner's point, which would be a different question.
    try:
        sc = s if len(s) else np.asarray(b.get("scale_lines") or [],
                                         float).reshape(-1, 4)
    except ValueError:
        sc = s
    mid = (sc[:, :2] + sc[:, 2:]) / 2.0 if len(sc) else None
    span = (float(np.median(np.hypot(sc[:, 2] - sc[:, 0], sc[:, 3] - sc[:, 1])))
            if len(sc) else None)

    def swing(u, w):
        """How far a mark of the family's own length swings between two points."""
        if mid is None or not span:
            return None
        d1, d2 = u - mid, w - mid
        d1 = d1 / np.maximum(np.linalg.norm(d1, axis=1), 1e-12)[:, None]
        d2 = d2 / np.maximum(np.linalg.norm(d2, axis=1), 1e-12)[:, None]
        sn = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])   # no 2-D cross
        return float(np.max(span * sn))

    if len(alt) == 2:
        cond["moves_px"] = round(float(np.linalg.norm(alt[0] - alt[1])), 1)
        sw = swing(alt[0], alt[1])
        if sw is not None:
            cond["swing_px"] = round(sw, 2)

    # Marked as well as derived, the gap between the two is a reading on the
    # camera -- the only one an askew object can give, and a weak one: the
    # swing above says how weak.
    if len(s) >= 2:
        vf = V.fit(s)
        if abs(vf[2]) > 1e-12:
            vf = vf[:2] / vf[2]
            cond["fitted_free"] = [round(float(x), 1) for x in vf]
            cond["gap_px"] = round(float(np.linalg.norm(vf - v)), 1)
            sw = swing(np.atleast_2d(vf), np.atleast_2d(v))
            if sw is not None:
                cond["gap_swing_px"] = round(sw, 2)
            im = doc.get("image") or {}
            mark = _mark_accuracy([doc.get("width") or im.get("w") or 1000,
                                   doc.get("height") or im.get("h") or 1000])[0]
            cond["mark_accuracy_px"] = round(float(mark), 1)
            cond["gap_why"] = (
                f"marked, these lines put their own point {cond['gap_px']:.0f} px "
                f"from the square one, which swings them "
                f"{cond.get('gap_swing_px', 0):.1f} px -- "
                + ("within the "
                   if cond.get("gap_swing_px", 0) <= mark else "beyond the ")
                + f"{mark:.0f} px a hand mark is worth here. An askew object is "
                  "a poor camera either way: the sweep above shows how little "
                  "this point minds f")
        else:
            cond["gap_why"] = ("marked, these lines come out parallel on the "
                               "page, so they have no point of their own to "
                               "set beside the square one")
    return jsonify(C.readout([float(v[0]), float(v[1])], "ok", **cond))


def _focal_span(cam, K):
    """(lo, hi, assumed): the focal lengths the marks still allow -- the sweep
    `_camera` ran, or a tenth either side of its f when it ran none."""
    c = cam.get("conditioning") or {}
    rg = c.get("focal_range_over_marks") or c.get("focal_range_over_pp")
    if rg and len(rg) == 2:
        return float(rg[0]), float(rg[1]), False
    f0 = float(K[0, 0])
    return 0.9 * f0, 1.1 * f0, True


def _span_words(lo, hi, assumed):
    """The focal span said as what it is: off the marks, or assumed."""
    if assumed:
        return (f"{lo:.0f}-{hi:.0f} px, a tenth either side of the fitted focal "
                "length because the marks give no range of their own")
    return f"{lo:.0f}-{hi:.0f} px, the range the marks allow"


def _with_f(K, f):
    Kx = np.array(K, float)
    Kx[0, 0] = Kx[1, 1] = f
    return Kx


# How far a drawn line may sit from the place that makes a square and still be
# taken for that side: a hand mark and a half. Further out it is somewhere else
# on purpose and is left alone. And how far from a right angle two families may
# be, in the world, before "square" is not a thing they can make.
SNAP_MARKS = 1.5
RIGHT_DEG = 5.0


def _family_segs(doc):
    """Every family's marked lines, from the document the page sent."""
    return {k: [s for s in (v or []) if s and len(s) == 4]
            for k, v in (doc.get("family_lines") or {}).items()}


@app.post("/api/square_snap")
def square_snap():
    """Where a line drawn into one family would close a SQUARE.

    The family has a line already (the first is free) and another family has
    two or more. Between them they place a second line that makes a square --
    which two vanishing points alone cannot: they name two directions without
    saying the angle between them, so a parallelogram is as good as a square
    to them. The camera says it, and the camera's refusal is this route's.

    A line within a hand mark and a half of such a place IS that line, and the
    answer is the drawn one moved onto it. Further out it is somewhere else on
    purpose and is left. A family with too few lines for a vanishing point of
    its own is given the one the camera puts square to the other, as
    `square_vp` does; a pair that is not at right angles in the world is
    refused with the angle, because a square is not what they can make.
    """
    b = _body()
    doc = b.get("doc") or {}
    into = b.get("into")
    try:
        d = np.asarray(b.get("line"), float).reshape(4)
    except (TypeError, ValueError):
        return _err("a line is four numbers")
    cam, K = _camera(doc)
    if K is None:
        return jsonify(C.undetermined(
            "a square needs the camera, because two vanishing points name two "
            "directions without saying the angle between them: "
            + (cam.get("why") or "")))
    names = doc.get("vp_names") or {}
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    segs = _family_segs(doc)
    mine = segs.get(into) or []
    if not mine:
        return jsonify(C.undetermined(
            "a square needs a line of this family to start from; the first is free"))
    im = doc.get("image") or {}
    mark = _mark_accuracy([doc.get("width") or im.get("w") or 1000,
                           doc.get("height") or im.get("h") or 1000])[0]
    reach = SNAP_MARKS * float(mark)
    askew = {}

    def cands(Kx):
        """(key, line, corners, angle, derived) for every square the lines on
        the page could close under camera `Kx`."""
        hz = None
        if into not in vps:
            a_, q_ = _horizon(doc, Kx)
            hz = None if a_ is None else V.horizon(a_, q_)
        for k, vb in vps.items():
            if k == into or len(segs.get(k) or []) < 2:
                continue
            va = vps.get(into)
            if va is None and hz is not None:
                va = R.square_partner(Kx, vb, hz)
            if va is None:
                continue
            da, db = (np.linalg.solve(Kx, np.append(v, 1.0)) for v in (va, vb))
            ang = float(np.degrees(np.arccos(min(
                1.0, abs(da @ db) / (np.linalg.norm(da) * np.linalg.norm(db))))))
            if abs(90.0 - ang) > RIGHT_DEG:
                askew[k] = ang
                continue
            vl = np.cross(np.append(va, 1.0), np.append(vb, 1.0))
            bl = [R.seg_line(s, vb) for s in segs[k]]
            for ai, s in enumerate(mine):
                a = R.seg_line(s, va)
                for i, j in combinations(range(len(bl)), 2):
                    if a is None or bl[i] is None or bl[j] is None:
                        continue
                    for n, got in enumerate(R.square_completions(Kx, va, vb, a, bl[i], bl[j])):
                        if got is None:
                            continue
                        line, q = got
                        side = [float(vl @ np.append(c, 1.0)) for c in q]
                        # a square with a corner past the vanishing line is
                        # half of it behind the viewer
                        if not (all(x > 0 for x in side) or all(x < 0 for x in side)):
                            continue
                        if min(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])) < 1.0:
                            continue
                        yield (k, ai, i, j, n), line, q, ang, vps.get(into) is None

    def off(line, p):
        return float(line @ np.append(p, 1.0))

    found = list(cands(K))
    if not found:
        why = ("no other family has two lines to make a square with"
               if not askew else
               "; ".join(f"{names.get(k, k)} is {a:.0f} deg from this family in "
                         "the world, not 90 -- a square is not what they make"
                         for k, a in askew.items()))
        return jsonify(C.undetermined(why))
    key, line, quad, ang, derived = min(
        found, key=lambda c: max(abs(off(c[1], d[:2])), abs(off(c[1], d[2:]))))
    gap = max(abs(off(line, d[:2])), abs(off(line, d[2:])))
    if gap > reach:
        return jsonify(C.undetermined(
            f"the nearest square is {gap:.0f} px from this line, beyond the "
            f"{reach:.0f} px a snap reaches",
            nearest_px=round(gap, 1), reach_px=round(reach, 1)))
    ends = [d[:2] - off(line, d[:2]) * line[:2], d[2:] - off(line, d[2:]) * line[:2]]

    # What the camera's own slack does to that side. It is the answer's
    # conditioning: a side that f can put anywhere within a hand mark's width
    # is no better placed than the hand put it, and a snap would only be
    # moving a mark toward a number no more certain than the mark.
    lo, hi, assumed = _focal_span(cam, K)
    swing = []
    for fx in (lo, hi):
        same = [c for c in cands(_with_f(K, fx)) if c[0] == key]
        if same:
            swing.append(max(abs(off(same[0][1], e)) for e in ends))
    k = key[0]
    cond = {"against": names.get(k, k), "against_id": k,
            "from_line": f"line {key[1] + 1} of this family",
            "between_lines": f"lines {key[2] + 1} and {key[3] + 1} of "
                             f"{names.get(k, k)}",
            "moved_px": round(gap, 2), "reach_px": round(reach, 1),
            "mark_accuracy_px": round(float(mark), 1),
            "right_angle_deg": round(ang, 2),
            "vp_of_this_family": "derived from the camera and the other family"
                                 if derived else "fitted from its own lines",
            "focal_span_px": [round(lo, 1), round(hi, 1)],
            "focal_span_assumed": assumed,
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in quad]}
    if cam.get("verdict") == "undetermined":
        cond["camera_undetermined"] = cam.get("why")
    span = _span_words(lo, hi, assumed)
    if len(swing) < 2:
        return jsonify(C.undetermined(
            f"at one end of the focal lengths in range ({span}) there is no "
            "such square, so where it goes is not settled", **cond))
    cond["f_moves_it_px"] = round(max(swing), 2)
    if max(swing) > mark:
        return jsonify(C.undetermined(
            f"over focal lengths of {span} this side lands anywhere within "
            f"{max(swing):.1f} px, which is no better than the {mark:.0f} px a "
            "hand mark is worth -- so a snap would be moving the line toward a "
            "place no more certain than the line",
            **cond))
    return jsonify(C.readout([round(float(x), 2) for e in ends for x in e], "ok", **cond))


@app.post("/api/square_cell")
def square_cell():
    """The circle in a square, for every place the page marked.

    A mark is a point. The cell is the one it sits in between two lines of
    each of the two families that span the plane -- the ground unless the
    feature names a surface, as an ellipse always has -- and the circle is the
    image of the one inscribed in a square, which four corners fix exactly
    and which needs no camera. What the camera is for is the OTHER question:
    whether that cell is a square at all, which the circle takes for granted.
    So a drawing with no camera still gets its circles and is told so.
    """
    b = _body()
    doc = b.get("doc") or {}
    feat = (doc.get("features") or {}).get(b.get("feature")) or {}
    pts = [p for p in (feat.get("cells") or []) if p and len(p) == 2]
    if not pts:
        return _err("click inside a cell: the space between two lines of each family")
    ids, why = _surface_vps(doc, feat.get("surface"), ground=True)
    if ids is None:
        return _err(why)
    names = doc.get("vp_names") or {}
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    segs = _family_segs(doc)
    sa, sb = (segs.get(i) or [] for i in ids)
    nm = [names.get(i, i) for i in ids]
    if len(sa) < 2 or len(sb) < 2:
        return _err(f"a cell needs two marked lines of each family that spans the "
                    f"plane: {nm[0]} has {len(sa)}, {nm[1]} has {len(sb)}")
    cam, K = _camera(doc)
    resolve = float(feat.get("resolve") or 0.05)
    if K is not None:
        lo, hi, assumed = _focal_span(cam, K)
        Ks = [K, _with_f(K, lo), _with_f(K, hi)]

    def reading(q):
        """How square the cell is, with what the focal slack does to it."""
        if K is None:
            return C.undetermined(
                "the circle is the one a SQUARE here would hold, and nothing "
                "can say this cell is one without a camera: "
                + (cam.get("why") or ""))
        got = [R.cell_reading(Kx, q) for Kx in Ks]
        if any(g is None for g in got):
            return C.undetermined("the cell has no size on its plane")
        asp = [g[0] for g in got]
        r = [max(a, 1.0 / a) for a in asp]             # long side over short
        along = nm[0] if asp[0] >= 1.0 else nm[1]
        cond = {"aspect": round(r[0], 3), "aspect_range": [round(min(r), 3), round(max(r), 3)],
                "longer_along": along, "angle_deg": round(got[0][1], 2),
                "focal_span_px": [round(lo, 1), round(hi, 1)],
                "focal_span_assumed": assumed, "resolve": resolve}
        if cam.get("verdict") == "undetermined":
            cond["camera_undetermined"] = cam.get("why")
        dev = [x - 1.0 for x in r]
        span = _span_words(lo, hi, assumed)
        if all(x <= resolve for x in dev):
            v, w = "ok", (f"a square to within {resolve * 100:.0f} %: its sides are "
                          f"{r[0]:.2f} : 1, and stay within that over focal lengths "
                          f"of {span}")
        elif all(x > resolve for x in dev):
            v, w = "off", (f"not a square: one side is {r[0]:.2f} times the other "
                           f"(longer along {along}), and no focal length of {span} "
                           "makes it one")
        else:
            v, w = "undetermined", (f"{r[0]:.2f} : 1 at the fitted focal length and "
                                    f"{min(r):.2f} to {max(r):.2f} over focal lengths "
                                    f"of {span}, which straddles what is worth "
                                    "calling square; the circle is drawn, whether it "
                                    "is round is not settled")
        return {"value": r[0], "verdict": v, "why": w, "conditioning": cond}

    cells = []
    for p in pts:
        q = R.square_cell(sa, sb, vps[ids[0]], vps[ids[1]], p)
        if q is None:
            cells.append({"why": "there is no cell here: the point is outside the "
                                 "lines marked, or the nearest lines cross before "
                                 "they close it"})
            continue
        cells.append({"quad": q.tolist(), "ring": R.cell_ring(q).tolist(),
                      "reading": reading(q)})
    return jsonify({"cells": cells, "families": nm})


@app.post("/api/tidy")
def tidy():
    """Absorb the hand's wobble into the point the family agrees on.

    Separate from fit_vp because it CHANGES the marks: the fit is a reading
    and this is an edit, and one route should not quietly do both.
    """
    b = _body()
    try:
        s = np.asarray(b.get("lines") or [], float).reshape(-1, 4)
    except ValueError:
        return _err("lines must be [[x1,y1,x2,y2], ...]")
    l = _horizon_line(b)
    if len(s) < 2:
        return _err("two lines at least")
    if len(s) < 3 and l is None:
        return _err("three lines at least -- two meet at exactly one point, so "
                    "there is no spread to tell the hand's wobble from the "
                    "drawing's. A horizon to hold them on would be a third "
                    "constraint, and then two is enough")
    v = V.fit(s)
    if abs(v[2]) < 1e-12:
        return _err("these lines are parallel on the page, so there is no "
                    "point to swing them onto")
    mark = _mark_accuracy(b.get("image"))[0]
    out, moved, worst = V.tidy(s, v, mark, line=l)
    r = {"lines": out.tolist(), "moved": [i + 1 for i in moved],
         "n": len(moved), "worst": round(worst, 1),
         "mark_accuracy_px": round(float(mark), 1)}
    if l is not None:
        r["on_horizon"] = True
    return jsonify(r)


@app.post("/api/tidy_row")
def tidy_row():
    """Absorb the hand's wobble into the even row the posts agree on.

    Separate from /api/check for the same reason /api/tidy is separate from
    /api/fit_vp: that one reads, this one edits the marks.
    """
    b = _body()
    try:
        q = np.asarray(b.get("pts") or [], float).reshape(-1, 2)
    except ValueError:
        return _err("pts must be [[x, y], ...]")
    vp = b.get("vp")
    if vp is None:
        return _err("say which vanishing point the row runs to")
    if len(q) < 4:
        return _err("four posts at least -- three always fit an even row "
                    "exactly, so there is nothing to tell the hand's wobble "
                    "from the drawing's")
    v = _xy(vp)
    mark = _mark_accuracy(b.get("image"))[0]
    _mid, _e, bend, off = R.row_frame(q, v)
    if off > mark:
        return _err(f"the vanishing point named is not the one this row runs "
                    f"to: it sits {off:.0f} px off the row's own line, more "
                    f"than a hand mark ({mark:.0f} px)")
    if bend > mark:
        return _err(f"these posts bend {bend:.0f} px off a straight row, more "
                    f"than a hand mark ({mark:.0f} px) -- an even row is only "
                    f"defined along a straight run")
    out, moved, worst, stray = R.tidy_row(q, v, mark)
    return jsonify({"pts": out.tolist(), "moved": [i + 1 for i in moved],
                    "n": len(moved), "worst": round(worst, 1),
                    "stray": [i + 1 for i in stray],
                    "mark_accuracy_px": round(float(mark), 1)})


# ------------------------------------------------------------------- the checks

N_BOX = 3        # the grid over a vanishing point's own spread, per axis


def _pull_into(apex, mid, w, h, margin=0.04):
    """Where the arch's upright leaves the picture, if the apex is outside it.

    Along the segment from the foot of the upright to the apex, which is part
    of the upright's own image, so the point returned is a point of the arch's
    own geometry and not a page fudge. None when the apex is already inside,
    or when clipping would leave the arch flat.
    """
    a, c = np.asarray(apex, float), np.asarray(mid, float)
    lo = np.array([margin * w, margin * h])
    hi = np.array([(1 - margin) * w, (1 - margin) * h])
    if np.all((a >= lo) & (a <= hi)) or np.any((c < lo) | (c > hi)):
        return None
    d, t = a - c, 1.0
    for k in (0, 1):
        if abs(d[k]) > 1e-9:
            t = min(t, max((lo[k] - c[k]) / d[k], (hi[k] - c[k]) / d[k]))
    return None if t < 0.05 else c + t * d


def _box_cameras(doc, vps, vert, horiz, pp):
    """A camera at each point of a grid over the upright family's own spread.

    Each point is paired with the f IT implies and nothing is mixed: an f taken
    from one point of the box with a vanishing point from another is not a
    camera at all, and swept that way an arch refuses at cameras that draw it
    perfectly well (half an hour lost to exactly that).

    A grid and not the four corners. f^2 is linear in the vanishing point, so
    over a box that straddles the horizon it runs from far below zero to far
    above and the corners are no guide to the inside: on the Worcester nave the
    corners all gave f = 26500 and the fitted camera, 14379, was not in the
    range they reported.
    """
    box = (doc.get("vp_ranges") or {}).get(vert)
    if not box or len(box) != 2 or horiz not in vps:
        return []
    xs, ys = (np.linspace(float(q[0]), float(q[1]), N_BOX) for q in box[:2])
    return [((x, y), C.camera_level([x, y], vps[horiz], pp_x=pp[0], pp_y=pp[1]))
            for x in xs for y in ys]


def _alt_cameras(doc, K, vps, vert):
    """(K, upright vanishing point) for each camera the marks still allow.

    The shape of an arch is read through f, so a readout taken at the one
    fitted camera is a readout without its conditioning. Same box as above,
    same pp -- only f and the upright point move, together.
    """
    horiz = [k for k in vps if k != vert]
    if K is None or not horiz:
        return []
    pp = (float(K[0, 2]), float(K[1, 2]))
    out = []
    for v, q in _box_cameras(doc, vps, vert, horiz[0], pp):
        Ka = q["conditioning"].get("K") if q["verdict"] != "undetermined" else None
        if Ka is not None:
            out.append((np.asarray(Ka, float), np.asarray(v, float)))
    return out


def _camera(doc):
    """The camera the document's vanishing points support, or a refusal.

    Three readouts need it and the rest do not, so it is built once here and
    anything needing it inherits the refusal rather than inventing a focal
    length of its own.
    """
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    vert = doc.get("vertical_vp")
    # A family DERIVED square to another one says nothing about the camera: it
    # came out of the camera. And of the rest, the two that make the camera are
    # the two the page already calls the horizon's own -- whichever the hand
    # drew first. Left to document order, a table standing askew to the room
    # took a wall's place and the camera came back confidently wrong.
    horiz = [k for k in vps if k != vert and k not in (doc.get("square_of") or {})]
    named = [k for k in (doc.get("horizon_vps") or []) if k in horiz]
    if len(named) >= 2:
        horiz = named + [k for k in horiz if k not in named]
    mode = doc.get("perspective") or "three"

    # The declared perspective type decides what a camera can be asked for at
    # all. One-point has no focal length to find -- the single vanishing point
    # IS the principal point, so f appears in no vanishing point and no number
    # of marks recovers it. Two-point has one, but only once the principal
    # point's position along the horizon is assumed, because sliding it there
    # trades for a different f and fits the marks exactly the whole way.
    if mode == "one":
        return C.undetermined(
            "a one-point drawing has no focal length to measure: its single "
            "vanishing point sits exactly at the principal point, so f enters "
            "no vanishing point and cannot be recovered however many lines are "
            "marked. The checks that need a camera need a second or third "
            "vanishing direction -- mark the diagonals of floor or ceiling "
            "squares, or check the drawing as two- or three-point"), None
    if mode == "two":
        if len(horiz) < 2:
            return C.undetermined(
                "two-point perspective needs its two horizontal vanishing "
                f"points; {len(horiz)} is marked"), None
        a, b = vps[horiz[0]], vps[horiz[1]]
        px = float(doc.get("width") or (doc.get("image") or {}).get("w") or 1000) / 2.0
        got = C.camera_level(a, b, pp_x=px)
        if got["verdict"] == "undetermined":
            return got, None
        cond = got["conditioning"]
        cond["assumed"] = (
            "camera level, as two-point perspective says, and the principal "
            f"point at x = {px:.0f} -- two-point cannot measure where it sits "
            "along the horizon, only its height")
        # Two-point throws the vertical family away: it is ASSUMED parallel on
        # the page and square to the horizon, which is what two-point means.
        # Mark one and the assumption becomes measurable -- and on a
        # photograph of a nave it was 7.0 deg out while this branch returned
        # f = 1538 "ok", because only the three-point branch ever looked. The
        # rib read through that camera leaned 39 deg and stood 2.1 spans tall.
        # K still goes back, as it does for the unpinned camera: the arch is
        # drawn from it, and a drawing is a guide, not a claim.
        if vert in vps:
            cons = C.horizon_consistency(vps[vert], a, b, 2.0 * px, pp=cond["pp"])
            cond["horizon_consistency"] = cons["conditioning"]
            if cons["verdict"] == "undetermined":
                cond["fitted_focal_px"] = round(float(got["value"]), 1)
                got = {"value": None, "verdict": "undetermined", "conditioning": cond,
                       "why": cons["why"] + ". Two-point perspective is the claim "
                       "that the camera is level, and a level camera keeps its "
                       "uprights square to its horizon. Re-mark whichever of the "
                       "three families is least sure, or take the tick off the "
                       "vertical one and two-point goes back to assuming what it "
                       "cannot then check"}
        return got, np.asarray(cond.get("K") or got["value"])

    # Say which of the three is missing, and where the switch for it is. The
    # one message for both cases sent a drawing with three good families back
    # to look for a fourth, when all it needed was the tick-box.
    if vert not in vps:
        return C.undetermined(
            "a camera needs three mutually perpendicular vanishing points, and "
            "no family is marked as the upright one: open the family of "
            "verticals and tick 'this is the vertical family'"
            + ("" if len(vps) >= 3 else
               f" -- {len(vps)} famil{'y is' if len(vps) == 1 else 'ies are'} "
               "marked, and three is the fewest that give a camera")), None
    w = float(doc.get("width") or (doc.get("image") or {}).get("w") or 1000)
    h = float(doc.get("height") or (doc.get("image") or {}).get("h") or 1000)
    if len(horiz) < 2:
        # One horizontal family and the uprights are two perpendicular
        # directions, and two give one equation for three unknowns -- f and
        # the principal point's two coordinates. Pinning the point at the
        # middle of the picture leaves f alone, and that is the best a view
        # along a nave or a street can do: there is no second horizontal
        # family in it to mark, however long it is looked at. The pin is an
        # assumption, so it travels with the answer and the sweep says what it
        # costs. Same equation as the level case, which is why that function
        # does it: f^2 = -(vA - pp).(vB - pp) holds for any two directions at
        # right angles once pp is fixed.
        if len(horiz) == 1:
            pp = (w / 2.0, h / 2.0)
            got = C.camera_level(vps[vert], vps[horiz[0]], pp_x=pp[0], pp_y=pp[1])
            if got["verdict"] != "undetermined":
                fs = []
                for dx in (-0.05, 0.0, 0.05):
                    for dy in (-0.05, 0.0, 0.05):
                        q = C.camera_level(vps[vert], vps[horiz[0]],
                                           pp_x=pp[0] + dx * w, pp_y=pp[1] + dy * h)
                        if q["verdict"] != "undetermined":
                            fs.append(float(q["value"]))
                cond = got["conditioning"]
                cond["assumed"] = (
                    "the principal point at the middle of the picture: one "
                    "horizontal family and the uprights fix f once it is "
                    "pinned, but cannot say where it sits")
                cond["pp_assumed"] = f"{pp[0]:.0f}, {pp[1]:.0f}"
                cond["pp_refused"] = (
                    "two horizontal vanishing points are what place it, and "
                    "one is marked")
                if fs:
                    cond["focal_range_over_pp"] = [round(min(fs), 1),
                                                   round(max(fs), 1)]
                # The pin above is an assumption and that sweep says what it
                # costs. This says what the MARKS cost, which on a view along a
                # nave is the bigger number by a factor of forty: ten uprights
                # running parallel to within the hand's own accuracy barely
                # place their vanishing point, and f appears in the geometry
                # only as a distance to a vanishing point.
                fv = [float(q["value"]) for _, q in
                      _box_cameras(doc, vps, vert, horiz[0], pp)
                      if q["verdict"] != "undetermined"]
                if (doc.get("vp_ranges") or {}).get(vert):
                    cond["focal_range_over_marks"] = (
                        [round(min(fv), 1), round(max(fv), 1)]
                        if len(fv) > 1 else None)
                    if len(fv) < N_BOX * N_BOX:
                        cond["focal_marks_refused"] = (
                            f"{N_BOX * N_BOX - len(fv)} of the "
                            f"{N_BOX * N_BOX} points across the upright "
                            "family's own spread give no camera at all, so f "
                            "is not measured by this drawing: it is whatever "
                            "that family's point happened to land on. Mark a "
                            "second horizontal family, or uprights that are "
                            "not all nearly parallel on the page")
                    elif max(fv) > 1.5 * min(fv):
                        cond["focal_unpinned_by_marks"] = (
                            f"f is anywhere between {min(fv):.0f} and "
                            f"{max(fv):.0f} px over the spread of the upright "
                            "family's own vanishing point: the value beside it "
                            "is one camera out of that range, not a "
                            "measurement. Everything read through the camera "
                            "-- an arch's rise over span, a viewing distance, "
                            "a circle's verdict -- inherits the range")
                why = (cond.get("focal_marks_refused")
                       or cond.get("focal_unpinned_by_marks"))
                if why:
                    # K still goes back, because the arch, the circle and the
                    # distance are DRAWN from it and a drawing is a guide, not
                    # a claim. The camera itself is not a measurement here, and
                    # a green tick beside 14378.9 was the whole of it: ten
                    # uprights down a nave, a lens no camera has, and a rib
                    # drawn from it that was visibly wrong.
                    cond["fitted_focal_px"] = round(float(got["value"]), 1)
                    got = {"value": None, "verdict": "undetermined",
                           "why": why, "conditioning": cond}
                return got, np.asarray(cond.get("K") or got["value"])
            return got, None
        return C.undetermined(
            "a camera needs two horizontal vanishing points at right angles "
            f"besides the vertical one; {len(horiz)} is marked"), None
    a, b = vps[horiz[0]], vps[horiz[1]]
    # Solve first, explain second. horizon_consistency used to gate this, but a
    # gate that runs BEFORE the solve can only ever refuse cameras the solve
    # would have found: it turned down an exactly orthogonal triple that
    # calibrate recovers to f = 1100.000. A camera that exists needs no gate,
    # so the check now says why there is none, rather than deciding whether
    # there may be one -- and once calibrate has answered, the roll can be
    # measured from the principal point it found instead of an assumed one.
    K = R.calibrate(vps[vert], a, b)
    if K is not None:
        cons = C.horizon_consistency(vps[vert], a, b, w,
                                     pp=(float(K[0, 2]), float(K[1, 2])))
        return (C.readout({"f": float(K[0, 0]),
                           "pp": [float(K[0, 2]), float(K[1, 2])]},
                          "ok", horizon_consistency=cons["conditioning"]), K)

    cons = C.horizon_consistency(vps[vert], a, b, w, pp=(w / 2, h / 2))
    if cons["verdict"] == "undetermined":
        return cons, None
    got = C.camera_level(a, b, pp_x=float(vps[vert][0]))
    if got["verdict"] == "undetermined":
        return got, None
    got["conditioning"]["note"] = ("the free three-VP solve found no real "
                                   "focal length; this assumes a level camera")
    return got, np.asarray(got["conditioning"].get("K") or got["value"])


def _surface_vps(doc, surface_id, ground=False):
    """The two vanishing points that span a surface, as ids, or a refusal.

    Both the vanishing line and the straightening need them, and they are not
    always the surface's own: a plane set level with the ground borrows the
    horizon's pair.

    `ground` lets one kind of feature fall back to the ground plane when no
    surface was named. Only the ellipse asks for it, and only because it is
    DRAWN: the ring appears on the picture, so a circle taken as flat when it
    was meant to stand in a wall is visible at once. Nothing that merely
    reports a number may default -- see bug 3.
    """
    surf = (doc.get("surfaces") or {}).get(surface_id)
    if not surf:
        if not ground:
            return None, "this feature has no surface label, and no plane is assumed"
        surf, surface_id = {"name": "the ground", "level": True}, None
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    name = "the ground" if surface_id is None else f"surface '{surf.get('name', surface_id)}'"
    if surf.get("level"):
        # A plane parallel to the ground shares the ground's vanishing line,
        # whichever way it is turned, so a table standing askew on the floor
        # needs no lines of its own -- only the horizon the room already gives.
        ids = doc.get("horizon_vps") or []
        if len(ids) != 2 or any(i not in vps for i in ids):
            lead = ("the ground's vanishing line is the horizon"
                    if surface_id is None else
                    f"{name} lies level with the ground, so its "
                    "vanishing line is the horizon")
            return None, (lead + " -- and the horizon takes two "
                          "horizontal vanishing points; mark a second horizontal "
                          "family, or name a surface whose two families span this "
                          "plane")
    else:
        ids = surf.get("vps") or []
        if len(ids) != 2 or any(i not in vps for i in ids):
            return None, (f"{name} needs two vanishing "
                          "points that span it")
    return list(ids), None


def _vanish_line(doc, surface_id, ground=False):
    """The vanishing line of the surface a feature was labelled with.

    Required, never defaulted, except where `ground` is passed. Arcs in a
    vertical wall were once measured against the horizon -- the vanishing line
    of HORIZONTAL planes -- and the numbers looked plausible for a whole
    session (bug 3).
    """
    ids, why = _surface_vps(doc, surface_id, ground=ground)
    if ids is None:
        return None, why
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    a, b = (np.append(vps[i], 1.0) for i in ids)
    l = np.cross(a, b)
    n = np.linalg.norm(l)
    if n < 1e-12:
        return None, "the surface's two vanishing points coincide"
    return l / n, None


def _run_one(doc, feat, K, Ks):
    kind = feat.get("kind")
    sid = feat.get("surface")
    try:
        if kind in ("arc", "ellipse"):
            # An ellipse drawn from its two ends IS the projection of a
            # circle -- that is how it was constructed -- so there is
            # nothing in it to measure. It is a guide to draw against; mark
            # round the rim of what was drawn to have the drawing measured.
            if kind == "ellipse" and len(_pts(feat.get("pts") or [])) < 5:
                if feat.get("cells") and not feat.get("ring_from"):
                    return C.undetermined(
                        "these circles were drawn as the ones inscribed in a square of "
                        "the lines marked, so they are round by construction -- there is "
                        "nothing here to check. Whether each square IS one is read beside "
                        "it, on the layer; mark round the rim of what you drew to have "
                        "the drawing measured",
                        n=len(feat.get("cells") or []))
                if feat.get("ring_from"):
                    return C.undetermined(
                        "this ellipse was drawn as the circle its plane and the camera "
                        "give, so it is round by construction -- there is nothing here "
                        "to check. It is a guide to draw against; mark round the rim "
                        "of what you drew to have the drawing measured",
                        n=len(feat.get("pts") or []))
                return C.undetermined("five marks at least round the rim: a conic "
                                      "has five degrees of freedom and four fit anything",
                                      n=len(feat.get("pts") or []))
            rim = _pts(feat["pts"])
            if K is None:
                return C.undetermined(
                    "a circle cannot be told from an oval without the camera: "
                    "every ellipse is some circle under some camera",
                    **C.arc_quality(rim))
            # An ellipse with no surface named is a circle lying flat on the
            # ground; an arc is not, and still must say which plane it is in.
            #
            # Reading it as a circle square to the viewer instead was measured
            # against a ring the user drew round a stool's base: 2.011 to 1,
            # which is a ground circle under a 70 deg lens, landing 0.5 px
            # from the drawn one. The facing plane gives 1.13 there.
            l, why = _vanish_line(doc, sid, ground=(kind == "ellipse"))
            if l is None:
                return C.undetermined(why, **C.arc_quality(rim))
            out = C.circle(rim, K, l, K_alt=Ks,
                           resolve=float(feat.get("resolve", 0.05)))
            surf = (doc.get("surfaces") or {}).get(sid) or {}
            if not surf and kind == "ellipse":
                out["conditioning"]["surface_assumed"] = (
                    "no surface was named, so this was read as a circle lying flat "
                    "on the ground, against the horizon. Name a surface if it stands "
                    "in a wall or on a slope.")
            elif surf.get("level"):
                out["conditioning"]["surface_assumed"] = (
                    f"'{surf.get('name', sid)}' was declared level with the ground, so "
                    "this was read against the horizon and not against lines marked "
                    "on the plane itself. Nothing here measures whether it is level.")
            return out

        if kind == "arch":
            vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
            vert, wall = feat.get("vertical_vp") or doc.get("vertical_vp"), feat.get("wall_vp")
            # Straight sides make it a roof, and a roof's slope vanishing
            # point, the vertical one and the one it rises along -- the wall's
            # own horizontal -- are collinear. No camera needed for that.
            if feat.get("shape") == "straight" and feat.get("slope_vp"):
                need = [feat["slope_vp"], vert, wall]
                if any(n not in vps for n in need):
                    return C.undetermined(
                        "a roof slope needs its own vanishing point, the vertical "
                        "one, and the wall's horizontal one it rises along")
                return C.roof(*[vps[n] for n in need])
            # A drawn arch is the arch its plane and the camera give, so there is
            # nothing in it to measure -- and that is true before any wall is
            # named, which a rib across the bay never names.
            if len(_pts(feat.get("pts") or [])) < 5:
                if feat.get("arch_from"):
                    return C.undetermined(
                        "this arch was drawn as the one its plane and the camera give, "
                        "so it is true by construction -- there is nothing here to "
                        "check. It is a guide to draw against; mark round the arch "
                        "you drew to have the drawing measured",
                        n=len(feat.get("pts") or []))
                return C.undetermined("five marks at least round the arch: a conic "
                                      "has five degrees of freedom and four fit anything",
                                      n=len(feat.get("pts") or []))
            if K is None or vert not in vps or wall not in vps:
                return C.undetermined(
                    "an arch is read on its own wall, so it needs the camera, the "
                    "vertical vanishing point and the wall's horizontal one")
            near = _pts(feat["pts"])
            q = C.arc_quality(near)
            far = _pts(feat["far"]) if feat.get("far") else None
            nvp = vps.get(feat.get("normal_vp")) if feat.get("normal_vp") else None
            got = R.arch(near, far, K=K, vert_vp=vps[vert], wall_vp=vps[wall],
                         normal_vp=nvp)
            out = {k: v for k, v in got.items() if k != "vanish_line"}
            out["depth_miss_px"] = (np.asarray(out["depth_miss_px"]).tolist()
                                    if "depth_miss_px" in out else None)
            if q["turning_deg"] < C.TURN_MIN_DEG:
                return C.undetermined(
                    f"the arc turns {q['turning_deg']:.0f} deg; below "
                    f"{C.TURN_MIN_DEG:.0f} deg its shape is not determined", **q)
            return C.readout(out, "ok", **q)

        if kind == "polygon":
            return C.polygon(_pts(feat["pts"]), idx=feat.get("idx"))

        if kind == "figures":
            a, b = _horizon(doc, K)
            if a is None:
                return C.undetermined(b)
            pairs = [(p["feet"], p["head"]) for p in feat["figures"]]
            return C.figures(pairs, a, b)

        if kind == "shadow":
            # Three marks an object -- base, top, shadow tip -- and the rays,
            # the ground lines and the uprights all come out of those three.
            # The horizon is wanted but not needed: without it the sun simply
            # cannot be told from a lamp, and the check says so.
            vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
            vid = feat.get("vertical_vp") or doc.get("vertical_vp")
            if vid in vps:
                vert = np.append(vps[vid], 1.0)
            elif doc.get("perspective") in ("one", "two"):
                vert = np.array([0.0, 1.0, 0.0])
            else:
                return C.undetermined(
                    "a light stands over its own foot, so the uprights are what "
                    "puts it there: mark the vertical vanishing point")
            a, b2 = _horizon(doc, K)
            im = doc.get("image") or {}
            size = (float(doc.get("width") or im.get("w") or 1000),
                    float(doc.get("height") or im.get("h") or 1000))
            return C.shadow(feat.get("pairs") or [], vert,
                            a, None if a is None else b2,
                            mark_px=_mark_accuracy(size)[0])

        if kind == "landmark":
            # The one measurement that leaves the paper: a real size in the
            # SCENE, turned into the viewer's distance from it. The world
            # direction it runs along is the vertical unless it says otherwise
            # -- a standing figure, a door, a lamp post.
            vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
            vid = feat.get("vp") or doc.get("vertical_vp")
            if vid in vps:
                v = np.append(vps[vid], 1.0)
            elif doc.get("perspective") in ("one", "two"):
                # Uprights that are parallel on the page have no vanishing
                # point to mark, which is not a gap: the direction is simply
                # parallel to the picture plane, and a homogeneous point at
                # infinity says exactly that.
                v = np.array([0.0, 1.0, 0.0])
            else:
                return C.undetermined(
                    "say which vanishing point this runs to -- the vertical "
                    "one for anything standing up. Without a direction a real "
                    "size says nothing about distance")
            im = doc.get("image") or {}
            size = (float(doc.get("width") or im.get("w") or 1000),
                    float(doc.get("height") or im.get("h") or 1000))
            pairs = [(p[0], p[1]) for p in (feat.get("pairs") or [])]
            sz = float(feat.get("size") or 0.0) * TO_MM.get(feat.get("size_unit") or "cm", 10.0)
            fr = None
            if len(Ks) >= 2:
                fr = [float(min(k[0, 0] for k in Ks)), float(max(k[0, 0] for k in Ks))]
            return C.landmark(pairs, v, K, sz,
                              mark_px=_mark_accuracy(size)[0],
                              unit=feat.get("size_unit") or "cm", f_range=fr,
                              on_ground=bool(feat.get("on_ground", True)))

        if kind == "posts":
            if feat.get("built"):
                return C.undetermined(
                    "this row was built from its two ends and a count, so it "
                    "is evenly spaced by construction -- there is nothing "
                    "here to check. It is a guide to draw against; tick each "
                    "post instead to have the drawing measured",
                    n=len(feat.get("pts") or []))
            a, b = _horizon(doc, K)
            if a is None:
                return C.undetermined(b)
            vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
            # A row standing on a surface has its vanishing point worked out
            # from that surface rather than named, and then there is no family
            # id to look up -- the page sends the point. The guard inside
            # C.spacing checks it against the row either way.
            row_vp = _xy(feat["row_vp"]) if feat.get("row_vp") else None
            if row_vp is None and feat.get("vp") not in vps:
                return C.undetermined("even spacing is measured toward a vanishing "
                                      "point; name the surface the row stands on, "
                                      "or say which family it runs to")
            # The document's declared real size is the scale unless this
            # feature overrides it: typing px_per_mm per feature was the only
            # way before, and two features could disagree about one picture.
            # The page sends px_per_mm only when it was typed -- it used to
            # send 1.0 always, which is truthy, so the declared size never
            # once reached this check.
            w = float(doc.get("width") or (doc.get("image") or {}).get("w") or 1000)
            h = float(doc.get("height") or (doc.get("image") or {}).get("h") or 1000)
            ppm = _px_per_mm(doc.get("scale"), w, h)
            vp = vps[feat["vp"]] if row_vp is None else row_vp
            return C.spacing(_pts(feat["pts"]), vp, a, b,
                             float(feat.get("px_per_mm") or ppm or 1.0),
                             resolve_mm=float(feat.get("resolve_mm", 2.0)),
                             mark_px=_mark_accuracy((w, h))[0])

        if kind == "axis":
            vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
            if feat.get("vp") not in vps:
                return C.undetermined(
                    "proportion along an axis is a cross-ratio against that axis's "
                    "own vanishing point; a standing figure uses the vertical one, "
                    "a reclining figure does not")
            alts = [vps[k] for k in (feat.get("vp_alt") or []) if k in vps]
            sp = tuple(feat.get("span", (0, -1)))
            un = tuple(feat.get("unit", (0, 1)))
            return C.proportion(_pts(feat["pts"]), vps[feat["vp"]], sp, un, alts)

        if kind == "pairs":
            pairs = [(p[0], p[1]) for p in feat["pairs"]]
            vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
            return C.symmetry(pairs, vps.get(feat.get("axis_vp")),
                              vps.get(feat.get("pair_vp")),
                              float(feat.get("floor", 0.03)))
    except (KeyError, ValueError, IndexError) as exc:
        return C.undetermined(f"this feature is incomplete: {exc}")
    return C.undetermined(f"no check knows what a '{kind}' is")


def _horizon(doc, K=None):
    """The horizon, from the two horizontal vanishing points the user named.

    With only one named, the camera still knows it: the horizon is the
    vanishing line of every plane square to the uprights, inv(K Kt) v. A
    photograph straight down a nave gives one horizontal family and no
    second one, and every check that wanted a horizon refused on a picture
    whose camera was already solved.
    """
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    ids = doc.get("horizon_vps") or [k for k in vps if k != doc.get("vertical_vp")]
    if len(ids) < 2 or any(i not in vps for i in ids[:2]):
        vert = vps.get(doc.get("vertical_vp"))
        if K is None or vert is None:
            return None, ("the horizon is where the horizontal vanishing points "
                          "line up; two of them are needed, or a camera and the "
                          "uprights")
        l = np.linalg.inv(np.asarray(K) @ np.asarray(K).T) @ np.append(vert, 1.0)
        n = float(np.hypot(l[0], l[1]))
        if n < 1e-12:
            return None, "the uprights and this camera give no horizon"
        p = -l[2] * l[:2] / (n * n)                 # the point nearest the origin
        return p, p + np.array([-l[1], l[0]]) / n
    return vps[ids[0]], vps[ids[1]]


@app.post("/api/export_pdf")
def export_pdf():
    """The overlay as a vector PDF.

    A marks file says where the lines are; it does not let anyone look at
    them. The geometry goes out as vectors and the picture as pixels on its
    own layer, so the line work survives deleting the photograph.
    """
    b = _body()
    doc = b.get("doc") or {}
    if not doc.get("w") or not doc.get("h"):
        return _err("no picture open, so there is nothing to draw the marks on")
    im = None
    if b.get("with_image") and doc.get("path"):
        try:
            im, _t = _load(_resolve(doc["path"]), b.get("trim"))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except (ValueError, json.JSONDecodeError) as exc:
            return _err(str(exc))
    out = EX.pdf(doc, im)
    name = Path(doc.get("path") or "marks").stem + ".lines.pdf"
    return send_file(io.BytesIO(out), mimetype="application/pdf",
                     as_attachment=True, download_name=name)


@app.post("/api/straighten")
def straighten():
    """The marked shape, seen as if the plane it lies in faced us.

    A drawing cannot be judged in perspective -- a circle is an oval and a
    square a trapezium -- so this undoes the plane's foreshortening and hands
    back a picture to look at. It grades nothing; the eye does that.
    """
    b = _body()
    doc = b.get("doc") or {}
    feat = (doc.get("features") or {}).get(b.get("feature")) or {}
    pts = feat.get("pts") or []
    if len(pts) < 3:
        return _err("put at least three points around the thing to straighten")
    ids, why = _surface_vps(doc, feat.get("surface"))
    if ids is None:
        return _err(why)
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    cam, K = _camera(doc)
    if K is None:
        return _err("straightening needs the camera: " + (cam.get("why") or
                    "three mutually perpendicular vanishing points"))
    # whichever of the two spanning directions is the vertical one stands up
    # in the view; on a floor neither is, and any turn in the plane will do
    up = vps.get(doc.get("vertical_vp")) if doc.get("vertical_vp") in ids else None
    try:
        im, _t = _load(_resolve((doc.get("image") or {}).get("path") or ""),
                       b.get("trim") or doc.get("trim"))
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        return _err(str(exc))
    out, _M = RC.straighten(im, K, np.asarray(pts, float).reshape(-1, 2),
                            vps[ids[0]], vps[ids[1]], vp_up=up)
    if out is None:
        return _err("this plane cannot be turned to face us from where the "
                    "camera is -- its vanishing line runs through the picture")
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        return _err("could not encode the straightened view")
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


@app.post("/api/ellipse")
def ellipse():
    """The circle that lies in a plane, as it falls on the page.

    The plane is the ground unless the feature names a surface.

    The page asks for this rather than working it out itself: two vanishing
    points name two directions in the plane without saying the angle between
    them, so a circle needs the camera, and the camera is solved here.
    """
    b = _body()
    doc = b.get("doc") or {}
    feat = (doc.get("features") or {}).get(b.get("feature")) or {}
    segs = feat.get("ring_from") or []
    if segs and not isinstance(segs[0], (list, tuple)):
        segs = [segs]                                  # one drag, the old shape
    segs = [q for q in segs if q and len(q) == 4]
    if not segs:
        return _err("drag once, right across the circle the long way")
    sid = feat.get("surface")
    l, why = _vanish_line(doc, sid, ground=True)
    if l is None:
        return _err(why)
    _cam, K = _camera(doc)
    if K is None:
        return _err("a circle needs the camera: " + (_cam.get("why") or
                    "three mutually perpendicular vanishing points"))
    # Horizontal planes level their own drag: the diameter parallel to the
    # picture plane is the one whose image runs parallel to the horizon, and
    # it is exactly parallel, so there is nothing for the hand to get right.
    surf = (doc.get("surfaces") or {}).get(sid) or {}
    level = (not sid) or bool(surf.get("level"))

    out = {"rings": [], "rings_true": [], "ends": []}
    for q in segs:
        ends = ((float(q[0]), float(q[1])), (float(q[2]), float(q[3])))
        if level:
            got = R.horizon_diameter(*ends, l)
            if got is not None:
                ends = (tuple(got[0]), tuple(got[1]))
        ring = R.plane_circle(K, l, *ends)
        if ring is None:
            return _err("a circle that size reaches this plane's vanishing line, "
                        "where its image is no longer an ellipse -- move it, or "
                        "make it smaller")
        out["ends"].append([ends[0][0], ends[0][1], ends[1][0], ends[1][1]])

        # The ring the user draws against is the circle seen SQUARE ON, because
        # that is the one whose LONG diameter ends at the two dots -- the thing
        # asked for above everything else. The circle this picture's camera casts
        # leans away from the centre of vision instead, missing them by 4 to 16 px.
        #
        # The leaning one is still drawn, faint, beside it: it is what a marked
        # ring is measured against, and the gap between them is the picture's own
        # marginal stretch. Only for a free ring, where the plane is the ground.
        square = R.square_on_circle(K, l, *ends) if not sid else None
        # how far apart they really are, not point against point: the two are
        # parameterized differently, so a pointwise difference means nothing
        gap = (0.0 if square is None else
               float(np.linalg.norm(square[:, None] - ring[None], axis=2).min(1).max()))
        if square is not None and gap > 0.5:
            out["rings"].append(square.tolist())
            out["rings_true"].append(ring.tolist())
        else:
            out["rings"].append(ring.tolist())
            out["rings_true"].append(None)
    return jsonify(out)


@app.post("/api/arch")
def arch_draw():
    """The arch two feet and a crown mark describe, standing on its own wall.

    Read on the same two vanishing points the arch CHECK uses, so the curve
    drawn here and the numbers reported later cannot disagree about which
    plane the arch is in. Nothing is defaulted: an arch measured against the
    wrong plane looked plausible for a whole session once (bug 3).
    """
    b = _body()
    doc = b.get("doc") or {}
    feat = (doc.get("features") or {}).get(b.get("feature")) or {}
    ms = feat.get("arch_from") or []
    if ms and not isinstance(ms[0], (list, tuple)):
        ms = [ms]                                      # one arch, the old shape
    ms = [q for q in ms if q and len(q) in (4, 6, 8)]
    if not ms:
        return _err("drag the springing line across, then move the top")
    vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items() if v is not None}
    vert = feat.get("vertical_vp") or doc.get("vertical_vp")
    wall = feat.get("wall_vp")
    if vert not in vps:
        return _err("an arch stands upright on something: name the vertical "
                    "vanishing point")
    _cam, K = _camera(doc)
    if K is None:
        return _err("a circular arch needs the camera: " + (_cam.get("why") or
                    "three mutually perpendicular vanishing points"))
    im = doc.get("image") or {}
    w = float(doc.get("width") or im.get("w") or 1000)
    h = float(doc.get("height") or im.get("h") or 1000)
    # Three planes an arch can stand in, and the drawing says which. Named a
    # wall, it stands in that wall -- an arcade. Named none, it stands in the
    # vertical plane through its OWN two feet, which is a rib across the bay.
    # Asked to lean, it does not stand upright at all: it tips about the
    # springing line, as a gothic rib leaves its respond, and the crown mark
    # gives the lean and the rise together.
    depth = vps.get(feat.get("normal_vp"))
    lean = bool(feat.get("lean"))
    # One crown mark is two numbers and the lean takes both: the arch tips
    # about the springing line, and how far up the mark then sits is the rise
    # in whatever plane that lands on. A vault wants the same mark to say how
    # far back it goes, and there is nothing left for it to say that with --
    # so the depth point is dropped and said, not made to kill the arch.
    note = None
    if lean and depth is not None:
        depth, note = None, (
            "the depth vanishing point is not read while this arch leans: one "
            "crown mark gives the lean and the rise together, and nothing is "
            "left over to say how far back a vault goes. Take the lean off and "
            "the same mark slides the arch back instead")
    wl = vps.get(wall)

    def one(m, K, vxy, with_bez=True):
        """The arch those marks make under THAT camera: (arch, lean deg, why)."""
        span = wl if wl is not None else R.span_vp_of(K, vxy, m[0:2], m[2:4])
        if span is None:
            return None, None, "level"
        l, deg = (None if wl is None else
                  np.cross(np.append(vxy, 1.0), np.append(wl, 1.0))), None
        if lean and len(m) >= 6:
            l, deg = R.lean_line(K, vxy, span, m[0:2], m[2:4], m[4:6])
            if l is None:
                return None, None, "sights"
        elif l is None:
            l = np.cross(np.append(vxy, 1.0), np.append(span, 1.0))
        got = R.plane_arch(K, l, m[0:2], m[2:4], m[4:6] if len(m) >= 6 else None,
                           pointed=bool(feat.get("pointed")), span_vp=span,
                           depth_vp=depth, shape=feat.get("shape"),
                           with_bez=with_bez,
                           through=m[6:8] if len(m) == 8 else None)
        return got, deg, None if got is not None else "nofit"

    # An arcade is several arches on ONE wall under ONE camera, so they come
    # out consistent because they were built that way, not because anything
    # reconciled them afterwards.
    alts = _alt_cameras(doc, K, vps, vert)
    out = {"arches": [], "feet": [], "crowns": [], "rise_over_span": [],
           "bez": [], "bez_off": [], "plane": [], "vault": [], "slope": []}
    out["lean_deg"], out["rise_over_span_range"] = [], []
    for m in ms:
        got, deg, why = one(m, K, vps[vert])
        if why == "level":
            return _err("that springing line runs level with the horizon, so it "
                        "has no vanishing point of its own: name the wall's "
                        "horizontal one and the arch will stand in the wall")
        if why == "sights":
            return _err("the top sights along the springing line, so nothing "
                        "there says how far the arch leans")
        if got is None:
            return _err("no arch fits those three marks: the top has to sit off "
                        "the springing line, and the span must not reach the "
                        "wall's own vanishing line")
        # The default top is a semicircle's, half the span up ON THE WALL --
        # and how high that is on the page is set by the camera's depth scale,
        # which a nearly end-on span barely pins. The same two feet down the
        # nave put it 456 px above the picture at f = 14379 and 409 px inside
        # it at f = 3968. An unreachable handle is no way to start, so the
        # mark starts where the upright leaves the frame and says so; drag it
        # to where the arch really goes.
        if len(m) == 4 and got["mid"] is not None:
            top = _pull_into(got["crown"], got["mid"], w, h)
            if top is not None:
                alt, deg2, _why2 = one(list(m) + list(top), K, vps[vert])
                if alt is not None:
                    off = float(np.hypot(*(np.asarray(got["crown"]) - top)))
                    got, deg, m = alt, deg2, list(m) + list(top)
                    out["crown_clamped"] = (
                        f"a semicircle on this span reaches {off:.0f} px past "
                        "the edge of the picture, so the top mark starts where "
                        "its upright leaves the frame. Drag it to the crown "
                        "you can see: how tall a receding span stands is read "
                        "through the camera, and this one is nearly end-on")
        out["lean_deg"].append(deg)
        # The same three marks under every camera the vanishing points still
        # allow. An arch's shape is read through f, so this IS the arch's
        # conditioning: one rib down a nave read 0.26 at the fitted camera and
        # 2.4 at the other end of what its own uprights allowed.
        rs = [got["rise_over_span"]] + [g["rise_over_span"] for g, _, _ in
                                        (one(m, Ka, va, with_bez=False)
                                         for Ka, va in alts)
                                        if g is not None]
        out["rise_over_span_range"].append(None if len(rs) < 3 else
                                           [min(rs), max(rs)])
        out["arches"].append(got["pts"].tolist())
        out["bez"].append(got["bez"])
        out["bez_off"].append(got["bez_off"])
        out["plane"].append(got["plane"])
        # a vault: the same arch slid back along the depth lines, which is one
        # homology, so the page can draw the far limb from the near one
        out["vault"].append(None if not got["mu"] else
                            {"homology": got["homology"],
                             "far_feet": got["far_feet"],
                             "far_crown": got["far_crown"],
                             "near_crown": got["crown"].tolist()})
        out["feet"].append([f.tolist() for f in got["feet"]])
        out["crowns"].append(got.get("far_crown") or got["crown"].tolist())
        out["rise_over_span"].append(got["rise_over_span"])
        out["slope"].append(got.get("slope_vps"))
    if note:
        out["depth_unused"] = note
    return jsonify(out)


@app.post("/api/check")
def check():
    doc = _body().get("doc") or {}
    cam, K = _camera(doc)
    Ks = []
    if K is not None and doc.get("vp_ranges"):
        try:
            vps = {k: _xy(v) for k, v in (doc.get("vps") or {}).items()}
            vert = vps[doc["vertical_vp"]]
            hz = [k for k in vps if k != doc.get("vertical_vp")][:2]
            # [[x lo, x hi], [y lo, y hi]] as the page sends it; f_range
            # slides the point along x, which is the horizon's own direction
            # for a horizontal family. Dead until the page started sending the
            # boxes at all -- it was reading the pair as one interval.
            rng = (doc["vp_ranges"].get(hz[0]) or [None])[0]
            if rng and len(rng) == 2:
                Ks, _ = C.f_range(vert, vps[hz[0]], vps[hz[1]], a_range=tuple(rng))
        except (KeyError, IndexError, ValueError):
            Ks = []
    results = {}
    for fid, feat in (doc.get("features") or {}).items():
        results[fid] = _run_one(doc, feat, K, Ks)
        results[fid]["kind"] = feat.get("kind")
        results[fid]["name"] = feat.get("name") or fid
    im = doc.get("image") or {}
    ppm = _px_per_mm(doc.get("scale"),
                     float(doc.get("width") or im.get("w") or 1000),
                     float(doc.get("height") or im.get("h") or 1000))
    return jsonify(_add_mm({"camera": cam, "results": results,
                            "undetermined": sum(1 for r in results.values()
                                                if r["verdict"] == "undetermined"),
                            "total": len(results)}, ppm))


# ------------------------------------------------------------------- the render

@app.post("/api/overlay")
def overlay():
    """The marks and the fits, drawn over the image.

    Not decoration. Rendering caught a conic RANSAC skimming curtain streaks, a
    column-tracked arch visibly missing the paint, and a ballooning ellipse from
    contaminated points -- none of which the residuals flagged (bug 15). The
    numbers agreed while the picture was wrong more than once.
    """
    b = _body()
    doc = b.get("doc") or {}
    try:
        p = _resolve(doc.get("image", ""))
    except ValueError as exc:
        return _err(str(exc), 404)
    try:
        im, _ = _load(p, doc.get("trim"))
    except (ValueError, json.JSONDecodeError) as exc:
        return _err(str(exc))
    ov = im.copy()
    h, w = im.shape[:2]
    results = b.get("results") or {}

    # the horizon, if two horizontal VPs are named
    a, bb = _horizon(doc)[0], None
    hz = _horizon(doc)
    if hz[0] is not None:
        a, bb = hz
        d = bb - a
        if abs(d[0]) > 1e-9:
            y0 = a[1] + (0 - a[0]) * d[1] / d[0]
            y1 = a[1] + (w - a[0]) * d[1] / d[0]
            cv2.line(ov, (0, int(y0)), (w, int(y1)), (0, 200, 0), 2)

    for fid, feat in (doc.get("features") or {}).items():
        res = results.get(fid) or {}
        und = res.get("verdict") == "undetermined"
        # An undetermined readout gets its marks drawn and NOTHING fitted. A
        # correction drawn from a refusal is a confident wrong answer at full
        # opacity, and that is the most expensive failure this project has had.
        col = (120, 120, 120) if und else (0, 215, 255)
        pts = feat.get("pts") or feat.get("far")
        if feat.get("kind") == "figures":
            for g in feat.get("figures") or []:
                cv2.line(ov, tuple(np.int32(g["feet"])), tuple(np.int32(g["head"])), col, 2)
            continue
        if feat.get("kind") in ("pairs", "shadow"):
            for q in feat.get("pairs") or []:
                P = np.int32(q)
                for i in range(1, len(P)):
                    cv2.line(ov, tuple(P[i - 1]), tuple(P[i]), col, 2)
                if len(P) > 2:                  # a shadow's three: closed round
                    cv2.line(ov, tuple(P[0]), tuple(P[-1]), col, 2)
            # where the light it settled on ends each shadow. Only ever present
            # on a readout that found a light, so a refusal still draws nothing.
            for q in ((res.get("conditioning") or {}).get("tip_at") or []):
                cv2.circle(ov, (int(q[0]), int(q[1])), 7, (255, 0, 255), 2)
            continue
        if not pts:
            continue
        P = np.asarray(pts, float).reshape(-1, 2)
        for q in P:
            cv2.circle(ov, (int(q[0]), int(q[1])), 3, col, -1)
        if und or feat.get("kind") not in ("arc", "arch"):
            continue
        Cc = R.conic(P)                       # the fit, only when it was believed
        ys, xs = np.mgrid[0:h, 0:w]
        v = (Cc[0, 0] * xs * xs + 2 * Cc[0, 1] * xs * ys + Cc[1, 1] * ys * ys
             + 2 * Cc[0, 2] * xs + 2 * Cc[1, 2] * ys + Cc[2, 2])
        gx = 2 * (Cc[0, 0] * xs + Cc[0, 1] * ys + Cc[0, 2])
        gy = 2 * (Cc[0, 1] * xs + Cc[1, 1] * ys + Cc[1, 2])
        ov[np.abs(v) / np.maximum(np.hypot(gx, gy), 1e-9) < 1.0] = (255, 0, 255)

    ok, buf = cv2.imencode(".png", ov)
    if not ok:
        return _err("the overlay did not encode")
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


def _families(b):
    """The marked families of a request, checked: (families, size, mode, roles).

    Shared by the readout and by the camera it can be turned into, so the two
    cannot disagree about what was marked.
    """
    fam = b.get("families") or {}
    try:
        fam = {k: np.asarray(v, float).reshape(-1, 4) for k, v in fam.items()}
    except ValueError:
        raise ValueError("each family must be [[x1,y1,x2,y2], ...]")
    for k, v in fam.items():
        if not np.isfinite(v).all():
            raise ValueError(f"family '{k}' has a non-finite coordinate")
    size = b.get("image_size")
    size = (float(size[0]), float(size[1])) if size else None
    mode = b.get("mode") or "three"
    if mode not in K.MODES:
        raise ValueError("mode must be one of " + ", ".join(K.MODES))
    return fam, size, mode, b.get("roles") or None


def _reference(b, size):
    """The reference camera a request carries, in this picture's pixels, or None."""
    if not b.get("reference"):
        return None
    ref = CIO.validate(b["reference"])
    return CIO.fit_to(ref, size[0], size[1]) if size else ref


@app.post("/api/consensus")
def consensus():
    """The nearest consistent perspective, and what reaching it costs.

    A hand-drawn picture is never exact, so refusing it as inconsistent says
    nothing useful. The question worth answering is the other one: which
    perspective is nearest to what was actually drawn, and how far would each
    line have to move to get there? A move under the ~9 px a hand mark is worth
    is no error at all, and saying "rectifiable" about it would send someone to
    redraw a line that was already right.

    Both readings come back. The camera free to roll is what the drawing says;
    forcing it level is an assumption, and its price is printed beside it.

    With a `reference` camera (read from a .blend, .fspy or .json) a third
    reading comes back: what the same marks cost against THAT camera. One
    family is then enough, because the reference supplies the others.
    """
    b = _body()
    try:
        fam, size, mode, roles = _families(b)
        ref = _reference(b, size)
    except (ValueError, TypeError, IndexError) as exc:
        return _err(str(exc))
    if len(fam) < (1 if ref else 2):
        return _err("two line families at least, or one against a reference camera")

    out = {}
    if len(fam) >= 2:
        out["free"] = K.fit(fam, image_size=size, mode=mode, roles=roles)
        if len(fam) == 2 and mode == "three":
            # fit_level is three-point with pitch and roll forced to zero, which
            # is a different assumption from the declared types and only worth
            # printing beside the free fit.
            out["level"] = K.fit_level(fam, image_size=size)
    if ref:
        r = out["reference"] = K.against_vps(
            fam, CIO.vps_h(ref), image_size=size,
            focal_px=ref["f"], principal_point=ref["pp"])
        c = r.get("conditioning") or {}
        c["camera"] = {"source": ref.get("source"), "up": ref.get("up"),
                       "lens_mm_on_36": CIO.describe(ref)["lens_mm_on_36"]}
        # The reference is somebody's claim about the picture, and the marks
        # are another. Say how far apart those two claims are, in the same
        # coin both are paid in: the line that has to move furthest.
        free = (out.get("free") or {}).get("conditioning") or {}
        if "worst_move_px" in c and "worst_move_px" in free:
            c["price_of_reference_px"] = round(
                c["worst_move_px"] - free["worst_move_px"], 2)
            c["price_of_reference_basis"] = (
                "worst move against the reference less the worst move the "
                "marks need on their own to be consistent")
            # a baseline of 0 px means something only if the marks had a line
            # to spare; without one it is the arithmetic of an exact fit
            c["own_redundancy"] = free.get("redundancy")
        # ...and the cameras themselves, when the marks measured one
        if free.get("axes_cam") and free.get("focal_px") is not None:
            c["versus_marks"] = CIO.compare(ref, free["focal_px"],
                                            free["principal_point"], free["axes_cam"])

    # Attach the corrected line beside each drawn one, so the page can draw the
    # change instead of only naming it. These are the CLIPPED segments, which
    # are what the move was measured on.
    w, h = size or (1000.0, 1000.0)
    for res in out.values():
        per = (res.get("conditioning") or {}).get("per_family") or {}
        for name, d in per.items():
            if d.get("vp"):
                v = np.array([d["vp"][0], d["vp"][1], 1.0])
            elif d.get("direction_deg") is not None:
                # a vanishing point at infinity is a direction on the page:
                # the family's lines are to be made parallel, and they were
                # drawn nowhere before this
                t = np.radians(d["direction_deg"])
                v = np.array([np.cos(t), np.sin(t), 0.0])
            else:
                continue
            clipped, _miss = K._clip(fam[name], w, h)
            fixed = []
            for seg in clipped:
                q, _ = K.rectify(seg, v)
                if q is not None:
                    fixed.append([round(float(x), 2) for x in q])
            d["corrected"] = fixed
    # After the corrected segments, so those stay in pixels -- they are drawn
    # on the canvas, not read.
    return jsonify(_add_mm(out, _px_per_mm(b.get("scale"), w, h)))


# ------------------------------------------------------------------ the camera

@app.post("/api/camera_in")
def camera_in():
    """A camera another program solved, read into this picture's pixels.

    .blend (Blender is run in the background, scripts off), .fspy (a JSON
    file with a picture inside) or a plain .json. The camera comes back with
    its three vanishing points, which is what the page draws -- so that what is
    drawn and what is priced are one set of numbers.
    """
    b = _body()
    try:
        got = CIO.read(_camera_file(b.get("path", "")), b.get("camera") or None)
        cam = got["camera"]
        size = b.get("image_size")
        if size:
            cam = CIO.fit_to(cam, float(size[0]), float(size[1]))
    except KeyError as exc:
        return _err(f"that file is missing {exc}")
    except (ValueError, TypeError, OSError) as exc:
        return _err(str(exc))
    return jsonify({"camera": cam, "describe": CIO.describe(cam),
                    "others": got["others"], "refused": got["refused"]})


def _script_notes(res, up_basis, yaw, size):
    """The header of an exported camera: what it rests on, said once, in plain lines."""
    c = dict(res["conditioning"], _size=size)
    mark = c.get("mark_accuracy_px")
    n = [f"{c.get('perspective', 'perspective')}, {c['n_lines']} lines marked: the "
         f"worst moves {c['worst_move_px']:.1f} px and the median {c['median_move_px']:.1f} px "
         f"to meet this camera (a hand mark is {mark:g} px) -- {res['verdict']}"]
    if res.get("why"):
        n.append("NOT TESTED: " + res["why"])
    pp = c.get("principal_point") or [0, 0]
    w_, h_ = c.get("_size", (None, None))
    if w_ and not (0 <= pp[0] <= w_ and 0 <= pp[1] <= h_):
        n.append(f"the principal point ({pp[0]:.0f}, {pp[1]:.0f}) lies OUTSIDE the "
                 f"{w_:.0f} x {h_:.0f} picture, which is what a cropped frame looks "
                 "like; with few marks it is a sign the fit is loose, not that the "
                 "photograph was cropped")
    if c.get("pp_refused"):
        n.append("principal point NOT measured, held at the image centre: " + c["pp_refused"])
    elif c.get("pp_assumed"):
        n.append(f"principal point NOT measured, assumed at {c['pp_assumed']}")
    if c.get("pp_lateral_assumed") is not None:
        n.append(f"principal point's x assumed at {c['pp_lateral_assumed']:.0f} px; its "
                 "height is the horizon's and was measured")
    rng = c.get("focal_range_over_pp") or c.get("focal_range_over_pp_x")
    if rng:
        n.append(f"focal length {c['focal_px']:.0f} px, anywhere from {rng[0]:.0f} to "
                 f"{rng[1]:.0f} px depending on where the principal point is assumed")
    if c.get("note"):
        n.append(c["note"])
    n.append("up: " + up_basis + f"; X runs along the picture's width, Y by the right "
             f"hand; yaw {yaw} quarter turns")
    n.append("to use it: Blender > Scripting > Open this file > Run Script")
    return n


@app.post("/api/camera_out")
def camera_out():
    """The camera the marks agree on, as a Blender script with the picture behind it.

    Same request as /api/consensus, because it IS that readout: the camera
    written is the one `fit` returned, so what the page showed and what Blender
    receives cannot be two different fits. Refuses where the readout refuses --
    a lens that was not measured is not written down as a number.
    """
    b = _body()
    try:
        fam, size, mode, roles = _families(b)
        if size is None:
            raise ValueError("the picture's size is needed to write a camera")
        yaw = int(b.get("yaw") or 0) % 4
    except (ValueError, TypeError, IndexError) as exc:
        return _err(str(exc))
    res = K.fit(fam, image_size=size, mode=mode, roles=roles)
    c = res.get("conditioning") or {}
    # A fit with no marks to spare is "undetermined" as a CHECK but it is still
    # a camera, and the one those lines define: it is written, with that said.
    if "axes_cam" not in c:
        return _err("no camera to write: " + (c.get("focal_refused") or res.get("why")
                                              or "the marks do not give one"), 422)

    per, A = c["per_family"], c["axes_cam"]
    flagged = [n for n, r in (roles or {}).items()
               if r == "vertical" and n in per]
    names = b.get("names") or {}          # the page's own names for its layers
    if mode in ("one", "two"):
        up = 1
        why = f"{mode}-point perspective: its upright axis is the vertical one"
    elif flagged:
        up = per[flagged[0]]["axis_index"]
        why = f"the family flagged vertical ('{names.get(flagged[0], flagged[0])}')"
    else:
        up = CIO.pick_up(A)
        why = ("the axis nearest the picture's vertical -- nothing was flagged "
               "vertical, and under 45 deg of pitch that is the upright")
        if up is None:
            return _err("which way is up cannot be read from this camera (no axis "
                        "is clearly the picture's vertical): flag the vertical "
                        "family and write it again", 422)
    Rcv = CIO.world_frame(A, up, yaw)
    cam = {"f": c["focal_px"], "pp": c["principal_point"], "size": [size[0], size[1]]}

    picture, trimmed = None, bool(b.get("trim"))
    if b.get("picture") and not trimmed:
        try:
            picture = _resolve(b["picture"])
        except ValueError:
            picture = None
    notes = _script_notes(res, why, yaw, size)
    if trimmed:
        notes.append("the picture was trimmed on the page and the trimmed picture "
                     "is not a file: no background image is attached")
    num = CIO.blender_numbers(cam, Rcv)
    stem = Path(b.get("picture") or "picture").stem
    return jsonify({
        "value": round(num["lens"], 3), "verdict": res["verdict"], "why": res.get("why"),
        "filename": stem + ".persimmon_camera.py",
        "script": CIO.blender_script(cam, Rcv, picture=picture, notes=notes),
        "conditioning": {
            "lens_mm": round(num["lens"], 4), "sensor_mm": num["sensor_width"],
            "shift": [round(num["shift_x"], 6), round(num["shift_y"], 6)],
            "resolution": num["resolution"], "up_from": why, "yaw_quarter_turns": yaw,
            "picture_attached": picture is not None, "notes": notes,
            "worst_move_px": c["worst_move_px"], "median_move_px": c["median_move_px"],
            "mark_accuracy_px": c["mark_accuracy_px"],
            "focal_px": c["focal_px"], "principal_point": c["principal_point"]}})


@app.post("/api/selfcheck")
def run_selfcheck():
    """Re-run every verification the core was built against, from the page."""
    import subprocess
    r = subprocess.run([__import__("sys").executable, str(HERE / "selfcheck.py")],
                       capture_output=True, text=True, cwd=str(HERE))
    return jsonify({"ok": r.returncode == 0, "output": r.stdout + r.stderr})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"Persimmon  ->  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
