"""The marks as a vector PDF.

A marks file says where every line is; it does not let anyone look at them.
This draws the same overlay the page draws, as real vector geometry, so it can
go into Illustrator or onto a sheet of paper beside the drawing.

Only the geometry is vector. The picture underneath is a photograph or a scan
and goes in as pixels, on its own layer, so it can be deleted and the line
work kept. Hidden layers are not drawn -- what is exported is what was on
screen.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Polygon, PathPatch              # noqa: E402
from matplotlib.path import Path                                 # noqa: E402

from labkit import plots as P                                    # noqa: E402

# `labkit.plots` sets pdf.fonttype 42 at import, so the key below goes out as
# text and not as outlines. Importing it is the whole reason it is imported.
_ = P.MM

LONG_SIDE_IN = 10.0          # the long edge of the page, in inches
# Widths are POINTS on the page, not pixels of the picture. Scaling them with
# the picture put a 3.9 pt stroke -- 29 picture pixels -- over a watercolour.
LW_PT = 0.9
POLY_KINDS = ("arc", "arch", "polygon", "axis", "posts")
PAIR_KINDS = ("figures", "pairs", "landmark", "shadow")


def _axes(w, h):
    """One axes filling the page, in picture coordinates, y downward."""
    matplotlib.rcParams["font.family"] = P.FONTS["arial"]
    long_px = max(w, h)
    fig = plt.figure(figsize=(LONG_SIDE_IN * w / long_px,
                              LONG_SIDE_IN * h / long_px))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _family(ax, L, lw=LW_PT):
    c = L.get("color", "#000000")
    for s in L.get("lines") or []:
        ax.plot([s[0], s[2]], [s[1], s[3]], color=c, lw=lw, solid_capstyle="round")
    for q in L.get("curves") or []:
        q = np.asarray(q, float)
        ax.plot(q[:, 0], q[:, 1], color=c, lw=lw, solid_capstyle="round")
    vp = L.get("vp")
    if not vp:
        return
    # the ray, not the point: a vanishing point is usually off the page, and
    # what the eye needs is where the lines are heading
    for s in L.get("lines") or []:
        ax.plot([(s[0] + s[2]) / 2, vp[0]], [(s[1] + s[3]) / 2, vp[1]],
                color=c, lw=lw * 0.5, ls=(0, (2, 4)), alpha=0.55)
    if 0 <= vp[0] <= ax.get_xlim()[1] and 0 <= vp[1] <= ax.get_ylim()[0]:
        ax.plot(vp[0], vp[1], marker="+", ms=9, mew=lw * 1.6, color=c)


def pdf(doc, image=None):
    """The overlay in `doc` as PDF bytes. `image` is the picture, or None.

    `doc` is what the page sends: {w, h, layers, horizon}. Each layer carries
    its own colour and, for a surface, the quad the page computed -- the
    geometry is worked out in one place and drawn in two, rather than written
    twice.
    """
    w, h = int(doc["w"]), int(doc["h"])
    fig, ax = _axes(w, h)
    if image is not None:
        ax.imshow(image, extent=(0, w, h, 0), interpolation="antialiased",
                  zorder=0)
    lw = LW_PT

    hz = doc.get("horizon")
    if hz and abs(hz[1][0] - hz[0][0]) > 1e-6:
        (ax0, ay0), (bx, by) = hz[0], hz[1]
        def y_at(x):
            return ay0 + (x - ax0) * (by - ay0) / (bx - ax0)
        ax.plot([0, w], [y_at(0), y_at(w)],
                color=doc.get("horizon_color") or "#00e676", lw=lw,
                ls=(0, (7, 5)), zorder=3)

    keys = []
    for L in doc.get("layers") or []:
        if L.get("hidden"):
            continue
        c = L.get("color", "#000000")
        kind = L.get("kind")
        if kind == "family":
            _family(ax, L, lw)
        elif kind == "surface":
            q = L.get("quad")
            if q:
                ax.add_patch(Polygon(q, closed=True, facecolor=c, alpha=0.18,
                                     edgecolor="none", zorder=1))
                ax.add_patch(Polygon(q, closed=True, fill=False, edgecolor=c,
                                     lw=lw, zorder=2))
        elif kind == "ellipse":
            # the rings are geometry the page worked out, like a surface's quad
            rings, marks = L.get("rings") or [], L.get("pts") or []
            for ring in rings:
                ax.add_patch(Polygon(ring, closed=True, fill=False, edgecolor=c, lw=lw))
            # the circle this picture's own camera casts, when it differs from the
            # one drawn against: faint, as on screen
            for true in L.get("rings_true") or []:
                if not true:
                    continue
                ax.add_patch(Polygon(true, closed=True, fill=False,
                                     edgecolor=c, lw=lw, ls=(0, (3, 3)), alpha=0.45))
            if marks:
                q = np.asarray(marks, float)
                ax.plot(q[:, 0], q[:, 1], ls="none", marker="o", ms=LW_PT * 2.4, color=c)
            if not rings and not marks:
                continue
        elif kind == "arch" and L.get("arches"):
            # an arcade: every arch on one wall, built under one camera
            # the curve goes out as the Bezier it is on the page, control
            # points and all, so the PDF carries the same path the hand
            # adjusted and a drawing program opens it with its handles on
            far = L.get("bez_far") or [None] * len(L["arches"])
            ridge = L.get("vault") or [None] * len(L["arches"])
            for a, bz, fz, rg in zip(L["arches"],
                                     L.get("bez") or [None] * len(L["arches"]),
                                     far, ridge):
                # a vault's far limb, and the ridge and springing joins to it
                for P4 in (fz or []):
                    v = [P4[0], P4[1], P4[2], P4[3]]
                    ax.add_patch(PathPatch(Path(v, [Path.MOVETO] + [Path.CURVE4] * 3),
                                           fill=False, edgecolor=c, lw=lw))
                if fz and bz:
                    for j in ([bz[0][0], fz[0][0]], [bz[-1][3], fz[-1][3]],
                              rg or []):
                        if len(j) == 2:
                            ax.plot([j[0][0], j[1][0]], [j[0][1], j[1][1]],
                                    color=c, lw=lw * 0.7, alpha=0.75)
                if bz:
                    v, co = [bz[0][0]], [Path.MOVETO]
                    for P4 in bz:
                        v += [P4[1], P4[2], P4[3]]
                        co += [Path.CURVE4] * 3
                    ax.add_patch(PathPatch(Path(v, co), fill=False,
                                           edgecolor=c, lw=lw))
                else:
                    q = np.asarray(a, float)
                    ax.plot(q[:, 0], q[:, 1], color=c, lw=lw)
            if L.get("pts"):
                q = np.asarray(L["pts"], float)
                ax.plot(q[:, 0], q[:, 1], ls="none", marker="o",
                        ms=LW_PT * 2.4, color=c)
        elif kind in POLY_KINDS:
            for run in (L.get("pts") or [], L.get("far") or []):
                if not run:
                    continue
                p = np.asarray(run, float)
                if kind == "polygon":
                    p = np.vstack([p, p[:1]])
                ax.plot(p[:, 0], p[:, 1], color=c, lw=lw)
                ax.plot(p[:, 0], p[:, 1], ls="none", marker="o",
                        ms=LW_PT * 2.4, color=c)
        elif kind in PAIR_KINDS:
            for q in L.get("pairs") or []:
                p = np.asarray(q, float)
                if len(p) > 2:                  # a shadow's three: closed round
                    p = np.vstack([p, p[:1]])
                ax.plot(p[:, 0], p[:, 1], color=c, lw=lw)
                ax.plot(p[:, 0], p[:, 1], ls="none", marker="o",
                        ms=LW_PT * 2.6, color=c)
        else:
            continue
        keys.append(Line2D([], [], color=c, lw=1.6, label=L.get("name", kind)))

    if keys:
        # Inside the axes, because the page is the picture's own shape and a
        # strip below it would change that.
        ax.legend(handles=keys, loc="lower left", fontsize=6, frameon=True,
                  framealpha=0.75, facecolor="white", edgecolor="none",
                  borderpad=0.4, labelspacing=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()
