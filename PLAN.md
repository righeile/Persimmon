# Persimmon — semi-automatic linear-perspective checker for artwork

Port **8830**. Checks linear perspective in drawings, paintings and photos:
vanishing points, horizon, tilted roofs, repeated spacing, figures against the
horizon, regular polygons, circles/ellipses, and arches including projecting ones.

Status: **core geometry is built and verified on synthetics. No app exists yet.**
`core/vp.py` and `core/rhythm.py` are working modules, not sketches. Read this
whole file before writing code — most of it is bugs already paid for.

---

## 1. The design rule that matters most

**Every readout ships with its conditioning, and refuses to answer when ill-posed.**

This is not a nicety. The single biggest failure in development was producing
confident numbers from measurements that could not support them. Four separate
results had to be withdrawn after being reported. A number without its
conditioning is worse than no number, because it gets believed.

Concretely, every check must return `(value, conditioning, verdict)` where
verdict may be `"undetermined"`, and the UI must render `undetermined` as
prominently as a value — never as a blank or a zero.

**Semi-automatic means the user marks what the pixels cannot give.** Automatic
extraction failed repeatedly on real paintings (occlusion, watercolor texture,
low contrast). Hand-marked input was measured at ~9 px median accuracy against
the paint, which is good enough for every check here. Design for marking first
and auto-detection as an accelerator, not the reverse.

---

## 2. What is verified, and how

Self-checked against synthetics with known ground truth.

| capability | function | verified result |
|---|---|---|
| conic fit | `rhythm.conic` | 0.0000 px on exact circles, offset circles, 90° arcs, tilted ellipses, 42° shallow arcs; 1 px noise in → 1.00 px out |
| conic residual | `rhythm.conic_resid` | Sampson distance, px |
| camera from 3 orthogonal VPs | `rhythm.calibrate` | recovers f=1000.00 and centre (40,−25) exactly |
| circle vs oval on a known plane | `rhythm.circle_check` | 1.00000 anywhere on the page; 8 % oval → 1.0800; noise floor 1.021 |
| arch rise/span | `rhythm.arch` | 0.5000 for a semicircle, ±0.010 under ±2 px noise, resolves ~2 % |
| arch crown offset | `rhythm.arch` | 0.0000, ±0.006, resolves ~1 % of span |
| arch springing level | `rhythm.arch` | 0.0000, 0.004, resolves ~1 % of span |
| arch depth (planar homology) | `rhythm.arch` | 0.0000 px exact, 5.6 px under noise, resolves ~3 % |
| polygon from marked corners | `rhythm.polygon_vertices` | true n beats runner-up 4–6× (5-gon 0.4 vs 12.7; 6-gon 2.1 vs 16.2; 8-gon 3.1 vs 13.0) |
| polygon sweep | `rhythm.polygon_which` | see limit below |
| proportion along a marked axis | `figure.axis_check`, `figure.metric_coords` | 7.5 heads recovered to 1e-16 through four cameras, up to 72 deg turned and 28 deg tilted, where reading the pixels gives 4.63 |
| projective midpoint | `figure.harmonic_mid` | 2e-13 px; halfway-along-the-line is 1.27 px off on the same figure |
| symmetry of landmark pairs | `figure.midline` | exact on a symmetric figure and on a pushed shoulder; with +-1.5 px clicks the odd pair is named 94 % at 0.071 of width, 100 % at 0.126, coin-flip below 0.03 |
| drawing against its reference | `figure.deviation` | 3e-13 px for a faithful drawing from a different viewpoint; a 28.7 px error found and localised among nine points |
| stage-to-stage transfer | `figure.homography`, `figure.transfer` | 1e-7 px exact; see the registration table below |

**Stage registration limit — measured, 20 geometries x 50 noise draws.** Worst of
12 transferred marks, in px, by points clicked and click accuracy:

| click sd | 4 pts | 6 pts | 8 pts | 12 pts | 20 pts |
|---|---|---|---|---|---|
| 0.5 px | 8.03 | 1.66 | 1.27 | 0.86 | 0.56 |
| 1.0 px | 10.02 | 3.25 | 2.41 | 1.63 | 1.14 |
| 2.0 px | 30.69 | 7.82 | 5.21 | 3.11 | 2.41 |
| 4.0 px | 63.71 | 13.62 | 9.08 | 6.25 | 4.48 |

**Rule: never register a stage on 4 points.** The fit is exact there, so click
noise passes straight through and amplifies — 8 px of error from half-pixel
clicks. Ask for 8 or more.

**Polygon sweep limit — measured, 200 trials at ±3 px.** Identification rate by
corners marked: hexagon 97 %/100 % at 5/6; octagon 26/83/100/100 at 5/6/7/8;
decagon 9/23/71/97; 12-gon 2/6/29/54. **Rule: you need about n−1 corners.**
Partial arcs of large polygons are mutually indistinguishable — a true octagon's
5 corners rank n=12 first and n=8 fifth. The sweep must report the whole ranking
and the corner count, never just a winner.

---

## 3. Bugs already paid for — do not re-introduce

1. **Normalize coordinates before fitting a conic.** On raw pixels the `x²`
   column reaches 1e5 against a constant column of 1. The fit came back *worse
   than a straight line* (15.35 px vs 7.96 px), which is impossible since a line
   pair is itself a conic. `rhythm.conic` now does Hartley normalization. That
   tell — a conic losing to a line — is a good permanent assertion.

2. **Collapse extracted point bands to ridges.** Edge extraction returns bands
   ~19 px thick, not curves. Per-column median took a fit from 7.78 px to
   **0.52 px**. Always collapse before fitting.

3. **The vanishing line must match the plane the feature lies in.** Arcs in a
   *vertical wall* were measured against the *horizon* (the vanishing line of
   *horizontal* planes). Combined with bug 1 the two errors partly cancelled and
   produced plausible-looking wrong numbers. Make the plane an explicit,
   user-labelled property of every marked feature — never a default.

4. **Shallow arcs carry no shape information.** Arcs turning 27–55° gave
   axis ratios that moved from 1.076 to **2.279** when 20 % was trimmed off each
   end. Every arc readout must report the **turning angle** (tangent direction
   at one end vs the other) and must run the trim test (full / middle 80 % /
   middle 60 %) and show the spread. Below roughly 60–90° of turning, refuse the
   shape readout. *The exact threshold is not yet measured — measure it.*

5. **A bootstrap on a dense smooth ridge under-represents model uncertainty.**
   It reported ±0.02 on ratios that trimming moved by 1.2. Bootstrap captures
   point noise only. Always pair it with a subset/trim test.

6. **The minor-axis rule is not a perspective invariant.** WITHDRAWN. Exact only
   when the circle's centre lies in the plane through the optical axis; 1 unit
   off-axis → 22° error, 8 units → 82°. Do not reintroduce it.

7. **No calibration-free test separates a circle from an oval.** Every ellipse is
   some circle under some camera. The circle check requires three mutually
   orthogonal VPs. Without them, refuse.

8. **Angle convention.** `(a % 180) - 90` is the angle from **vertical**, not
   horizontal. For angle-from-horizontal use `min(|a|, 180−|a|)` (0 = horizontal,
   90 = vertical), as `rhythm.is_vertical` does. This error silently inverted a
   near-horizontal filter and was only caught by re-deriving it.

9. **Short segments cannot fit a distant VP.** 28–46 px mouldings gave a VP at
   x=−153 with bootstrap [−178, −49]; four user-drawn 644–917 px lines gave −18.
   The short-segment fit was wrong and was briefly believed over a correct
   earlier answer. Weight by segment length, report the bootstrap interval, and
   prefer long marked lines over many short detected ones.

10. **Under occlusion, intersect — do not track.** A corner hidden behind a
    chandelier was located to ±6 px by intersecting an arc fit with a line fit,
    after column-tracking had failed entirely. Build intersection as a primitive.

11. **Edge tracking with an adaptive slope runs away.** A per-step slope update
    snowballed within 8 steps. Track only inside a fixed corridor derived from a
    user stroke or a prior fit.

12. **Check for contamination before fitting.** 311 of 1020 extracted "arc"
    points were chandelier and cornice. Print the coordinate distribution and
    look for gaps and off-trend blobs before any fit. A 75 px gap and a 282-point
    off-curve blob were both invisible in the residual alone.

13. **Turning angle ≠ angular spread about the centroid.** The latter reads
    355–358° for arcs that turn 35–55°. Use tangent-direction change.

14. **Interior angle vs turning angle.** Smooth = 0° turning = 180° interior.
    These were mislabelled once; state which one you mean in the function name.

15. **Verify by rendering, always.** Rendering caught a conic RANSAC skimming
    curtain streaks, a column-tracked arch visibly missing the paint, and a
    ballooning ellipse from contaminated points — none of which the residuals
    flagged. Numbers agreed while the picture was wrong more than once.

16. **Image proportion is not proportion.** Pixel heights gave 4.63 heads for a
    7.5-head figure turned 72 deg with the camera 28 deg down. Proportion along a
    marked axis must be read as a cross-ratio against that axis's vanishing
    point — `figure.metric_coords` is the primitive, and every proportion
    question is a ratio of differences of its output. Permanent assertion worth
    keeping: a figure parallel to the picture plane must read its true ratio from
    the pixels alone. My first synthetic silently failed that and I nearly
    reported conditioning numbers from a scene where the figure pointed at the
    camera instead of standing up in it.

17. **Perpendicular scatter about a fitted midline is the wrong symmetry
    measure.** A displaced landmark moves its midpoint mostly *along* the
    midline: 5.48 px of shift showed up as 0.98 px of perpendicular residual, and
    adding pairs made the detector *less* sensitive — the tell that the measure,
    not the data, was wrong. Measure where the midline crosses each pair as a
    fraction of that pair's width; symmetric is exactly 0.5, and the answer comes
    out in body units a person can act on.

18. **Leave-one-out does not localise when the fit still contains the bad
    point.** With four landmark pairs the leakage beat the real signal at every
    magnitude tested (0.071 of width at the pushed shoulder against 0.105
    somewhere else). Identify the odd one first — the one whose removal leaves
    the rest collinear — then measure everything against that clean fit. In
    `figure.deviation` (nine points, one plane) LOO does pick the right point,
    but leakage still reached 58 % of the true error, so never read the
    second-worst point as a real finding.

19. **A contact point is not the nearest visible edge of the thing it rests on.**
    Marking where a seated figure meets a bench, I put the mark on the bench's
    front top edge, because that is the crisp line the eye goes to. The artist
    caught it: his body meets the seat 14 px higher up, and the two are at
    different depths because the seat recedes. The same trap waits on every
    ground contact, every object on a table, every foot on a step. **The mark
    belongs on the figure's own silhouette, where it terminates against the
    support** — the UI must say so, and the overlay should draw the support's
    edge too, so the gap is visible rather than assumed away.

    The attempt to recover from it is worth recording as well. The seat contact
    is the better proxy for a seated figure's *head* depth (the head sits over
    the buttocks, not over the feet), so I tried to pin the horizon from the
    bench's two parallel top edges, which must meet on it. Three extractions
    gave y = 933, 958 and 996 under a trim test, all of which put the viewer's
    eye about 40 cm off the floor in a painting that plainly is not. Causes, all
    already in this list: the edges span 86-96 px and meet 370 px away (bug 9),
    one "edge" turned out to be a cast shadow and so is not parallel to the
    bench at all (bug 12), and the bootstrap reported +-4 px while the trim test
    moved 38 (bug 5). A cast shadow's edge is not parallel to the edge that
    casts it — worth its own line in any auto-extraction.

20. **A tolerance in pixels is not a tolerance.** `MARK_PX = 9.0` — the hand's
    marking accuracy — is compared against distances in *image* coordinates, in
    `core/consensus.py` (the "already consistent" verdict) and in the support-edge
    gap check in `static/app.js`. Nine image pixels means a different thing in
    every picture. Read as the fraction of 3.jpg's diagonal that it is (0.578 %),
    the same tolerance would be 17 px on an A4 scan at 150 dpi, 35 px at 300 dpi
    and 61 px on a 6000 px scan. All five testdata images are 1280 px on the long
    edge, which is why this was invisible: the suite cannot see it.

    The failure is in the dangerous direction. A mark placed 2.5 px from a support
    edge at 1280 — indistinguishable from it by hand — reads 9.6 px at A4/300 dpi
    and 16.9 px on a big scan, and both **clear** the bug-19 flag. False clears
    begin at an ordinary 300 dpi scan. The opposite case (a small picture viewed
    zoomed in) costs only false alarms, which are safe.

    The two uses need different fixes, because they measure different things. The
    gap check has the view scale to hand, and the hand's precision is in *screen*
    pixels at the zoom the mark was placed at: compare `gap * view.k`, which is
    scale-free. `consensus.py` has no zoom, but `fit` and `fit_level` already take
    `image_size` and ignore it here; scale by the diagonal. Permanent assertion:
    no threshold in this app is a bare pixel count unless the quantity it bounds
    is itself resolution-independent.

---

## 4. Checks to expose

Built and verified (§2): polygons, circles on a known plane, arches (rise/span,
crown offset, springing level, depth), vanishing points and per-line residuals.

Proportion, symmetry, deviation and stages: sections 8 and 9.

Built but **never demonstrated on a real image** — do this early:
- figures against the horizon (`rhythm.figure_ratio`)
- equidistant repeated objects (`rhythm.spacing`)
- tilted roofs (`rhythm.tilted`)

New, from the vault work, worth generalising:
- **apex angle** where two marked limbs meet — the robust readout when global
  shape is unavailable. Local tangents are well determined even when the conic
  is degenerate.
- **arc ∩ line intersection** for occluded corners.

Known broken / open:
- `rhythm.arch_shape` axis assignment flips on near-circular conics (0.571 read
  between neighbours at 0.41–0.44; 1/(4×0.86)=0.58 is the signature). Sidestep
  via `circle_check`, which has no axis to assign, or fix it properly.
- VP families over-fragment: 8 families where ~4 are real, pairs 2–3° apart,
  pulling in chairs and background. `vp.merge` exists and is not wired in.
- Sequential RANSAC splits badly on near-one-point images (testdata/3.jpg).
- Automatic arc extraction on real paintings: failed by column tracking and by
  conic RANSAC. Treat as marking-first.

---

## 5. Worked example — testdata/4.jpg (the chandelier room)

Keep as a regression fixture; these were measured, not assumed.

- horizon **y = 844** (eye level of the seated figures, confirmed by the artist)
- vertical VP ≈ (287, −69267) — essentially no pitch, so the principal point
  must lie on the horizon. (I was using pp_y = 828; it should be 844. Fix.)
- right VP (frontal wall horizontals) ≈ 3486 on the horizon, 57 segments
- left VP (depth) ≈ **−18 to −55**; the long-line fit gives −18
- **f is not settled**: 1024 → 1150 → 1230 as the VPs firmed up. Re-derive it
  once the VPs are final, and treat every absolute 3-D number as provisional
  until it stops moving.
- window arcs: four concentric, centres (295–302, 559–604), turning only 27–55°
  → **shape undetermined for all four**
- vault arch: left limb fitted to **0.52 px**, apex at **(412, 141) ± 6**,
  apex angle **90° ± 15°** (a groin, not a smooth crown)
- table is a polygon, **32.49°** off the left wall

**Withdrawn and must not be reused:** axis ratios 1.149 / 1.152 / 1.219 / 1.146
and 2.154. All rest on bugs 1 and 3.

**Open question for the artist:** the two marked vault limbs end 188 px either
side of the axis (which implies a frontal arch) while the apex sits 92 px off
that axis (which implies an oblique one). These cannot both hold. Ask whether
those limb ends are real springing points or just where the stroke stopped,
then run the full arch check.

---

## 6. App shape

Follow the house pattern of the other apps in `~/Documents/Apps`:

- FastAPI backend + vanilla JS front end; `app.js` is **one IIFE** — a stray
  apostrophe kills the whole client. Run `node --check` after every JS edit,
  then load the page; a `const` read before its line parses fine and still dies.
- Shared `labkit` vendored in. **Edit labkit only at `Apps/labkit/`**, never in
  an app copy — `python -m labkit.sync` overwrites it.
- Runs on `~/Documents/Apps/.venv-apps` (py 3.13, numpy 2.5). Note numpy 2
  removed `ndarray.ptp()` (use `np.ptp(arr)`) and 2-D `np.cross`.
- Double-click launcher that stops any running copy first.
- `core/figure.py` carries proportion, symmetry, deviation and stage transfer;
  `python core/figure.py` re-runs its own self-check and must stay silent-pass.
- American spelling in user-facing text. Do not rename identifiers, API routes
  or saved-data keys for spelling alone.

Workflow: load image → detect segments and VP families → **user accepts, rejects
and merges them** → user labels surfaces (wall / ground / table, with which VPs
span each) → user marks features (arc strokes, polygon corners, arch limbs,
figures) → run checks → overlay results on the image with every value's
conditioning beside it.

Persist a run manifest next to results so a methods paragraph can be generated
from what actually ran (see the `writing-methods-paragraphs` skill).

---

## 7. Suggested order

1. Port `core/` into the app, add an assert-guarded self-check module that
   re-runs every verification in §2 — these must keep passing.
2. Measure the turning-angle threshold below which arc shape is refused (bug 4).
3. Marking UI: arcs, polygon corners, arch limbs, surface labels. This is the
   backbone; auto-detection plugs in later.
4. Wire the checks with conditioning and `undetermined` verdicts.
5. Demonstrate the three never-tested checks on testdata/1.jpg and 2.jpg.
6. Wire `vp.merge` and fix the family over-fragmentation.
7. Revisit testdata/3.jpg (near one-point) and the 4.jpg table polygon.
8. Proportion and symmetry marking (section 8) — the axis plus pair tool. The
   geometry is done and verified; what is missing is the marking UI.
9. Stages and corrections (section 9). Do the correction overlay before the
   multi-stage comparison: it is useful on its own, and it is what makes a
   second stage worth photographing.

---

## 8. Proportion and symmetry — `core/figure.py`

Built and verified (section 2 rows, `python core/figure.py` re-runs them).

**Image proportion is not proportion**, so none of this can be read off the
pixels. A figure turned away images its far half shorter: a 7.5-head figure read
4.63 heads tall from its pixels at 72 deg of turn. What survives projection is
the cross-ratio, and the fourth point it needs is the vanishing point of the
figure's own axis — which this app already computes.

**`metric_coords(pts, v)` is the primitive.** Given points marked along one axis
and that axis's vanishing point, it returns their real spacing, up to an overall
scale and offset. Every proportion question is then a ratio of differences:
heads tall, where the navel falls, whether two limbs match, whether a repeated
element keeps its size. One function, verified exact; do not add a second way to
ask.

What to expose:
- **axis tool** — the user drags the figure's axis (feet to crown) and drops
  ticks on it (chin, shoulder, navel, crotch, knee). Which vanishing point the
  axis runs to is an explicit label, exactly as surfaces are (bug 3): a standing
  figure uses the vertical one, a reclining one does not.
- **canon comparison** — the ticks against a chosen canon (8-head, 7.5-head,
  whatever the user keeps). Report the difference per landmark in head lengths,
  because that is the unit the correction gets made in.
- **contact marks** — ground contacts, seats, anything resting on anything.
  Mark the figure's silhouette where it ends against the support, never the
  support's own edge (bug 19), and draw both so the difference is visible.
- **pair tool** — left/right landmark pairs. `midline` finds the projective
  midpoints, identifies the odd pair, and reports each one as a fraction of its
  own width. Symmetric is 0.5. Halfway along the line is not the midpoint and
  was 1.27 px wrong on the test figure.
- **limb comparison** — two marked axes, each with its own vanishing point,
  compared through their metric coordinates. This is how a left forearm is
  checked against a right one when they point in different directions.

Conditioning, all already returned:
- `naive_image_ratio` and `correction` — how much work the perspective
  correction is doing. When they agree the figure is near-parallel to the picture
  plane and the vanishing point barely matters; when they diverge, it matters a
  lot and must be trustworthy.
- `vp_spread` — pass the vanishing point's own uncertainty in as `v_alt` and this
  is the resulting spread in the answer. A vanishing point known to +-10 % moved
  heads-tall by 0.11. **This is the readout that decides whether a proportion
  claim can be made at all**, and it is why bug 9 matters here too: a vanishing
  point from short segments will quietly ruin a proportion.
- `off_axis_frac` — refuses when the marked points are not actually on one
  straight axis (a bent pose is not an axis).
- `floor` for symmetry — 0.03 of width by default, measured from +-1.5 px
  clicking. Below it the verdict is `"symmetric"`, not a small number.

Known limit: `midline` finds **one** odd pair. Two simultaneous asymmetries rank
the wrong pairs first (6 cm at the shoulders and 6 cm at the knees ranked hips
first). It returns `"undetermined"` when no pair stands out, which is that case
showing up honestly, but it cannot separate two.

---

## 9. Stages and corrections

A session is **one artwork and an ordered list of stages**, each stage a
photograph of it plus the marks and results for that photograph. This is what
turns a check into something that improves a drawing instead of just grading it.

**Marks carry forward; they are not re-made.** A drawing on paper is flat, so the
map between two photographs of it is a homography — exact to 1e-7 px on
synthetics. The user clicks registration points on the new stage (corners,
tape marks, anything stable), `homography` + `transfer` bring every mark over,
and the user adjusts the few that moved because the *drawing* changed. Register
on **8 or more points**: at 4 the fit is exact, click noise passes straight
through, and half-pixel clicks cost 8 px (table in section 2).

**Corrections out.** Every failing check knows what right looks like, because it
fitted it:

| check | the correction it can emit |
|---|---|
| line to a vanishing point | the line from the same anchor that actually meets the VP, plus the angle error |
| proportion along an axis | ticks at the canon's positions, in metric coordinates |
| symmetry | the pair endpoint moved so the midline crosses at 0.5 |
| arch | the arc whose rise/span is the intended one; the apex at the projective midpoint of the span |
| polygon | the vertices of the best-fitting regular n-gon |
| circle on a labelled plane | the ellipse a true circle would project to |

Two rules for these, both learned the hard way:

1. **A correction needs an explicit anchor, and the UI must show it.** Fixing a
   proportion by moving the head and fixing it by moving the feet are different
   drawings. The arithmetic does not choose; the artist does.
2. **No correction may be emitted from an `undetermined` readout.** A corrected
   line drawn from a vanishing point that is not pinned is a confident wrong
   answer rendered at full opacity — the most expensive failure mode this project
   has had (bugs 3, 4, 9). If the check refuses, the overlay refuses.

Render corrections as a layer over the stage at the artwork's own resolution so
they can be printed or traced, and keep the measured and corrected geometry in
separate colors with a legend. **Every overlay gets looked at before it is
believed** (bug 15): rendering caught three failures that residuals did not.

**Did the correction land?** For each check, the session shows the value per
stage with its conditioning, and the change between stages. Report a change only
when it exceeds the combined uncertainty of the two stages — otherwise
`"no measurable change"`, which is the honest reading of a correction that did
not take, and must not be rendered as a small improvement. Where a check has no
uncertainty estimate yet, the stage comparison inherits the refusal: no interval,
no verdict.

This closes the loop the app exists for: measure, emit the correction, the artist
works on the piece, a new stage comes in, and the app says whether the thing that
was wrong is now right.

---

## 10. Consensus — a drawing is not a diagram

Added 2026-10-07, from the user: *"it's hand drawn so it won't be exact. The
idea behind this app is to find some sort of consensus between what was drawn
so that the drawing could be rectified with least possible changes to the
composition."*

This changes inconsistency from a refusal into a measurement. The question is
not "do these lines agree?" — they never will — but **which perspective is
nearest to what was actually drawn, and how far would each line have to move
to reach it?** A move smaller than the ~9 px a hand mark is worth is not an
error, and the verdict is `already consistent`, not `rectifiable`. Saying
otherwise sends someone to redraw a line that was right.

`core/consensus.py`. `fit` takes two or three perpendicular families and
returns the camera, the reconciled vanishing points, and every line's travel.
`fit_level` imposes a level eye on two horizontal families and prints the price
of that assumption. `rectify` rotates one line about its own midpoint — the
least change that fixes it, because the midpoint stays where the composition
put it.

### What this cost to get right

**Random seeds cannot find a real camera.** Seeded from `N(0, 0.4)` rotation
vectors, the two-family fit ran to **f = 1.7e37 and 164 px of movement** on a
drawing that reconciles at 0.00 px. The real camera's rotation vector had norm
2.4 rad. `_seeds` now starts from each family's own drawn vanishing point,
snapped to the nearest rotation at a trial f. Locked by an assertion.

**Line length leaked into the headline number.** `_move_px` measures travel at
the segment's ends, so a construction line dragged past the canvas reports a
move scaled by how far it was dragged: one real line ran 6176 px for 888 px of
picture, and its "3.5 px" was measured 2600 px off-canvas. `_clip` cuts every
line to the picture first. This dropped the forced-level answer on the worked
example from 11.44 px to **1.94 px** — the two long lines had been dominating
the fit — and improved the synthetic's recovered f from 1115 to 1104 against a
true 1100. Locked by an assertion: a line stretched 5x must read the same.

**Two families do not determine the focal length.** With both vanishing points
fixed, `(va - pp)·(vb - pp) + f² = 0` is one equation and the principal point
has two degrees of freedom, so f slides along a one-parameter family. Fitting it
directly returned 611 for a true 1100. `fit` now pins pp for two families and
reports `focal_range_over_pp`; `fit_level` fits `[u_a, u_b, pp_y]` and derives f
from an assumed `pp_x`. **Quote the reconciliation and the moves from a
two-family fit, never the absolute 3-D numbers.**

**The focal length is a separate question and may fail on its own.** When both
vanishing points land the same side of the principal point, `f² < 0` and no
level camera exists — but the reconciled vanishing points, the horizon and every
line's movement are still good. Refusing the whole result because a dependent
number failed is the same mistake in the other direction.

**A bootstrap over two lines is meaningless.** Half the resamples draw the same
line twice; the page showed "x within -0 to 3686". `/api/fit_vp` now refuses it
below three lines and says why.

### The worked example, answered

`testdata/4.jpg`, the user's own six marked lines (four left-wall, two
right-wall, no verticals):

| | as drawn (camera free to roll) | forced level |
|---|---|---|
| verdict | already consistent | already consistent |
| worst line moves | **0.00 px** | **1.94 px** |
| median | 0.00 px | 0.16 px |
| f | 1250 px | 1239 px |

**The drawing needs no change.** The drawn horizon tilts +0.428°, but removing
that tilt costs under 2 px — it is a camera roll, not a drawing error.

On the focal length, which §5 left unsettled at 1024 → 1150 → 1230: the
convergent value is **~1240**, but it is *not measured*. Over the middle 30% of
the frame the principal point moves it anywhere in **1035–1394**. Two or three
vertical lines would pin it; two horizontal families alone cannot.

Marked by hand in the app at 45% zoom instead of in Illustrator, the same six
lines give f 1260 and worst move 1.28 px — **0.8% from the precise marks**,
which is the measurement that justifies the semi-automatic design.

---

## 11. The marking tool — built 2026-10-07

`static/index.html`, `static/app.js` (one IIFE), `static/app.css`. Port 8830.

A layer's kind *is* the tool, so there is no separate palette: pick what the
layer holds and click on the picture. Line families drag; arcs, arches,
polygons, axes and spaced rows take a run of points, **by dragging to trace or
clicking point by point** — clicking a vault limb is 139 clicks, so a stroke
samples every 6 screen px. Placed points and line ends drag to adjust;
`⌘`-click deletes one. `Enter` or double-click ends a clicked run.

Things found by looking at the rendered page rather than the code:

- labkit's `--sel` is a **text** colour (`#f2f5f8`), not a background. Using it
  for the selected layer row painted the row near-white and hid the name.
- A readout that arrives from the server after `showEditor()` has drawn the
  panel never appears until the layer is re-selected.
- Drawing a correction for every line, including those already inside marking
  accuracy, covers the original in red and implies work that is not there. Only
  lines over the floor are drawn, and the card says so in words.
- The browser pane suppresses `confirm()`, and a screenshot taken straight after
  a click shows the previous frame. Two apparent bugs were neither.

### Bug 19 in the marking tool

The `figures` and `posts` layers are contact layers: their marks *are* ground
contacts, so bug 19 is their default failure.

- Both blurbs now state the rule where it is read while marking: the mark goes
  on the **figure's own silhouette** where it meets the ground, never on the
  ground's own edge, which is at a different depth.
- A contact layer can hold **one support edge** — toggle to `marking the
  SUPPORT EDGE` and drag it out. It is drawn dashed and dimmed, as reference
  rather than as a mark.
- Every contact's perpendicular distance to that edge is reported live, and any
  contact within a hand mark of it is flagged `check these`: at that distance
  the mark cannot be told from the edge. Verified on `testdata/3.jpg` — a
  contact placed on the edge reads 0.82 px and is flagged; dragged 14 px up the
  silhouette, the distance bug 19 actually measured, it reads 14.83 px and
  clears.

The gap card lives in its own `#supportHost` and is re-rendered whenever the
marks change. The first version was built once, when the layer was selected,
and so showed a stale gap — the same mistake as the vanishing-point readout
above, and the reason the editor panel now refreshes the parts that depend on
marks without touching the selects that do not.

### What a hand mark is worth is a fraction, not a pixel count

Found by the session that owns §3, whose note there has the arithmetic. The
`9 px` of marking accuracy was measured on **one** scan, and both users of it
treated it as an absolute pixel count. All five testdata images are 1280 on
the long edge, so nothing here could ever have caught it: the same hand error
that reads 9 px at 1280 reads 35 px on an A4 at 300 dpi, where a flat 9 px
calls the drawing exact. It failed in the dangerous direction — bigger scan,
more forgiving verdict — and in the marking tool it was bug 19 walking past
the check built to catch it.

It does two different jobs and needed two different fixes:

- **`core/consensus.py`** compares it against travel in image pixels, so it
  scales with the picture: `_mark_px()` multiplies by the image diagonal over
  the `884x1280` scan it was measured on. Both fits already took `image_size`
  and ignored it for this. The scaled figure goes out as `mark_accuracy_px`,
  so the page needs no change, and `mark_accuracy_basis` says in words what it
  rests on — an unexplained `36 px` reads as arbitrary. The assumption is that
  marking was done at a zoom that fits the picture, which is what the tool
  does by default; marking at 400% beats it, which only makes the verdict
  cautious, never generous.
- **`static/app.js`** compares it against a gap on screen, where the hand's
  precision actually lives, so it does not scale with the picture at all —
  but it *does* depend on the zoom the mark was made at. Comparing against the
  **live** `S.view.k` would have been a second bug of the same shape: zoom in
  to inspect a flagged contact and the flag clears, the verdict changing with
  nothing but where the viewport happens to be. Each contact's placement zoom
  is stamped into a UI-only `L.markK`, parallel to the contacts and spliced,
  re-dated and saved alongside them, so a reloaded marks file reproduces the
  verdict it had. The wire format the server sees is untouched.

Both are locked. `_demo` scales the whole synthetic by k and asserts the
verdict, the moves and the threshold all scale with it; reintroducing the bug
fails it at k = 2 with the exact flip — 11.83 px against a flat 9.0, calling a
drawing `rectifiable` for nothing but being scanned at higher dpi. In the page,
three contacts placed at 41% read `2 of 3` flagged, and still read `2 of 3`
after zooming to 281%; under the live-zoom version the flagged contact would
have read 20.5 px and cleared.

The general rule, for every threshold still to be added: **a tolerance measured
on one image is a fraction of that image.** Store what it was measured on.

## 12. Undo, and automatic edge detection — 2026-10-07

### Undo / redo (cmd-Z, cmd-shift-Z, ctrl-Y)

Snapshots, not inverse operations. The marks are a few hundred points, so a
JSON copy of `{layers, nextId, pending, sel, cand}` costs less than a command
stack's bookkeeping and cannot drift out of step with what it is undoing. The
top of the stack is always the state on screen; `commit()` collapses identical
states, so it is safe to call from any path that *might* have changed
something, and a click that did nothing leaves no dead undo step.

- One entry per interaction, not per event. A drag of 20 mousemoves produced
  exactly one entry (measured by counting undos to the floor: 5 from
  `[empty, +layer, +3 lines, +drag]`), and a traced stroke undoes whole.
- Geometry is restored exactly: a line drawn to `(652, 230)`, dragged to
  `(764, 342)`, read back `(652, 230)` after one cmd-Z.
- A vanishing point can arrive from the server *after* the commit that should
  have carried it, which would leave a restored state showing the fit of a
  different set of lines. `restore()` re-fits every family rather than trying
  to snapshot something asynchronous; the stale reply lands on a layer object
  that no longer exists, so it cannot corrupt the restored marks.
- The offered segments are in the snapshot too: taking a family and undoing it
  puts those segments **back on offer**, instead of losing them from the layer
  and the pool at once.
- Loading a marks file starts a fresh history — opening a document is not an
  edit, and cmd-Z should not walk back into the previous picture's marks.
- `Clear marks` now clears `markK` as well, or the next contacts inherit the
  placement zooms of the marks that were cleared.

### Detection, finally wired to the page

`/api/detect` has existed since the first week — LSD segments, sequential
RANSAC, merged families, all tested — and **nothing in the UI ever called it**.
It is now a button, and it stays an accelerator: nothing enters a layer until
the user clicks it.

- Candidates are drawn thin and faint, coloured by the family they fell into;
  ungrouped segments stay grey and stay clickable, because a line RANSAC did
  not group is often exactly the one that is wanted. Dashing them instead was
  tried and reverted — a 58 px segment at 45 % zoom is 26 screen px, and a dash
  pattern erases it.
- Click one to take it; `take N lines` takes a whole family. Both are one undo
  step. On `testdata/4.jpg`: 141 segments, 2 families, 505 ms.
- `min length` is the one knob that matters, and the short-segment warning
  names it. Sweeping it on 4.jpg: 40 → 141 segs (vertical family median 58 px,
  **warned**); 55 → 30 lines, median 82, no warning; 70 → 19 lines, median 88;
  90 → 13 segs, no family at all.

### Two conditioning bugs this uncovered

**The bug-9 guard was anisotropic and absolute.** `median < 60 and not (0 <=
vp_x <= 4000)` tested *x only*, so 4.jpg's vertical family — 54 segments of
median 58 px aimed at a vanishing point **59 397 px above** the picture — read
as nearby and went out unflagged. Both numbers were also pixel counts read off
one 1556 px diagonal scan, the same disease as §11. Now `V.short_for_vp()`:
distance from the image centre over `3 x diag`, median under `0.039 x diag`,
with eight checks in `selfcheck.py` including scale invariance at 0.5x, 2x, 4x.

**A free principal point collapses on a sliver triangle.** Three vanishing
points locate pp only if their triangle is well shaped. A near-vertical family
puts one of them tens of thousands of pixels away, the triangle degenerates,
and the solver buys a *lower* residual by sliding pp onto another vanishing
point — where that family's constraint goes slack and f collapses with it. On
4.jpg with a detected vertical family, `fit()` returned **f = 7 px with pp four
image widths outside the picture, at 1.08 px of movement against the honest
answer's 2.34**. The better-looking residual is the whole danger. `fit()` now
refuses an implausible pp, pins it at the image centre and prints why, as
loudly as the number it replaces. Removing the gate fails `_demo` with
`pp = (3421, 808), f = 17`.

This mattered immediately, because `/api/consensus` calls `fit()` with
`free_pp=True` by default and detection had just made a third family one click
away.

### The focal length, measured at last

§5 left f unsettled (1024 → 1150 → 1230) and §10 could only bracket it:
**1035–1394**, because the user's six hand-marked lines are two horizontal
families and two families cannot determine f. A detected vertical family
supplies the third:

| | lines | median | f |
|---|---|---|---|
| hand marks alone | 6 | — | 1250, anywhere in 1035–1394 |
| + detected vertical, min_len 55 | 30 | 82 px | **1238** |
| + detected vertical, min_len 70 | 19 | 88 px | **1242** |

Stable to 0.3 % across two detections whose vertical vanishing points differ by
20 000 px (−65 105 vs −44 279) — at that distance only the direction matters.
The verdict stays `already consistent` (worst move 2.34 px, under the 9 px a
hand mark is worth), so the 19–30 machine-found lines **agree with the six
hand-marked ones**. f is now a measurement, on the stated assumption that the
principal point is the image centre, which the readout says in words.

---

## 13. Trimming the picture to the artwork — 2026-10-08

> *"make it trim the image edges first (automatic/manual). as some images are
> photographs, they are not necessarily flat scans"*

### 13.1 Two problems, not one

A photograph of a drawing is not the drawing.

**The cheap problem is pollution.** The line detector will happily fit a
vanishing-point family to the edge of the desk. On `testdata/1.jpg` it did
worse than that: the *second-largest* family in the picture, 25 segments with
a median length of 110 px — longer than any family inside the artwork — was
the sheet's own paper edge. All 25 of 25 segments lay within 13 px of the
sheet outline (7 of them outside the sheet entirely). That family was being
offered to the user as a vanishing point.

**The expensive problem is the keystone.** Photographing a flat sheet off-axis
applies a homography to it, and that homography composes with the perspective
the artist drew. Lines stay lines and families stay concurrent, so nothing
looks broken — but every vanishing point has moved and the horizon has tilted.
Asking "is this horizon level" of such an image answers a question about where
the photographer was standing.

Both are fixed by mapping the sheet's quadrilateral back to the rectangle it
must have been. The target rectangle is not assumed: it is recovered from the
quadrilateral, along with the focal length of the camera that took the
photograph (`core/trim.py:shape_of`, the standard single-view rectangle
construction, assuming square pixels and the principal point at the image
centre).

### 13.2 The side test — finding a sheet, versus inventing one

Brightness finds the sheet: paper on a desk is lighter than most of what
surrounds it. Brightness alone also finds a sheet in a painting that has no
sheet in it, because an already-cropped artwork always contains some bright
patch of paint whose outline approximates to four corners. On `testdata/4.jpg`
that was a quadrilateral spanning the whole painting, returned with exactly as
much confidence as the one correct answer in the set.

What separates them is whether the sides are **real edges**. For each side,
the median brightness step across it, divided by the median step the picture's
own texture produces over the same distance:

| picture | what it is | per-side ratios | verdict |
|---|---|---|---|
| `1.jpg` | drawing taped to a board | 7.2, 16.5, 20.9, 13.2 | **sheet found** |
| `2.jpg` | sheet on a beige desk | one side 12.4, three off-frame | refused → manual |
| `3.jpg` | artwork, already cropped | no candidate at all | refused |
| `4.jpg` | artwork, already cropped | 1.2, 1.4, 1.1 | refused |
| `5.jpg` | artwork, already cropped | 1.5 | refused |

Nothing landed between 1.5 and 7.2. The gate is 3.0 and **all four sides must
clear it**, so only `1.jpg` is auto-accepted. Three detectors were measured
(Otsu, Canny-with-flood-fill, and a "bright *and* grey" saturation score);
Canny is useless here because the drawing's own pencil lines flood it, the
saturation score ties with Otsu, and **none of the three finds the sheet in
`2.jpg`** — the beige desk is nearly as bright as the paper, so the bright
region runs to the frame. Otsu was kept for being the simplest of the three.

So auto-detection succeeds on 1 of the 2 photographs and correctly refuses on
all 3 scans. That is the shape the design rule predicted: **the manual corners
are the backbone and detection is the accelerator.** The refusal falls straight
into corner-dragging with a suggested inset, and corners may be dragged outside
the picture for a sheet that runs off the frame.

### 13.3 One parallel edge pair is already fatal

The camera is recovered from the two vanishing points of the rectangle's own
edge pairs, which are orthogonal directions:
`(v₁ − pp)·(v₂ − pp) + f² = 0`.

The first version refused only when **both** pairs came out parallel (a
square-on shot, where there is no keystone and the trim is a plain crop). That
is wrong. Send *one* vanishing point to infinity and the constraint degenerates
into a statement about where the **other** one lies — f drops out of it
entirely. On a hand-placed quadrilateral for `2.jpg` (one pair parallel to 4
parts in 10 000, the other converging) it returned **f = 41 716 px for a
picture 1600 px corner to corner**: 26 diagonals, read out of rounding noise in
the corner positions, with no sign that anything was wrong.

Three gates now, all three falling back to the quadrilateral's own mean side
lengths and saying so:

- both pairs parallel → shot square-on, nothing to undo, plain crop;
- either pair parallel → one constraint, nothing fixed about the camera;
- `f > 8 × diagonal` → the projection is all but orthographic and f is noise.
  (The real pictures sit at 0.6–1.7 diagonals; `1.jpg` gives 1.72.)

This is the fifth time in this project that an ill-posed fit returned a
confident number, and the sixth conditioning refusal in the app.

### 13.4 The trim is a parameter, never a file

`_load(path, trim)` in `app.py` applies it once for every route that touches
pixels — `/api/image`, `/api/imageinfo`, `/api/detect`, `/api/overlay`. Nothing
is written to disk, the user's original picture is never modified, and there is
no derived file to fall out of step with the marks. `core/vp.py:lines` now
takes an array as well as a path, so detection runs on what the user is
actually looking at.

Marks are stored against the picture they were placed on, so changing the trim
afterwards would move every one of them off the paint. Rather than forbid that
or silently corrupt the document, `/api/retrim` carries them through: out of
the old trim, back into the original photograph, and into the new one.

**Verified in the page**, not just on the server:

- four marks clicked on the sheet's corners in the untrimmed photograph landed
  on the corners of the trimmed rectangle to within **3.2 px** — and the clicks
  themselves were already ~2 px off target, because at 41 % zoom one screen
  pixel is 2.4 image pixels. The transform contributes nothing measurable.
- a document carrying all four geometry fields (2 `lines`, 3 `pts`, 3 `far`,
  2 `pairs` = 14 coordinates) had **every** coordinate move, 163–182 px, with
  none left behind; clearing the trim returned all 14 to better than 1e-6 px.
- `cmd-Z` across a trim restores the pixels *and* the marks, because the
  picture on screen is a different picture, not a different view of one.
  Redo restores both exactly.

### 13.5 What it bought

With `1.jpg` trimmed, detection at `min_len 55` returns **two** families — the
ceiling run at (−3141, 1054) and the verticals at (948, −19 211) — both
entirely inside the artwork. The sheet-border family is gone.

### 13.6 Still open

- `2.jpg` has no automatic answer. Three detectors failed on it; a fourth idea
  (strongest four near-border lines from LSD, which the app already has) was
  not tried, and is the obvious next thing if auto-trim on desk photographs
  turns out to matter.
- The found quad traces the *sheet*, which on `1.jpg` includes a white paper
  margin outside the drawing. Trimming to the drawing proper is a corner drag
  away, but nothing offers it.

---

## 14. One-, two- and three-point perspective — 2026-10-09

**The request:** "we need to be able to choose 1, 2 and 3 point perspective to
do the check."

### 14.1 It is one model, not three

The perspective type is a constraint on the camera's **rotation**, nothing
else. Writing it as `R = Rz(roll) @ Rx(pitch) @ Ry(yaw)` makes both
constraints "set an angle to zero":

| type | free angles | what it forces |
|---|---|---|
| three-point | yaw, pitch, roll | nothing |
| two-point | yaw, roll | pitch = 0 → uprights parallel on the page |
| one-point | roll | pitch = yaw = 0 → uprights *and* one horizontal parallel |

So `fit(..., mode=...)` is one solver with one, two or three free angles, and
three-point reaches the old six-parameter Rodrigues path verbatim. The
three-point self-check is unchanged to the digit: f 1104 vs 1100 true, worst
move 5.9 px, price of consistency 0.8 px.

Declaring the type is **the check, not a hint**. The moves reported become
"how far is this from the perspective I meant to draw", which is a different
and more useful question than "how far from any perspective at all".

### 14.2 One-point has no focal length. At all.

In one-point perspective R is a pure roll about the optical axis, so the depth
axis maps onto the optical axis and its vanishing point **is** the principal
point. f therefore appears in no vanishing point. This is not
ill-conditioning; f is absent from the model, and no number of marked lines
recovers it. `fit` returns `focal_px: None` with `focal_refused` saying so,
and `_camera` in `app.py` refuses outright for a one-point document. What a
one-point drawing *does* give is the centre of vision, exactly — recovered to
0.0 px on synthetic data.

The way out, if f is wanted, is the distance-point construction: mark the
diagonals of floor or ceiling squares; their vanishing point sits f from the
centre of vision along the horizon. The refusal text says this.

### 14.3 Two-point cannot locate pp along the horizon

With pitch = 0 both finite vanishing points lie on the horizon through pp, at
signed distances whose product is −f². Slide pp along that line and f changes
to match, fitting the marks **exactly** the whole way. Measured on a synthetic
camera (true pp = (560, 470), f = 1150):

| pp slid along the horizon | pp | implied f |
|---|---|---|
| −400 px | (161, 442) | 1253.8 |
| −200 px | (360, 456) | 1219.6 |
| 0 | (560, 470) | 1150.0 |
| +200 px | (760, 484) | 1038.1 |
| +400 px | (959, 498) | 867.7 |

Every one of those is an exact fit. Left free, the solver duly returned
pp = (234, 447) and f = 1245 at 0.00 px — a confident wrong camera.

So two-point gets a **lateral pin**: pp_x is assumed (image centre, or
whatever is passed), pp_y is *measured* — it is the horizon's height, which
the two vanishing points do fix. `_take_pp` carries a half-pinned pp for
this. Afterwards pp_y came back 472.8 against a true 470.

The assumption's worth is printed beside f as `focal_range_over_pp`, obtained
by sliding pp_x across the middle 30 % of the frame. On the synthetic that
range is 1024–1203 and contains the true 1150, which is a self-check row.
`camera_level` in `core/checks.py` already knew all of this — it takes `pp_x`
as an input — and `_camera` now routes declared two-point documents to it
instead of using it only as a fallback.

### 14.4 The nested-model trap, and the gate that catches it

A *more general* claim fits data from a *more specific* one perfectly, because
the specific model is nested in the general one. A true one-point drawing
reconciles at 0.00 px as two-point and as three-point — correct, and the f
printed alongside was **1781** and **3402** px, pure solver noise.

The honest gate is local geometry, no re-fitting: f enters only through
`−f² = (a − pp)·(b − pp)`, so it needs **two** finite vanishing points that
are not at pp. A VP at the principal point calibrates nothing.

Finding that gate exposed a second bug behind it. `at_infinity` was testing
the homogeneous `|w| > 1e-12`, which does not scale: a vanishing point
2 × 10¹¹ px away passed as "finite" and calibrated the lens, which is how
`f = 1202604 px` got printed for a 1556 px picture. A vanishing point is at
infinity when its *distance* says so — now `_vp_xy(v, diag)`, with `FAR = 50 ×
diagonal`, used by the per-family report and the f gate alike. `FMAX = 8 ×
diagonal` caps f itself, matching the ceiling `core/trim.py` already puts on a
focal length read off a quadrilateral.

### 14.5 The axis a mode singles out need not be marked

`_assignments` originally forced one family onto the upright axis for
two-point. But two-point perspective is drawn by marking the two **horizontal**
families — the uprights are parallel and have no vanishing point to mark. On
`testdata/4.jpg`, where both marked families are horizontal, that forcing gave
**322 px** of moves. Enumerating every labelling instead: **126 px**.

Then the second pin came off. Two families used to trigger a full pp pin,
which forces the horizon through the image centre; two-point needs only the
lateral pin (4 parameters against 4 observations). `testdata/4.jpg` as
two-point: **0.00 px**, pp = (442, 834) — the horizon measured at y = 834, not
assumed at 640 — and f = 1266 against the three-point fit's 1250.

### 14.6 Speed: 34 s → 0.9 s

The first version swept 5 f × 8 yaws × 2 rolls = 80 starts per axis
assignment, up to 240 `least_squares` solves: **18–34 s**.

Measured per assignment on three synthetic drawings: the *correct* assignment
solved in **1 ms** from a closed-form seed, while *wrong* assignments burned
2.7–5.6 s grinding `max_nfev = 4000`. So:

- **Seed two-point in closed form.** Roll from the upright marks; intersect the
  horizon through the two families' own vanishing points with the assumed
  pp_x to get pp_y; then f = √(−d₀d₂) and yaw = atan2(d₂, f). The sweep never
  beat it.
- **The closed form doubles as the axis test.** It only solves when the family
  called upright really is upright, so it fired on the correct assignment and
  on no other. Sweep only where it says nothing.
- **Cap `max_nfev` at 300** for the constrained modes. Three-point keeps 4000.

Worst case now **0.9 s**; one-point 30 ms, three-point unchanged.

### 14.7 Where it shows

A **Perspective** panel above Marks, with the three types and a line saying
what each one measures and what it refuses. Changing it re-runs the check and
drops the previous answer — including on undo, which is why the clearing lives
in `setPerspective` rather than in the selector's handler: undo left a card
reading "checked as one-point" beside a selector saying 2-point. The type
travels in the document, the saved marks file and the undo snapshot.

### 14.8 Measured, all three types, synthetic cameras of known type

Drawn from W,H = 1200×900, pp = (560, 470) deliberately off-centre, f = 1150.

| drawn as | claimed one | claimed two | claimed three |
|---|---|---|---|
| **one-point** | **0.00 px**, pp (560, 470), f refused | 0.00 px, f refused | 0.00 px, f refused |
| **two-point** | 11.98 px | **0.00 px**, pp_y 473, f 1131 (range 1024–1203) | 0.00 px, f 1230 |
| **three-point** | 20.32 px | 14.15 px | **0.00 px**, pp (560, 470), f 1150 |

The correct claim always reconciles at nothing; a tighter claim costs real
pixels; a looser one refuses the f it cannot have measured. Sixteen rows of
this are in `selfcheck.py:perspective_modes`. Whole suite: **106 passed, 0
failed**.

### 14.9 Still open

- A two-point drawing claimed as three-point gives f = 1230 against a true
  1150 and pp (317, 453) against (560, 470) — the extra pitch trades against
  pp. It reconciles at 0.00 px, so nothing flags it. The honest reading is
  that the user should claim two-point; nothing says so yet.
- Nothing detects the type. Declaring it is the point, but offering "this
  looks like two-point" after a check would be cheap now that all three run in
  under a second.

---

## 15. Pixels into millimetres — 2026-10-09

**The request:** "let's also make it possible to convert px deviations into mm
by providing an option to give the size in si or imperial units."

### 15.1 One declaration, every number

A **Real size** panel takes a number, a unit (mm, cm, m, in, ft) and which
edge it measures. From that and the picture's live pixel size comes one
`px_per_mm`, and the server puts a millimetre twin beside **every** `*_px`
number in the response — `_add_mm` walks the whole JSON rather than the
handful of fields anyone remembered, so the twenty-six `*_px` keys already in
the app are covered, and so is the next one written.

`px_per_mm` already existed, but only as a hand-typed field on the posts
feature, where two features could disagree about one picture. The posts check
now inherits the document's declaration and keeps the per-feature field only
as an override.

### 15.2 It is resolved against the picture *as it is now*

The declaration is stored as (value, unit, edge) and turned into px/mm on
every request against the live width, never cached as a number. So trimming a
photograph to the sheet keeps the same declaration meaning the same edge.
Measured on `testdata/1.jpg`, declared 297 mm wide:

| | pixels | px per mm |
|---|---|---|
| as photographed | 960 × 1280 | 3.232 |
| trimmed to the sheet | 824 × 1090 | 2.774 |

The panel also states the size of the edge that was *not* declared — "the
picture is 393 mm tall". That is the cheapest guard against declaring the
short edge by mistake, which otherwise scales every millimetre in the app
without a symptom.

### 15.3 Two numbers that must not be converted

- **`focal_px` is not a lens.** In the picture's own units f is the station
  point: how far the eye belongs from the paper. It is reported as
  `viewing_distance_mm`, never `focal_mm`, because "focal length in mm" reads
  as the lens of whatever photographed the sheet. On `testdata/4.jpg` at
  297 mm wide: f 1250 px → **the eye belongs 420 mm from the picture**, with
  the two-point range quoted in mm as well (349–470 mm), since a sentence that
  switches to mm and then quotes a span of pixels reads as one number with two
  meanings.
- **`photo_focal_px`** is the photographing camera's f, living in the
  untrimmed photograph's pixels. Converting it to the artwork's millimetres
  would be meaningless, so it stays in pixels and says why.

Counts and booleans are left alone (`n_lines`, `at_infinity`), and `mm_per_px`
— which ends in `_px` by accident of naming — is excluded so that wrapping a
readout twice cannot mint `mm_per_mm`.

### 15.4 Measured

`testdata/4.jpg`, 884 px wide declared 297 mm → 2.976 px/mm:

| readout | px | mm |
|---|---|---|
| worst move, one-point | 30.874 | 10.373 |
| hand-marking accuracy | 9 | 3.02 |
| arch conic residual | 6.157 | 2.069 |

Declared instead as 17 in tall → 2.964 px/mm, and the panel's cross-check
says the picture is 298 mm wide. The same length given in mm, cm, m, inches
and feet agrees to 1e-6. Nineteen rows in `selfcheck.py:real_size`; whole
suite **125 passed, 0 failed**.

### 15.5 A camera pitched down was being refused

Verifying the `/api/check` path turned up a bug of its own, unrelated to
units. `horizon_consistency` compared the horizon's tilt to the roll implied
by the vertical vanishing point without folding either into one range. A
vertical vanishing point *below* the picture — which is simply a camera
pitched down — points the other way along the same line, so on a synthetic
camera of f = 1100 it read −175.55 deg against a +4.25 deg horizon and
reported a **179.8 deg disagreement**, refusing a perfectly good camera with
"no square-pixel camera does both".

A direction is a line, so both are now folded into (−90, 90] and compared
mod 180. The same synthetic now recovers f = 1100.0 exactly. Nothing else
moved: 125 passed, 0 failed.

## 16. A known real size into the viewer's distance — 2026-10-09

**The request:** "can we supply real life size (not on the image) of certain
lines and convert this focal point into the distance between the viewer and
the scene (or rather that particular line). e.g. if I have a 180 cm tall human
figure, it should serve as a landmark."

### 16.1 The geometry

§15 scales the **paper**; this scales the **scene**. Mark the base and the top
of something whose real size is known, say how big it really is, and with a
camera already solved the distance follows — nothing is assumed about where it
stands or which way it faces, only that it runs along a known world direction
(the vertical vanishing point for a standing figure).

The two marked ends are rays `r̂1`, `r̂2` from the camera; the direction is `d`.
Being the same segment,

    t1·r̂1 + s·d = t2·r̂2

Three coplanar vectors, so `M = [r̂1 | d | −r̂2]` has a one-dimensional null
space. Fix `s = 1` and `t1`, `t2` come out as multiples of the segment's own
length — multiply by 1.80 m and they are metres. That is `rhythm.segment_depth`,
thirty lines, no optimiser.

The eye height falls out of the same solve for free: the ground plane's normal
*is* the segment's direction, so the base point's extent along it is how high
the camera sits above the ground the figure stands on. That assumption is
named in the conditioning (`assumed`) and switchable, rather than hidden
inside the distance — a figure on a balcony gives a true distance and a
meaningless eye height.

### 16.2 The answer is a range, because it has to be

Distance is proportional to f and to one over the marked height, so both
uncertainties pass straight through undamped. Measured on a synthetic camera,
a 180 cm figure perturbed by the hand-marking accuracy of 9 px:

| marked height | distance is good to |
|---|---|
| 625 px | ±3 % |
| 240 px | ±8 % |
| 78 px | ±25 % |
| 28 px | ±111 % — refused |

f carries its own share on top: 1000–1250 px moved every one of those by a
further ±11 %, independently of how well the figure was marked. So the readout
quotes `distance`, `range`, `±%` and the marked height together, and refuses
outright once the span exceeds the answer: at 28 px the honest span was 42 m
to 198 m for a true 70 m, and printing "70 m" beside it is exactly the
confident wrong number this app exists to avoid.

Imperial in, imperial out: a size given in feet or inches is answered in feet,
metric in metres (`SCENE_UNIT`).

### 16.3 Measured end to end

Through the real client, on a synthetic camera (f 1100, pp 442,640) with two
180 cm figures:

| | reported | true |
|---|---|---|
| figure #1 | 3.78 m (3.69–3.88, ±3 %, 664 px tall) | 3.780 m |
| figure #2 | 8.78 m (8.20–9.45, ±8 %, 250 px tall) | 8.778 m |
| eye above the ground | 1.65 m from each figure independently | 1.65 m |

The camera card in the same run read f 1100.000, pp (442.000, 640.000).
Thirteen rows in `selfcheck.py:scene_distance`, including the four refusals —
no camera, no real size, nothing marked, marked too small. Whole suite
**138 passed, 0 failed**.

### 16.4 Three bugs found by verifying

- **The headline distance was a corner of its own error bar.** `mid` was read
  off the first *perturbation* rather than the nominal solve, so the f = 1000
  corner was printed as the answer: 3.543 m for a true 3.750 m, with eye
  heights of 2.80 m and 5.20 m instead of 1.65 m. The nominal is now computed
  separately and the perturbations never touch it.
- **`buildDoc` sent no marks at all.** It carried `pairs` only for the layer
  kind literally named `pairs`, so a new pair-drawing kind silently posted an
  empty feature and the check refused with "nothing marked". It now sends
  pairs for anything whose `draw` is `'pair'` — the bug was a name test where
  a capability test belonged.
- **`horizon_consistency` was being used as a gate.** `_camera` refused an
  exactly-orthogonal triple over a 0.324° "disagreement" that was an artefact
  of measuring roll from `y = 0` instead of from pp. `calibrate` recovers that
  camera's f = 1100.000 and pp = (442, 640) exactly. The function now takes pp,
  and `_camera` **solves first and explains second**: a diagnostic that vetoes
  the solver it is meant to describe will eventually veto a correct answer.

## 17. A line that does not belong to its family — 2026-10-09

**The report:** "one of the blue lines is inconsistent but it is said as
consistent." On the user's own marks, ten lines in one family, the readout
showed a green `ok` chip beside the words *worst line off by 4.287 deg*.

### 17.1 The verdict was not reading the conditioning

`fit_vp` returned `ok` whenever a finite point came out of the solve. The
residuals went into the conditioning and nothing ever looked at them. So the
chip said consistent while the panel underneath said otherwise, which is the
one thing this app is not allowed to do.

### 17.2 Degrees are the wrong yardstick

The obvious fix — flag the worst residual — flags the wrong line. In this
family:

| line | length | residual | its ends must swing |
|---|---|---|---|
| 6 | 1149 px | 4.287 deg | **45.7 px** |
| 9 | 409 px | 2.436 deg | 8.8 px |

Line 9 is second-worst in degrees and entirely innocent: on a segment that
short, 2.4 deg is 9 px of movement, well inside the hand that drew it. What a
line is actually asking for is that its own ends move, so that is what
`vp.strays` measures — half its length times the tangent of its residual.

### 17.3 Measured leave one out, or the stray hides in its own baseline

A long stray drags a length-weighted fit toward itself and inflates everyone
else's residual with it. Twisting each line of the clean family by 4 deg and
asking whether it was caught:

| twisted line | in its own fit | left out |
|---|---|---|
| 2548 px | 3.2x the median — missed | 5.2x — caught |
| 2278 px | 3.7x the median — missed | 4.7x — caught |
| 1487 px | caught | caught |

So each line is measured against the point the *other* lines agree on. The
clean family's worst line scores 2.3x that way, and the real stray 5.4x.

### 17.4 Both conditions, and the floor they set

A line is called a stray when it needs more than a hand mark is worth **and**
more than four times what the rest of the family needs. The ratio alone would
condemn the worst member of a uniformly careless family; the absolute test
alone would condemn every line on a large picture. On this 6583 px diagonal a
hand mark is worth 38 px, so the honest floor is about 1200 px of line: below
that, 4 deg of error does not move the ends far enough to tell. That is
stated rather than hidden — a 1030 px line twisted 4 deg scores 5.1x the
median, is still not called, and becomes a stray the moment the declared hand
accuracy drops to 3.8 px.

### 17.5 What it says, and where

The verdict becomes `check these` — not a refusal, the point is still returned
and still used, because a stray is something to say about a fit, not a reason
to throw it away. The panel names the line and the canvas draws it with a
white halo, since an index in a side panel does not find one line among ten of
the same color. On the user's file:

> line 6 of 10, marked white on the picture, must swing 46 px to meet the
> point the others agree on, where the rest need 28 px or less and a hand mark
> here is worth 38 px.

Verified on the rendered canvas rather than by eye: the pixels flanking line 6
read 226 against 169 two pixels further out, which is exactly the 0.65-alpha
white the code draws over that background (0.65 x 255 + 0.35 x 169 = 225). No
other line in the family exceeds 156, and exactly one haloed band appears.

### 17.6 Two bugs alongside

- **The fit was about to be told the wrong picture size.** `applyMarks` fits
  every family before `openImage` has resolved, so `S.image` is still null
  there and the new size argument would have gone out as nothing — leaving the
  server to assume the 9 px hand of a 1556 px picture and judge a 6583 px one
  four times too harshly. The marks file carries the size, so it is passed in
  explicitly on that path.
- **`selfcheck.check(..., want=[], tol=0)` passed whatever it was given.**
  numpy broadcasts a one-element array against an empty one and `np.all([])`
  is `True`, so every "flags none" check was vacuous — and it hid that one of
  my own test premises was wrong. `check` now compares shapes first.

`vp.fit` was lifted out of the `fit_vp` endpoint on the way, since the stray
test needs the same length-weighted solve and two copies of a weighting rule
is one too many. Eleven rows in `selfcheck.py:stray_lines`, on a fixture of
the user's ten lines; whole suite **149 passed, 0 failed**.

## 18. Absorbing the hand, and getting at the lines again — 2026-10-09

**The request:** "after a certain number of lines (3), make it possible to
auto-adjust small deviations without flagging: manual marking is often less
precise than a drawing. also, when marks are loaded, it should be still
possible to modify the lines."

### 18.1 Marking is coarser than drawing

A line drawn true can still be marked a degree off, and that wobble belongs to
the marker, not the artist. With three lines the family is over-determined, so
the two can be told apart, and §17 already measures the difference in the
right unit: the swing a line's ends must make. **Tidy marks** turns every line
whose swing is within what a hand mark is worth here onto the point the family
agrees on, and leaves everything beyond that exactly as marked, because that
is the drawing talking.

Nothing about the flagging rule changed — a deviation inside the hand was
never flagged. What changes is that it no longer sits in the fit either.

### 18.2 The stray must not be the thing they are pulled onto

The first version aimed each line at the family's own vanishing point. Run it
repeatedly and the app launders a bad line into agreement:

| tidy pass | vanishing point | strays | worst of the rest |
|---|---|---|---|
| 0 | 12619, 2465 | line 6 | 27.9 px |
| 1 | 12577, 2513 | line 6 | 13.1 px |
| 2 | 12538, 2557 | line 6 | 12.4 px |
| 4 | 12470, 2638 | **none** | 36.0 px |

Each pass let the stray pull the consensus and the tidied lines followed it,
174 px in four passes, until nothing was wrong any more. The target is now the
point the lines that are **not** strays agree on, and tidying becomes a fixed
point: five passes land within 2e-8 px of one, the stray stays flagged, and
nothing moves further than a hand mark from where it was drawn.

On the user's family: nine of ten absorbed, worst swing 27.9 px against a
38 px hand, worst of the rest 27.9 -> 13.8 px, line 6 untouched at 43 px and
still named.

### 18.3 It has to say that it did it

Afterwards those lines agree with the point by construction, so their spread
no longer measures the marking and the bootstrap under it is narrower than the
marks ever were. That is exactly the kind of borrowed confidence this app
exists to refuse, so the readout carries it:

> 9 of 10 lines were swung onto this point, each by at most 28 px -- within
> what a hand mark is worth here, so it was marking and not drawing. Those
> lines now agree by construction and the spread below is narrower than the
> marking was.

The stamp lives on the layer, so it is saved with the marks and it travels
through undo: cmd-Z puts the marks back **and** the note disappears, because
`snapshot()` serialises whole layers and `restore()` re-fits from them.

### 18.4 Editing a loaded file was possible, and invisible

It was never broken — a drag within 8 screen px of an end moves that end, on a
loaded file as much as a fresh one, verified by dragging one and watching the
vanishing point follow. But nothing was drawn at those ends. On a 3756 x 5406
picture at fit-to-window, an unmarked 8 px target is a guess, and a miss does
not do nothing: it draws a new line. That is the whole of the complaint.

The ends of the selected family's lines are now drawn as handles, the same
dots every other mark already had, and the hint says what they do: *drag out
each line; drag a dot on its end to move that end, cmd-click it to delete the
line.*

Eight rows in `selfcheck.py:stray_lines`; whole suite **157 passed, 0 failed**.

### 18.1 Only the selected family's marks are grabbable — 2026-10-09

`nearestPoint` swept every layer. A corner where a blue line ends is exactly
where a yellow line wants to start, so the mousedown grabbed the blue end and
the yellow line could not be drawn at all. The hit test now takes the selected
layer and scans only that one; `onDown` already had it in hand.

Measured on the user's file (3 families, 10/2/2 lines), family 2 selected:
a drag starting on blue line 4's end (2584.2, 1318.3) — inside the 76 world px
grab radius at fit-to-window — added a third yellow line and left family 1 at
10 lines. A second drag from that new line's own end moved it: still 3 lines,
not 4. Handles are drawn only on the selected family, so what can be grabbed
is what is shown.

### 18.2 Hiding a layer — 2026-10-09

The swatch in the layer list is the switch: filled means drawn, hollow means
not, and the row's text dims. Hiding is a view and nothing else -- the layer
is still marked, still fitted, still part of a Check. Measured with family 1
hidden, the two-point check still returned f = 2139.4 px, which it could not
have done without that family's vanishing point.

Three rules fell out of trying it:

* A hidden layer is not drawn, so it cannot be marked on blindly: hiding the
  selected layer deselects it, and selecting a hidden one shows it again.
* Hiding is not an edit, so it is not an undo step; and `restore` carries the
  hidden flags across by layer id, or an undo would put back on screen what
  the person had just put away. Verified: draw a line in family 2, hide
  family 3, cmd-Z -- the line goes, family 3 stays hidden.
* Showing a layer again repaints it bit for bit (0 pixels different from
  before it was hidden). Hiding families 1/2/3 changes 19105 / 3791 / 2823
  canvas pixels, and all three at once 25694 -- the 25 px short of the sum
  are where two families cross.

### 19. A surface that is level with the ground — 2026-10-09

A surface was a pair of vanishing points, which meant a table standing askew
on the floor needed two line families of its own before anything on it could
be read. It does not. A plane parallel to the ground has the ground's
vanishing line -- the horizon -- however it is turned, so the horizon the room
already gives is the table's vanishing line too.

The surface panel now has a *level with the ground* switch. Ticked, the two
selectors go away and the surface takes `doc.horizon_vps`, the first two
horizontal families. Untickable when fewer than two are marked, and the panel
says how many there are.

Measured (selfcheck `askew_table`, f = 1000, camera rotated 0.17/0.41/0.06):
a table turned 35 deg about the vertical gives a vanishing line identical to
the room's and to the floor's own, to **1e-12** in every component; a round
tabletop projected on it reads **ratio 1.0000** against the room's horizon.
The contrast is the point of the rule: the same marks read against a *wall's*
vanishing line are not even an ellipse, and `circle_check` returns None.

It is a declaration, not a measurement -- nothing here checks that the table
really is level -- so every arc on such a surface carries
`surface_assumed: "'<name>' was declared level with the ground, so this arc
was read against the horizon and not against lines marked on the plane
itself."` Verified end to end on the user's file: a traced ellipse on a level
surface returned *oval 2.158* with that note in its conditioning, and with
only one horizontal family left it refused: *"the horizon takes two horizontal
vanishing points; mark a second horizontal family, or name the two that span
this plane"*.

### 19.1 A surface you can see — 2026-10-09

A surface was two dropdowns and nothing on the picture, which made it the one
layer whose meaning you had to take on trust. It is drawn now: **drag across
the plane** and the two ends are opposite corners of a translucent patch whose
other two corners are where the edges from each end, run to the surface's two
vanishing points, meet. So the patch is the perspective image of a
parallelogram lying in the plane -- if it does not sit on the table, the
vanishing points are not the table's, and that is visible at a glance instead
of being discovered three readouts later.

Measured on the user's picture with a level surface over the bench: all four
edges aim at their vanishing point to **0.0 deg** (the fourth to 1.2e-6, which
is the intersection solve), and the painted outline is the layer's own colour
at every corner. One patch per surface, like a contact layer's support edge,
and the diagonal's two ends are the only draggable corners -- the other two
follow. Candidate segments are not offered on a surface: it is drawn from the
vanishing points, not from a detected edge.

### 20. A family held on the horizon — 2026-10-09

Horizontals all share one horizon, whichever way they run, so a family added
later is over-determined by the horizon as well as by its own lines. The
family panel has *its vanishing point is on the horizon*; ticked, the fit is
reported held on the line the **other** horizontal families give -- never on
one this family helped draw -- and Tidy swings its lines onto that instead of
onto their own free consensus. `V.fit_on_line` is the free solve restricted to
l . v = 0, using the right singular vectors of `l` past the first as a basis,
so the constraint costs a 3x2 multiply and there is no penalty weight to
choose.

Measured (selfcheck `horizon_families`, five lines aimed at 8000, 2400 with
the horizon at y = 2400):

| marking jitter | free fit misses the truth by | held on the horizon |
|---|---|---|
| 0.7 deg | 253 px | **167 px** |
| 2.0 deg | 722 px | **469 px** |

over 300 markings each, because one draw says nothing about a prior. The held
fit sits on the horizon to 1e-9, and tidying onto it is a fixed point: five
further passes move nothing (1e-13), which it is only because the horizon is
never this family's own fit.

End to end on the user's file, a fourth family of four lines aimed 115 px
below the horizon: free fit **6158.3, 2548.6**; held, **6592.2, 2435.5**,
which the readout reaches by saying the lines swing at most 20 px to follow,
"within what a hand mark is worth here (38 px), so Tidy can absorb it". Tidy
then reports *4 lines absorbed onto the horizon, worst 20.1 px*, and afterwards
the held fit asks for 0 px. Moving a line in family 1 tilts the horizon and
family 4's held point follows by itself, 6592, 2436 -> 7582, 2167: a held
family refits whenever a family that defines the horizon changes.

**A number that was wrong, and is now right.** `tidy` reported as its worst
swing the leave-one-out figure that *chose* each line, not the rotation it
actually applied. With a free target the two are close; with a horizon to aim
at they are not -- the page predicted 20 px and then printed 13. It now
reports the displacement it applied, checked against `|out - s|`. The honest
figure for the §18 tidy of the user's family 1 is **10.4 px**, not the 27.9 px
printed then; 27.9 px was the leave-one-out swing of the line that moved
furthest, and the 10.1 px measured there by hand was the real displacement all
along.

### 20.1 What the held family actually shows — 2026-10-09

Two faults, both visible in one screenshot of a two-line family ticked onto
the horizon: the cross was drawn **off** the horizon it was said to be held
on, and there was no Tidy button.

* The point reported was the free fit, with the held one in the small print.
  Held on the horizon, the point on the horizon IS the answer -- that is what
  the declaration means -- so it is the value now, and the free fit became
  conditioning: *"Left free the lines put it at 7932, 2239, 982 px away; on
  the horizon they swing at most 0.5 px to meet the point above."* The cross
  now lands on the horizon: 6969.5, **2437.3** against 2437.35 computed by
  hand from the two defining vanishing points, and on screen the crosshair
  sits on the dashed horizon.
* Tidy refused under three lines because two meet at exactly one point and
  there is no spread to tell the hand from the drawing. With a horizon that
  reasoning does not hold: two lines plus a line they must meet on is one
  constraint more than the point has freedom. Tidy now takes two lines **when
  a horizon is given**, and the refusal says so.

The 982 px / 0.5 px pair is the useful number here: two short marks pin a
vanishing point so weakly that half a pixel of marking slides it a thousand
pixels along the page. That is the case for holding it on the horizon, and
also why the free point was never worth drawing.

One more gate while in there. The tidy gate was the leave-one-out residual,
which is identically zero for two lines -- so a horizon could have swung them
any distance it liked. The applied swing is now itself the gate: nothing moves
further than a hand mark, by construction rather than by consequence.

### 21. The marks as a vector PDF — 2026-10-10

A marks file says where every line is; it does not let anyone look at them.
**Save PDF**, beside Save marks, draws the same overlay the page draws, as
real vector geometry: the segments in their layer colours, a dashed ray from
each one toward its vanishing point, the vanishing point itself where it falls
on the page, the horizon, surfaces as translucent quads, and every other
layer's points and pairs. The picture goes in as pixels on its own object, so
deleting the photograph in Illustrator leaves the line work.

`core/export.py` draws; it derives nothing. The page already knows how to work
out a surface's quad and which two families give the horizon, so it sends
those with the layers -- the geometry is worked out in one place and drawn in
two, rather than written twice in two languages.

Checked by reading the PDF back, because "it opened" is not a measurement: an
overlay rasterised into the page, or text turned into outlines, looks the same
on screen and is useless in Illustrator. selfcheck `pdf_export` inflates the
page's content stream and asserts stroked paths (23 `l` ops), three dash
arrays for the rays, each visible layer's colour as an `RG` operator, and the
key as text in `ArialMT` -- `labkit.plots` sets `pdf.fonttype 42` at import,
which is the whole reason it is imported. A hidden layer contributes no
colour, and showing it again adds exactly that one.

**Points are not pixels.** Widths scaled with the picture first, which put a
3.9 pt stroke -- 29 picture pixels -- over a watercolour. They are 0.9 pt on
the page now, and the page's long edge is 10 in whatever the picture's size.

## 22. The spaced row, made honest — 2026-10-10

Three rounds of "I don't understand how evenly spaced row works" were three
real defects, not three explanations.

**The maths was never the problem.** Sending the vanishing point to infinity
turns world-even into page-even: project each post base onto the horizon
direction (`t`), do the same for the VP (`t_vp`), and `u = 1/(t_vp - t)` is
affine in the real distance along the row. Equal world spacing means `u` is an
arithmetic sequence. Verified: a six-post row 2 m apart under a real camera
reads **0.00**, and one post moved 20 cm out of step reads **6.02**.

**The tolerance could not be met by a hand.** `px per mm` defaulted to 1, so
the millimetres were pixels and `resolve 2` was a two-pixel tolerance. The same
dead-even row marked to ±4 px reads **6.23 px out** — perfect geometry, called
uneven. `C.spacing` now takes `mark_px` and floors the tolerance at it, the way
`figure_ratio` already did, and reports `mark_noise_floor`, `tolerance_mm` and
the `resolve_mm` that was asked for, so raising it is visible rather than
silent. On a 2400×1800 picture the floor is 17.36 px and the row passes.

**The declared real size could never reach it.** `feat.get("px_per_mm") or ppm`
— and the page sent `1.0` on every check, which is truthy. The field is blank
by default now and sent only when typed. Measured through the live server: no
declared size gives tolerance 17.36 px; declaring the picture 600 mm wide gives
2.474 mm against a 4.339 mm floor; typing `1` still overrides both.

**`worst_at` named the wrong post.** The fit runs along the row, which is the
reverse of the marking order whenever the far end was marked first, and the
index was never mapped back. A row marked right to left reported post 2 for a
fault at post 3. `err_mm` and `worst_at` are in the marked order now.

**The support edge is gone from spaced rows**, kept on figures. It was built
for things standing *on* a surface, where the surface's front edge is nearer
than any contact. A spaced row is usually features *in* a plane — cabinet
dividers along the shelf they hang under — so the contacts belong on the edge,
and the check called correct marking a mistake. Moving the edge to an unrelated
vertical line then turned it green, which is the measure of what the test was
worth there: it only ever reported distance to whatever line was drawn, and fed
no verdict.

**The row has to run to the vanishing point it was told to.** A vanishing
point lies ON the image of every line running to it, so the family named must
be the one the run itself goes along -- name the cross direction and every
number below is about some other direction. It was answered with a number
before. Now the swing is measured at the row's own ends, the way strays are,
and anything past a hand mark refuses and says which way to go. Measured
live on a six-door cupboard run: the across family sits 20 px off the row and
refuses; the along family reads **2.27 px** against a 17.36 px floor, even.
The same gate catches a run that is not straight in the world.

`selfcheck.spaced_rows()`, 16 rows. Suite **199 passed, 0 failed**.

## 23. Two ways to say where a row is, and a tidy for it — 2026-10-10

**Ends and a count.** Mark only the first and last post, say how many there
are, and the row in between is drawn where even spacing puts it. The
construction is the check's own model read backwards: `u = 1/(t_vp - t)` is
affine in the real distance along the run, so stepping `u` evenly between the
two ends places every post. Proven against a real camera: six posts 0.6 m
apart, rebuilt from the two ends alone, land on their true projections to
**2.3e-13 px**. It refuses when the count is under three and when the
vanishing point falls between the ends, which would put a post infinitely far
away.

A built row is **not graded**. It is even by construction, so a verdict on it
would be a verdict on arithmetic; `/api/check` refuses and says to tick each
post instead if the drawing is what wants measuring. Measured in the client:
the six built posts come out 212.9, 148.6, 109.6, 84.2, 66.7 px apart,
shrinking toward the vanishing point, and the server returns the refusal.

**Tidy marks**, the row's version of the family tidy, same bargain: a post
asking for less than a hand mark is the marker's wobble and is absorbed, more
than that is the drawing and is left exactly where it is. Through the real
button on a row marked with a few pixels of wobble: **6 absorbed, worst
4.7 px**.

Two things had to be got right, both already learned once on the family tidy:

*Least squares with leave-one-out is the wrong selector here.* The model has
two parameters, so with six posts one post really out of place threw the fit
far enough to flag its two neighbours as strays too (`[0, 4, 5]` for a fault
at 4). The median of all pairwise slopes does not care: `[4]`.

*The target must not be fitted to the points it moves.* Taking the row's line
from all the marks, strays included, meant the line shifted as the others were
pulled onto it, and a second tidy moved everything another 0.11 px. The line
now comes from the kept posts alone, so afterwards they lie exactly on it and
it fits itself: a second tidy moves **0.0 px**.

Three client gaps found by using it: setting a row's family or its count
redrew nothing, and the Tidy button did not appear until the layer was
re-selected. `refRow`, `numRow` and `finishRun` now refresh the panel and the
picture.

`selfcheck.spaced_rows()`, 24 rows. Suite **207 passed, 0 failed**.

## 24. A row that stands on a surface, and looking at a shape straight on — 2026-10-10

**A surface can name the row's direction, so no family has to.** A row of
lamp posts along a pavement often has no family drawn for the pavement
itself; what the page does have is the surface they stand on. The line
through the first and last foot meets that surface's vanishing line at the
row's own vanishing point, and `rowVp` returns it. The page sends the point
rather than a family id, and `/api/check` takes either — `C.spacing` checks
the point against the row the same way whichever arrived. Measured on a
synthetic camera: the vanishing point derived this way sits **4.311e-10 px**
from the true one, the row built from its two ends lands on the true feet to
**2.728e-12 px**, and ticking each post instead reads **EVEN 0.000**.

**The objects' own family gives their tops.** Lamp posts are the same height,
so their tops lie on a world line parallel to the one their feet lie on —
same vanishing point. Name the upright family, mark the top of the first post
only, and `rowUprights` meets each post's upright with that line for every
other top. Nothing else has to be marked, and the rendering is the check: a
top that will not sit on the drawn post is the drawing disagreeing.

**Straightening** — `core/rectify.py`. `H = K R K⁻¹`, where `R` is the
rotation that turns the camera square on to the plane. This is a *metric*
rectification of the whole plane, not a warp fitted to four marked corners,
and that distinction is the point: it straightens a shape that has no
parallel sides at all, which a four-corner warp cannot. `fit` then pins the
arbitrary scale and offset so the marks land in a picture of their own, and
`upright_error` flips a view that came out upside down.

Verified through the real client on a wall panel at obliqueness 2.74, with
the camera solved exactly (f **800.000**, pp **540.000**/**830.000**):

| on the page | straightened |
|---|---|
| 6x6 checkerboard, cell sides spread **79.20 %** | **6.65 %** |
| its corner angles **72.39 ± 4.49°** | **89.85 ± 0.74°** |
| a triangle, side ratios off **33.16 %** | **0.15 %** |
| its angles off **20.64°** | **0.14°** |

Two client faults found only by using it. A polygon layer had no way to name
a surface, so Straighten could never appear on one — `surfRow` now sits in
the polygon editor. And `row_vp or vps[...]` raised numpy's truth-value error
on an array, which reached the user as a verdict reading *"this feature is
incomplete: The truth value of an array…"*; the test is now `is None`.

## 25. An ellipse is a circle lying in a surface — 2026-10-10

Two halves, and the tool does the first by default.

**Drawn.** `R.plane_circle` takes the surface's vanishing line and builds
`plane_frame(K, l)`. One drag, right across the ring the long way, and the
ellipse a real circle in that surface would cast appears — a guide to draw
against.

**The two dots are the ring's ends, not its middle and rim.** A centre and a
radius sound like the natural handle and are the wrong one: the image's
middle is not the image of the middle, so the drawn ring ran up to **17 %**
past the point it was dragged to, and the user saw it at once — *"real
circles don't look like that. the dots should specify the long diameter"*.
Reading the dots as the two ends of a diameter **on the plane** fixes the
easy case and not the hard one: on a steeply foreshortened ring the image of
a diameter's end sits **85 %** of the span inside the ring's real end.

What is right has a closed form. On the plane, the ring touches the line
through each dot that crosses the drag, and touches it AT that dot, so the
centre lies one radius along each dot's inward normal. Two statements of the
same centre: three unknowns in four equations, `r = -(a₁-a₂)·(n̂₁-n̂₂)/|n̂₁-n̂₂|²`,
exact when the dots really are some circle's ends and the nearest circle when
the hand was not that precise. Drag the two ends of a real circle and that
very circle comes back, to **2e-4 of its radius**, with its ends on the dots
to **0.0013 px**. It refuses by name when the circle reaches the plane's
own vanishing line, which is where the projection stops being a closed curve.
Measured against the scene it was drawn in: on the level floor radius
**0.5726 m**, out of round **1.11e-12 %**, centre off 8.01e-16 m; on the
oblique side wall radius **0.6784 m**, out of round **1.54e-12 %**, centre
off 1.49e-15 m. Both read back exactly the radius that was dragged.

A circle needs the **camera**, not two vanishing points: two points name two
directions without saying the angle between them. So the tool refuses
wherever the camera does.

**The plane is the ground unless a surface is named.** Most circles an artist
draws lie flat — a table top, a pond, a rim on the floor — and making the
user draw a surface layer first for the commonest case is the wrong default.
Naming a surface still wins, for one standing in a wall or on a slope.
Measured: the ring returned with no surface named is **identical to 0.000e+00
px** over all 192 points to the one returned for the floor surface, and in
the world it is a circle of radius 0.5726 m, out of round 1.11e-12 %. The
same drag with the side wall named comes out 191 px away and 157x207 px
instead of 230x7 — naming the surface still matters, and the difference is
unmissable on the page.

This is the one place the vanishing line may be defaulted, and only because
the ellipse is **drawn**: the ring lands on the picture, so a circle taken as
flat when it was meant to stand in a wall is wrong to look at. An arc only
reports a number, so it still refuses — *"this feature has no surface label,
and no plane is assumed"* — which is bug 3 and stays guarded. `_surface_vps`
takes `ground=` for exactly this, and nothing else passes it.

**Measured.** Switch the layer to marking and click round the rim of what was
actually drawn, and the same conic machinery the arc uses reports the axis
ratio on the plane. Five marks at least — a conic has five degrees of freedom
and four fit anything.

**The arc's circle-or-oval verdict was wrong, and widening the tolerance was
not the fix.** A true circle marked through the UI with ±2 px of hand read
**OVAL 1.065**. The cause is bias, not scatter: every wobble makes the fitted
conic less round and none makes it rounder, so the reading sits above 1 no
matter how many marks are taken, and no tolerance centred on it can undo
that. Widening it was measured and rejected — it bought 21.7 % false ovals
down at the price of 17.1 % missed ovals.

What the check does now is refuse. The lift marking noise puts on the axis
ratio was measured over rings 16 to 70 px across, 13 to 60 marks, 1 to 4 px
of hand: in units of `resid / (semi-minor · √n)` the median is 2.6 and the
ninetieth percentile 5, which is `MARK_BIAS`. The check computes that floor
for the marks in hand and, when it exceeds the `resolve` asked for, returns
`undetermined` naming the number it *could* resolve — *mark more points round
it, mark a bigger one, or ask for this instead*. When it does answer: **1.3 %
false ovals, 0 % missed ovals**.

The ring's page size has to come from the marks' own spread
(`σ_minor·√2`), not from the fitted conic's semi-minor. A big ellipse on a
steeply slanted plane projects to a page conic that is not an ellipse at all
and has no semi-minor — asking for one returned `None` and the floor came out
`inf` for every oval.

Live, on the floor circle with 13 marks at ±2 px: at `resolve` 0.05 the
client refused with *"marks this shaky on a ring this size move the axis
ratio by about 0.08 on their own"*; asked for 0.09 instead it answered
**CIRCLE 1.065**. A drawn ellipse with no rim marks says so plainly — it is
round by construction, there is nothing there to check.

`selfcheck.ellipses()`, 22 rows, and `selfcheck.ellipse_plane_default()`,
6 more. Suite **248 passed, 0 failed**.

One of those six was wrong at first and says something general: two
vanishing lines cannot be compared by the dot product of their
normalized coefficient vectors. Any two lines passing far from the
origin are both nearly `(0, 0, 1)`, so the floor and a wall agreed to
1e-5 that way while lying **3424 px** apart at the corners of the
picture. Compare them where it matters, in pixels, at points in the
frame.

## 26. An arch, built rather than marked — 2026-10-10

The ellipse's two halves again, for the arch: build the right one from three
marks, or mark the one in the picture and have it measured. The user's spec,
verbatim: *"it has to have two supports (most likely vertical but any line
family would do) ending on a line of a different family, and then either a
sharp (if it's a pointed arch) or a blunt point snapped to the middle between
the two supports."*

**Three marks, and two of them are snapped.** Drag the springing line between
the two supports: the far foot is pulled onto the line that runs from the near
one to the springing family's own vanishing point, so the two feet really do
end on one line of a family that is not the supports'. The top is then pulled
onto the upright over the middle of the span — the middle **on the wall**, not
halfway between the feet on the page. In the test scene those differ by
**3.2 % of the span**, and in the picture the app was driven on, by **28.1 px
on a 262 px span**; the crown lands on the upright through the true middle to
**0 px**. With no top mark at all it goes where a semicircle's would, which is
`rise/span` **0.500** exactly, and not the 0.332 that half the span up the
*page* gave on that wall.

**Round and pointed differ by one line of geometry.** A round arch has one
centre, on the crown's own upright, at `y = (h²-s²)/(2h)` — zero is a
semicircle, below it segmental, above it horseshoe, one formula for all three,
and both halves leave the top with the same tangent, so the top is blunt. A
pointed arch has two, both on the springing line itself, at `±(h²-s²)/(2s)`
with `ρ = (h²+s²)/(2s)`, so the halves arrive at an angle and the top is
sharp. Measured at the crown: the round arch turns **0.03°** there, the
pointed one **18.9°** at rise/span 0.575 and **98.5°** at 1.200. At rise/span
exactly 0.500 the two centres meet in the middle and the arch IS the
semicircle either way — the point vanishes, as it should.

**Each half is its own cubic, and that is a measurement, not a preference.**
A quarter turn in one cubic Bézier is **0.027 %** of the radius out (0.04 px
on a 300 px arch), and the 105° a horseshoe's half needs is **0.069 %**. A
whole semicircle in a single cubic is **1.84 %** out — **2.8 px** on that same
arch, which would be a worse arch than the drawing it is there to check. So
the curve is cut at the crown, which is also where a pointed arch has to be
cut anyway.

Against a true arch projected into the scene, the drawn curve is the same
curve: **0.06 px** for a semicircle, 0.002 for segmental, **0.16** for a
horseshoe (its halves sweep 105°), 0.007 equilateral, 0.002 lancet. It refuses
when the two feet are in one place, when the top is on the springing line, and
when the span reaches the wall's own vanishing line.

**Nothing is defaulted.** Unlike the ellipse, an arch names its vertical and
its wall vanishing point — the same two the arch CHECK reads it on, so the
curve drawn and the numbers reported cannot disagree about the plane (bug 3).
And a drawn arch is not a measured one: the check says so in the same words
the built row and the drawn ellipse use, instead of the `'pts'` KeyError it
used to leak.

## 27. A free ellipse stays on the ground — 2026-10-10

Tried reading a surface-less circle as **turned square to the viewer**, and
reverted it the same day. The geometry is right and the default was wrong.

**What decided it.** The user drew a ring by hand in Illustrator over a
Persimmon export — round the base of a stool standing on the floor of a
lab watercolour — and said *this is what the circle should look like*. It is
an axis-aligned ellipse, semi-axes 27.03 × 13.44 pt on a 500 × 720 pt page,
centre (330.97, 511.74), horizon at 321.3. **Ratio 2.011.**

| read as | ratio there | long axis |
|---|---|---|
| a circle on the ground, f 357 pt (70° field) | 2.011 | 8.7° off level |
| a circle square to the viewer | 1.13 | 62° |
| the ring drawn by hand | 2.011 | level |

The ground reproduces the hand-drawn ring to 0.5 pt of where it was put.
Facing the viewer is half as flat. It is the ground, and the user's earlier
*"bound … just to the horizon line (!)"* said so before the picture did.

**The geometry that lost.** A plane's vanishing line is the polar of its
normal's vanishing point in the image of the absolute conic, `l = ω v` with
`ω = inv(K Kᵀ)`, and a plane facing the viewer has its normal vanishing **at
the point looked at**, so `l = ω p̃` needs no horizon at all -- the handprint
article's sphere construction in one step instead of two. It was measured
(the projection square's two edge families land on that line to 1e-7 px; the
ring stretches radially from the principal point by `1/cos` of the off-axis
angle, 1.2194 against a predicted 1.2207, long axis within 0.03° of radial),
and then deleted, because nothing can reach it. Worth remembering only as the
reason the facing plane is the wrong default twice over: it is not even the
roundest one. Parallel to the picture images every circle as an exact circle,
1.000000, anywhere on the page.

**What "looking straight at it" turned out to mean.** Not a different plane --
the ring stays on the ground -- but the ring **drawn to work against**: the
circle seen square on, whose long diameter ends at the two dots. *"The most
important point is to draw the circle with the two points ON its longer
diameter."* Nothing else satisfies that. The circle this picture's own camera
casts leans away from the centre of vision -- 8.7° on an arm's-length ring in
the watercolour -- so its ends sit 4.1 to 15.8 px from the dots, and the dots
are up to 3.1 px off that ring altogether: pinning a circle to two page points
is four equations in three unknowns, consistent only in the symmetric case.

`square_on_circle(K, l, a, b)` in `core/rhythm.py` does it. Two things had to
be got right.

*Moving the principal point is not a camera rotation.* The obvious reading of
"look straight at it" is to shift pp onto the ring, and it passes a level
camera, which is why it survived an afternoon. `plane_frame` reads
`n = Kᵀ l`, so editing pp asks for a **different world plane**. In a camera
with roll 0.06 and pitch 0.17 it flattened a ring reading 2.255 to 1.055
(pp along page x) or 1.172 (pp yawed along the horizon), with the dots 16 to
169 px adrift. Replaced by the slant, `sin θ = |d̂ · n̂|` with `d = inv(K) c̃`
the ray to the ring and `n = Kᵀ l` the plane normal -- one dot product, and a
rotation invariant, so it holds under any roll or tilt.

*A cosine squash is not a projection.* Foreshortening the ring by `sin θ`
orthographically is 1.3 % wrong on a ring wide enough to matter: 9.2834 where
the answer is 9.4075, at the centre of vision, where the square-on ring and
the camera's own ring must be the **same curve**. Drawing out the turned eye
instead -- optical axis on the ring, no roll, the plane at the same slant, its
horizon then `f tan θ` off the middle -- and asking `plane_circle` for what
that camera casts gives 1.6e-13 px at half-widths 20, 60 and 120.

**Both rings are drawn.** Away from the centre of vision the picture plane
stretches a circle further than the eye sees it -- 2.99 against 1.68 at 56°
off axis -- so the square-on ring is a drawing guide and nothing more. What a
marked rim is checked against is still `plane_circle`, the leaning one, and it
is drawn faint and dashed beside the solid guide, on the page and in the PDF
export, so the lean is there to be looked at. Only for a free ring: name a
surface and there is one ring, the true one. `/api/ellipse` returns
`{ring, ring_true}` when the two differ by more than 0.5 px and `{ring}`
otherwise. (§28 widened that to `{rings, rings_true, ends}` — one entry per
drag — and made `rings_true` carry a `null` for the rings that need no second
curve, so a free ring and one in a named surface come back the same shape.)

`ellipse_drawn_square_on()` in `selfcheck.py` holds the 19 rows above. It uses
the rolled, pitched camera on purpose.

## 28. Families, and the one diameter that is level — 2026-10-10

Three instructions in a row, and they turned out to be one thing:

> we need ellipse families: they should all be consistent against the horizon
> ellipses on the horizontal planes should have strictly horizontal diameters
> what is going on with the arches? make them as I suggested

— the last with ArtGraphica Fig. 215/216 (a pointed arch on two supports
standing on a line marked TO VP) and Fig. 208/209 (a round arch with
voussoirs, and **an arcade of four receding round arches** off one horizon and
one VP).

**A family is several members in one layer on one plane under one camera.**
That is the whole mechanism, and it is why they agree: an arcade is not four
arches reconciled afterwards, it is four spans read through the same
`plane_frame`. `/api/ellipse` and `/api/arch` each take a **list** of drags and
return a list of curves; the old single-drag payload is still accepted
(`if segs and not isinstance(segs[0], (list, tuple)): segs = [segs]`) so
nothing already drawn had to be remade. `ONE_LINE = {surface: 1}` is now the
only kind limited to one drag.

**The level diameter is exact, and it is exact for a reason.** On any plane,
the diameter **parallel to the picture plane** images **parallel to that
plane's vanishing line** — measured at **2.2e-6 deg** across rolls, pitches and
yaws, which is machine noise, not a tolerance. The reason is in one number:
its two ends sit at **the same depth**. Back-projected onto the ground, the
levelled drag's ends come back with camera-frame Z ratio **1.000000**; the
slanted drag actually made, **0.918**. Ends at one depth recede by the same
amount, so the chord cannot tilt.

Nothing *else* about the ellipse is level. The fitted ellipse's own major axis
leans **5.69°** on the centre line of the test camera and **15.04°** out at the
side; the camera's own ring through those same two points leans **29–56°** down
in the corner of the frame. So "strictly horizontal diameters" is not a
simplification of the geometry, it is the one chord that has it — and since the
hand cannot hit it by eye, `R.horizon_diameter(a, b, l)` turns the drag onto it
**about its middle** (turning it about one end would carry the circle off as
well as its diameter: same midpoint to 0 px, same length to 0 px). The levelled
ends go back to the page as `ends`, so the grab handles sit on the diameter the
ring really has — but not while a hand is still on one (`if (!S.dragPt)`),
which is the same guard the arch's feet-snap already used.

Only **horizontal** planes level their drag: `(not surface) or
surfaces[sid].level`. A ring in a wall keeps the drag as made, because there
the level diameter is not the one the hand is drawing.

**The arch was not broken, it was defaulting to the wrong tool and refusing
with the wrong sentence.**

- `S.archDraw` started `false`, so a new arch layer *marked* points while a new
  ellipse layer *drew* a ring. Now `true` on both the default and the selection
  reset. An arch is built by default, as asked.
- The camera refusal said "a camera needs three mutually perpendicular
  vanishing points" for two quite different situations, and sent a drawing with
  **three good families** back to look for a fourth. The missing thing was the
  **tick-box**: the document's vertical family is declared by "this is the
  vertical family" on the family layer (`doc.vertical_vp`), not by the arch's
  own `vertical_vp` dropdown. `_camera` now splits the refusal in two and names
  the switch. The symptom cost an hour; the message was the bug.
- An arcade reads straight off Fig. 209: the pier spacing comes from the
  **spaced row** (§22, diagonals and all), and each bay's springing line is one
  more drag in the one arch layer. Three spans on a wall, under the rolled test
  camera, come back `rise_over_span` 0.500 each.

**Still two cubics, not one.** The spec says a round arch "should be a single
symmetric bezier curve". It is drawn as two mirrored cubics meeting at the
crown, and §26 is why: a quarter turn in one cubic is **0.027 %** of the radius
out, a whole semicircle in one cubic is **1.84 %** — **2.8 px** on a 300 px
arch, which is a visible error on a tool whose job is to measure error. Under
an Illustrator reading, "a single symmetric bezier curve" is one symmetric
*path*, which is what exists. Flagged, not silently kept.

**The Marks menu is built from `KINDS` now.** "remove the idiotic names of
the marks, keep them concise and accurate": the menu had been hand-written
beside the labels it duplicated, so it carried a sentence of explanation per
line — *"arch (near limb, optional far limb)"*, which stopped being true the
moment the arch started building itself, and *"arc &mdash; circle or oval?"*,
a question where a name belongs. The option now holds the name and the blurb
hangs on it as hover text, exactly as the perspective menu already did, and
both come from the one `KINDS` entry that already named the editor's title and
a new layer. Twelve hard-coded lines deleted; one label corrected, `polygon`
to **regular polygon**, which is the check that actually runs
(`checks.polygon` ranks regular n-gons).

`ellipse_family_level()` holds 13 rows: the level diameter, the equal-depth
reason, the turn about the middle, the refusal on a zero-length drag, three
rings from one call all levelled and foreshortened in the order of their
distance from the horizon, and three arches from one call all half-round.

## 29. One Bezier curve, with handles on it — 2026-10-10

> make the round arch a single bezier curve. make bezier curves adjustable
> like in illustrator

§28's number was already known — one cubic over a whole semicircle is 1.84 %
out — and it was asked for anyway. The user's decision; here is what it cost
and what was done to keep the cost down.

**The obvious way is the worst way.** Draw the cubic in the arch's own plane
and map its four control points through the homography: that is two errors
stacked, because the image of a cubic is not a cubic. Measured against the
arc as it really images, the classic 4/3 semicircle construction lands
**3.0–4.9 % of the rise** out, and gets *worse* with foreshortening.
Optimising the plane-frame handle length (k = 1.3157 beats 4/3's 1.8350 %
radial error) barely helps once it is mapped: 2.3–4.1 %.

**Fit it on the page instead.** `fit_cubic` keeps the two endpoints and the
two tangent *directions* — the real arch's tangents at its springing, which
an arch is not allowed to trade away — and fits only the two handle
**lengths** against a dense sampling of the arc as it images. A cubic's point
is linear in those two lengths, so a pass is one `lstsq`; between passes each
sample takes the parameter of the nearest point on the curve so far. Six
passes, about 10 ms for the whole arch. Result: **1.8–2.0 % of the rise**,
and steady under foreshortening instead of growing with it. Three passes
leaves 0.15 % on the table; ten passes and twice the samples buy 0.02 % for
three times the time, and a minimax polish on top buys 0.2 % for 90 ms —
both measured, both dropped.

**What it replaced, in one number.** The old curve was the projected arc
sampled: **0.055–0.128 px** off the true arch, against this one's
**1.4–3.3 px**. A pointed arch keeps a cubic per side, as specified, and
pays **0.4–1.3 px**. That is the trade — a path with handles instead of a
polyline — and it is not hidden: `curve_off` travels with the curve and the
editor prints "2.25 px off the true arch" beside rise/span. `pts` is still
the exact arc, so every readout is unchanged; only what is *drawn* is the
Bezier.

**Handles, as in a drawing program.** A bar from each anchor out to a hollow
knob; filled dots stay the anchors (they move the arch), hollow knobs only
shape the curve. Hit-tested *after* the anchors so an anchor underneath still
wins. The hand's change is stored as an **offset** from the fitted control
point, not as a position, so nudging a foot, moving a vanishing point or
undoing carries the adjustment along with the curve instead of throwing it
away; `reset handles` clears it. Once a handle has moved, the departure is
re-measured in the page against the same arc, so the number never goes stale
— and it goes up, which is the point of printing it.

The PDF carries the real path: `PathPatch` with `CURVE4`, control points and
all, so a drawing program opens it with its handles on rather than as a
polyline of 191 points.

`arch_one_bezier()` holds 13 rows: one curve round and one a side pointed,
the ends on the springings, the two halves meeting at the point, the tangent
it leaves by (0.03°), the 1.90 % and 0.79 %, the reported number against an
independent measurement, the 20x-closer polyline it replaced, the mapped 4/3
construction being 1.67x worse, the refusal on a flat arc, and three paths
with three numbers from one endpoint call. The suite is 316 passed, 0 failed.

## 30. Persimmon, a mirrored limb and a vault — 2026-10-10

**The name.** PerspectiveCheck is **Persimmon** — it finds the most
parsimonious perspective. The header carries `by RíEile` under the wordmark,
as ArtStitch does, reusing labkit's `.brand-by` rather than new CSS. The
folder, `.claude/launch.json`, labkit's own app list in `sync.py`, both run
scripts, the README, the `/api/health` payload and PLAN's own title moved with
it; eleven occurrences of the old name, none left.

**Round or pointed is a toggle, not a flip.** It was a button that said what
it would do next, which means the panel never said what the arch *is*. Now
labkit's `.seg` holds `round | pointed` with the current one lit. The arch
editor's numbers and `reset handles` moved to a second row — the wider control
crowded them, which is only visible by looking at the page.

**One limb follows the other, in perspective.** The arch's own mirror is the
reflection across the perpendicular bisector of its span *in the wall plane*,
carried to the page: `H · Rp · H⁻¹`. It is not a page mirror. A page mirror
moves both handles by exactly the same amount (measured: 1.000); the wall's
mirror moves the far limb by **0.823** and the near limb by **1.602**, because
the far limb is foreshortened and the near one is not.

What follows is the hand's **move**, not the handle's **position**. The
reason is measured, not stylistic: `fit_cubic` optimises each handle length
independently in the page, and the best page cubic is genuinely *not*
wall-symmetric — **17.97 px** of asymmetry on a mild arch, **116.45 px** on a
steep one. Forcing symmetry onto the fit costs **1.09x–2.33x** the accuracy.
So the fit keeps its freedom and the mirror carries only the offset, which has
the side benefit that nothing jumps when a handle is first touched.

**A vault is the arch slid back.** Name a depth vanishing point on the arch
layer and the crown is free to leave the wall. Where it lands says the rise
*and* the depth at once: its wall position is where the depth line through it
crosses the upright over the middle of the span — one two-line intersection —
and the slide follows linearly as `x' ~ x + mu(l·x)v`, a planar homology whose
axis is the wall's vanishing line and whose vertex is the depth VP. That is
the same homology the arch *check* already uses for `normal_vp`, so the thing
built and the thing checked are provably one: the check reads the built vault
as a single slide at **rms 0.0000 px**, at the depth it was built at
(ratio 1.0 ± 1e-6). The far limb draws as the near limb's cubics through the
homology — the image of a cubic Bezier is not a cubic, but it costs no more
than the near limb's own error (ratio < 1.2) — plus a ridge between the two
crowns and a join at each springing. The PDF carries all of it as real
`CURVE4` paths.

`arch_mirrored_handles()` and `arch_vault()` hold 13 rows each: the mirror
swapping the feet, holding the crown, being an involution and mapping the arch
onto itself (all 0.0000 px) against 58.20 px for a page mirror; the free fit's
asymmetry and what symmetry would cost; the two pull ratios against the page
mirror's exact 1.000; the far limb sitting on its depth lines; the check's
agreement; the 0.726 foreshortening; and the refusal when the crown sights
along the wall. Three more rows in `pdf_export()` count curve operators as
*differences* against a baseline — absolute counts pick up the page's own
furniture — showing +1 cubic for the arch over the polyline it replaced, +1
for the far limb, and +3 lines for the ridge and springings. The suite is
**345 passed, 0 failed**.

## 31. Arches and roofs — 2026-10-10

**A roof is an arch with straight sides.** The roof slope used to be its own
thing; it is now the third `shape` on the arch layer, beside `round` and
`pointed`, and the layer is called *arches and roofs* in the menu. The reason
it costs almost nothing is that a straight line in the arch's plane images as
a straight line: the limb needs no curve at all. Measured, a straight side
images straight to **0.0000 px**, where a round arch's side is **44.06 px**
off its chord; the path drawn IS the roof (**0.0 px**, exactly), where one
cubic over a round arch costs **2.39 px**; and every handle sits on the slope
itself (**0.0000 px**), so dragging one cannot bend a rafter.

With no top marked the gable stands at **45°**, which is a guess and says so.
Slid back on a depth vanishing point it stops being a gable and becomes a
roof: **2** slopes at the pitch the gable had (**0.5000**), near and far rake
meeting at points **7855.61 px** apart — they are different points, which is
the whole of what a roof looks like — and each pair meets exactly where the
camera puts that 45° direction (**0.0000 px**), wherever the far gable is slid
to (**0.0000 px** under a change of depth).

The check is `C.roof`, and it needs the slope's own vanishing point, the
vertical one, and the wall's horizontal one that the roof rises along: those
three are collinear and no camera is needed to say so. A gable drawn in the
wall has **no slope to measure** and is refused rather than scored. The
numbers that make it a real check: a slope is **1773.96 px** off the horizon,
so a roof slope marked as a horizontal family cannot pass; a steeper roof
reads **7.70** steeper and sits **0.55** nearer the vertical vanishing point.
`arch_straight_roof()` holds 17 rows.

## 32. An arch that leaves its wall, both ways — 2026-10-10

A vault (§30) slides the crown back and the arch stays flat against its
supports. A gothic rib does not: it leaves the plane its feet stand in. Both
ways of leaving it are now built, and which one is meant is said by what is
marked, not by a mode.

**Way one: no wall at all.** Leave the wall's vanishing point empty and the
arch stands in the vertical plane through *its own two feet* — a rib across
the bay, which names no wall because it runs diagonally over one. The plane
comes from `span_vp_of`: the springing line's own vanishing point is where the
line through the two feet cuts the horizon, and the horizon is
`inv(K Kᵀ)·v_vert`. It is **0.0000 px** off true, and the rib built on it lies
on the real rib to **0.1225 px** — the plane cubic's own approximation, not an
error of the plane — at the rise it really has (**0.4200**), landing on both
marks exactly (**0.0000 px**). Forcing the wall's plane onto the same three
marks slides the far foot **4.68 px** and the top **8.41 px** off the marks
the user placed: the symptom of a wrong plane is the drawing leaving your
marks, not the curve bending, and that is what the self-check measures. A
springing line that runs level with the horizon has no vanishing point at all,
and is refused (**None**).

**Way two: lean.** Tick `leaning` and the crown mark says two things at once,
the same trick the vault uses for rise and depth. Every plane through the
springing line has its vanishing line through `span_vp`; the arch's upright —
perpendicular to the span, in the arch's own plane — therefore has its
vanishing point on `L_perp = ω·span_vp`. The line from the projective middle
of the span through the crown *is* that upright drawn, so
`up = (M × crown) × L_perp` and the plane is `span_vp × up`. Measured: a rib
tipped **28.0000°** out of its feet's plane comes back at 28.0000°, the arch
it tips into is the real one (**0.1161 px**), the same arch as before only
leaning (rise **0.4200**), with its top exactly on the mark (**0.0000 px**).
Built flat instead, that top is **19.81 px** from where it was put and the
rise reads **0.3491** instead of 0.42. A crown over the middle leans
**0.0000°** and returns the plane it was already in (**0.0000 px**), so
nothing moves for an upright arch. Leaning it the other way is a different
arch: **116.09 px**.

Two refusals hold the thing honest. One crown mark cannot say four things, so
a lean *and* a depth vanishing point together are refused — take one away. And
a crown sitting on the springing line itself says nothing about the lean
(**None**), rather than returning the degenerate plane. `arch_gothic()` holds
19 rows. The checking side needs no change: a built arch is true by
construction and says so, and it says so before any wall is named, which a rib
across a bay never names.

## 33. Put through a photograph — 2026-10-10

A photograph of Worcester nave (the user's, now `testdata/worcester.jpg` with
its marks beside it). It was chosen as a gothic vault to try the new arch on,
and what it actually taught was about the camera.

**A camera where only one horizontal family exists.** Looking straight down a
nave there is nothing to mark but the uprights and the nave axis: two
perpendicular directions, one equation, three unknowns. Pinning the principal
point at the middle of the picture leaves f alone, and the same function the
level case uses solves it, because `f² = −(vA − pp)·(vB − pp)` holds for any
orthogonal pair once pp is fixed. The pin is an assumption, so it travels with
the answer: the readout carries *the principal point is not measured, pinned
at 600, 600* and the sweep over ±5 % of the picture (**3855 – 4075 px**). On
this photograph **f = 3967.6 px**. The EXIF says 13.715 mm on the DiMAGE A1's
8.8 mm sensor, which is 3990 px across a 2560 px frame — **0.6 %** away — on
the assumption that the square was cropped from the full frame and not
resampled; the file has been through Paint Shop Pro, so that is an assumption,
and the camera's own 35 mm-equivalent of 53 mm would say 3769 px instead, 5 %
the other way. Two EXIF routes disagreeing by 5 % is the honest size of this
corroboration.

What the pin costs is not a constant: f is near enough the product of the two
vanishing points' distances from it, so moving the pin by δ changes f by
`δ(1/B − 1/A)` with B the **nearer** vanishing point — measured 6.0 % against
the rule's 5.6 %, and 2.2 % against 2.0 %. A nave is the cheap end of that,
its uprights 14 000 px off the page. The danger is elsewhere: two directions
carry no redundancy, so 5° off square reads **f = 711.5** instead of 1000 with
nothing in the marks to catch it, and only at 20° does it refuse.

**The horizon the camera already knew.** Every check that wants a horizon
asked for two horizontal vanishing points and refused on a nave, although the
camera has it: the horizon is the vanishing line of every plane square to the
uprights, `inv(K Kᵀ)v`. `_horizon(doc, K)` falls back to it, exact to
**0.00 px** against the third vanishing point of a synthetic camera, and still
refuses when there is no camera either.

**The pin was 100 px off centre.** The page sends the picture's size as
`image.w`, never as `width`, so `_camera` fell through to a default 1000 and
pinned the principal point at x = 500 on a 1200 px photograph — in both the
two-point branch and the new one. Here it moved f by only 3.2 px, exactly as
the rule above predicts for vanishing points this far away, which is why it
could sit there unnoticed.

**The bay rhythm, graded.** The ridge bosses are gilded, so they are found by
saturation along the ridge rather than by eye: 8 of them, and the spaced-row
check reads them as **even** — worst **1.54 px** against a marking floor of
**9.82 px**, gap ratios 0.88 – 1.13. That is the first readout this app has
given on a photograph of a real building, and the answer a real building is
entitled to.

**The arch: refused, with numbers.** The vault's ribs cannot be marked end to
end in this frame — the near bays' left springings are outside the picture,
and the right wall is so oblique its arcades are slivers. Where both feet are
in frame, 104 combinations of left foot and crown boss were built and scored
against the paint (the rib images as a dark line, so the picture is read along
each curve's normal): the best reached **40 %** of samples within 5 px,
against **48 – 92 %** for lines actually traced on this picture and **10 – 23
%** for a line drawn through nothing. At 4× the best arch runs parallel to a
painted rib, **10 – 15 px** to one side of it, for the whole length of the
limb: the shape is right and the springing mark is not. Worth keeping: at 1×
that same overlay looked like a hit, and only the measurement said otherwise.
And the construction that should say which boss is the crown is ill-conditioned
here — the upright over the span's middle and the ridge are nearly parallel on
the page, so **5 px** on the two feet moves the crown **±50 px** along the
ridge while the bosses are 110 – 160 px apart. The marks file keeps the camera
and the bosses, and no arch.

## 34. "leaning makes the back limb lean the wrong way"

The report came with a screenshot: an arch tipped off its wall, the near limb
bowing as drawn and the back one bowing the other way, into an S.

Four things were measured before anything was touched, and all four cleared
the geometry:

- against real leaning arches built in 3-D and projected, `span_vp_of` +
  `lean_line` + `plane_arch` recover the lean over +-60 deg either way and
  the built curve sits **0.114-0.134 px** from the true projected arc;
- the drawn curve never reaches its plane's own vanishing line, so no part
  of it is coming back from behind the viewer (780-963 px clear in the reported
  case, and `page()`'s depth-sign guard never fires);
- the feet and the crown come back through `plane_frame` with the same sign,
  so `back()` is not returning an antipodal crown and the `if h < 0` flip
  is not silently mirroring the arch;
- lifting the S-shape back into its own plane gives a **perfectly symmetric**
  arch -- 0.0000 % of the span -- whose plane is 11.99 deg from edge-on. The
  S is what a symmetric arch looks like seen nearly along its own plane.

The bug was in the page. `bezAdj` keeps the hand's handle offsets, and the
drag handler stored TWO of them: the hand's own, and the far limb's, got by
mirroring the move through `L.bezMirror` **at drag time**. That mirror
belongs to the plane the arch was in when the hand moved. Tip the arch off
the wall and the stored far offset is the wall's answer applied to a limb
that now runs somewhere else. Measured on the user's marks: the far handle
lands **362.5 px** from where the leaning plane puts it, and the arch goes
from 8.21 % to **67.47 %** of its span out of symmetry. On an arch seen
broadside the same stale offset is worth 1.3 px, which is why nothing
caught it and why it only ever showed on a rib seen nearly edge-on.

So the far offset is not stored at all any more. `bezOf` derives it when it
draws: a handle the hand has not touched follows its partner (`n - 1 - j`,
which also re-reads itself when the arch changes between round and pointed)
through whatever `L.bezMirror` says NOW. Nothing to go stale. What is
mirrored is still the hand's move and not the handle, so the fitted curve
stays where the camera put it and nothing jerks on first touch.

Checked on the page and not only in the suite: the user's arch loaded into the app,
`bezAdj` set as a drag would leave it, the canvas pixels scanned. The drawn
far limb sits 0.7 px (worst 1.5) from the leaning plane's answer and 10 px
(worst 18.2) from the wall's.

## 35. "the camera is not wrong. it's a rib"

The user sent marks of the Worcester rib with "this is what it should look
like": all four Bezier handles dragged, one of them 609 px. My first reading
was that the arch had only been drawable because the camera was wrong. The
answer was that the camera is not wrong, and that was right twice over.

Wrong once: I swept f while holding the vertical vanishing point fixed, which
is not a camera at all, and got refusals at f = 3990 and 5000 that do not
exist. Built coherently the arch draws at every camera tried. Wrong again in
the framing: the camera is not refuted. It is *unmeasured*, and the app had
printed it as if it were measured.

The chain, each step a number:

| | |
|---|---|
| the ten marked uprights down the nave, tidied | each swung at most 1.6 px |
| the point the bootstrap reported | +- 2e-7 px |
| f reported | 14378.9, swept +-3 % over the pp pin |
| f over what the marks actually allow | 843 .. 26554, and 4 of 9 cameras across that spread do not exist |
| the rib's rise over span, across them | 0.14 .. 5.53 |

So the one rib the app drew was one member of a family it never mentioned. At
the camera it certified, the arch sails 156 px past the marked crown (crown
y 236, curve top 80) -- which is what the handle drags were trying to beat. At
the camera this same photograph was measured on earlier, f = 3968, the apex
lands **on** the crown mark and the rib comes out rise/span 0.917 against
the hand-drawn curve's 83 px. No handle drag needed.

**The root cause is the tidy.** `fit_vp` bootstraps by resampling the lines as
stored, and a tidied family's lines were swung onto one point: they agree by
construction, so the resample returns that point to 1e-7 px and certifies a
camera the marks never pinned. The tidy's own note even said the spread would
be "narrower than the marking was" -- and nothing downstream read that
sentence. Fixed by putting back what the tidy took out: the draws jitter every
endpoint by the tidy's own worst swing, which is a floor on the disagreement
that was really there.

Then the honesty has to reach the eye, so it travels:

- the page sends each family's box as `vp_ranges` (it was already read by
  `/api/check` and never sent -- that f-sensitivity path had been dead, and
  was reading the [x, y] pair as one interval when it did fire);
- `_camera` sweeps f over a 3x3 grid across the upright family's own box, not
  only over the principal-point pin. A grid and not the corners: f^2 is linear
  in the vanishing point, so over a box straddling the horizon the corners all
  said 26500 and the fitted 14379 was not even inside the range they gave;
- when a camera across that box is unpinned (>1.5x) or partly non-existent,
  the camera READOUT goes undetermined -- but K still goes back, because the
  arch is drawn from it and a drawing is a guide, not a claim;
- `/api/arch` re-reads each arch under every camera the box allows and returns
  `rise_over_span_range`, shown on its own line in the refusal's colour.

The sensitivity run costs 1.1 ms, not 85: `plane_arch(with_bez=False)` skips
fitting the drawn path, which is five sixths of an arch (9.4 ms against 1.6),
and nine of those on every drag frame was 102 ms.

Ten rows in `camera_unpinned_by_marks()`, both ways round: a tidied
near-parallel family must report a spread of 1.5e6 px and an undetermined
camera, and a tidied family that does place its point must still come back
f = 1100.0 with nothing called unpinned -- or the jitter cries wolf on every
drawing. 411 passed, 0 failed.

Still open, and for the user to answer: even at f = 3968 the drawn arch is
83 px from the rib drawn by hand. A vault rib need not be a plane circular
arc, and 3968 is itself only measured to 3542 .. 6375 on this photograph.

## 36. "by default a pointed arch shoots very high up in the beginning" — 2026-10-10

Their screenshot showed a pointed arch whose apex sat in the black margin
above the photograph. Nothing was wrong with the fit: a two-mark arch gets a
default crown, the default is a semicircle **on the wall**, and how tall a
semicircle stands on the page is read through the camera's depth scale. For a
span that recedes nearly end-on that height is enormous. Same two feet, two
cameras:

| f (px) | default apex, page y |
|--------|----------------------|
| 14379  | -456                 |
| 3968   | 409                  |

So the first thing the user saw was a number never pinned down, drawn at
full size off the top of the picture.

`_pull_into(apex, mid, w, h, margin=0.04)` clips the segment from `mid` -- the
springing line's middle on the plane, imaged, which `plane_arch` now returns --
to the apex against the picture inset by 4 %. The clipped point lies on that
upright's own image, so the starting crown is still geometry and not a page
fudge; it is simply the highest point of the arch you can actually see. On the
Worcester rib the apex moves to (822, 48), rise/span 0.288, and the panel says
what was cut:

> a semicircle on this span reaches 504 px past the edge of the picture, so
> the top mark starts where its upright leaves the frame. Drag it to the crown
> you can see: how tall a receding span stands is read through the camera, and
> this one is nearly end-on.

`mid` is half way between the feet **on the plane**, not on the page -- half
way on the page is a different point under perspective, and clipping towards
it would walk the crown off the upright.

## 37. "two halves of the arch have to move SIMULTANEOUSLY" — 2026-10-10

§32 derived an untouched handle from its partner through the wall's own mirror
at draw time. The hole was the word untouched: each handle kept its own stored
offset, so once all four had been dragged the mirror had nothing left to
derive and the two limbs were independent curves that happened to meet.

One line in the drag handler:

```js
adj[2 * segs.length - 1 - d.j] = null;
```

A drag nulls its partner's offset, so the partner is derived again from here
on and both limbs always move together. Proved on the user's marks in the pane:
dragging handle 0 left

```
bezAdj[0] = [[-37.15,-143.57],[444.42,457.11],[609.22,546.31], null]
```

and then grabbing the far handle -- at the mirrored position the app had drawn
it, image (711.7, 554.3) -- gave `[null, ..., [-71.44,-11.30]]`. Reciprocal,
and the hit itself confirms the handle is rendered where the mirror puts it.

Both of these are server-side as well as client-side, so a running instance
needs a relaunch, not a reload.

## 38. "leaning arches are still messed up" — 2026-10-10

Their marks file is a nave photograph with three families: down the nave,
the uprights, across the nave. The document is declared two-point, and the
camera panel said **f = 1538.2, ok**. Checked as three-point the SAME three
families are refused:

> the horizon tilts -6.07 deg but the verticals say +0.94 deg; no square-pixel
> camera does both, so no focal length exists

Two-point never looked. `_camera`'s two-point branch builds the camera from
the two horizontal families and *assumes* the uprights are square to the
horizon -- which is what two-point means -- and when a vertical family is
marked that assumption is measurable. Here it is **7.01 deg** out, which is
**128 px** of horizon across this picture: the longitudinal point sits 847 px
below the horizon the uprights allow, and neither family's own spread comes
close to covering that. Everything downstream was read through a camera that
cannot exist: the rib leaned 39 deg and stood 2.104 spans tall.

`horizon_consistency` now runs in the two-point branch too, after the solve
(same order as three-point: a gate before the solve can only refuse cameras
the solve would have found). The level camera still goes back as `K`, as it
does for the unpinned camera of §35 -- the arch is drawn from it and a
drawing is a guide, not a claim -- so nothing disappears from the picture;
the camera panel says UNDETERMINED and the arch panel keeps its range.

Three smaller things the same file showed:

- **lean + a depth vanishing point** used to refuse outright, leaving the
  drawing empty: *"one crown mark cannot say both how far back the vault goes
  and how far the arch leans"*. True, and no reason to kill the arch. The
  lean wins now, the depth point is dropped, and the panel says so. A vault
  is the thing that can be left out and still leave an arch.
- **the range and the notes outlived their fit**: `fitArch` cleared `arches`,
  `riseSpan`, `bez` and the rest but not `riseSpanRange` or `crownNote`, so a
  confident "rise/span 0.97 to 1.91" from the last good fit sat directly above
  the refusal that replaced it. That is what their message quoted.
- **the perspective selector did not refit the arches**, so switching the mode
  left the old rib lying on the picture beside the new mode's refusal.
  `setPerspective` already dropped a stale consensus card for this reason.

What is NOT wrong is the lean. Against a real circular rib in a real plane
under a known camera -- a 12 m nave, a 7 m bay, the boss 8 m above the
springing, which is the configuration they are marking (two springings a bay
apart on one wall and the boss between them, the pair of half-diagonals seen
as one arch leaning in over the nave) -- the lean comes back at 36.869898 deg
against 36.869898 true, the rise over span exact to 1e-9, and the built rib
lies **2.13 px** off the real one on a springing line only 198 px long.

Those 2.13 px are the PLANE cubic, not the camera, and they are a known
ceiling. Each half of an arch is one cubic; a cubic holds a circle while the
sweep is small, and a round arch's sweep grows with its rise while a pointed
one's shrinks:

| round rise/span | 0.25 | 0.50 | 0.75 | 1.00 | 1.43 |
|---|---|---|---|---|---|
| sweep per half | 53 deg | 90 deg | 113 deg | 127 deg | 141 deg |
| cubic off, % of radius | 0.001 | 0.027 | 0.105 | 0.216 | 0.418 |

A pointed arch at rise/span 2.10 is 0.00002 %. So no real horseshoe loses
more than 0.1 % of its radius (0.3 px on a 300 px arch) and the user's own
ribs lose nothing; subdividing further would change the exported Bezier path
that was asked for, so it stays as it is, measured.

Their rib draws nearly straight because their three marks say so: a crown
430 px from a 127 px springing line is a pointed arch at rise/span 2.1, and a
pointed arch that tall has nearly straight limbs. Whether it is the right
shape cannot be known until one of the three families is re-marked -- which
is now what the camera panel says.

19 rows in `two_point_never_looked_at_its_verticals()` and
`arch_lean_keeps_the_crown()`, both ways round: the nave's three families
must refuse in both modes while the arch still draws, and a level camera's
own three points must still come back f = 1100.0 with nothing called
inconsistent. 430 passed, 0 failed.

## 39. "the farther side of the rib still moves outward" -- it is the handle, not the limb

> "when drawing a ribbed vault, the farther side of the rib still moves
> outward, away from the closer side. it should move in the same direction,
> because it's a rib!"

Measured, and the coupling is right. What is wrong is what the handle looks
like it is doing.

**The configuration is the normal one, not a broken arch.** Their two feet
are two capitals one bay apart down the nave and the crown is the boss
between them -- the same configuration §38 verified the lean against. The
curve from one capital over the boss to the next is two half-ribs of two
different ribs, and it really is symmetric about the boss's upright in the
plane those three points span. In that plane it stands tall: span is the bay
(7 m), rise is the boss's distance from the middle of the springing line,
out over the nave and up (sqrt(6² + 8²) = 10 m). rise/span 1.43, and 1.92
for a 12 m boss. Nothing is degenerate about that.

**The gain is large and that is geometry.** Under true cameras, on a 12 x 7 m
bay with the camera 12° up, the mirror moves the partner handle by

| boss above springing | rise/span | handles 0,1 (near limb) | handles 2,3 (far limb) |
|---|---|---|---|
| 2.0 m | 0.90 | 0.89, 0.82 | 1.66, 2.02 |
| 3.5 m | 0.99 | 0.75, 1.01 | 2.05, 1.84 |
| 5.0 m | 1.12 | 0.81, 1.23 | 2.52, 2.19 |
| 8.0 m | 1.43 | 1.08, 1.68 | 3.56, 3.08 |
| 12.0 m | 1.92 | 1.47, 2.33 | 5.11, 4.34 |

so the old comment in `bezOf` -- "a pull on the near limb moves the far one
less, by however much further off it stands" -- is true of a wall arch
(`arch_mirrored_handles` still asserts 0.2--0.9x there) and false of a vault
rib. The comment now says the measured thing.

**The invariant that matters.** Push a handle out, so its own limb bulges
fuller: the partner limb bulges out too, on every handle of every rib tried
(worst +11.8 px of outward component for a 28 px push). The two ways of
making the far handle "move in the same direction" both cave that limb in:
the same page offset by -26.6 px, carrying the hand's move over to the far
handle without the plane's reflection by -14.2 px. One limb out and the other
in is an S, not a rib. `arch_rib_pair_handles()`, 7 rows, 437 passed.

The derived move also keeps the two halves mirror images as curves: mapping
the near half through the arch's own mirror lands 0.2--0.9 px from the far
half for a pointed arch, before and after a drag (the user's arch: 0.6 px, 2.1 px
after). A round arch is one cubic over the whole arc and its two halves are
3.7--48.8 px apart before anything is touched -- that is the single cubic,
which `bez_off` already reports, not the mirror.

So: the mirror is faithful and the comment was wrong, and the comment and the
check are fixed. **The coupling itself was then changed anyway -- see §40.**
Everything measured above still holds; what it does not settle is which
symmetry a rib wants, and that was not mine to decide.


## 40. "the plane of the arch should bend itself" -- carry the move, don't reflect it

> "I need a different kind of symmetry. the plane of the arch should bend
> itself"

§39 answered the report by showing the mirror was faithful. It was, and
that was not the question. A reflection makes the arch **rigid**: push a
handle out and both limbs bulge out together, about the arch's own upright.
That is what a stone arch in a wall does. A rib is not that. The hand that
pushes one limb of a rib is saying the plane the rib stands in has bent, and
a bend pinned at the two feet and the crown runs one way between the near
foot and the top and the other way past it. Both answers draw the same marks;
the page cannot tell them apart; the user says which one the app means.

**The rule.** Take the hand's move off the page onto the arch's plane where
the hand is, and put it back on the page where the other handle stands:

    fwd( back(c_j) + ( back(c_p + d) - back(c_p) ) ) - c_j

with `p = n - 1 - j`. Two homographies instead of one, and the reflection
`Rp` drops out entirely -- `plane_arch` now ships `plane = [H, Hi]` in place
of `mirror`, and `app.py` sends that. Still derived at draw time, never
stored: §38's 362 px stale-plane bug is a property of storing, not of
reflecting, and it comes straight back otherwise.

**What it does.** The partner handle follows the hand's own direction to
within 0.1--9.3° on every handle of every rib tried, at 0.56--0.66x coming
from the near limb and 1.46--1.75x from the far one; on the steep wall-arch
fixture 0.763x and 1.480x at 17--20°. The reflection, on the same ribs, sent
it 0.52x--6.05x and usually the other way across the page -- which is the
report. The far **limb** now caves in by -14.2 px for a 28 px push where the
reflection bulged it out by +11.8 px: the S-shape §39 called wrong is the
bend, and is what was asked for.

**Live, on the user's Worcester marks** (one arch layer, 3-point, a 25 px drag
left on a near handle; canvas rows scanned for the curve's own green, near
limb / far limb):

| row | 364 | 451 | 538 | 625 | 712 | 799 | 886 | 973 | 1060 |
|---|---|---|---|---|---|---|---|---|---|
| near | -31 | -45 | -48 | -48 | -59 | -53 | -45 | -38 | -32 |
| far | -66 | -57 | -48 | -38 | -29 | -19 | -10 | 0 | 0 |

Same sign at every row, decaying to nothing at the far limb's pinned foot.

`arch_handle_pairs()` (renamed from `arch_mirrored_handles`) keeps the
geometry rows that prove the page-fitted cubic is deliberately asymmetric and
now tests the carry; `arch_rib_pair_handles()` measures both rules side by
side on three real bay ribs so the decision stays visible. 435 passed, 0
failed.


## 41. "add an option to add one more point on the curve of the arch"

Two feet and a top leave the arch **nothing to decide**. The centre is forced
onto the springing line (pointed) or onto the crown's own upright (round), and
in both cases a circle through the foot and the apex with its centre on that
line is unique -- which is to say the round/pointed toggle was not a hint, it
was the entire say over the shape. Real arches are not all built either way: a
drop or a stilted arch raises its centres, and the only way to follow one was
to drag the Bezier handles, which is drawing, not measuring. The user's own
Worcester arch carries two hand offsets of 17 and 21 px for exactly that.

**The rule.** One more mark, ON the curve. Three points name one circle, so the
limb is the circle through foot, mark and apex, and the other limb is its
mirror -- an arch is symmetric about its own upright on its plane, so a mark
on either limb is the same mark. The toggle is not overruled, it is answered:
with the mark, round and pointed give curves **0.0000 px** apart.

**Split at the mark**, four cubics instead of one or two. The curve then goes
through it rather than near it (0.000 px), a cubic a quarter is truer than one
a half, and a limb that sweeps past a half turn no longer has a short way
round to get wrong. On the live Worcester arch the drawn path went from
**6.54 px off the true arch to 0.24 px** on the mark being added alone.

**Measured** (`arch_point_on_the_curve`, a real arch imaged at 34 deg off, its
centres raised off the springing line by a known amount):

| centres raised | the forced arch misses | with the mark |
|---|---|---|
| 0.00 m (the app's own) | 1.5 px | 1.6 px |
| 0.45 m | 26.3 px | 1.6 px |
| 0.90 m | 60.1 px | 1.6 px |

so nothing is lost where the toggle was already right, and 37x is gained where
it was not. The residual 1.5 px is the plane cubic against the true arc, which
`pts` has always carried. Marked on either limb: 0.001 px apart. A roof ignores
it, there being no circle in a gable.

**It costs nothing to add.** The mark is put down ON the curve the camera drew,
a quarter along, so the circle through it is the circle already there: adding
one moves the arch 1.3 px (pointed) or 1.7 px (round), which is the cubic's own
error being re-cut, not the arch changing. Hand offsets on the handles ARE
dropped, because the path is re-cut into four and their indices no longer mean
what they meant; the button says so.

Drawn as a dot in a ring -- it moves the arch like an anchor but it is not the
crown and not a handle. Dragged like any mark, cmd-click drops it, and dropping
one of an arcade leaves a hole rather than sliding the next arch's onto it.
Live, a 29 px drag outward lifted the near limb 62 px and carried the whole
curve with it, tapering to 4 px at the far end, with the arch still 0.29 px
from a real circular arch on its plane and rise/span untouched at 0.500 --
which is the point: the mark says what shape the limbs are, the crown still
says how high the arch is. `rise/span 0.500 is a semicircle` is the toggle's
reading, so the tooltip now says that it stops being true once a mark is on
the curve.


## 42. "a fourth vp on the horizon" — the one square to a marked horizontal — 2026-10-11

A third horizontal family is often not part of the camera at all. It is an
object **standing askew** to the room -- a table turned off the wall, an arch
in a wall that is not the room's -- and the question was whether its *other*
horizontal direction has to be marked. It does not. Two world directions at
right angles have vanishing points conjugate in `w = inv(K Kt)`: `u' w v = 0`.
So the partner of `v` lies on the line `w v`, which is the vanishing line of
every plane square to `v`, and a partner that is **horizontal** lies on the
horizon as well. Two lines meet in one point:

    v4 = (w v3) x horizon            core/rhythm.py, square_partner

**Exact**, not fitted: 0.0000 px off the true perpendicular for turns of 8 to
80 deg (tol 1e-8), and 0.0000 px off the horizon (tol 1e-9). Two directions
have no partner and both say so rather than returning a number -- the upright
direction (every horizontal is square to it, so `w v` *is* the horizon and the
two lines coincide) and a partner parallel to the picture (no point on the
page at all). The test is done on normalised lines, so `q[2]` is the sine of
the angle between them and the threshold is in radians rather than in whatever
scale `K` happened to carry.

**The hazard the declaration fixes, measured.** `_camera` took the first two
non-vertical vanishing points in document order. An askew family could
therefore take a wall's place in the camera's own triple, and on a camera built
for the test it came back **f = 769.2 px against a truth of 1400.0 px, verdict
"ok"** -- confidently wrong, no flag. Declaring the family square excludes it,
and the camera is then the room's own to 1e-6 (1400.0000). Of the rest, the two
the camera uses are now the two the page already calls the horizon's definers
(`doc["horizon_vps"]`), not whichever the hand drew first.

**A derived point is hopeless as a position and fine as a direction**, and both
numbers go out. f out by a tenth moves it 35.7 px on the test camera (hundreds
to tens of thousands on real ones), while the **swing** it costs a mark of the
family's own length -- which is all a line drawn to it actually suffers -- is
1.06 px. A ratio of **33.7x**. The readout says so in those words: *a point to
draw to, not a position to quote.* A derived family usually has no marks of its
own and so no length on the page to say what a swing is worth, so the sweep
borrows the partner's lines (`scale_lines`); the *gap* never does, because the
partner's lines run to the partner's point, which is a different question.

**Derived at use time, never stored** -- the same rule the arch's mirrored
handle follows. A stored point would go on answering for the camera it was made
in. It is a *family* whose `vp` the server fills in, so `refRow`, `buildDoc`,
the arch, the ring, the row and the surface all take it with no further change.

**Marked as well as derived**, the gap between the two is the only reading an
askew object can give on the camera, and a weak one -- the sweep says how weak.
When the family's own marks come out parallel on the page there is no point to
compare and the row says that instead of vanishing.

**One horizontal family and the uprights are now a horizon too**, through the
camera (`inv(K Kt) v`), server-side in `_horizon_line` and gated client-side by
`camHorizon`. Without it the third family of a view down a nave could not be
held on a horizon the check path already knew.

**Live, on the user's own Worcester photograph** (three-point, one horizontal
family): picking "square to down the nave" gave the empty layer a vanishing
point at **53566.4, 3381.7** with zero lines marked, over a camera that is
itself undetermined (f anywhere in 3542..6375 px over the spread of the upright
family's vanishing point) -- across which the point moves **94673 px** and
swings a mark **2.78 px**. Two throwaway marks on it then put their own point
50737 px away, swinging them 55.3 px, beyond the 10 px a hand mark is worth
there. Both sentences are on screen, the undetermined camera in warn colour.

Checks: `square_partner` in selfcheck.py, nine rows; 457 passed, 0 failed.

## 43. "add shadows" — a cast shadow is a perspective relation — 2026-10-11

**Three rules, and they are exact.** The light rays all pass through one point
on the page (`l`), the shadow lines on the ground all pass through another
(`s`), and — this is the one that does the work — a light stands over its own
foot, so `l`, `s` and the vertical vanishing point are **collinear**. True for
a lamp in the room and for the sun alike: the sun is just the case whose foot
has gone all the way to the horizon. One construction answers both, and which
one it was is itself a reading on the picture.

Three clicks an object: base on the ground, top, tip of the cast shadow.
Nothing else is needed — `shadow_tip(l, s, base, top)` is two crosses of
crosses, homogeneous throughout, so the sun is not a special case. In
selfcheck the light comes back to **0.013 px**, its foot to **0.036 px**, the
foot sits on the horizon to **0.000 px** and the three points are collinear to
**0.000 px**.

**One free parameter, and a scan cannot find it.** Collinearity leaves the
light one degree of freedom: which line through `vert` it stands on. That
angle is a brutal lever — **1e-5 rad of slack costs 26 px of tip movement** on
the first fixture. Measured on the same marks: a 721-step scan gets the sun to
**92.5 px**, 7201 steps to **19.5 px**, and the construction to **8.4e-12 px**.
The construction holds the foot where the ground lines want it (`fit(grd)` for
a lamp, `fit_on_line(grd, hz)` for the sun), takes the line through `vert` that
follows, and puts the light at the rays' best point on it; a bounded Brent
polish (`xatol=1e-13`) over ±0.05 rad finishes it. **Every polish start must
pass through `vert`** — projecting a line that misses it into the pencil basis
is a nearest in coefficient space, not on the page, and a jittered lamp started
43 px out and stayed there until the starts were fixed.

**What to measure in was the whole difficulty.** Three currencies were built
and two deleted, each killed by a measurement:

- *Tip movement* (how far the crossing point moves) carries 1/sin of the
  crossing angle. A 9 px hand mark is worth **146.2 px** out at the tip, so a
  floor that size waves through a 40 px blunder.
- *Swing about a mark's midpoint* hides a blunder on a short mark: a shadow
  pushed 40 px across its own direction swung **8 px** and was called sunlight.
- **Kept: the perpendicular distance from the marked tip to each of the two
  lines it must lie on.** The same 9 px mark is **34.0 px** here — px of mark
  movement, not amplified. The crossing angle goes out as conditioning
  (`crossing_sin`, with a sentence when any is under 0.15); the tip positions
  go out as *what the drawing would have to change*, never as the test.

**The ladder, strictest first, never gated on the free fit.** marks → a lamp →
the sun, each priced the same way against what a hand mark is worth here. It is
not gated on the free fit because a *constrained* light can beat it on the
maximum: the free fit minimises the algebraic residual over all the lines and
this is a maximum. Calling the fixture's lamp the sun costs **125.8 px**, so
the two are not close.

**Naming the culprit needs least-median-of-pairs.** Two objects determine a
light exactly, so `shadow_stray` fits every pair and keeps the light whose
*median* miss is smallest — the one the majority drew — then names whoever
misses it farthest. Both cheaper methods get it wrong and are pinned as rows:
the worst miss inside one fit says **object 1**, leave-one-out says **object
5**, the truth is **object 3**. The blunder stays inside the fit on every fold,
and the length-weighted solve swings toward it once the longest honest shadow
is removed. Needs three objects; with two there is nothing to outvote.

**The measured blind spot, and it is the right answer.** A tip pushed 40 px
**along** its own shadow on the longest (221 px) shadow reads **8.5 px** —
absorbed, because lengthening a shadow is nearly the same picture as a lower
light. The app is asked for the least change that makes the drawing consistent,
and that *is* the least change. It is a selfcheck row so it stays known.

**Refusals draw nothing.** `tip_at` is emitted only from a branch that actually
found a light; the numbers (`tip_px`) go out either way, the positions do not.
A correction drawn from a refusal is a confident wrong answer at full opacity,
which is the most expensive failure this project has had.

**Live, on `testdata/shadowtest.png`** (synthetic, f 1400, pp 600/450, five
posts, sun at 35°/200°; the truth is light **1599.7, -935.9** and foot
**1498.4, 335.1**, and `shadowtest.marks.json` beside it marks four of the five
objects). Four marked objects plus a fifth clicked by hand in the browser:
**consistent / the sun**, light_vp 1585.6, -914.5 and foot 1487.0, 334.9, with
the ladder reading marks 1.0 / lamp 1.0 / sun 1.3 px against an 8.7 px floor —
the hand-clicked object costs 1.4 px. With all five from the file: light_vp
**1599.700, -935.900** and foot **1498.400, 335.100**, exactly the truth.

Dragging one tip 60 px further along its own shadow: **inconsistent**, worst
object **5** (the one moved), "left out of the fit, its own tip is 39 px off
the light the others agree on", and no rings drawn. Dragging it back to ~25 px
out: **consistent — a lamp, at that foot**, because held over its own foot it
costs 7.6 px but driving that foot to the horizon costs 9.8 px. That is the
blind spot above, shown from the other side: the app does not pretend it is
still the sun, it moves the light into the room and says where it stands.

The three marks close as a triangle on the canvas, in the PDF (`PAIR_KINDS`)
and in the server overlay — base→top is the upright, top→tip is the ray,
tip→base is the shadow, all three meaningful. The upright itself is checked for
free from the same marks (`upright_move_px`): a shadow read off something that
is not upright is being read against a ray that does not belong to it.

Also fixed here, found while verifying: `fitView` with a zero-size canvas set
`S.view.k = 0` and blanked the picture for good — one guard, `if (!(k > 0))
return`, so a picture opened before the pane has layout stays visible at the
previous zoom.

Checks: `shadows` in selfcheck.py, 23 rows; 480 passed, 0 failed.

## 44. "step 1" — a camera in, a camera out — 2026-10-14

Asked: "think whether we can have some 3d capabilities like reading a blender
file?", then "check whether the app actually works. go on to step 1". Step 1
was the smallest thing that is 3D and still *this* app: **a camera in and a
camera out**, nothing modelled.

*In* — a `.blend`, an `.fspy` or a plain `.json` becomes a **reference
camera**: its three vanishing points and horizon are drawn on the picture, and
the marks are priced against it in the one coin the app already pays in, **px a
line must move**. *Out* — the camera the marks agree on, written as a Blender
script with the picture behind it. Both are in `core/camera_io.py`; the pricing
is `consensus.against_vps`.

**Verified first, and three things were broken.** The launcher took a conda
`python3` whose scipy would not import (`.venv` now links to the shared
`../.venv-apps`, which the launcher already preferred); `fitView` blanked the
picture for good on a zero-size canvas (one guard); and opening a second
picture left the first one's marks on it (`openFresh` — a different picture is
a different document, and it asks before discarding layers). An instance
started before this needs `./run_mac.command` again and a page reload.

**The conventions are where a mirror hides, so each is a row, not a comment.**

| | convention | the trap |
|---|---|---|
| picture | x right, y **down**, px, origin top-left | fSpy is centre-origin, half-width units, y **up** |
| camera frame | x right, y down, z forward | Blender's is x right, y **up**, looks down **−Z**: `diag(1,−1,−1)` |
| lens | `sensor_fit AUTO` lays the sensor along the picture's **larger** side: `f_px = lens·S/36`, `S = max(W,H)` | `lens·W/36` is 31 % off on a portrait picture |
| shift | `cx = W/2 − shift_x·S`, `cy = H/2 + shift_y·S` | both divide by `S`, never by the side they move |

Blender's own projection of seven known points agrees with the pinhole formula
to **7.6e-5 px** (1280×960), **9.6e-5** (884×1280, portrait) and **6.4e-5**
(1000×1000, shifted) — float32 is the floor. A Workbench render's sphere
centroids sit **0.19–0.40 px** from the formula. `.blend` is read by running
Blender in the background (`--factory-startup -b -Y`, scripts off) and parsing
one `@@JSON@@` line; nothing was pip-installed.

**Two independent files agree.** `Untitled.fspy` (fSpy's own numbers) gives f
**1054.167 px** = 29.648 mm on 36, pp (640, 480), VPs (−201.3, 817.9),
(2085.4, 790.1), (598.8, −2911.5). The scene camera *inside*
`pteranodon.blend` — an object named `Untitled.fspy`, so presumably put there
by fSpy's add-on, and in any case Blender's representation, not fSpy's — gives
**1054.168 px** and the same VPs to 0.1 px. Neither file was modified.

**A reference is a claim, so it is priced, not believed.** `against_vps` gives
each family the axis it costs least on (one family → one axis, least total
movement) and prints the runner-up beside it: a family that fits two axes about
equally says so (`ambiguous` when the margin is under a hand mark) instead of
being quietly placed. One family is enough here, which `fit` cannot say — the
reference supplies the other two. A vanishing point at infinity (a level
camera's uprights) is a *direction*, and its corrections are now drawn: they
were drawn nowhere before.

**Up is not in a camera file.** `pick_up` takes the axis with the largest
vertical component in the camera frame and returns `None` when the top two are
within 0.1. It is a **prior** — wrong beyond 45° of pitch — and a family
flagged vertical overrides it; one/two-point perspective fixes axis 1 as
upright by definition. The script header says which was used.

**Out is the readout's camera, not a second fit.** `/api/camera_out` takes the
same request as `/api/consensus` and writes what `fit` returned, so what the
page showed and what Blender receives cannot be two fits. It refuses where the
readout refuses: one-point (f is absent from the geometry, "not for want of
marks"), a refused lens, no `axes_cam`. The origin is 10 units ahead of the
camera and *says* it is a choice; yaw is quarter turns because nothing in a
picture says which wall is front. A trim drops the reference with a note —
a homography is not a pinhole camera, and a camera in the wrong pixels would be
priced as if it were in the right ones.

**The finding about the fixture, and a hole in the verdict.** Marked *in fSpy*
(2 lines for each of three families), the verticals against the file's own
camera must move **29.61 px** worst, **27.46** median — three hand marks, with
the horizontals at **0.00**. The file's camera is built from the two
horizontals only (`TwoVanishingPoints`); its third VP is above the picture at
(598.8, −2911.5) while the verticals drawn in the file converge *below*, near
(627, 2162). The free fit of the same six lines is exact and absurd: f **644.7**
px, pp (621, 1692) — **1212 px** below the centre — 41.3° from the file's
camera, and it said **"already consistent"**.

That verdict was a claim with nothing under it. Each mark gives one angle and
the camera has `len(sol.x)` numbers; with none to spare any marks fit, and 0 px
is arithmetic. `fit` now reports `parameters` and `redundancy`
(`n_lines − parameters`), and **a fit with redundancy ≤ 0 whose lines are all
within a hand mark returns `undetermined`**, with the sentence "6 lines for the
camera's 6 numbers … Mark a third line in a family to test it". The camera is
still written (it is the one those lines define; fSpy does the same) and the
script header says `NOT TESTED`. Removing the gate fails exactly the three rows
that pin it. Nothing in the 480 existing rows depended on the old answer.

Not built, on purpose: using the reference in place of the marked VPs for the
other checks (`use_reference`); an overlay of the model; a lamp's height from a
cross-ratio; refusing an obtuse vanishing triangle on the way in; reflections;
the circle of view in degrees stays an offer.

Checks: `camera_in_out` in selfcheck.py, 62 rows — the conversions with no
program in the loop; the fSpy fixture; `against_vps` (the true camera, relabelled
axes, 8° of yaw out, a level camera); the routes; and an export run through a
real Blender and read back (skipped, and said so, when none is installed).
Three broken exports — lens on the width, `shift_y` negated, axes not flipped —
fail 5, 4 and 4 rows. **542 passed, 0 failed.**

## 45. "make it possible to change colors for the lines" — 2026-10-14

*The request:* "sometimes they are invisible against a drawing".

The palette is ten saturated colours handed out in turn (`COLORS`), with no
dark ones and no white, so on pale paper the yellow vanishes and on a
watercolour nothing is guaranteed. `L.color` already existed on every layer
and already reached the canvas, the layer swatch, the saved marks, undo and
the PDF; what was missing was only a control. So the change is a control, not
a model:

- **A layer's colour** — a well and its hex code (`LK.hexBox`, the shared
  control) in the layer editor. Live while the picker is dragged, one undo step
  when it is let go. Saved with the marks, as before.
- **The two lines that are not a layer** — the horizon (green) and the
  correction lines (red) had fixed colours and could vanish the same way, so
  each has a well under the picture. The reference camera's own X/Y/Z and violet
  stay fixed: their colours are a convention (fSpy's), not a styling.
- **The PDF follows the page.** The horizon's colour travels with the document
  (`horizon_color`); a layer's already did. A colour that was only a screen
  choice would be undone by the first export.

Not built: persistence of the two overlay colours between sessions (they reset
like the other view toggles), line width, and a contrasting outline for a line
that crosses both light and dark paint — one colour per layer cannot be seen
on both, an outline could, but it would cover 1–2 px of the paint the mark is
measured against, and it would collide with the white halo that already means
"stray line".

Checks: four rows in `pdf_export` read the PDF back — a picked layer colour is
the one stroked, the palette colour it replaced is gone, the horizon takes the
colour it was given, and is green when none is. The horizon row fails on the
old export. On the real page (launched as the user launches it): typing a hex
code recolours the canvas, swatch and well together; one cmd-Z restores the old
colour and cmd-shift-Z the new; Save marks carries it; the PDF request carries
`horizon_color` and the layer colours and the server answers 200; canvas pixel
counts move 1325 → 0 (old horizon colour) and 3859 → 0 (old correction
colour) as the wells change. One screenshot was a stale frame and disagreed with
the pixel counts; a second frame agreed with them. **546 passed, 0 failed.**

## 46. "auto circle insertion into a square formed by two line families" — 2026-10-14

*The request:* put the circle into a square two families of lines make, and
let a second line of a family snap to a square when two lines of another
family are there.

One piece of geometry, used two ways. Two vanishing points name two
directions and not the angle between them, so a parallelogram is as good as a
square to them — **the camera is what says "square"**, and where it cannot the
answer is `undetermined`. The circle itself needs no camera at all.

- **The circle.** The four corners of a cell fix one homography from the unit
  square, and the inscribed circle is carried along with it (`cell_ring`).
  It touches each side at the image of its middle, which is *not* the middle
  of the side on the page — 0.7–1.5 px apart on a 173 px tile, 3.4–14 px on a
  460 px one — so a page ellipse fitted to the quad would be wrong by that
  much. On an ellipse layer, "circle in a square", click inside a cell: the
  cell is the one round the click between the nearest line of each family on
  each side (`square_cell`); the click is kept as an anchor, draggable, and
  `⌘`-click deletes it. A click with no cell under it is withdrawn, not left
  as a dot.
- **Whether the cell is a square** (`cell_reading`). With `M = K⁻¹H`, the
  length ratio of its first two columns is the cell's aspect and the angle
  between them its angle — no plane named, no distance known. It is read at
  the fitted f and at both ends of the focal span, and the verdict (`ok` /
  `off` / `undetermined`) is against the layer's resolve, 5 % by default;
  where the span straddles it, the circle is drawn and the card says it is not
  settled. With no camera the ring is still drawn and the card says it is
  unearned.
- **The snap** (`square_completions`, `POST /api/square_snap`). Given a line
  of this family and two of another: the feet are where the first meets the
  other two, the diagonal out of one foot runs to `K(d_a ± d_b)` and meets the
  second line, and the far side is the line from this family's vanishing point
  through that meeting — one cross product after another, nothing to converge,
  two answers (one each side). A drawn line within a hand mark and a half of
  one (15 px on a 1400 px picture) is moved onto it; further out it is
  somewhere else on purpose and left, and the answer says how far the nearest
  square was. A family with fewer than two lines gets its vanishing point from
  the camera and the other family, as `square_vp` does, and says so. A pair
  not at right angles in the world (more than 5° off) is refused with the
  angle. The snap **edits** marks, as tidy does, so it is a checkbox on the
  family ("snap new lines to a square") and off until ticked; so is "fill the
  squares a snap closes", which puts the circle in the square a snap just
  closed, once.
- **What the camera's slack costs, in the answer.** The focal span is the
  marks' own range when the camera has one and a tenth either side of f when
  it has not — and *says which*. What f moves the side by travels with the
  snap (2.5 px in the test scene, against a hand mark of 10 px), and a snap
  is refused where f moves it by more than a hand mark: moving a line toward a
  place no more certain than the line would be an edit with nothing behind it.
  Also refused, with the reason, where one end of the span has no such
  square.

Measured, on a synthetic floor of unit tiles under f = 1400 px, 1400 × 1000:
the two completions are exact to 1e-13 px; the ring lies on the projected
circle to 0.023 px (a bound set by the 8192 samples of the truth) and touches
the sides at the images of their middles to 1e-13 px; a square reads 1.000000
at 90.00°, a 1 × 1.5 rectangle 1.500, a 70° rhombus 70.00° with equal sides;
a focal length a tenth out reads the same true square 1.094 : 1. A line 6 px
off at one end and 5 px at the other snaps to within 0.005 px of the true
side; a line 40 px off is left; one line plus a derived point lands on the
same side to 0.005 px. A near tile (side 460 px) is refused, f moving it
13.2 px; a strongly oblique one, because one end of the span has no square.

**The honest limit.** With three families the camera has no measured f range,
so ±10 % is assumed and flagged, and a true square then reads `undetermined`
at the default 5 % (aspect 1.000, range 1.000–1.104) and `ok` at 12 %; a 1 × 2
cell is `off` at 2.0 : 1 and no focal length in the span makes it square. A
measured range for the three-family camera would sharpen the snap and the
reading together; it is not built.

Not built, on purpose: squares from lines that are not neighbours (only the
nearest flank each way is used); a live ghost of the snap while the line is
dragged; a "fill every cell" button; comparing a rim marked round a drawn
circle against the inscribed one (a marked rim is measured as before, and the
Check says a cells-only layer is round by construction rather than checking
the drawing against itself).

Checks: `circle_in_a_square` in selfcheck.py, 58 rows — completions, ring,
reading, the cell finder, the snap through its route (ok / hand error / too
far / derived point / askew / no camera / no line / one line / near tile /
one end gone), the cell route (verdicts at two resolves, a rectangle, no
camera, no cell), and the Check route. Six deliberate breaks — a diagonal at
0.9 of the second side, corners in bow-tie order, aspect upside down,
farthest lines for nearest, a page circle for the ring, a cells-only layer
re-checked as a circle — fail 10, 14, 2, 9, 3 and 1 rows. Driven on the real
page, launched as the user launches it: the ring sits on a painted circle; a
snap lands to 0.03 px and refits the family; the closed square fills; a
dragged anchor refits; undo, redo and save → load round-trip the anchors; the
PDF carries the rings; Check and Consensus read the same camera (f 1400.000,
principal point 700 / 500). The page found five things the routes had not: a
second circle where a snap's square already held one; a vanishing point left
stale by two quick deletions (an async refit answering late); undo and redo
cut by a counter kept on the layer and a commit inside the refit; a Check
message that read a cells-only layer as "five marks needed"; and the assumed
focal span worded as if the marks had given it. **604 passed, 0 failed.**

## 47. "can we implement fisheye perspective?" — measured, not built — 2026-10-14

A feasibility study on synthetic scenes with known truth (1280 × 960, a hand
mark ≈ 9.3 px, an equidistant lens with f = 509 px unless stated, which is
180° over the diagonal). Nothing was added to the app.

1. **Today's fit says nothing wrong, silently.** Straight two-point lines
   marked on a fisheye drawing are read by the pinhole camera as `already
   consistent` in 28 of 30 scenes (the other two `rectifiable`); the axes it
   returns are off by a median 10.1° (p90 16.5°, worst 33°), and its f of
   435 px claims a 123° diagonal field for a 180° lens. Marking only the
   middle of the frame is no better in verdict — 30 of 30 consistent — and
   2.4° off. This is a wrong answer without a refusal, and stands whether or
   not a fisheye mode is ever built.
2. **A lens-aware fit recovers it.** A marked curve is a set of rays that must
   lie on one plane through the camera's centre, and the planes of one family
   share the family's direction. With the centre pinned at the picture's, 12
   curves × 5 marks, f from 400 to 1500 px and 3–9 px of mark noise: f to a
   median 0.3–2.2 % (p90 0.8–6.8 %) and the axes to 0.16–0.73° (p90
   0.3–1.3°); from as few as one curve a family (three curves) × 3 marks at
   f = 509 and ±3 px, f 0.9 % and axes 0.9°.
3. **It refuses when ill-posed.** Of 600 fits (one or two curves a family
   with 3 or 5 marks, and four with 5), 20 came out more than 10 % off in f,
   and all 20 were flagged by the fit's own standard error on f (over 10 % of
   f), by f sitting at its floor, or by no redundancy; two good fits of 580
   were flagged by mistake. The wildest ran to 15 617 % and said so.
4. **The lens model is a question the marks can answer only partly.** With the
   truth stereographic, the right model won in 40 of 40 at 0–3 px noise and
   36 of 40 at 9 px (rectilinear won the rest). A wrong centre by 40 px costs
   f 1.4 % median and shows as 4.9 px of rms on marks that were ±3.
5. **Centre free versus pinned:** free puts the centre 5 px off (p90 8) and
   the axes 0.56° against 0.20°, and f no worse; pinned is better conditioned
   and is the default a build would take.

A build, if wanted: A) the core — lens models, the sphere fit, the checks,
the refusals; B) page marks of three or more clicks per curve (a straight
line is no longer two clicks) and drawing the fitted curves back; C) Blender
camera out with a fisheye lens. Open questions put to the user: photographs
through a fisheye lens, or hand-drawn curvilinear perspective; polyline marks
acceptable; camera-out in the first pass.

## 48. "build top view" — the drawing from above — 2026-10-29

*The request:* "if vanishing points and horizon are established, it should
translate onto a top view in a small window in the bottom right".

A top view is the plan a perspective is set up from on paper: the eye, the
picture plane seen edge-on as a line, and from the eye a ray parallel to each
family, meeting the picture plane at that family's vanishing point. All of it
follows from the camera the checks already use, so nothing new is solved:

- **`core/rectify.plan`** — the ground frame is `frame()` with the horizon's
  two points and the principal ray as "up", so the plan's forward is the line
  of sight laid flat. A page point goes onto the picture plane as
  f·K⁻¹[x, y, 1] and is read off that frame. The eye stands f / cos(pitch)
  back from where the picture plane meets eye level; the picture's width there
  is where the horizon crosses its sides, which gives the field of view at eye
  level, off-centre principal point included. The uprights are left out: from
  above they are a point.
- **`/api/plan`** — refuses where `_camera` refuses (one-point; three-point
  with no upright family), and where the camera is only a guide (two-point
  under uprights that contradict a level camera) it draws the plan,
  `undetermined`, dimmed on the page. The focal slack is carried through as a
  range of fields of view and eye distances; with no range from the marks it
  is the usual ±10 %, said to be assumed.
- **The page** — a 230 px inset anchored above the bar under the picture,
  toggled by **top view** there. It appears once there is a horizon (two
  horizontal families, or one and the uprights) and is asked for again
  whenever a vanishing point moves; an answer for points that have moved since
  is dropped, and none of it enters the undo history. The picture plane and
  the picture's span take the horizon's colour, each ray its family's, and the
  eye's possible distances are a pale bar through the eye. The label is
  `top&nbsp;view`: as two words it wrapped on a narrow window and made the bar
  a line taller.

Not built: footprints of marked things on the ground (the plan of the room,
not only its directions); a family whose lines stay parallel on the page,
which has no vanishing point and so no ray, though in plan it runs along the
picture plane; the 60° cone of comfortable vision.

Checks: `top_view`, 25 rows. Each number is held against the world the
synthetic camera was built in (directions to 1e-9°, distances to 1e-6 px,
level and pitched-rolled-off-centre), a slope runs where its level run does,
and each refusal is reached through the endpoint. Seven deliberate breaks were
each caught: no left-right flip, f in place of f / cos(pitch), a symmetric
field of view, forward taken along a family, a slope's mark left off the
picture plane, a guide passed as measured, no focal slack. The flip was caught
only once a row fed the horizon's two points the other way round. On the real
page (launched as the user launches it): the fSpy fixture gives 60.6°
(58.9–62.3°) as three-point and 60.3° as a two-point guide, dimmed and
labelled, with the vanishing points 29.5 px apart between the two cameras;
4.jpg as three-point says "no top view: … no family is marked as the upright
one", and as two-point gives 38.5°, which is the f = 1266 px the check reports
across 884 px. The toggle hides and shows it; no console errors. **629 passed,
0 failed.**

### 48.1 "when real sizes are specified, the top view should give a real scale" — 2026-10-29

The plan's units are the picture's own pixels, so the picture's declared real
size (the Real size panel) is already its scale; nothing new is measured. The
endpoint now goes through `_add_mm` like every other readout, so the plan's
lengths carry millimetre twins: `distance_mm` and `mm_per_px` beside
`distance_px`, and `distance_range_mm` beside `distance_range_px`. Both had to
be renamed to end in `_px` first: the plan was the one readout that broke the
naming the twin pass relies on. With a size given, the inset draws a bar of
1, 2 or 5 × 10ⁿ mm, at most 60 px long, at its lower left and writes the eye's
distance beside the line of sight; the tooltip gives the range. The guide plan
is to scale too, dimmed with the rest.

The distance written is the plan's own: the eye to where the picture plane
crosses eye level, f / cos(pitch). The camera readout's "the eye belongs …
from the picture" is f, measured square to the sheet; the two agree only for
a level camera.

Two page faults found on the way: typing a size repainted nothing unless a
consensus was on screen, so the plan would not have changed until the next
mark; and the inset drew its eye and labels in white, which vanished on the
light theme. It now draws in the panel's own text colour and repaints on a
theme change.

Not built: a scale for the scene from a known-size layer, which would put the
plan in metres of the room rather than millimetres of the sheet. It needs the
marked thing placed on the plan (the footprints left out of §48) before a
metre on it means anything.

Checks: three rows in `top_view` — no millimetres without a size; with
35 cm across 1400 px, the eye's distance at 4 px to the mm, its range, and
`mm_per_px` 0.25 on the measured plan and on the guide. Each fails on its own
break: the twin pass dropped on the measured plan or on the guide, width and
height swapped, the range left misnamed. On the real page: the fSpy fixture
declared 42 cm wide gives 0.328 mm per px, the eye 359 mm back (347–372),
and a 200 mm bar; in a magnified screenshot the line of sight and the bar
measure about 180 and 100 px, the 1.8 to 1 that 359 to 200 should give;
clearing the size takes the scale off again; the
light theme draws in dark ink. **632 passed, 0 failed.**

## 49. "can we improve automatic edge detection?" — joined pieces, families by length, merges that agree — 2026-10-29

*The request:* "can we improve automatic edge detection?"

Detection stays what §1 and §12 made it: candidates the user takes or ignores.
The question was how many right ones it offers, and how close they land.

**Two benchmarks.** Real pictures, with nine vanishing points known another
way: shadowtest.png's uprights, across and into-the-picture families (from its
marks; the picture is synthetic, §43) and its shadows' foot (1498.4, 335.1);
the fSpy camera's X and Y for fspy_untitled.jpg (its Z disagrees with the
drawn columns), read inside the sheet because the paper's edge runs along
the horizon; 4.jpg's two walls from its marks and the vertical from §5. Error
is the angle between the rays K⁻¹v; a point is found under 2°. And 40
synthetic pictures, 120 vanishing points: boxes and long lines drawn as broken
pencil or wash strokes of mixed ink and width, with ellipses, stray strokes,
grain, blur and JPEG. There a family no vanishing point lies within 3° of is
junk.

**What was wrong.**

- *LSD leaves crumbs.* On 4.jpg, of the line it finds in pieces of 10 px or
  more, 81 % is in pieces under 40 px (44 645 px against 8 479 px in pieces of
  40 or more), and the minimum length threw them away.
- *Each segment was one vote, and a family needed twelve.* shadowtest's ten
  pole edges, 166–255 px each, were never offered. Pairs were drawn uniformly
  and refits were unweighted, unlike every marked family's fit.
- *The merge was the big family's to decide.* The pooled median stays low
  whatever joins a big family. On 2.jpg it pooled a family whose own lines sat
  4.23° from the pooled point (on the fSpy picture 2.28°), and once the
  families were found by length it pooled the curtain folds of 4.jpg into its
  verticals, moving them 1°.

**What changed** (`core/vp.py`; `/api/detect` passes the picture's size):

- `lines` drops pieces under 0.6 % of the diagonal, which have no direction to
  join by, then `_join`s the rest. Two pieces join when they run within 2° of
  each other, end to end with under 1.5 % of the diagonal between them (2 px
  of overlap at most), and one straight line passes within 1 px of every end
  of every piece joined so far. Testing only the newest pair would walk round
  a gentle arc. The minimum length applies afterwards.
- `vps` draws and scores lines by length, refits by length (`fit`, as marked
  families are), and stops at a family holding under 0.7 picture diagonals of
  line, or under 4 lines.
- `merge` pools two families only if the pooled median is under tol (as
  before) **and** their two vanishing points lie within 2·tol, 3°, of one
  direction, read as a lens as long as the picture's diagonal sees them.
  Families come out sorted by total length.

Synthetic, 40 pictures:

| | found of 120 | error mean / median | families a picture | junk | segments a picture | family median length | warned short |
|---|---|---|---|---|---|---|---|
| before | 101 | 0.36° / 0.27° | 2.7 | 3 | 123 | 53 px | 37 |
| after | **104** | **0.20° / 0.10°** | 2.9 | 5 | 154 | 82 px | 1 |

The floor of line a family must hold, swept with everything else as shipped:
0.5 diagonals finds 105 with 18 junk, 0.6 finds 104 with 9, **0.7 finds 104
with 5**, 0.8 finds 101 with 2, 1.0 finds 100 with 2. At 0.5 the junk
families have a median of 8 lines and 0.59 diagonals of line; the real ones
have 40 lines and 2.76 diagonals. The synthetic set never splits a family at
any of these floors, so `merge` has nothing to do there.

Real pictures, degrees from the known point:

| | before | after |
|---|---|---|
| shadowtest uprights | not offered | **0.40** |
| shadowtest across | 0.70 | 0.39 |
| shadowtest into the picture | 0.07 | 0.04 |
| shadowtest shadows | not offered | not offered |
| fSpy X | 0.95 | 0.85 |
| fSpy Y | 1.92 | 1.57 |
| 4.jpg right wall | 1.78 | 1.82 |
| 4.jpg vertical | 0.17 | 0.40 |
| 4.jpg left wall | not offered | not offered |

That is six found before and seven after. The 4.jpg vertical is 0.23° worse,
and the cause was not traced. Detection takes longer: 0.43 → 0.60 s a
synthetic picture, 0.39 → 0.82 s on 4.jpg.

**The merge, pair by pair.** Every pair of raw families with a pooled median
under tol, on the seven test pictures, with the angle between their points:
4.jpg's right wall and its table and frame edges, 2.37° (one family, pooled);
3.jpg's split, 2.89° (pooled); 1.jpg's drawn verticals and the keystoned paper
edges of the untrimmed photograph, 5.75° (kept apart; trimming removes the
edges anyway); 4.jpg's verticals and the curtain folds, 15.3° (kept apart);
2.jpg's swallow, 63.4° (kept apart). Residuals cannot tell the curtain from a
true split: pooled, the curtain's own median is 2.23° and the table's 2.14°.
Only the direction separates them, which is why a per-family median under
2·tol was tried as the guard and dropped.

**Tried and dropped: folding a stroke's two edges into one.** LSD returns both
edges of a drawn stroke, 3.1–3.7 px apart (126 pairs on 1.jpg, 143 on 2.jpg,
157 on the fSpy picture). Folding each pair into its centre line would have
tidied the candidates, but with families assigned by the true point it doubled
the error, median 0.10° → 0.22°. So every stroke is still offered as its two
edges.

A slip worth recording: the fSpy picture is read inside the sheet, and its
vanishing points were first compared in the crop's coordinates against the
whole picture's camera. Every variant read about 2.5° off there, and the
fixture was nearly written off as weak. Shifted back, X is 0.85° and Y 1.57°.

Not built: the left wall of 4.jpg and shadowtest's shadows are still not
offered. The shadows hold 0.58 diagonals of line (10 lines of median 87 px),
under the floor. A floor of 0.5 finds them at 1.70°, at the price of 18 junk
families on the synthetic set instead of 5. The direction test assumes a
normal lens, so on a very wide picture the halves of one family could read
more than 3° apart and be offered twice.

Checks: `detection`, 13 rows.
- The join: a line in 12 pieces comes back whole, 595 px end to end. The two
  edges of a stroke stay two, and a wider gap stays open. Round an 8000 px
  radius the longest join is 245 px, against the 358 px that 1 px allows.
- The merge: a curtain-like family 15° off is kept apart, though the pooled
  median, 0.89°, would take it alone. A split 2.4° apart is pooled.
- End to end through `/api/detect`: shadowtest's three families within 0.6°,
  the uprights among them; 4.jpg's verticals within 0.6°, and its right wall
  offered once, not as halves.

Two deliberate breaks were each caught. Opening the direction gate to 20°
fails the curtain rows (4.jpg's verticals go to 1.42°). Dropping the
straightness test fails the arc row (545 px) and three end-to-end rows.

On the real page (`run_mac.command` on a spare port):
- 4.jpg: 256 segments, four families merged to three. The verticals are 68
  lines aimed at (405, −49893), median 64 px and not warned (§12's 54 lines of
  58 px were). The right wall is 62 lines at (3948, 907). The curtain folds
  are 23 lines, warned. Taking the right wall puts 62 lines in a family layer,
  fitted to (3948, 907), ok.
- shadowtest.png: 142 segments, three families. The 10 uprights, median
  184 px, are drawn along the posts' edges.

**667 passed, 0 failed.**

### 49.1 "can we manually adjust the merging and other parameters of automatic detection for different kinds of images" — 2026-10-29

Four of §49's numbers are now on the page, under *detection settings* below
Detect edges (closed until opened). Each starts at the measured default:

- *spread*, 1.5°: how far a line may miss its family's vanishing point
  (`tol`, in both `vps` and `merge`).
- *merge within*, 3°: how close, as directions, two families' points must lie
  to be offered as one (`apart`). 0 merges nothing.
- *least line*, 0.7 × the diagonal: the line a family must hold to be offered
  (`floor`).
- *join gaps*, 1.5 % of the diagonal: the widest gap `_join` closes (`join`).
  0 joins nothing.

They reach `/api/detect` as `tol`, `merge_deg`, `floor` and `join_pct`. One
left out is the default, and one that is not a number gets the usual 400. On
the page a blank field sends the default but 0 is sent as 0, since for two of
them 0 is a setting; *defaults* puts all four back. *merge within* no longer
follows spread: §49 had it at 2·tol. The settings last as long as the page and
are not saved with the marks. The toast now gives the count before merging
when the merge took any.

**What spread and join gaps do**, swept one at a time with the rest at the
defaults, on §49's two benchmarks (found under 2°, junk with no point within
3°):

| | synthetic found of 120 | junk | error median | real found of 9 | 4.jpg right wall | 4.jpg verticals |
|---|---|---|---|---|---|---|
| spread 0.75° | **112** | 1 | 0.08° | 7 | **0.90°** | 0.67° |
| spread 1.0° | 109 | **0** | 0.08° | 7 | 1.57° | 1.22° |
| spread 1.5°, default | 104 | 5 | 0.10° | 7 | 1.82° | 0.40° |
| spread 2.0° | 105 | 6 | 0.12° | 6 | 2.46° | 0.62° |
| spread 3.0° | 98 | 11 | 0.14° | 7 | 1.83° | 1.18° |
| join gaps 0 | 89 | 1 | 0.26° | 7 | 1.78° | **0.17°** |
| join gaps 0.75 % | 102 | 2 | 0.12° | 6 | 2.96° | 0.55° |
| join gaps 1.5 %, default | 104 | 5 | 0.10° | 7 | 1.82° | 0.40° |
| join gaps 3 % | 109 | 3 | 0.10° | 7 | 1.99° | 0.89° |

- The synthetic lines are drawn true, and there a tighter spread finds more
  with less junk. On 4.jpg, the one hand drawing with known points, it is
  mixed: 0.75° halves the right wall's error but moves the verticals from
  0.40° to 0.67°. The default stays at 1.5° until more drawings with known
  points say otherwise. The tooltip now says what was measured: lower it for a
  ruled or rendered picture, raise it when a loose drawing's lines fall into
  no family. The second half is the mechanism, not a measurement.
- A wider join finds more broken strokes (3 % finds 109) and moves 4.jpg's
  verticals. That traces §49's open question: with join gaps 0 and everything
  else as shipped, the verticals are 0.17° off, as before §49.
- *merge within* and *least line* were measured in §49. A split family's
  halves come 2.4–2.9° apart and the curtain 15°; least line 0.5 offers the
  shadows, with 18 junk families against 5.

Checks: six rows added to `detection`, each through `/api/detect`.
- merge within 0: 4.jpg's right wall comes back as its two halves.
- least line 0.5: shadowtest's shadows are offered, 1.70° from the sun's foot
  (not offered at 0.7).
- join gaps 0: shadowtest gives LSD's own 230 segments, against 142 joined.
- spread 0.5°: 4.jpg's verticals are found from 32 lines, against 68 at the
  default.
- A setting that is not a number is refused.

Each setting dropped on its way to the detector fails its own row: merge
within (1 family where 2 are expected), least line (20.0°), join gaps (142
segments) and spread (68 lines).

On the real page (`run_mac.command` on port 8832; 8831 was the other
session's):
- 4.jpg at the defaults: "256 segments, 3 families (4 before merging)". At
  merge within 0: "256 segments, 4 families", the right wall offered as
  43 lines at (4173, 951) and 19 at (4223, 786). Emptied, the field sends 3
  and the three come back. *defaults* resets all four.
- shadowtest.png: three families at the defaults; at least line 0.5 a fourth,
  10 lines at (1556, 319), the shadows.

A layout fault found on the way: labkit's `width: 100%` on number fields beats
`.num`, so each setting's unit wrapped onto a line of its own. The settings box
now sets its own widths. The min length and Real size fields have the same
fault and were left as they were.

**673 passed, 0 failed.**

### 49.2 "go add presets" — 2026-10-30

*detection settings* now opens on a *picture* choice that fills the four
settings:

| preset | spread | merge within | least line | join gaps |
|---|---|---|---|---|
| freehand | 1.5° | 3° | 0.7 | 1.5 % |
| ruled or rendered | 0.75° | 2° | 0.7 | 3 % |
| sparse | 1.5° | 3° | 0.5 | 1.5 % |

Freehand is the defaults. A setting changed by hand turns the choice to *your
own*, and choosing a preset again puts its values back, which retires the
*defaults* button.

Measured together, on §49's benchmarks:

| | synthetic found of 120 | junk | error median | real found of 9 | 4.jpg right wall | 4.jpg verticals | shadows |
|---|---|---|---|---|---|---|---|
| freehand | 104 | 5 | 0.10° | 7 | 1.82° | **0.40°** | — |
| spread 0.75°, join 1.5 % | 112 | 1 | 0.08° | 7 | 0.90° | 0.67° | — |
| spread 0.75°, join 3 %, merge 3° | 114 | 0 | 0.09° | 7 | 0.76° | 1.58° | — |
| **ruled or rendered** | **114** | **0** | 0.09° | 7 | **0.76°** | 0.85° | — |
| **sparse** | 105 | 18 | 0.10° | **8** | 1.82° | 0.40° | **1.70°** |
| least line 0.5, spread 1.0° | 110 | 12 | 0.08° | 8 | 1.57° | 1.22° | 1.78° |
| least line 0.5, spread 0.75° | 113 | 7 | 0.08° | 7 | 0.90° | 0.70° | — |

- *Ruled or rendered.* A tight spread and wide joins together find 114 of 120
  with no junk, 57 of 60 for each stroke style, pencil and wash. The synthetic
  set never splits a family, so *merge within* changes nothing there. On
  4.jpg at 3° the verticals took in a family 2–3° off them and went to 1.58°;
  at 2° they stay at 0.85°. On the rendered shadowtest its families are where
  they were or closer (across 0.39° → 0.33°); on the fSpy picture X moves
  0.85° → 0.98°. It is not for a painting: on 4.jpg it offers five families.
- *Sparse.* Least line 0.5 alone keeps every default result and adds the
  shadows, 8 of 9, at 18 junk families against 5. A tighter spread with it is
  a trade: 12 junk at 1.0° but the verticals at 1.22°, 7 junk at 0.75° but the
  shadows lost again. So it changes the one setting.
- *Freehand* stays the defaults, where 4.jpg's verticals are best.

Checks: three rows in `detection`, reading the presets from index.html itself.
Freehand equals the fields' own defaults, which a blank field falls back to;
ruled or rendered keeps shadowtest's families within 0.6° (worst 0.40°);
sparse offers its shadows (1.70°). Each fails on its own break, fed through
the page's text: freehand's spread at 1.0, sparse's least line back at 0.7
(20.0°), ruled's least line at 3 (90°).

On the real page (port 8832): the choice opens on freehand. Ruled or rendered
fills 0.75, 2, 0.7 and 3, and Detect edges sends them. Typing a spread turns
the choice to *your own*; freehand puts 1.5, 3, 0.7 and 1.5 back. Sparse on
shadowtest offers four families, the fourth 10 lines at (1556, 319), the
shadows.

Not built: the choice saved with the marks, and a preset for photographs,
which the benchmarks hold none of with known points.

**676 passed, 0 failed.**

## 50. "what about curvilinear perspective?" — 5-point, read as the stereographic projection, and a guard for the straight modes — 2026-10-29

*The request:* "what about curvilinear perspective?", then "go build". §47's
questions went unanswered, so the defaults held: hand-drawn 5-point first, a
line marked by a run of three or more clicks, a refusal in 1-, 2- and 3-point
when such runs bend, and no camera out yet. 4- and 6-point were not asked for
and are not built.

**The model.** A 5-point drawing made with a compass draws every straight edge
as an arc through two opposite points of the circle 90° out from the centre.
That is exactly the stereographic projection: a ray (x, y, z) lands at
c + 2f·(x, y)/(1 + z), and the circle 90° out has radius 2f. §47 found the
stereographic model picked 40 of 40 at up to 3 px. The projection is
conformal, so a mark's distance from its curve is its angle off the line's
plane times 2f/(1 + z) px a radian, whichever way the curve runs.

**The fit** (`core/curvi.py`). A run of marks is a set of rays on one plane
through the eye; a family's planes all hold the family's direction; the
directions are square to each other. The unknowns are the centre, log f, the
turn, and one angle a line for its plane. Least squares from several starting
f; a second start that lands elsewhere at the same cost is a rival, and the
answer is refused. Standard errors come from the Jacobian. If the centre is
refused, the fit runs again with it pinned at the picture's centre, which §47
found better conditioned.

**Refusals**, in order. Any one withholds both the camera (f, the circle, the
field) and the directions:
- a rival start as good;
- f past 8 picture diagonals (the arcs are straight to within the marks, so
  the picture is linear) or under 0.02;
- fewer than 4 points to spare, below which the fit's own error bars mean
  nothing;
- f placed only to worse than ±10 %;
- the turn placed only to worse than ±2°.

With no point to spare at all (lines marked by their two ends) the verdict is
undetermined. Otherwise it is already consistent when no line moves more than
a hand mark, and rectifiable when one does.

**Measured** on synthetic rooms with known truth: 1200 × 900, f 150, 220 or
400 px, 10 seeds each, marks scattered by 3 px (a careful hand) or 6 px. A fit
is wrong when f is more than 10 % off or a direction more than 5°.

| families × lines × points | 3 px: quoted | wrong, flagged | wrong, not flagged | 6 px: quoted | wrong, flagged | wrong, not flagged |
|---|---|---|---|---|---|---|
| 3 × 2 × 3 | 27 | 1 | 0 | 7 | 4 | 0 |
| 2 × 2 × 3 | 0 | 3 | 0 | 2 | 5 | 0 |
| 2 × 3 × 3 | 29 | 0 | 0 | 16 | 1 | 1 |
| 3 × 3 × 5 | 30 | 0 | 0 | 25 | 1 | 0 |
| 2 × 2 × 5 | 23 | 2 | 0 | 7 | 5 | 1 |

Of 300 fits, 166 were quoted and 2 were wrong without a flag, both at 6 px with
two families. Those two were turned 5.9° and 5.2° off, against quoted errors of
±1.8° and ±1.9°; their f was 2.7 % and 6.4 % off, within its bars. 112 right
answers were withheld. That is the price of the rules on spare points and on
the turn, which cut the unflagged wrong answers from 9 to 2. Turn limits of
2.5° and 3° were compared, and 2° kept.

The turn's quoted error is honest. On one scene (3 × 3 × 4 at 3 px), over 100
draws of the scatter, the turn came out a median 0.72° off (p90 1.56°) against
a median quoted ±0.74°, and inside its quoted error 54 % of the time.

**The guard.** In 1-, 2- and 3-point, each run counts as its straight chord:
the total-least-squares line, with the farthest mark from it as the bend. So a
straight line clicked three times counts toward its vanishing point like a
dragged one (`/api/fit_vp`, the family fit, the checks). If any run bends more
than a hand mark, Consensus answers undetermined and names the lines
(`curved_lines`, by family). Camera out refuses for the same reason. On the
same synthetic 5-point rooms at 3 px, 60 of 60 pictures were caught with 3, 4
or 5 clicks a line. Straight lines at 3 px were refused 0, 1 and 5 times in
1000 at 3, 4 and 5 clicks, since the farthest of more marks strays further. The
message says that a line meant straight is misdrawn by that much.

**In 5-point,** Check answers undetermined for every check, since the checks
are built on linear perspective's straight lines. Camera out refuses, because
the camera it writes for Blender is a pinhole. A camera read in is a pinhole
too, so it is not compared.

**The page.** In a family layer a drag is a line, as before. Three or more
clicks along a line, ended by Enter or a double-click, make a run; two clicks
make a line. A double-click's repeated point is dropped. ⌘-click removes a
point from a run, a run's points drag, and Backspace takes the last run back.
In 5-point the overlay draws the circle 90° out, the horizon, each line's arc
and its whole circle, and the vanishing points at both ends of each direction.
The readout gives f, the circle's radius and centre, and the field through the
centre. Saved marks, the PDF, Clear and undo carry runs.

**The test picture,** `testdata/5point.png`, with its marks beside it: a room
drawn in 5-point, centre (596, 452), f 250 px, turned by the rotation vector
(0.10, −0.16, 0.02). Three families, 3, 3 and 4 runs of five points, scattered
1.5 px. The fit: already consistent, f 249.5 ± 2.9 px, the centre 2.3 px off
(± 9.2), the turn 0.31° off (± 1.21°), the worst line 3.72 px.

The first version was wrong at the bottom left, as the user saw. The drawing
ran on past the circle 90° out, to 123°, so lines behind the eye were drawn,
and a floor line behind the eye ran down through the floor like an upright.
Stopping at the circle left a second fault: a cross-wall grid plane passed
through the eye. Everything in a plane through the eye lands on one circle, so
the wall's nearest upright and the floor line beyond it drew as one smooth arc,
which reads as an upright running on through the floor. That is true to the
projection but wrong for a test picture. The grid now steps half a unit off the
eye.

Checks: `five_point`, 22 rows.
- The projection and its inverse undo each other.
- Exact marks give f within 0.1 %, the centre within 0.5 px and the directions
  within 0.05°. At 3 px, f is within 3 % and the directions within 1.5°.
- A line drawn 30 px off its arc is found and named.
- Lines marked by their ends are refused, and two lines of three points a
  family quote no camera.
- The guard refuses a 5-point drawing checked as 3-point, names the lines, and
  writes no camera. Checked as 5-point, the same drawing is consistent and
  draws its arcs.
- Straight 3-click lines are not refused, and count toward their vanishing
  point.
- The checks refuse in 5-point.

Three deliberate breaks were each caught. Switching the guard off fails the
3-point refusal row. Fitting with an equidistant lens fails the round trip, and
the exact marks then get no camera at all. Lifting the refusals fails the
sparse row.

On the real page (`run_mac.command` on port 8831):
- The test picture and its marks load. Consensus in 5-point is already
  consistent at 249.5 px, and draws the circle, the horizon, the arcs and the
  vanishing points.
- Runs are clicked, double-clicked and taken back.
- Check answers undetermined with the reason, and Camera out shows its
  refusal.
- Switched to 3-point, the guard's refusal shows.

Not built: 4- and 6-point; fisheye photographs, whose lenses are mostly
equidistant or equisolid rather than stereographic (the marks choose the model
only partly, §47.4); and camera out for 5-point.

**654 passed, 0 failed.**
