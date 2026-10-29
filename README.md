# Persimmon

Finds the most parsimonious perspective.

Persimmon checks the linear perspective of traditional artwork: sketches,
watercolours, paintings, and photographs of them. You mark what you see:
lines, rims, figures, shadows. It finds the vanishing points and the horizon
and says how consistent the drawing is. No hand drawing is exact, so it also
finds the nearest consistent perspective and shows how far each line would
have to move to reach it.

It is a small local web app: a Python server on your own machine and a page in
your browser.

## What it checks

- **Vanishing points and the horizon**, from families of lines, in 1-, 2- or
  3-point perspective.
- **Consensus**: the nearest consistent perspective, with the move each line
  needs, in pixels.
- **Circles and ellipses** on a plane, including the circle inscribed in a
  square formed by two line families.
- **Arches and roofs**: round or pointed arches, arcades, gables and pitched
  roofs, including the slope's vanishing point.
- **Regular polygons**, **evenly spaced rows**, **proportion along an axis**
  (by cross-ratio) and **symmetry**.
- **Figures** standing on one ground, checked against the horizon.
- **Cast shadows** from a lamp or the sun.
- **Known sizes**, which turn the camera into a distance.
- **Cameras**: it can read a camera solved in Blender or fSpy, and write the
  camera the marks agree on back out as a Blender script.
- **Photographs of a sheet** can be trimmed to the sheet, with the keystone
  undone.

## Two rules the code keeps

**Every readout comes with its conditioning, and refuses when the question is
ill-posed.** Every check returns `{value, verdict, conditioning}`, and the
verdict may be `"undetermined"`. The page shows that as plainly as a number,
never as a blank or a zero. During the project, four results had to be
withdrawn because they had been reported with a confidence they had not
earned. The refusals are there for a reason.

**It is semi-automatic: you mark what the pixels cannot give.** Automatic
extraction failed again and again on real paintings, because of occlusion,
watercolour texture and low contrast. Hand marks land within about 9 px of the
paint, which is plenty for every check here. So marking is the backbone, and
edge detection is a shortcut that you can accept, reject or ignore.

## A drawing is not a diagram

A hand-drawn picture is never exact, so calling it inconsistent tells you
nothing. **Consensus** asks a different question: which perspective is nearest
to what was actually drawn, and how far would each line have to move to reach
it? A move smaller than the ~9 px a hand mark is worth is not an error. In that
case the verdict says `already consistent`, rather than sending you to redraw a
line that was right.

Consensus gives two readings:

- **As drawn** lets the camera roll, which is what the picture says.
- **Forced level** assumes the eye is level, and prints what that assumption
  costs.

The worked example is `testdata/4.jpg`. Marked by hand in the app at 45 %
zoom, it came out already consistent: the worst line was 1.3 px off, and the
focal length was 1260 px. The same lines drawn precisely in Illustrator give
1250 px.

## Getting started

You need Python 3.10 or newer. Development used Python 3.13.

```bash
git clone https://github.com/righeile/Persimmon.git
cd Persimmon
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Then open <http://127.0.0.1:8830>. To use another port, start it with
`app.py --port 8831`.

There are also launchers you can double-click:

- **`run_mac.command`** (macOS) uses `.venv` if there is one, and offers to
  install any missing packages.
- **`run_windows.bat`** (Windows) runs the `python` on your PATH, so install
  the requirements into that one.

Both launchers stop an older copy of the app on the same port, open the
browser, and honour a `PORT` environment variable.

By default the server listens on 127.0.0.1 only. It keeps nothing between
requests: the marks live in the page until you save them.

[Blender](https://www.blender.org) is optional. You only need it to read a
camera from a `.blend` file.

## Using it

1. **Open** a picture, or pick one of the test pictures.
2. If it is a photograph of a sheet, **Find sheet** or **Place corners**, then
   **Apply**. This trims the picture to the sheet and undoes the keystone.
3. Choose the perspective the drawing claims: **3-point**, **2-point** or
   **1-point**. This is the claim being checked, not a hint to the solver.
4. **Add** a layer, choose what it holds, and mark on the picture.
   **Detect edges** can offer line families, but nothing is adopted until you
   click it.
5. **Check** runs every check against the marks. **Consensus** finds the
   nearest consistent perspective.
6. **Save marks** writes a `.json` file that you can load again. If a marks
   file sits next to a picture, the app offers it when you open that picture.
   **Save PDF** writes the marks as vector lines over the picture.

You can also give the picture's real size, in mm, cm, m, in or ft, so that
readouts are in those units instead of pixels.

| layer | you mark | it gives |
|---|---|---|
| line family | drag each line | a vanishing point, and, with three or more lines, how well they agree |
| surface | two vanishing points, or "level with the ground" | the plane that an arc, ellipse or arch is read in |
| arc | click round the rim | circle or oval, once the rim turns through 120° |
| ellipse | drag across a circle, click round a drawn one, or click inside a cell | the ellipse a real circle would cast, how far a drawn one is from it, or the circle in a square |
| arches and roofs | the springing line and the top, or click round a drawn one | the arch or roof a real one would cast; for a drawn arch: its rise, springing, crown offset and depth |
| regular polygon | the corners, in order | whether it is regular |
| figures | feet, then head | whether they stand on one horizon |
| spaced row | one point per post | whether the spacing is even |
| axis | points along one axis | proportion, by cross-ratio |
| cast shadows | base, top and shadow tip of each object | the light and its foot, whether they line up with the vertical, and whether it is the sun or a lamp |
| landmark pairs | left, then right | symmetry |
| known size | base, then top, of something whose real size you know | the distance from the viewer |

An arc is read against its own surface. It is only read against the horizon
when it lies flat, and with no surface the check refuses rather than assume
one.

Each layer has its own colour, set in its editor. The horizon and the
correction lines have colour wells under the picture. Recolour a line when you
cannot see it against the paint; the PDF uses the same colours.

Once there is a horizon, a small **top view** at the bottom right shows the
drawing from above: the eye, the picture plane, and a ray from the eye to each
vanishing point, with the field of view beneath. Where the camera is refused,
the window says why; where the camera is only a guide, the plan is dimmed and
says so. Given the picture's real size, the plan is drawn to scale, with a
scale bar and the eye's distance in millimetres.

| to | do |
|---|---|
| pan | drag with no layer selected, middle-drag, alt- or shift-drag, or scroll |
| zoom | pinch, or ⌘/ctrl + scroll |
| fit the picture | `f`, or **Fit** |
| end a run of points | `Enter`, or double-click |
| remove the last mark | `Backspace` |
| delete a placed point | ⌘-click |
| cancel a pending mark | `Escape` |
| undo / redo | ⌘Z / ⇧⌘Z or ⌘Y (ctrl on Windows) |

## A camera in, a camera out

**Camera in** reads a solved camera from a Blender `.blend`, an fSpy `.fspy`
or a plain `.json`. It draws the camera's three vanishing points and horizon on
the picture, and prices the marks against it in pixels of movement. One family
of lines is enough. The camera is somebody's claim, not the truth: the card
says how far the marks are from it, and how far they are from being consistent
on their own.

To read a `.blend` file, Blender runs in the background with the file's
scripts switched off. Persimmon looks for Blender in this order:

1. the path in `PERSIMMON_BLENDER`
2. `/Applications/Blender*.app` on macOS
3. `C:\Program Files\Blender Foundation\Blender*` on Windows
4. `blender` on your PATH

**Camera out** writes the camera the marks agree on as a Blender script, with
the picture as the background. To use it, open Blender and go to
Scripting > Open, choose the file, then Run Script. It refuses wherever the
readout refuses: a lens that was not measured is never written down. If the
marks had no line to spare, the header says `NOT TESTED`. A fit with nothing
to spare is `undetermined`, because any set of marks fits a camera that has as
many numbers as the marks have angles.

## Running the checks

```bash
.venv/bin/python selfcheck.py        # about 600 checks, every verification in PLAN.md (~25 s)
.venv/bin/python core/consensus.py   # the consensus fit's own self-check
.venv/bin/python core/figure.py      # proportion and symmetry's own self-check
```

Checks that need something missing are skipped, and the output says so. If
Blender is not installed, for example, the camera export is not run through a
real Blender.

## Layout

```
app.py          the Flask server and the API the page calls
core/           the geometry, with no web code
  vp.py           lines to vanishing points to residuals
  consensus.py    the nearest consistent perspective
  checks.py       every readout, with its conditioning
  rhythm.py       spacing, roofs, figures, ellipses
  figure.py       proportion and symmetry, projectively
  rectify.py      a plane turned to face the viewer; the plan from above
  trim.py         a photograph trimmed to the artwork, keystone undone
  camera_io.py    cameras in from Blender, fSpy and JSON; out as a Blender script
  export.py       the vector PDF
static/         the page: index.html, app.js, app.css
selfcheck.py    the verification suite
testdata/       the pictures and marks that the checks run on
PLAN.md         the specification and lab notebook
```

`labkit/` and `static/labkit.*` are a vendored copy of a small toolkit shared
with other apps. Persimmon uses its file browser, theme and plot style. The
copy is overwritten whenever the toolkit is synced, so edits made here do not
last.

## Status and limits

`PLAN.md` is the specification. It holds the design rules, the verification
table, the bugs found and what each one cost, and the worked example.

Persimmon models a pinhole camera, so it handles linear perspective only. It
does not yet detect a fisheye or curvilinear drawing, and can call one
consistent when it is not. PLAN.md §47 measures what supporting fisheye would
take.
