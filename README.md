# Persimmon

Finds the most parsimonious perspective.

Persimmon checks the perspective of pictures made by hand: drawings,
watercolours, paintings, and photographs of them. You mark what you see
(lines, rims, figures, shadows), and it finds the vanishing points and the
horizon, checks the rest of the picture against them, and shows how far each
line would have to move for the whole picture to be consistent.

It runs on your own machine: a small Python server and a page in your browser.

## What it does

- **Vanishing points and the horizon**, from families of lines, in 1-, 2- or
  3-point perspective.
- **5-point perspective**, the curvilinear kind drawn with a compass. Lines
  marked as runs of points give the nearest consistent 5-point picture, the
  circle 90° out, and the move each line needs.
- **Consensus**: the nearest consistent perspective, and the move each line
  needs to reach it.
- **Checks against that perspective**: circles and ellipses (including the
  circle in a square), arches and roofs, regular polygons, evenly spaced rows,
  proportion along an axis, symmetry, figures standing on one ground, cast
  shadows from the sun or a lamp, and things of known size.
- **A top view** of the scene from above: the eye, the picture plane, a ray to
  each vanishing point, and the field of view. It is drawn to scale once you
  give the picture's real size.
- **Cameras**: reads a camera solved in Blender or fSpy and measures your marks
  against it, and writes the camera your marks agree on as a Blender script.
- **Photographs of a sheet** are trimmed to the sheet, with the keystone
  undone.
- **Output**: the marks as a `.json` file, and the overlay as a vector PDF.

## How to read the results

No hand drawing is exact, so Persimmon does not ask whether a picture is
perfect. It asks which perspective is nearest to what was drawn, and how far
each line is from it. A move smaller than a hand mark is worth (about 9 px) is
not an error, and the picture is reported as **already consistent**.

Every result comes with how well the marks pin it down. When the marks cannot
settle a question, the answer is **undetermined**, with the reason, rather
than a confident number.

Marking is done by hand on purpose. Automatic edge detection fails on real
paintings (occlusion, brush texture, low contrast), so it is offered only as
suggestions you can take or ignore.

For example, `testdata/4.jpg` marked by hand at 45 % zoom comes out already
consistent: the worst line is 1.3 px off, and the focal length is 1260 px. The
same lines drawn precisely in Illustrator give 1250 px.

## Getting started

You need Python 3.10 or newer.

```bash
git clone https://github.com/righeile/Persimmon.git
cd Persimmon
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Then open <http://127.0.0.1:8830>. Use `--port` to pick another port. The
server listens on 127.0.0.1 only and keeps nothing: your marks stay in the page
until you save them.

You can also double-click a launcher:

- **`run_mac.command`** (macOS) uses `.venv` if there is one, and offers to
  install missing packages.
- **`run_windows.bat`** (Windows) uses the `python` on your PATH.

Both open the browser, replace a copy of Persimmon already running on the
port, and honour a `PORT` environment variable.

[Blender](https://www.blender.org) is optional. You only need it to read a
camera from a `.blend` file.

## Using it

1. **Open** a picture, or pick one of the test pictures.
2. If it is a photograph of a sheet, use **Find sheet** or **Place corners**,
   then **Apply**.
3. Choose the perspective the picture claims: **3-point**, **2-point**,
   **1-point** or **5-point**. This is the claim being checked, not a hint to
   the solver.
4. **Add** a layer, choose what it holds, and mark on the picture.
   **Detect edges** offers line families; nothing is used until you click it.
5. **Check** runs every check against the marks. **Consensus** finds the
   nearest consistent perspective.
6. **Save marks** writes a `.json` file. When one sits next to a picture, the
   app offers it on opening. **Save PDF** writes the marks as vector lines over
   the picture.

Give the picture's real size (in mm, cm, m, in or ft) to get results in those
units as well as pixels.

| layer | you mark | it gives |
|---|---|---|
| line family | drag each line, or click along it (three points or more, then `Enter`); in 5-point, click | a vanishing point, and, with three or more lines, how well they agree |
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

Each layer's colour is set in its editor. The bar under the picture has colour
wells for the horizon and the correction lines, and the switch for the top
view.

## Cameras in and out

**Camera in** reads a solved camera from a Blender `.blend`, an fSpy `.fspy` or
a plain `.json`, draws its vanishing points and horizon, and shows how far your
marks are from it. One family of lines is enough. Blender opens the file in the
background with the file's own scripts switched off, and is looked for in this
order: the `PERSIMMON_BLENDER` path, `/Applications/Blender*.app` on macOS,
`C:\Program Files\Blender Foundation\Blender*` on Windows, then `blender` on
your PATH.

**Camera out** writes the camera your marks agree on as a Blender script, with
the picture as the background. In Blender, choose Scripting > Open, pick the
file, and click Run Script. It refuses when the marks give no camera, so a lens
that was not measured is never written down. When the marks had no line to
spare for testing the camera, it is still written, with `NOT TESTED` in the
header.

## Running the checks

```bash
.venv/bin/python selfcheck.py        # over 600 checks; under a minute
.venv/bin/python core/consensus.py   # the consensus fit's own self-check
.venv/bin/python core/figure.py      # proportion and symmetry
```

Checks that need something you do not have, such as Blender, are skipped, and
the output says so.

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
testdata/       the pictures and marks the checks run on
PLAN.md         the specification and lab notebook
```

`labkit/` and `static/labkit.*` are a vendored copy of a small toolkit shared
with other apps (file browser, theme, plot style). The copy is overwritten
whenever the toolkit is synced, so edits made here do not last.

## Limits

Besides linear perspective, Persimmon handles 5-point, read as the
stereographic projection a compass draws. In 5-point only Consensus answers.
The other checks and Camera out are built on straight lines, and say so. 4- and
6-point, and photographs through a fisheye lens, are not handled.

A curvilinear picture checked as 1-, 2- or 3-point is refused only when its
lines are marked by three or more clicks. A line dragged as two points shows no
bend, so such a picture can still be called consistent when it is not.

`PLAN.md` holds the design rules, the verification behind every check, and the
bugs found along the way.
