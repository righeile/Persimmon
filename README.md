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
- **A guess at the perspective**: Detect edges says which type the lines fit
  best, the lens, and how far the camera rolled and tipped.
- **Straighten** turns the picture to the type you select: the horizon level,
  the uprights parallel, and in 1-point the wall facing you. The marks move with
  it, and the file is not touched.
- **Patterns on floors and walls**: Detect edges sorts a parquet or a tiled wall
  from the room's own lines, and uses it as a check on the horizon. A pattern
  layer lays a repeat across its surface as a guide, and measures how far the
  real pattern drifts from a true, even one. **Show flat** turns a surface to
  face you.
- **Arches**: Detect edges finds them, and names the wall each stands in when
  it can tell. Curves are kept out of the line families.
- **Vaults**: Detect edges finds the point a vault runs to and sorts its ribs into
  **flat**, in one plane across the vault, and **3D**, running through depth from
  foot to crown, such as diagonals and tiercerons. When the ribs follow a regular
  vault it says how closely, and **build** draws the whole vault, dashed where the
  rule adds a rib that was not found. Any other vault is read by what repeats along
  it: ribs that are copies of each other at one bay, with no pattern assumed. When
  nothing runs to a vault's point, three lines you mark give it.
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
   **Detect edges** offers line families, arches, a vault's ribs and patterned
   surfaces, with a guess at the perspective; nothing is used until you click it.
   **Take** puts the guess into layers. When a vault shows no point, mark three lines
   running down it as a line family, select that layer and press **Detect edges**
   again. When the families look wrong, pick the
   kind of picture under *detection settings* (freehand, ruled or rendered,
   photograph, sparse), or set its four numbers.
5. **Check** runs every check against the marks. **Consensus** finds the
   nearest consistent perspective.
6. **Straighten** turns the picture to the selected type, from the marked
   families with the uprights flagged. **Save picture** writes the result as a
   PNG. Undo takes it back.
7. **Save marks** writes a `.json` file. When one sits next to a picture, the
   app offers it on opening. **Save PDF** writes the marks as vector lines over
   the picture.

Give the picture's real size (in mm, cm, m, in or ft) to get results in those
units as well as pixels.

| layer | you mark | it gives |
|---|---|---|
| line family | drag each line, or click along it (three points or more, then `Enter`); in 5-point, click | a vanishing point, and, with three or more lines, how well they agree |
| surface | two vanishing points, or "level with the ground" | the plane that an arc, ellipse, arch or pattern is read in; **Show flat** turns it to face you |
| pattern | one repeat (a corner, then the next corner along each side), then corners further off | the repeat laid across the surface as grid, checker, brick or herringbone, and how far the marks drift from it |
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
wells for the horizon and the correction lines, and the switches for the top
view and for the picture itself, to see the marks alone.

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
.venv/bin/python selfcheck.py        # 939 checks; about a minute and a half
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
  rectify.py      a plane turned to face the viewer; a camera stood up; the plan from above
  pattern.py      a repeat on a plane: laid across it, checked, found in the edges
  arches.py       arches and a vault's ribs found among the edges; the vault they belong to
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

An arch is read for its shape, and its wall named, only once it turns through
120°. Less of a curve than that is offered and left to you. A found arch can
run a little way down a straight jamb, so its springing may want moving.

A vault's ribs are sorted only in a view along the vault, and it takes lines
running to its point as well as the ribs, so a small or tightly cropped picture
may give no vault at all. When the ribs lie on three lines no better than chance
and no regular vault vouches for them, the sort is refused: 3 of 19 blurred,
rolled, compressed or cropped versions of one nave were. Wall ribs, curves that
stop short of the crown and ribs that could be either of two of the vault's stay
undetermined, so a sort is partial even when it succeeds.

A vault is built only when its ribs repeat at one bay. A flat rib a side at every
other boss with two 3D ribs a side at the rest is read by its own rule, which adds
the ribs it did not find. Any other pattern is read as families of ribs, each one rib
moved along the vault, and it adds only a rib missed between two found. Bays at
random are refused, not fitted, though bays a little uneven are built as the nearest
regular ones. Wall ribs and the ridge are not built, and the vault is drawn in the
picture only: with no lens there is no plan of it. Of drawn naves with regular crowns
and their feet at random, 2 of 72 were still taken for a vault.

A family is called flat or 3D only as far as its point is known, and its run along
the vault is read to about a third of a bay. 30 px sideways on a barrel turned a run
of -.10 bays into -.35. A family stays undetermined unless it stands clear by twice
what splitting its curves and moving the point as far as its lines allow can do, so on
drawn naves with the true point most are left undetermined: of 182 families 24 were
called right and 1 wrong. That only knows how well the lines agree, not where the point
truly is. On a heavily molded vault a point a few pixels out can still call a flat
family 3D.

Of ten photographs of vaults tried, one read by itself, two with three lines marked
(one of them left undetermined), and seven gave nothing firm. Soft or small ribs give
too few curves. A fan vault is refused or left undetermined, and a rule with too few of
its own ribs there is refused. On a heavily molded Gothic vault only about a third of
the curves follow the rule, since each rib has several edges, and a line moved by a
pixel can tip the read between the rule and families, or change the ribs built from 34
to 28. Three lines marked to a wrong point gave a read 14 times in 120, 5 of them firm.
A vault in a picture claimed as 5-point is refused, since its rays bend.

A pattern's long planks are given by their length in widths, which you type. It
is not measured.

**Show flat** needs a measured lens, so a room that reads 1-point with nothing to
measure one is refused.

`PLAN.md` holds the design rules, the verification behind every check, and the
bugs found along the way.
