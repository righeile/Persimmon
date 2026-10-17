"""A camera in, and a camera out.

Persimmon's camera is a focal length in pixels, a principal point and a
rotation. Other programs keep the same three things in other words: Blender as a
lens in millimetres on a sensor, a shift as a fraction of the picture's larger
side and the 4x4 matrix of the camera object; fSpy as a focal length in
half-widths, a principal point from the centre and a copy of that matrix. This
is the one translation between them, in both directions. Nothing else in the
app knows the words.

Every convention below is a place to be wrong by a mirror, so they are written
once and the round trip through a real Blender (selfcheck) is what vouches for
them:

  picture   x right, y DOWN, pixels, origin at the top-left corner.
  camera    x right, y down, z FORWARD -- the frame the page's K lives in.
  Blender   x right, y UP, looks down -Z: the same frame turned half a turn
            about x, diag(1, -1, -1). That is the only flip.
  lens      millimetres, and f = lens * S / sensor where S is the picture's
            LARGER side and sensor is sensor_width (sensor_fit AUTO). On a
            portrait picture S is the height: lens * W / sensor is wrong there.
  shift     fractions of that same S: cx = W/2 - shift_x*S, cy = H/2 + shift_y*S.

A vanishing point fixes an axis only up to sign, so a camera's rotation is kept
as three unit directions in the camera frame, `axes`, and nothing here pretends
to know which way along one is "positive" beyond what a source file said.

Reading a .blend runs Blender itself, in the background, with the file's own
scripts switched off (-Y). A .blend is a memory dump whose layout changes with
every release; the only reader that is right is the version of the program
that wrote it, and 0.4 s of its time is cheaper than a parser that is wrong.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

AXES = ("X", "Y", "Z")
_FLIP = np.diag([1.0, -1.0, -1.0])
FAR = 50.0                  # x the picture diagonal: consensus.FAR, same meaning
ASPECT_TOL = 0.005          # a camera is read against a picture of its own shape
FSPY_ID = 2037412710


# ----------------------------------------------------------------- the camera

def _K(cam):
    f, (cx, cy) = float(cam["f"]), cam["pp"]
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def _orthonormal(M):
    """The nearest rotation to M, and how far its columns were from unit length."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, float(np.max(np.abs(np.linalg.norm(M, axis=0) - 1.0)))


def pick_up(axes, margin=0.1):
    """Which of three directions is the upright, as a picture shows it: the one
    that runs most nearly along the picture's vertical -- the largest |y| in
    the camera frame. An index, or None when the top two are within `margin`
    (a camera turned 45 deg about its axis has no upright to read).

    A prior and not a measurement. Three perpendicular directions are a
    symmetric triple; nothing in them says which one gravity is along. Under
    45 deg of pitch the upright is the one that reads as vertical; looking
    steeper than that, this picks the depth axis and cannot know. A family
    flagged vertical on the page is the only thing that settles it.
    """
    y = np.abs([float(np.asarray(v, float)[1]) for v in axes])
    order = np.argsort(-y)
    if y[order[0]] - y[order[1]] < margin:
        return None
    return int(order[0])


def from_pose(f, pp, size, Rwb, source=None, up=None):
    """A camera from f, pp and a camera object's 3x3 (columns: its x, y, z in the world).

    `up` is an axis label, or None to read it off the picture (pick_up)."""
    if not (np.isfinite(f) and f > 0):
        raise ValueError("a focal length has to be a positive number")
    R, off = _orthonormal(np.asarray(Rwb, float).reshape(3, 3))
    Rcv = _FLIP @ R.T                    # world -> camera, x right, y down, z forward
    src = dict(source or {})
    if off > 1e-3:
        src["scaled"] = (f"the camera object is scaled (a column of its matrix is "
                         f"{off:.3f} off unit length); only its direction was used")
    axes = {a: [float(x) for x in Rcv[:, i]] for i, a in enumerate(AXES)}
    if up is None:
        i = pick_up([axes[a] for a in AXES])
        up = None if i is None else AXES[i]
        src["up_basis"] = ("the axis nearest the picture's vertical; a camera file "
                           "does not say which way is up" if up else
                           "no axis is clearly the picture's vertical")
    return {"f": float(f), "pp": [float(pp[0]), float(pp[1])],
            "size": [float(size[0]), float(size[1])], "axes": axes,
            "up": up, "source": src}


def validate(cam):
    """A camera dict as the page sent it back, or ValueError. Nothing is trusted."""
    try:
        f = float(cam["f"])
        pp = [float(cam["pp"][0]), float(cam["pp"][1])]
        W, H = float(cam["size"][0]), float(cam["size"][1])
        A = np.array([[float(x) for x in cam["axes"][a]] for a in AXES])
    except (KeyError, TypeError, ValueError, IndexError):
        raise ValueError("that is not a camera: it needs f, pp, size and the three axes")
    if not (np.isfinite([f, *pp, W, H]).all() and f > 0 and W > 0 and H > 0):
        raise ValueError("that camera has a non-finite or non-positive number in it")
    if A.shape != (3, 3) or not np.isfinite(A).all() or \
            np.abs(A @ A.T - np.eye(3)).max() > 1e-3:
        raise ValueError("the three axes of that camera are not at right angles")
    return {"f": f, "pp": pp, "size": [W, H],
            "axes": {a: A[i].tolist() for i, a in enumerate(AXES)},
            "up": cam.get("up") if cam.get("up") in AXES else None,
            "source": cam.get("source") or {}}


def vps_h(cam):
    """The three vanishing points, homogeneous, in the picture's pixels."""
    K = _K(cam)
    return {a: K @ np.asarray(cam["axes"][a], float) for a in AXES}


def describe(cam):
    """What the page draws: each axis' point (or its direction when it has none),
    the horizon, and the lens as a photographer would say it."""
    W, H = cam["size"]
    diag = float(np.hypot(W, H))
    out = {"vps": {}, "far": {}}
    for a, h in vps_h(cam).items():
        p = h[:2] / h[2] if abs(h[2]) > 1e-12 else None
        if p is None or not np.isfinite(p).all() or np.hypot(*p) > FAR * diag:
            n = float(np.hypot(h[0], h[1])) or 1.0
            out["vps"][a], out["far"][a] = None, [h[0] / n, h[1] / n]
        else:
            out["vps"][a], out["far"][a] = [float(p[0]), float(p[1])], None
    out["horizon"] = None
    up = cam.get("up")
    if up in AXES:
        # the vanishing line of the plane square to `up`: inv(K)^T n
        l = np.linalg.inv(_K(cam)).T @ np.asarray(cam["axes"][up], float)
        n = float(np.hypot(l[0], l[1]))
        if n > 1e-12:
            out["horizon"] = [float(x) for x in l / n]
    out["lens_mm_on_36"] = round(cam["f"] / max(W, H) * 36.0, 3)
    return out


def fit_to(cam, W, H):
    """The camera in the pixels of a picture W x H, or ValueError.

    Same shape or no camera: a frame of another aspect has no pixel-for-pixel
    relation to the picture (crop? stretch? bars?) and any answer would be a
    guess dressed as a measurement.
    """
    w, h = cam["size"]
    if abs((w / h) / (W / H) - 1.0) > ASPECT_TOL:
        raise ValueError(
            f"this camera's frame is {w:.0f} x {h:.0f} (aspect {w / h:.3f}) and "
            f"the picture is {W:.0f} x {H:.0f} ({W / H:.3f}): a camera belongs to "
            "a frame of the same shape. Render the scene at the picture's "
            "shape, or open the picture the camera was solved on")
    s = W / w
    out = dict(cam)
    out["f"] = cam["f"] * s
    out["pp"] = [cam["pp"][0] * s, cam["pp"][1] * s]
    out["size"] = [float(W), float(H)]
    if abs(s - 1.0) > 1e-9:
        out["source"] = dict(cam.get("source") or {},
                             scaled_by=round(s, 6),
                             scaled_from=[w, h])
    return out


def compare(cam, f, pp, axes):
    """How far a fitted camera is from this one, in the numbers that matter.

    `axes` are the fitted camera's three directions in its own frame. Lines
    have no arrow, so the comparison is between UNDIRECTED axis sets: the
    smallest rotation, over the six ways of pairing the axes, that brings one
    onto the other. Handedness is not a fact a vanishing point carries.
    """
    A = np.array([cam["axes"][a] for a in AXES], float)
    B = np.asarray(axes, float).reshape(3, 3)
    best = None
    for perm in itertools.permutations(range(3)):
        c = np.abs(np.einsum("ij,ij->i", A, B[list(perm)])).clip(0, 1)
        worst = float(np.degrees(np.arccos(c.min())))
        if best is None or worst < best[0]:
            best = (worst, perm)
    return {"rotation_deg": round(best[0], 3),
            "axis_pairing": {AXES[i]: int(p) for i, p in enumerate(best[1])},
            "f_ratio": round(float(f) / cam["f"], 4),
            "pp_offset_px": round(float(np.hypot(pp[0] - cam["pp"][0],
                                                 pp[1] - cam["pp"][1])), 2)}


# --------------------------------------------------------------------- Blender

def _blender_scale(fit, W, H, sw, sh):
    """(S px, sensor mm): the side the lens' sensor figure is laid along."""
    if fit == "VERTICAL":
        return H, sh
    if fit == "HORIZONTAL":
        return W, sw
    return (W, sw) if W >= H else (H, sw)         # AUTO: the larger side


def from_blender(c, res, source=None):
    """A camera from the numbers BLEND_READER reads off a Blender camera."""
    if c.get("type", "PERSP") != "PERSP":
        raise ValueError(f"camera '{c.get('name')}' is {c.get('type')}: only a "
                         "perspective camera has vanishing points")
    W, H, pax, pay = float(res[0]), float(res[1]), float(res[3]), float(res[4])
    if abs(pax - pay) > 1e-9:
        raise ValueError(f"the render's pixels are not square (aspect {pax:g}:{pay:g}); "
                         "this reads square pixels only")
    S, mm = _blender_scale(c.get("sensor_fit", "AUTO"), W, H,
                           float(c["sensor_width"]), float(c["sensor_height"]))
    f = float(c["lens"]) * S / mm
    cx = W / 2 - float(c["shift_x"]) * S
    cy = H / 2 + float(c["shift_y"]) * S
    M = np.array(c["matrix_world"], float).reshape(4, 4)
    src = dict(source or {}, kind="blend", camera=c.get("name"),
               lens_mm=float(c["lens"]), sensor_mm=mm,
               sensor_fit=c.get("sensor_fit", "AUTO"),
               shift=[float(c["shift_x"]), float(c["shift_y"])])
    return from_pose(f, (cx, cy), (W, H), M[:3, :3], src)


# Run inside Blender. Prints one line, "@@JSON@@" + JSON, which is all the
# parent looks at: everything else Blender says on the way in is ignored.
BLEND_READER = r'''
import bpy, json
bpy.context.view_layer.update()
sc = bpy.context.scene
r = sc.render
out = {"blender": bpy.app.version_string, "scene": sc.name,
       "scenes": len(bpy.data.scenes), "frame": sc.frame_current,
       "resolution": [r.resolution_x, r.resolution_y, r.resolution_percentage,
                      r.pixel_aspect_x, r.pixel_aspect_y],
       "active": sc.camera.name if sc.camera else None, "cameras": []}
for o in bpy.data.objects:
    if o.type != 'CAMERA':
        continue
    c = o.data
    out["cameras"].append({
        "name": o.name, "type": c.type, "lens": c.lens,
        "sensor_width": c.sensor_width, "sensor_height": c.sensor_height,
        "sensor_fit": c.sensor_fit, "shift_x": c.shift_x, "shift_y": c.shift_y,
        "matrix_world": [list(row) for row in o.matrix_world]})
print("@@JSON@@" + json.dumps(out))
'''


def blender_exe():
    """Where Blender is, or None. PERSIMMON_BLENDER wins over looking."""
    cands = [os.environ.get("PERSIMMON_BLENDER")]
    cands += ["/Applications/Blender.app/Contents/MacOS/Blender"]
    cands += sorted(glob.glob("/Applications/Blender*.app/Contents/MacOS/Blender"),
                    reverse=True)
    cands += sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender*\blender.exe"),
                    reverse=True)
    cands += [shutil.which("blender")]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def run_blender(args, timeout=120):
    """Blender in the background, nothing of the user's startup file, no
    scripts out of the .blend. Returns stdout, or ValueError saying why not."""
    exe = blender_exe()
    if exe is None:
        raise ValueError(
            "reading a .blend runs Blender in the background and none was found "
            "(looked in /Applications, Program Files and PATH). Set "
            "PERSIMMON_BLENDER to its path, or give a .fspy or .json instead")
    try:
        r = subprocess.run([exe, "--factory-startup", "-b", "-Y", *args],
                           capture_output=True, text=True, errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ValueError(f"Blender did not answer in {timeout} s")
    return r.stdout, r.stderr, r.returncode


def _marked(stdout):
    for line in stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    return None


def read_blend(path):
    """Every camera in a .blend, the scene's own first: [(name, camera | why)]."""
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "read_camera.py"
        script.write_text(BLEND_READER)
        out, err, code = run_blender([str(path), "--python", str(script)])
    got = _marked(out)
    if got is None:
        tail = [l for l in (out + "\n" + err).splitlines() if l.strip()][-4:]
        raise ValueError("Blender could not read that file: " + " | ".join(tail))
    src = {"file": Path(path).name, "blender": got["blender"],
           "scene": got["scene"], "frame": got["frame"]}
    if got["scenes"] > 1:
        src["note"] = (f"the file has {got['scenes']} scenes; the active one "
                       f"('{got['scene']}') gives the frame size")
    cams = sorted(got["cameras"], key=lambda c: c["name"] != got["active"])
    out = []
    for c in cams:
        try:
            cam = from_blender(c, got["resolution"], src)
            cam["source"]["active"] = c["name"] == got["active"]
            out.append((c["name"], cam))
        except (ValueError, KeyError) as e:
            out.append((c["name"], str(e)))
    return out


# ------------------------------------------------------------------------ fSpy

def from_fspy(raw, name=None):
    """A camera from an .fspy file: a 16-byte header, a JSON state, a picture."""
    if len(raw) < 16:
        raise ValueError("that is not an fSpy file (too short)")
    ident, ver, ns, ni = struct.unpack("<IIII", raw[:16])
    if ident != FSPY_ID or 16 + ns > len(raw):
        raise ValueError("that is not an fSpy file (wrong header)")
    st = json.loads(raw[16:16 + ns].decode("utf-8"))
    cp = st.get("cameraParameters")
    if not cp or "cameraTransform" not in cp:
        raise ValueError("this fSpy file has no solved camera in it yet")
    W, H = float(cp["imageWidth"]), float(cp["imageHeight"])
    f = float(cp["relativeFocalLength"]) * W / 2.0
    # fSpy measures from the centre, in half-widths, y UP -- vanishing points too
    pp = (W / 2 + float(cp["principalPoint"]["x"]) * W / 2,
          H / 2 - float(cp["principalPoint"]["y"]) * W / 2)
    M = np.array(cp["cameraTransform"]["rows"], float).reshape(4, 4)
    cam = from_pose(f, pp, (W, H), M[:3, :3],
                    {"kind": "fspy", "file": name, "version": ver,
                     "mode": (st.get("globalSettings") or {}).get("calibrationMode")})
    # Its own vanishing points are a second statement of the same camera; when
    # the two disagree the file was edited or this reading is wrong.
    worst = 0.0
    for v, ax in zip(cp.get("vanishingPoints") or [], cp.get("vanishingPointAxes") or []):
        a = str(ax)[:1].upper()
        if a not in AXES:
            continue
        p = np.array([W / 2 + v["x"] * W / 2, H / 2 - v["y"] * W / 2, 1.0])
        d = np.linalg.solve(_K(cam), p)
        d /= np.linalg.norm(d)
        c = abs(float(d @ np.asarray(cam["axes"][a])))
        worst = max(worst, float(np.degrees(np.arccos(min(c, 1.0)))))
    cam["source"]["vp_vs_matrix_deg"] = round(worst, 4)
    if worst > 0.5:
        cam["source"]["note"] = (f"fSpy's matrix and its own vanishing points "
                                 f"disagree by {worst:.2f} deg")
    return cam


# ------------------------------------------------------------------ plain JSON

def from_json(d):
    """Either what BLEND_READER prints for one camera (plus "resolution": [w, h])
    or Persimmon's own: {"f", "pp", "size", "axes": {"X": [...], ...}}."""
    if isinstance(d, dict) and "matrix_world" in d and "lens" in d:
        res = d.get("resolution") or d.get("size")
        if not res or len(res) < 2:
            raise ValueError("a Blender camera in JSON needs its resolution: [w, h]")
        res = list(res[:2]) + [100, 1, 1]
        return from_blender(dict(d, sensor_width=d.get("sensor_width", 36.0),
                                 sensor_height=d.get("sensor_height", 24.0),
                                 sensor_fit=d.get("sensor_fit", "AUTO"),
                                 shift_x=d.get("shift_x", 0.0),
                                 shift_y=d.get("shift_y", 0.0)), res,
                            {"file": None})
    if isinstance(d, dict) and "axes" in d:
        cam = validate(d)
        cam["source"] = dict(cam["source"], kind="json")
        return cam
    raise ValueError("that JSON is neither a Blender camera (lens, matrix_world, "
                     "resolution) nor a Persimmon camera (f, pp, size, axes)")


def read(path, camera=None):
    """{camera, others, refused} from a .blend, .fspy or .json file."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".fspy":
        return {"camera": from_fspy(p.read_bytes(), p.name), "others": [], "refused": {}}
    if ext == ".json":
        cam = from_json(json.loads(p.read_text(encoding="utf-8")))
        cam["source"]["file"] = p.name
        return {"camera": cam, "others": [], "refused": {}}
    if ext != ".blend":
        raise ValueError(f"{ext or 'that'} is not a camera file: .blend, .fspy or .json")
    got = read_blend(p)
    good = [(n, c) for n, c in got if isinstance(c, dict)]
    refused = {n: c for n, c in got if not isinstance(c, dict)}
    if not got:
        raise ValueError("there is no camera in that .blend")
    if not good:
        raise ValueError("no perspective camera in that .blend: "
                         + "; ".join(f"{n}: {w}" for n, w in refused.items()))
    pick = good[0]
    if camera:
        named = [g for g in good if g[0] == camera]
        if not named:
            raise ValueError(f"no camera called '{camera}' in that .blend")
        pick = named[0]
    return {"camera": pick[1], "others": [n for n, _ in good if n != pick[0]],
            "refused": refused}


# -------------------------------------------------------------------- the out

def world_frame(axes, up, yaw=0):
    """The world as the picture says it should be: Z up, right-handed.

    `axes` are three orthonormal directions in the camera frame (a fit's, in
    any order and either sign) and `up` says which one is vertical. Up is the
    sign that points toward the top of the picture. X is the horizontal one
    that runs most across it, Y follows from the right hand, and `yaw` turns
    the pair by quarter turns: nothing in a picture says which wall is "front",
    so it is the one choice left to be made by the person who knows.

    Returns the 3x3 whose columns are the world's X, Y, Z in the camera frame.
    """
    d = [np.asarray(v, float) for v in axes]
    z = d[up] * (-1.0 if d[up][1] > 0 else 1.0)
    a, b = sorted((i for i in range(3) if i != up), key=lambda i: -abs(d[i][0]))
    x = d[a] * (1.0 if (d[a][0] > 1e-9 or (abs(d[a][0]) <= 1e-9 and d[a][2] > 0))
                else -1.0)
    y = np.cross(z, x)
    for _ in range(int(yaw) % 4):
        x, y = y, -x
    return np.column_stack([x, y, z])


def blender_numbers(cam, Rcv, distance=10.0, sensor_mm=36.0):
    """lens, shifts and the 4x4 a Blender camera object needs: the inverse of from_blender.

    The camera is put `distance` units from the world's origin, straight
    behind it along the optical axis. A picture has no scale, so that number
    is a choice and is written into the script as one.
    """
    W, H = cam["size"]
    S = max(W, H)
    Rwb = Rcv.T @ _FLIP                            # columns: camera x, y, z in the world
    C = -Rcv.T @ np.array([0.0, 0.0, float(distance)])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Rwb, C
    return {"lens": cam["f"] * sensor_mm / S, "sensor_width": sensor_mm,
            "shift_x": (W / 2 - cam["pp"][0]) / S,
            "shift_y": (cam["pp"][1] - H / 2) / S,
            "matrix_world": M.tolist(), "resolution": [int(round(W)), int(round(H))]}


def blender_script(cam, Rcv, *, picture=None, distance=10.0, sensor_mm=36.0,
                   notes=()):
    """A Python file for Blender: the camera, the picture behind it, an origin."""
    n = blender_numbers(cam, Rcv, distance, sensor_mm)
    W, H = n["resolution"]
    pic = repr(str(picture)) if picture else "None"
    head = ["# Persimmon -> Blender: a camera read off a picture",
            f"# picture    {Path(picture).name if picture else '(none)'}, {W} x {H} px",
            f"# f          {cam['f']:.3f} px = {n['lens']:.3f} mm on a "
            f"{sensor_mm:g} mm sensor (AUTO fit)",
            f"# principal point ({cam['pp'][0]:.2f}, {cam['pp'][1]:.2f}) px; shift "
            f"({n['shift_x']:.6f}, {n['shift_y']:.6f})",
            f"# placement  the world origin is {distance:g} units straight ahead of "
            "the camera. A picture has no scale: that is a choice, not a measurement"]
    head += [f"# {line}" for line in notes]
    body = f'''
import bpy
from mathutils import Matrix

PICTURE = {pic}
RES = ({W}, {H})
LENS_MM = {n['lens']!r}
SENSOR_MM = {n['sensor_width']!r}
SHIFT = ({n['shift_x']!r}, {n['shift_y']!r})
MATRIX_WORLD = {json.dumps(n['matrix_world'])}

scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = RES
scene.render.resolution_percentage = 100

data = bpy.data.cameras.new("Persimmon camera")
data.sensor_fit = 'AUTO'
data.sensor_width = SENSOR_MM
data.lens = LENS_MM
data.shift_x, data.shift_y = SHIFT
cam = bpy.data.objects.new("Persimmon camera", data)
scene.collection.objects.link(cam)
cam.matrix_world = Matrix(MATRIX_WORLD)
scene.camera = cam

origin = bpy.data.objects.new("Persimmon origin", None)
origin.empty_display_type = 'ARROWS'
scene.collection.objects.link(origin)

if PICTURE:
    try:
        bg = data.background_images.new()
        bg.image = bpy.data.images.load(PICTURE)
        bg.display_depth = 'BACK'
        bg.frame_method = 'FIT'
        data.show_background_images = True
    except Exception as exc:
        print("Persimmon: picture not attached:", exc)
'''
    return "\n".join(head) + "\n" + body
