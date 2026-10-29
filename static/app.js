/* Persimmon -- the marking tool.
 *
 * Semi-automatic means the person marks what the pixels cannot give. Hand
 * marking measures about 9 px against the paint, which is ample for every
 * check here, so this page is the backbone and detection is an accelerator.
 *
 * One IIFE, on purpose: a stray apostrophe anywhere in here kills the whole
 * client, so it is checked with `node --check` and then loaded, every time.
 */
(function () {
  'use strict';

  // What a hand mark is worth, in SCREEN pixels. The hand's precision is a
  // hand-eye limit, so it is constant on screen and shrinks in the picture as
  // you zoom in. Comparing it against image pixels silently under-flags every
  // image bigger than the testdata -- all five of which are 1280 on the long
  // edge, which is why that went unseen. Measured against the paint at 1:1.
  var MARK_PX = 9.0;
  var TRACE_PX = 6;                        // screen px between traced points

  // Every layer kind, and what it holds. `draw` is how marks arrive:
  //   line     drag a segment        poly  click a run of points
  //   pair     click two points      none  nothing to draw, it refers to others
  var KINDS = {
    family:  { draw: 'line', label: 'line family',  blurb: 'Lines that run to one vanishing point. Two make a vanishing point; three or more also measure how well they agree. Click along a line instead of dragging, three points or more, then Enter: that tests whether it is straight, and in 5-point it is how a line is marked.' },
    surface: { draw: 'line', label: 'surface',      blurb: 'A plane: either the two vanishing points that span it, or level with the ground, which gives it the horizon. An arc is read against its own surface -- never the horizon unless it lies flat.' },
    arc:     { draw: 'poly', label: 'arc',          blurb: 'A rim that should be circular. Mark round it; below 120 deg of turning its shape is not determined and the check refuses.' },
    ellipse: { draw: 'line', label: 'ellipse',      blurb: 'Circles lying flat on the ground, or in a surface you name. Drag once right across one, the long way, and the ellipse a real circle of that width would cast is drawn for you \u2014 on a horizontal plane the drag is levelled onto the horizon, because that is the one diameter whose image runs parallel to it. Drag again for another: they are all built on the one plane under the one camera, so a row of them agrees with the horizon by construction. To have one you drew MEASURED instead, switch to marking and click round its rim. Or choose "circle in a square" and click inside a cell between two lines of each of two families: the circle inscribed in that square is drawn, and the card beside it says how square the cell is.' },
    arch:    { draw: 'poly', label: 'arches and roofs', blurb: 'Drag the springing line between the two supports and move the top, and the arch a real round or pointed one would cast on that wall is drawn for you. Straight sides make it a roof instead: a gable, and slid back along the depth lines a pitched roof, whose slope vanishing point is then read off the drawing. Drag again for the next bay: an arcade is one layer, every arch on the one wall under the one camera. To have one you drew MEASURED instead, switch to marking and click round it \u2014 the near limb, and the far one if the vault shows it.' },
    polygon: { draw: 'poly', label: 'regular polygon', blurb: 'The corners, in order round the shape.' },
    figures: { draw: 'pair', label: 'figures',      blurb: "Feet then head, for each figure. The feet mark goes on the FIGURE's own silhouette where it meets the ground -- never on the ground's own edge, which is at a different depth." },
    posts:   { draw: 'poly', label: 'spaced row',   blurb: "One point per post, in order along the row, each where the POST's own silhouette meets what it stands on -- not on that surface's edge." },
    axis:    { draw: 'poly', label: 'axis',         blurb: 'Points along one straight axis, in order.' },
    shadow:  { draw: 'triple', label: 'cast shadows', blurb: "Three clicks an object: where it meets the ground, its top, and the TIP of its cast shadow. Two objects are enough. The rays run to the light, the shadows run along the ground to the light's own foot, and a light stands over its foot \u2014 so the light, its foot and the vertical vanishing point are in one line, for a lamp in the room and for the sun alike. The sun is the case whose foot goes all the way to the horizon, and the check says which it was." },
    pairs:   { draw: 'pair', label: 'landmark pairs', blurb: 'Left then right, for each pair that should be symmetric.' },
    landmark:{ draw: 'pair', label: 'known size',   blurb: "Base then top, for each thing whose REAL size you know -- a 180 cm figure, a 2 m door. That size turns the camera into the distance from the viewer to it. Mark the base where the thing's own silhouette meets the ground, and mark it as tall on the page as you can: a figure 80 px tall is worth +-29 %, one 30 px tall is refused." }
  };

  var COLORS = ['#2ca8ff', '#ffd400', '#ff6b35', '#00e676', '#c77dff',
                '#ff2d55', '#4dd0e1', '#ffb4a2', '#9ccc65', '#f06292'];

  var S = {
    image: null,            // {path, w, h}
    bitmap: null,
    layers: [],             // see newLayer()
    sel: null,              // selected layer id
    nextId: 1,
    view: { x: 0, y: 0, k: 1 },
    pending: [],            // points of the run being drawn
    dragLine: null,         // {x1,y1,x2,y2} while dragging a family line
    dragPt: null,           // {layer, path, i} while moving a placed point
    results: null,
    consensus: null,
    plan: null, planKey: null,  // the top view and the points it was asked for: see paintPlan()
    archFar: false,         // marking the far limb of an arch
    ringMark: false,        // marking an ellipse's rim rather than drawing it
    inSquare: false,        // an ellipse layer is taking clicks inside cells
    // An arch is BUILT by default, like an ellipse is drawn by default: both
    // tools exist to put the right curve on the page. Marking round one
    // already drawn is the other half, and it is one button away.
    archDraw: true,         // building an arch from three marks rather than marking it
    tracing: false,         // the mouse is down, sampling a curve
    supportEdge: false,     // marking what the contacts rest on
    straight: null,         // the last straightened view, {id, url}
    undo: [], redo: [],     // see history, below; current state is undo's top
    cand: null,             // detected segments on offer: {segs, famOf, families}
    perspective: 'three',   // the type being CHECKED, not detected: see /api/consensus
    scale: null,            // {value, unit, along}: how big the picture really is
    ref: null,              // a camera read from a file: see cameraIn()
    refNote: null,          // why there is none, when one was taken away
    trim: null,             // {corners:[[x,y]x4], out:[w,h]} in the ORIGINAL photo
    trimEdit: false,        // placing those corners, which shows the whole photo
    trimDraft: null,        // the corners being dragged
    dragCorner: -1          // which one, while dragging
  };

  // --------------------------------------------------------------- utilities

  var $ = function (q) { return document.querySelector(q); };
  var el = function (t, a) { var n = document.createElement(t); if (a) Object.assign(n, a); return n; };

  function toast(msg, bad) {
    var t = $('#toast');
    t.textContent = msg;
    t.className = 'status' + (bad ? ' bad' : '');
    clearTimeout(toast._t);
    toast._t = setTimeout(function () { t.className = 'status hidden'; }, 4200);
  }
  function busy(on) { $('#busy').className = on ? 'spinner' : 'spinner hidden'; }

  function post(url, body) {
    busy(true);
    return fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(function (r) { return r.json(); }).finally(function () { busy(false); });
  }

  function num(v, d) {
    if (v === null || v === undefined || (typeof v === 'number' && !isFinite(v))) return '--';
    return typeof v === 'number' ? v.toFixed(d === undefined ? 2 : d) : String(v);
  }

  // ----------------------------------------------------------------- history
  // Snapshots, not inverse operations. The marks are a few hundred points, so
  // a JSON copy costs less than a command stack's bookkeeping would -- and it
  // cannot drift out of step with the thing it is meant to undo. The TOP of
  // the undo stack is always the state on screen.

  var HIST = 80;

  function snapshot() {
    // the offered segments travel with the marks: taking a family and undoing
    // it must put those segments back on offer, not lose them from both places
    return JSON.stringify({ layers: S.layers, nextId: S.nextId,
                            pending: S.pending, sel: S.sel, cand: S.cand,
                            trim: S.trim, trimEdit: S.trimEdit,
                            trimDraft: S.trimDraft,
                            perspective: S.perspective, scale: S.scale,
                            ref: S.ref });
  }

  // Safe to call from any path that MIGHT have changed something: identical
  // states collapse, so a click that did nothing leaves no dead undo step.
  function commit() {
    var s = snapshot();
    if (S.undo.length && S.undo[S.undo.length - 1] === s) return;
    S.undo.push(s);
    if (S.undo.length > HIST) S.undo.shift();
    S.redo.length = 0;
  }

  function restore(s) {
    var d = JSON.parse(s);
    // what is hidden is a view, not a mark: undo must not bring a layer back
    // on screen that the person put away
    var hid = {};
    S.layers.forEach(function (L) { hid[L.id] = !!L.hidden; });
    S.layers = d.layers; S.nextId = d.nextId;
    S.layers.forEach(function (L) { if (L.id in hid) L.hidden = hid[L.id]; });
    S.pending = d.pending; S.sel = d.sel; S.cand = d.cand || null;
    S.dragLine = null; S.dragPt = null; S.tracing = false; S.dragCorner = -1;
    // Undoing across a trim change has to re-fetch the pixels, because the
    // picture on screen is a different picture -- not just a different view
    // of one. Compare what is DISPLAYED, so leaving corner-editing (which
    // shows the whole photograph) reloads too.
    var wasShown = JSON.stringify(shownTrim());
    S.trim = d.trim || null;
    S.trimEdit = !!d.trimEdit; S.trimDraft = d.trimDraft || null;
    setPerspective(d.perspective || 'three');
    setScale(d.scale || null);
    S.ref = d.ref || null; camState();
    if (S.image && JSON.stringify(shownTrim()) !== wasShown) {
      showPicture(S.image.path, true, function () { trimState(); paint(); });
    }
    trimState();
    // A vanishing point can arrive after the commit that should have carried
    // it, which would leave a restored state showing the fit of a different
    // set of lines. Re-fit instead of trying to snapshot something asynchronous.
    S.layers.forEach(function (L) { if (L.kind === 'family') fitFamily(L); });
    refreshLayers(); showEditor(); paint();
  }

  function undo() {
    if (S.undo.length < 2) { toast('nothing to undo'); return; }
    S.redo.push(S.undo.pop());
    restore(S.undo[S.undo.length - 1]);
  }

  function redo() {
    if (!S.redo.length) { toast('nothing to redo'); return; }
    var s = S.redo.pop();
    S.undo.push(s);
    restore(s);
  }

  // ------------------------------------------------------------------ layers

  function newLayer(kind) {
    var k = KINDS[kind];
    var n = S.layers.filter(function (L) { return L.kind === kind; }).length + 1;
    var L = {
      id: 'L' + (S.nextId++), kind: kind, name: k.label + ' ' + n,
      color: COLORS[(S.layers.length) % COLORS.length],
      lines: [], pts: [], far: [], pairs: [], crown: [], refs: {},
      curves: [],              // a family's lines marked by a run of points
      cells: [], cellOut: [],  // an ellipse's circles in squares: where, and what came back
      through: [],             // per arch: one more point, ON the curve
      bezAdj: [],              // per arch, per handle: the hand's offset
      markK: [],
      resolve: 0.05, px_per_mm: '', resolve_mm: 2.0, count: '',
      size: 180, size_unit: 'cm', on_ground: true,
      span: [0, -1], unit: [0, 1]
    };
    S.layers.push(L);
    return L;
  }

  function layer(id) {
    for (var i = 0; i < S.layers.length; i++) if (S.layers[i].id === id) return S.layers[i];
    return null;
  }
  function selected() { return S.sel ? layer(S.sel) : null; }
  function families() { return S.layers.filter(function (L) { return L.kind === 'family'; }); }

  // The horizon, from the horizontal families that are not themselves held on
  // it. Holding a family on a line its own lines helped draw would measure
  // nothing, so the two that define it are always other families.
  function horizonVps(exclude) {
    var h = families().filter(function (F) {
      return F.vp && !F.vertical && !F.on_horizon && !(F.refs || {}).square_of
             && F !== exclude;
    });
    return h.length >= 2 ? h.slice(0, 2) : null;
  }

  // One horizontal family and the uprights give a horizon too, through the
  // camera: inv(K Kt) v. The server derives it from the document; the page
  // only has to know it is there, or the third family of a view down a nave
  // cannot be held on a horizon the check path already knows.
  function camHorizon(exclude) {
    var fs = families();
    return !horizonVps(exclude)
      && fs.some(function (F) { return F.vertical && F.vp; })
      && fs.some(function (F) {
           return F !== exclude && !F.vertical && F.vp && !(F.refs || {}).square_of;
         });
  }

  // Only figures. A spaced row is normally a row of features IN a plane --
  // cabinet dividers along the shelf they hang under -- so its contacts
  // belong ON the edge, and the check called correct marking a mistake.
  // Moving the edge somewhere unrelated then turned it green, which says
  // what the test was worth here.
  // Two ends and a count give the whole row. u = 1/(t_vp - t) is affine in
  // the real distance along the run, so evenly spaced out there means u steps
  // evenly along here. Built on the page, not the server, because dragging an
  // end has to redraw it at once and nothing else needs the construction --
  // the server only ever has to refuse to grade it.
  function builtRow(L) {
    var V = L.kind === 'posts' ? rowVp(L) : null;
    var n = Number(L.count);
    if (!V || L.pts.length !== 2 || !(n >= 3)) return null;
    var A = L.pts[0], B = L.pts[1];
    var ex = B[0] - A[0], ey = B[1] - A[1];
    var len = Math.hypot(ex, ey);
    if (len < 1e-9) return null;
    ex /= len; ey /= len;
    var tv = (V[0] - A[0]) * ex + (V[1] - A[1]) * ey;
    // a row cannot pass through its own vanishing point: that is infinitely
    // far away along it, so no post can be marked on the far side of it
    if (tv > -1e-6 && tv < len + 1e-6) return null;
    var out = [];
    for (var i = 0; i < n; i++) {
      var u = 1 / tv + (1 / (tv - len) - 1 / tv) * i / (n - 1);
      out.push([A[0] + (tv - 1 / u) * ex, A[1] + (tv - 1 / u) * ey]);
    }
    return out;
  }

  // The row's own vanishing point, from the plane it stands on. The feet lie
  // on a line IN that plane, so that line's vanishing point is where it meets
  // the plane's vanishing line -- nothing to name and nothing to get wrong.
  // Naming a family by hand stays as the fallback, for a row on no surface.
  function rowVp(L) {
    var S2 = L.refs.surface && layer(L.refs.surface);
    if (S2 && L.pts.length >= 2) {
      var v = surfaceVps(S2);
      var m = v && meet(L.pts[0], L.pts[L.pts.length - 1], v[0], v[1]);
      if (m) return m;
    }
    var F = L.refs.vp && layer(L.refs.vp);
    return (F && F.vp) ? F.vp : null;
  }

  function rowPts(L) { return builtRow(L) || L.pts; }

  // Lamp posts are the same height, so their tops lie on a world line
  // parallel to the one their feet lie on -- the same vanishing point. Mark
  // the top of the first and every other top is the meet of its own upright
  // with that line. Returns one segment per post, or null.
  function rowUprights(L) {
    var U = L.refs.upright && layer(L.refs.upright);
    var R = rowVp(L);
    var top0 = L.far[0];
    if (!U || !U.vp || !R || !top0) return null;
    var feet = rowPts(L);
    if (feet.length < 2) return null;
    var out = [];
    for (var i = 0; i < feet.length; i++) {
      var tk = meet(feet[i], U.vp, top0, R);
      if (tk) out.push([feet[i][0], feet[i][1], tk[0], tk[1]]);
    }
    return out.length ? out : null;
  }

  function contactful(L) { return L.kind === 'figures'; }

  // What a click does on this layer right now. The arch and the contact layers
  // both have a second thing to mark, and the toggle says which.
  function drawMode(L) {
    if (contactful(L) && S.supportEdge) return 'line';
    // an ellipse is given two ways: dragged (the circle its surface gives) or
    // marked round the rim (the circle the drawing actually has)
    if (L.kind === 'ellipse') return S.inSquare ? 'cell' : S.ringMark ? 'poly' : 'line';
    // and so is an arch: drawn from three marks, or marked round the one in
    // the picture. The far-limb toggle only applies while marking.
    if (L.kind === 'arch' && S.archDraw) return 'line';
    return KINDS[L.kind].draw;
  }

  // Each contact's perpendicular distance to the support edge. A contact that
  // lands ON that edge is bug 19: the edge is the crisp line the eye goes to,
  // but the figure meets the support further back, at a different depth.
  function contactGaps(L) {
    if (!contactful(L) || !L.lines.length) return null;
    var s = L.lines[0];
    var dx = s[2] - s[0], dy = s[3] - s[1];
    var n = Math.hypot(dx, dy);
    if (n < 1e-9) return null;
    var pts = L.kind === 'figures' ? L.pairs.map(function (q) { return q[0]; }) : L.pts;
    return pts.map(function (q, i) {
      var g = Math.abs((q[0] - s[0]) * dy - (q[1] - s[1]) * dx) / n;
      // in the picture, and as the hand saw it -- the test is on the second,
      // taken at the zoom this mark was placed at, not the zoom you are at now
      return { px: g, screen: g * (L.markK[i] || S.view.k) };
    });
  }

  function markCount(L) {
    if (L.kind === 'family') return L.lines.length + L.curves.length;
    if (KINDS[L.kind].draw === 'pair' || KINDS[L.kind].draw === 'triple') return L.pairs.length;
    if (L.kind === 'arch') return L.pts.length + L.far.length;
    if (L.kind === 'surface') return L.lines.length;
    if (KINDS[L.kind].draw === 'none') return 0;
    if (L.kind === 'ellipse') return L.pts.length + L.cells.length;
    return L.pts.length;
  }

  // ------------------------------------------------------------- coordinates

  function toScreen(p) {
    return [(p[0] - S.view.x) * S.view.k, (p[1] - S.view.y) * S.view.k];
  }
  function toWorld(sx, sy) {
    return [sx / S.view.k + S.view.x, sy / S.view.k + S.view.y];
  }
  function canvasXY(ev) {
    var r = $('#view').getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top];
  }

  function fitView() {
    if (!S.image) return;
    var c = $('#view');
    var k = Math.min(c.clientWidth / S.image.w, c.clientHeight / S.image.h) * 0.96;
    if (!(k > 0)) return;   // no layout yet: k = 0 blanks the canvas for good
    S.view.k = k;
    S.view.x = S.image.w / 2 - c.clientWidth / (2 * k);
    S.view.y = S.image.h / 2 - c.clientHeight / (2 * k);
    paint();
  }

  // --------------------------------------------------------------- rendering

  function paint() {
    var c = $('#view'), g = c.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var W = c.clientWidth, H = c.clientHeight;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    $('#zoomLbl').textContent = Math.round(S.view.k * 100) + '%';
    paintPlan();
    if (!S.bitmap) return;

    g.imageSmoothingEnabled = S.view.k < 2;
    var o = toScreen([0, 0]);
    g.drawImage(S.bitmap, o[0], o[1], S.image.w * S.view.k, S.image.h * S.view.k);

    drawCand(g);
    drawTrim(g, W, H);

    var showFits = $('#showFits').checked;
    var showMoves = $('#showMoves').checked;

    S.layers.forEach(function (L) {
      if (L.hidden) return;
      var on = L.id === S.sel;
      g.lineWidth = on ? 2.4 : 1.6;
      g.strokeStyle = L.color;
      g.fillStyle = L.color;
      if (L.kind === 'surface') {
        var q = surfaceQuad(L);
        if (q) {
          g.beginPath();
          q.forEach(function (c, i) {
            var t = toScreen(c);
            if (i) g.lineTo(t[0], t[1]); else g.moveTo(t[0], t[1]);
          });
          g.closePath();
          g.save(); g.globalAlpha = on ? 0.22 : 0.12; g.fill(); g.restore();
          g.stroke();
          // only the diagonal's own ends are grabbable; the other two follow
          if (on) { dot(g, q[0], 3.4); dot(g, q[2], 3.4); }
        } else if (L.lines.length) {
          g.save(); g.setLineDash([5, 5]); g.globalAlpha = 0.7;
          strokeSeg(g, L.lines[0]); g.restore();
        }
      } else if (L.kind === 'ellipse') {
        // the circle this picture's own camera casts, faint, when it
        // differs: its long diameter leans away from the centre of vision and
        // the solid one's does not, so the lean is there to be looked at
        (L.ringsTrue || []).forEach(function (q) {
          if (!q) return;
          g.save(); g.globalAlpha = 0.45; g.setLineDash([3, 3]);
          strokeRing(g, q); g.restore();
        });
        (L.rings || []).forEach(function (q) { strokeRing(g, q); });
        // the circles in squares, each with its cell drawn faint beside it so
        // the tangent points on the sides can be looked at, not trusted
        (L.cellOut || []).forEach(function (c) {
          if (!c.ring) return;
          strokeRing(g, c.ring);
          g.save(); g.setLineDash([2, 4]); g.globalAlpha = 0.5; g.lineWidth = 1.2;
          strokeRing(g, c.quad); g.restore();
        });
        if (on) (L.cells || []).forEach(function (c) { dot(g, c, 3.4); });
        // the drag each one was asked for by: the ring's long diameter,
        // faint, whose two ends are what a drag grabs
        g.save(); g.setLineDash([4, 4]); g.globalAlpha = 0.6; g.lineWidth = 1.4;
        L.lines.forEach(function (q) { strokeSeg(g, q); });
        g.restore();
        if (on) L.lines.forEach(function (q) {
          dot(g, [q[0], q[1]], 3.4); dot(g, [q[2], q[3]], 3.4);
        });
      } else if (L.kind === 'arch' && (L.arches || L.crown.length)) {
        (L.arches || []).forEach(function (q, i) {
          var segs = bezOf(L, i);
          if (!segs) { drawRun(g, q, false, false); return; }
          strokeBez(g, segs);
          var fs = bezFar(L, i), V = (L.vault || [])[i];
          if (fs) {
            strokeBez(g, fs);
            g.save(); g.globalAlpha = 0.75;
            [[segs[0][0], fs[0][0]],
             [segs[segs.length - 1][3], fs[fs.length - 1][3]],
             [V.near_crown, V.far_crown]].forEach(function (e) {
              if (e[0] && e[1]) strokeSeg(g, [e[0][0], e[0][1], e[1][0], e[1][1]]);
            });
            g.restore();
          }
          if (!on) return;
          // the handles, as in a drawing program: a bar out from the anchor
          // to a knob. Only on the selected layer, or an arcade would be all
          // knobs and no arches.
          g.save(); g.lineWidth = 1.2; g.globalAlpha = 0.85;
          segs.forEach(function (P) {
            [[0, 1], [3, 2]].forEach(function (e) {
              strokeSeg(g, [P[e[0]][0], P[e[0]][1], P[e[1]][0], P[e[1]][1]]);
              knob(g, P[e[1]], 4);
            });
          });
          g.restore();
        });
        // the springing line each arch stands on and the two supports under
        // it, faint: the construction, not the drawing
        if (L.lines.length) {
          g.save(); g.setLineDash([4, 4]); g.globalAlpha = 0.6; g.lineWidth = 1.4;
          L.lines.forEach(function (q) { strokeSeg(g, q); });
          archLegs(L).forEach(function (q) { strokeSeg(g, q); });
          g.restore();
          if (on) L.lines.forEach(function (q) {
            dot(g, [q[0], q[1]], 3.4); dot(g, [q[2], q[3]], 3.4);
          });
        }
        L.crown.forEach(function (c) { dot(g, c, on ? 5 : 3.4); });
        (L.through || []).forEach(function (c) {
          if (!c) return;
          dot(g, c, on ? 3.6 : 2.6);
          var a = toScreen(c);
          g.beginPath(); g.arc(a[0], a[1], on ? 7 : 5, 0, 6.2832); g.stroke();
        });
      } else if (contactful(L)) {
        g.save(); g.setLineDash([6, 4]); g.globalAlpha = 0.6; g.lineWidth = 1.4;
        L.lines.forEach(function (s) { strokeSeg(g, s); });
        g.restore();
      } else {
        var bad = strayList(L);
        L.lines.forEach(function (s, i) {
          if (bad.indexOf(i) < 0) { strokeSeg(g, s); return; }
          g.save();
          g.strokeStyle = '#fff'; g.lineWidth = (on ? 2.4 : 1.6) + 4;
          g.globalAlpha = 0.65; strokeSeg(g, s);
          g.restore();
          strokeSeg(g, s);
        });
        // The ends are what a drag grabs. Unmarked they are an invisible 8 px
        // target, and a miss draws a new line instead of moving the old one,
        // which is why editing a loaded file felt impossible.
        if (on) L.lines.forEach(function (s) {
          dot(g, [s[0], s[1]], 3.4); dot(g, [s[2], s[3]], 3.4);
        });
        (L.curves || []).forEach(function (q) { drawRun(g, q, on, false); });
      }
      var run = rowPts(L);
      drawRun(g, run, on, L.kind === 'polygon' || L.kind === 'ellipse');
      // the built posts are a guide; only the two ends are anyone's marks
      if (run !== L.pts) L.pts.forEach(function (q) { dot(g, q, on ? 5 : 3.4); });
      var ups = L.kind === 'posts' && rowUprights(L);
      if (ups) ups.forEach(function (s) { strokeSeg(g, s); });
      if (L.far.length) drawRun(g, L.far, on, false);
      L.pairs.forEach(function (p) {
        // a pair is one segment; a shadow's three marks are the upright, the
        // shadow along the ground, and the ray from the top closing them
        for (var j = 1; j < p.length; j++) {
          strokeSeg(g, [p[j - 1][0], p[j - 1][1], p[j][0], p[j][1]]);
        }
        if (p.length > 2) {
          strokeSeg(g, [p[0][0], p[0][1], p[p.length - 1][0], p[p.length - 1][1]]);
        }
        p.forEach(function (q) { dot(g, q, on ? 4 : 3); });
      });
      // where the light this check settled on actually ends each shadow:
      // the answer drawn, beside what was marked
      var tipAt = L.kind === 'shadow' && showFits && S.results
        && (((S.results.results || {})[L.id] || {}).conditioning || {}).tip_at;
      if (tipAt) {
        g.save(); g.setLineDash([5, 4]); g.globalAlpha = 0.85;
        tipAt.forEach(function (q, i) {
          var pr = L.pairs[i];
          if (!pr || pr.length < 3) return;
          strokeSeg(g, [pr[0][0], pr[0][1], q[0], q[1]]);
          var a = toScreen(q);
          g.beginPath(); g.arc(a[0], a[1], on ? 7 : 5, 0, 6.2832); g.stroke();
        });
        g.restore();
      }
      if (showFits && L.kind === 'family' && L.vp) drawVP(g, L);
    });

    // the run being drawn, and the line being dragged
    var cur = selected();
    if (cur) {
      g.strokeStyle = cur.color; g.fillStyle = cur.color; g.lineWidth = 2.4;
      if (S.pending.length) {
        drawRun(g, S.pending, true, false);
        g.setLineDash([4, 4]);
        g.strokeStyle = 'rgba(255,255,255,.7)';
        g.setLineDash([]);
      }
      if (S.dragLine) strokeSeg(g, [S.dragLine.x1, S.dragLine.y1, S.dragLine.x2, S.dragLine.y2]);
    }

    if (showFits) drawHorizon(g, W);
    if (showFits && S.ref) drawReference(g, W);
    if (showFits && S.consensus) drawCurvi(g);
    if (showMoves && S.consensus) drawMoves(g);
  }

  // 1-based in the readout because that is how a person counts lines; 0-based
  // here because that is how the array is indexed.
  function strayList(L) {
    var c = L.vpInfo && L.vpInfo.conditioning;
    if (!c || !c.stray_lines) return [];
    return c.stray_lines.map(function (n) { return n - 1; });
  }

  // The plane a surface stands for, as something to look at. Two clicks give
  // the diagonal; the other two corners are where the edges from each end,
  // run to the two vanishing points, meet. So the patch is a perspective
  // image of a parallelogram lying IN the plane -- if it does not sit on the
  // table, the vanishing points are not the table's.
  function meet(p, vp, q, wp) {
    var l = [p[1] - vp[1], vp[0] - p[0], p[0] * vp[1] - p[1] * vp[0]];
    var m = [q[1] - wp[1], wp[0] - q[0], q[0] * wp[1] - q[1] * wp[0]];
    var x = l[1] * m[2] - l[2] * m[1];
    var y = l[2] * m[0] - l[0] * m[2];
    var w = l[0] * m[1] - l[1] * m[0];
    return Math.abs(w) < 1e-9 ? null : [x / w, y / w];
  }

  function surfaceVps(L) {
    if (L.level) {
      var h = horizonVps(null);
      return h ? [h[0].vp, h[1].vp] : null;
    }
    var a = L.refs._a && layer(L.refs._a), b = L.refs._b && layer(L.refs._b);
    return (a && a.vp && b && b.vp) ? [a.vp, b.vp] : null;
  }

  function surfaceQuad(L) {
    var v = surfaceVps(L);
    if (!v || !L.lines.length) return null;
    var s = L.lines[0], A = [s[0], s[1]], C = [s[2], s[3]];
    var B = meet(A, v[0], C, v[1]), D = meet(A, v[1], C, v[0]);
    return (B && D) ? [A, B, C, D] : null;
  }

  function strokeSeg(g, s) {
    var a = toScreen([s[0], s[1]]), b = toScreen([s[2], s[3]]);
    g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
  }
  function strokePath(g, q) {
    g.beginPath();
    q.forEach(function (c, i) {
      var t = toScreen(c);
      if (i) g.lineTo(t[0], t[1]); else g.moveTo(t[0], t[1]);
    });
    g.stroke();
  }
  function strokeRing(g, q) {
    g.beginPath();
    q.forEach(function (c, i) {
      var t = toScreen(c);
      if (i) g.lineTo(t[0], t[1]); else g.moveTo(t[0], t[1]);
    });
    g.closePath(); g.stroke();
  }
  function dot(g, p, r) {
    var a = toScreen(p);
    g.beginPath(); g.arc(a[0], a[1], r, 0, 6.2832); g.fill();
  }
  function strokeBez(g, segs) {
    g.beginPath();
    segs.forEach(function (P, s) {
      var a = toScreen(P[0]), b = toScreen(P[1]),
          c = toScreen(P[2]), d = toScreen(P[3]);
      if (s) g.lineTo(a[0], a[1]); else g.moveTo(a[0], a[1]);
      g.bezierCurveTo(b[0], b[1], c[0], c[1], d[0], d[1]);
    });
    g.stroke();
  }
  // a handle: the bar out from its anchor, and a hollow knob to take hold of.
  // Hollow, because the anchors are already filled dots and the two do
  // different things -- an anchor moves the arch, a knob only shapes it.
  function knob(g, p, r) {
    var a = toScreen(p);
    g.beginPath(); g.arc(a[0], a[1], r, 0, 6.2832);
    g.save(); g.fillStyle = '#fff'; g.fill(); g.restore(); g.stroke();
  }
  function drawRun(g, pts, on, close) {
    if (!pts.length) return;
    g.beginPath();
    pts.forEach(function (p, i) {
      var a = toScreen(p);
      if (i === 0) g.moveTo(a[0], a[1]); else g.lineTo(a[0], a[1]);
    });
    if (close && pts.length > 2) g.closePath();
    g.stroke();
    if (on || pts.length < 40) pts.forEach(function (p) { dot(g, p, on ? 3.4 : 2.2); });
  }

  function drawVP(g, L) {
    // The vanishing point is usually off the picture, so draw the rays toward
    // it rather than the point: an arrow off-screen says nothing.
    var v = L.vp;
    g.save();
    g.setLineDash([3, 5]); g.globalAlpha = 0.55; g.lineWidth = 1;
    L.lines.forEach(function (s) {
      var m = [(s[0] + s[2]) / 2, (s[1] + s[3]) / 2];
      strokeSeg(g, [m[0], m[1], v[0], v[1]]);
    });
    g.restore();
    var p = toScreen(v);
    if (p[0] > -50 && p[1] > -50 && p[0] < g.canvas.clientWidth + 50) {
      g.save(); g.strokeStyle = L.color; g.lineWidth = 2;
      g.beginPath(); g.arc(p[0], p[1], 7, 0, 6.2832); g.stroke();
      g.beginPath(); g.moveTo(p[0] - 11, p[1]); g.lineTo(p[0] + 11, p[1]);
      g.moveTo(p[0], p[1] - 11); g.lineTo(p[0], p[1] + 11); g.stroke();
      g.restore();
    }
  }

  function drawHorizon(g, W) {
    var f = horizonVps(null);
    if (!f) return;
    var a = f[0].vp, b = f[1].vp;
    if (Math.abs(b[0] - a[0]) < 1e-6) return;
    var y = function (x) { return a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]); };
    var w0 = toWorld(0, 0)[0], w1 = toWorld(W, 0)[0];
    g.save(); g.strokeStyle = $('#colHorizon').value; g.lineWidth = 1.6; g.setLineDash([7, 5]);
    strokeSeg(g, [w0, y(w0), w1, y(w1)]);
    g.restore();
  }

  // The reference camera's claim, drawn where the marks can be held against
  // it: a dotted ray from each marked line toward the point its axis says the
  // line should run to (a short fan from the picture while an axis has no
  // family yet), and the horizon it implies. These are the very numbers the
  // server priced the marks against -- describe() worked them out once, in
  // Python, from the one camera -- so what is drawn and what is quoted cannot
  // be two sets.
  var AXIS_COLOR = { X: '#ff3352', Y: '#8bdc00', Z: '#2890ff' };
  var REF_COLOR = '#d070ff';

  function refAnchors(a) {
    var rc = S.consensusBoth && S.consensusBoth.reference;
    var per = (rc && rc.conditioning && rc.conditioning.per_family) || {};
    var out = [];
    Object.keys(per).forEach(function (id) {
      var L = layer(id);
      if (L && per[id].axis === a) {
        L.lines.forEach(function (s) { out.push([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2]); });
      }
    });
    if (out.length) return out;
    var w = S.image.w, h = S.image.h;
    return [[0.5, 0.5], [0.25, 0.3], [0.75, 0.3], [0.25, 0.75], [0.75, 0.75]]
      .map(function (q) { return [q[0] * w, q[1] * h]; });
  }

  function drawReference(g, W) {
    var d = S.ref.describe, diag = Math.hypot(S.image.w, S.image.h);
    var cw = g.canvas.clientWidth, ch = g.canvas.clientHeight;
    g.save();
    g.font = '11px system-ui';
    ['X', 'Y', 'Z'].forEach(function (a) {
      var vp = d.vps[a], far = d.far[a];
      if (!vp && !far) return;
      var anchors = refAnchors(a);
      g.strokeStyle = g.fillStyle = AXIS_COLOR[a];
      g.lineWidth = 1.3; g.globalAlpha = 0.75; g.setLineDash([2, 5]);
      anchors.forEach(function (m) {
        if (vp) strokeSeg(g, [m[0], m[1], vp[0], vp[1]]);
        else strokeSeg(g, [m[0] - far[0] * diag, m[1] - far[1] * diag,
                           m[0] + far[0] * diag, m[1] + far[1] * diag]);
      });
      g.setLineDash([]); g.globalAlpha = 1;
      // its name where it can be read: on the point if that is on screen,
      // otherwise a little way along the first ray
      var at, tag = 'ref ' + a + (vp ? '' : ' (parallel)');
      var m0 = anchors[0];
      if (vp) {
        var p = toScreen(vp);
        if (p[0] > 0 && p[1] > 0 && p[0] < cw && p[1] < ch) {
          g.lineWidth = 2; g.beginPath(); g.arc(p[0], p[1], 7, 0, 6.2832); g.stroke();
          at = [p[0] + 10, p[1] - 8];
        } else {
          var n = Math.hypot(vp[0] - m0[0], vp[1] - m0[1]) || 1;
          at = toScreen([m0[0] + (vp[0] - m0[0]) / n * 0.18 * diag,
                         m0[1] + (vp[1] - m0[1]) / n * 0.18 * diag]);
        }
      } else {
        at = toScreen([m0[0] + far[0] * 0.18 * diag, m0[1] + far[1] * 0.18 * diag]);
      }
      g.fillText(tag, at[0] + 4, at[1] - 4);
    });
    var hz = d.horizon;
    if (hz && Math.abs(hz[1]) > 1e-6) {
      var y = function (x) { return -(hz[0] * x + hz[2]) / hz[1]; };
      var w0 = toWorld(0, 0)[0], w1 = toWorld(W, 0)[0];
      g.strokeStyle = g.fillStyle = REF_COLOR; g.lineWidth = 1.8;
      g.setLineDash([12, 4, 2, 4]);
      strokeSeg(g, [w0, y(w0), w1, y(w1)]);
      g.setLineDash([]);
      var lab = toScreen([w0 + 8 / S.view.k, y(w0 + 8 / S.view.k)]);
      g.fillText('ref horizon', lab[0], lab[1] - 5);
    }
    g.restore();
  }

  function drawMoves(g) {
    // Only the lines that actually need moving. Drawing a correction for a
    // line already inside marking accuracy covers the original in red and
    // tells someone to redraw a line that was right. What the marks need to
    // be consistent is red; what the reference camera asks is violet.
    var both = S.consensusBoth || {};
    g.save(); g.lineWidth = 2.2; g.setLineDash([9, 5]);
    [['free', $('#colMoves').value], ['reference', REF_COLOR]].forEach(function (kv) {
      var c = (both[kv[0]] || {}).conditioning || {};
      var per = c.per_family || {};
      var floor = c.mark_accuracy_px || MARK_PX;
      g.strokeStyle = kv[1];
      Object.keys(per).forEach(function (id) {
        var d = per[id];
        // a segment, or in 5-point the arc the line is moved onto
        (d.corrected || []).forEach(function (s, i) {
          if ((d.move_px || [])[i] > floor) (Array.isArray(s[0]) ? strokePath : strokeSeg)(g, s);
        });
      });
    });
    g.restore();
  }

  // A 5-point answer drawn the way the picture is built: the circle 90 deg
  // out from the centre, the horizon's great circle, each family's two
  // vanishing points, and the whole line each mark belongs to, through both.
  // Not drawn when the marks did not settle the camera.
  function drawCurvi(g) {
    var c = S.consensus.conditioning || {};
    if (c.mode !== 'five' || c.axes_refused || !c.horizon) return;
    g.save();
    var o = toScreen(c.principal_point);
    g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = 1; g.setLineDash([2, 4]);
    g.beginPath(); g.arc(o[0], o[1], c.rim_radius_px * S.view.k, 0, 6.2832); g.stroke();
    g.strokeStyle = $('#colHorizon').value; g.lineWidth = 1.6; g.setLineDash([7, 5]);
    strokePath(g, c.horizon);
    Object.keys(c.per_family || {}).forEach(function (id) {
      var L = layer(id), d = c.per_family[id];
      if (!L || L.hidden) return;
      g.strokeStyle = L.color; g.lineWidth = 1; g.globalAlpha = 0.55; g.setLineDash([3, 5]);
      (d.guides || []).forEach(function (q) { strokePath(g, q); });
      g.globalAlpha = 1; g.lineWidth = 2; g.setLineDash([]);
      [d.vp, d.vp_back].forEach(function (v) {
        if (!v) return;
        var p = toScreen(v);
        g.beginPath(); g.arc(p[0], p[1], 7, 0, 6.2832);
        g.moveTo(p[0] - 11, p[1]); g.lineTo(p[0] + 11, p[1]);
        g.moveTo(p[0], p[1] - 11); g.lineTo(p[0], p[1] + 11); g.stroke();
      });
    });
    g.restore();
  }

  function overFloor(r) {
    var c = (r || {}).conditioning || {};
    return (c.worst_move_px || 0) > (c.mark_accuracy_px || MARK_PX);
  }

  // The drawing from above, laid out the way a perspective is set up on paper:
  // the eye, the picture plane, and a ray from the eye to each vanishing point.
  // Derived like a fit -- asked for again whenever a point moves, never kept in
  // the history -- and an answer for points that have moved since is dropped.
  function planKey() {
    if (!S.image || !(horizonVps(null) || camHorizon(null))) return null;
    return JSON.stringify([S.perspective, S.scale, S.image.w, S.image.h, families().map(function (L) {
      var c = (L.vpInfo || {}).conditioning || {};
      return [L.id, L.vp, !!L.vertical, !!L.on_horizon, (L.refs || {}).square_of || null,
              c.bootstrap_x || null, c.bootstrap_y || null];
    })]);
  }

  function paintPlan() {
    var key = planKey();
    if (key !== S.planKey) {
      S.planKey = key;
      if (!key) S.plan = null;
      else post('/api/plan', { doc: buildDoc() }).then(function (r) {
        if (S.planKey === key) { S.plan = r; paintPlan(); }
      });
    }
    var box = $('#plan'), r = S.plan, v = r && r.value;
    box.classList.toggle('hidden', !key || !r || !$('#showPlan').checked);
    if (box.classList.contains('hidden')) return;
    var cond = r.conditioning || {}, why = r.why || '';
    box.classList.toggle('und', !v);
    box.classList.toggle('guide', !!v && r.verdict !== 'ok');
    $('#planNote').textContent = !v ? 'no top view: ' + why.split(/[:;]/)[0]
      : r.verdict !== 'ok' ? 'a guide only: ' + why.split(/[:;]/)[0]
      : 'field of view ' + num(v.fov_deg, 1) + '°, ' + cond.fov_deg_range.join('–') + '°'
        + (cond.focal_span_assumed || cond.camera_assumed ? ', partly assumed' : '');
    box.title = [why, cond.focal_span && 'focal length ' + cond.focal_span, cond.camera_assumed,
                 cond.distance_range_mm && 'the eye stands ' +
                 cond.distance_range_mm.map(Math.round).join('\u2013') + ' mm back from the picture plane']
      .filter(Boolean).join('\n\n');
    if (!v) return;

    var c = $('#planView'), g = c.getContext('2d'), dpr = window.devicePixelRatio || 1;
    var cw = c.clientWidth, ch = c.clientHeight;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) {
      c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr);
    }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, cw, ch);
    var d = v.distance_px, pic = v.picture, back = cond.distance_range_px || [d, d], far = back[1];
    var fams = families().filter(function (L) { return v.families[L.id]; });
    // the eye, the picture, and every vanishing point not too far off to the side
    var half = Math.max(Math.abs(pic[0][0]), Math.abs(pic[1][0])), xs = [0, pic[0][0], pic[1][0]];
    fams.forEach(function (L) {
      var at = v.families[L.id].at;
      if (at && Math.abs(at[0]) < 4 * half) xs.push(at[0]);
    });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min(0, d - far);
    var m = 9, k = Math.min((cw - 2 * m) / Math.max(x1 - x0, 1e-9), (ch - 2 * m) / (d - y0));
    var P = function (q) { return [cw / 2 + (q[0] - (x0 + x1) / 2) * k, ch - m - (q[1] - y0) * k]; };
    var line = function (a, b) {
      a = P(a); b = P(b); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
    };
    var eye = P([0, 0]), R = 2 * (cw + ch) / k;               // off the canvas, whichever way
    var ink = getComputedStyle(c).color;                      // the panel's, light or dark
    // what the picture takes in, and the picture plane it is cut from
    g.fillStyle = ink; g.globalAlpha = 0.07;
    g.beginPath(); g.moveTo(eye[0], eye[1]);
    pic.forEach(function (q) { var s = P(q); g.lineTo(s[0], s[1]); });
    g.fill(); g.globalAlpha = 1;
    g.strokeStyle = $('#colHorizon').value; g.lineWidth = 1;
    g.setLineDash([5, 4]); line([x0 - R, d], [x1 + R, d]);
    g.setLineDash([1, 3]); line([0, 0], [0, d]);              // the line of sight
    g.setLineDash([]); g.lineWidth = 2.5; line(pic[0], pic[1]);
    fams.forEach(function (L) {
      var F = v.families[L.id];
      g.strokeStyle = g.fillStyle = L.color; g.lineWidth = 1.4;
      line([0, 0], F.at || [F.dir[0] * R, F.dir[1] * R]);
      if (F.at) { var s = P(F.at); g.beginPath(); g.arc(s[0], s[1], 3, 0, 2 * Math.PI); g.fill(); }
    });
    // where the eye stands, and how far back it might stand
    g.strokeStyle = g.fillStyle = ink; g.globalAlpha = 0.35; g.lineWidth = 5; g.lineCap = 'round';
    line([0, d - far], [0, d - back[0]]);
    g.globalAlpha = 1; g.lineCap = 'butt';
    g.beginPath(); g.arc(eye[0], eye[1], 3.5, 0, 2 * Math.PI); g.fill();
    // To scale once the picture's real size is given: a bar of 1, 2 or 5 x 10^n
    // mm, no longer than 60 px, and the eye's distance along the line of sight.
    var mpp = v.mm_per_px;
    if (!mpp) return;
    var mm = 60 * mpp / k, p10 = Math.pow(10, Math.floor(Math.log10(mm)));
    mm = (mm >= 5 * p10 ? 5 : mm >= 2 * p10 ? 2 : 1) * p10;
    var x = m + mm / mpp * k, y = ch - m;
    g.lineWidth = 1; g.font = '10px system-ui';
    g.beginPath(); g.moveTo(m, y - 4); g.lineTo(m, y); g.lineTo(x, y); g.lineTo(x, y - 4); g.stroke();
    g.fillText(mm >= 1000 ? mm / 1000 + ' m' : mm + ' mm', m, y - 6);
    g.fillText(num(v.distance_mm, 0) + ' mm', eye[0] + 5, (eye[1] + P([0, d])[1]) / 2);
  }

  // ---------------------------------------------------------------- drawing

  // Only the selected layer's marks are grabbable. Every layer used to be,
  // so a blue line's end sitting on the corner a yellow line starts from
  // swallowed the drag and there was no way to draw the yellow one.
  function nearestPoint(L, w, tol) {
    var best = null;
    // a line's two ends, so a line that went down slightly wrong can be
    // nudged instead of deleted and drawn again
    L.lines.forEach(function (s, i) {
      [0, 1].forEach(function (j) {
        var d = Math.hypot(s[2 * j] - w[0], s[2 * j + 1] - w[1]);
        if (d < tol && (!best || d < best.d)) best = { d: d, layer: L, path: 'lines', i: i, j: j };
      });
    });
    ['pts', 'far', 'crown', 'through', 'cells'].forEach(function (path) {
      (L[path] || []).forEach(function (p, i) {
        if (!p) return;                        // a dropped point on the curve
        var d = Math.hypot(p[0] - w[0], p[1] - w[1]);
        if (d < tol && (!best || d < best.d)) best = { d: d, layer: L, path: path, i: i };
      });
    });
    ['pairs', 'curves'].forEach(function (path) {
      (L[path] || []).forEach(function (p, i) {
        p.forEach(function (q, j) {
          var d = Math.hypot(q[0] - w[0], q[1] - w[1]);
          if (d < tol && (!best || d < best.d)) best = { d: d, layer: L, path: path, i: i, j: j };
        });
      });
    });
    // the Bezier handles, last, so an anchor sitting under one still wins:
    // moving the arch is the commoner thing to want
    (L.bez || []).forEach(function (_, i) {
      var segs = bezOf(L, i);
      if (!segs) return;
      segs.forEach(function (P, sg) {
        [1, 2].forEach(function (k) {
          var d = Math.hypot(P[k][0] - w[0], P[k][1] - w[1]);
          if (d < tol && (!best || d < best.d)) {
            best = { d: d, layer: L, path: 'bez', i: i, j: 2 * sg + (k - 1) };
          }
        });
      });
    });
    return best;
  }

  function finishRun() {
    var L = selected();
    if (!L || !S.pending.length) { S.pending = []; return; }
    // a family's run is one line marked by its points: two are a line as if
    // dragged, and three or more test whether it is straight
    if (L.kind === 'family') {
      var run = S.pending.filter(function (p, i, a) {     // a double-click lands twice
        return !i || Math.hypot(p[0] - a[i - 1][0], p[1] - a[i - 1][1]) * S.view.k > 3;
      });
      S.pending = [];
      if (run.length === 2) addLine(L, run[0].concat(run[1]));
      else if (run.length > 2) { L.curves.push(run); fitFamily(L); }
      refreshLayers(); showEditor(); paint(); commit();
      return;
    }
    // `far` is the arch's far run and the row's one marked top: both are a
    // second thing to mark on one layer, and one toggle serves both
    var target = ((L.kind === 'arch' || L.kind === 'posts') && S.archFar) ? 'far' : 'pts';
    if (contactful(L) && target === 'pts') {
      for (var i = 0; i < S.pending.length; i++) L.markK[L.pts.length + i] = S.view.k;
    }
    L[target] = L[target].concat(S.pending);
    if (L.kind === 'posts' && target === 'far') L.far = L.far.slice(-1);  // one top
    S.pending = [];
    // showEditor too: a panel can depend on how many marks there are -- the
    // row's Tidy button appears at four, and only on re-selecting the layer
    // until this was here
    refreshLayers(); showEditor(); paint(); commit();
  }

  function onDown(ev) {
    if (!S.image) return;
    var p = canvasXY(ev), w = toWorld(p[0], p[1]);
    if (ev.button === 1 || ev.altKey || ev.shiftKey) { startPan(ev); return; }
    if (S.trimEdit) {
      // placing the sheet's corners takes the whole canvas: no marking, and
      // a miss pans, so the corners can be dragged well outside the picture
      var ci = nearestCorner(p[0], p[1], 14);
      if (ci >= 0) { S.dragCorner = ci; paint(); return; }
      startPan(ev); return;
    }
    var L = selected();
    if (!L) { startPan(ev); return; }

    var hit = nearestPoint(L, w, 8 / S.view.k);
    if (hit && ev.metaKey) {                       // cmd-click removes a point
      // the placement zooms are indexed like the contacts, so they splice too
      if (contactful(hit.layer) && (hit.path === 'pairs' || hit.path === 'pts')) {
        hit.layer.markK.splice(hit.i, 1);
      }
      if (hit.path === 'pairs') hit.layer.pairs.splice(hit.i, 1);
      else if (hit.path === 'curves') {
        // the point, not the line: two points left are still a line
        var cv = hit.layer.curves[hit.i];
        cv.splice(hit.j, 1);
        if (cv.length < 2) hit.layer.curves.splice(hit.i, 1);
        fitFamily(hit.layer);
      }
      else if (hit.path === 'lines') {
        hit.layer.lines.splice(hit.i, 1);
        if (hit.layer.kind === 'family') fitFamily(hit.layer);
      }
      // the point on the curve is one per arch, so dropping one must not
      // slide the next arch's point onto it
      else if (hit.path === 'through') hit.layer.through[hit.i] = null;
      else hit.layer[hit.path].splice(hit.i, 1);
      if (hit.path === 'cells') hit.layer.cellOut.splice(hit.i, 1);
      refreshLayers(); showEditor(); paint(); commit(); return;
    }
    if (hit) { S.dragPt = hit; return; }

    if (candTakers(L)) {                       // adopt a detected segment
      var ci = nearestCand(w, 6 / S.view.k);
      if (ci >= 0) { takeCand(L, ci); return; }
    }

    var how = drawMode(L);
    if (how === 'cell') { L.cells.push(w); fitCells(L, true); return; }
    if (how === 'line') { S.dragLine = { x1: w[0], y1: w[1], x2: w[0], y2: w[1] }; return; }
    if (how === 'poly') { S.pending.push(w); S.tracing = true; paint(); return; }
    if (how === 'pair' || how === 'triple') {
      S.pending.push(w);
      if (S.pending.length >= (how === 'pair' ? 2 : 3)) {
        L.pairs.push(S.pending.slice());
        L.markK[L.pairs.length - 1] = S.view.k;
        S.pending = []; refreshLayers();
      }
      paint(); return;
    }
    startPan(ev);
  }

  function startPan(ev) {
    var p = canvasXY(ev), v0 = { x: S.view.x, y: S.view.y };
    $('.stageWrap').classList.add('panning');
    var move = function (e) {
      var q = canvasXY(e);
      S.view.x = v0.x - (q[0] - p[0]) / S.view.k;
      S.view.y = v0.y - (q[1] - p[1]) / S.view.k;
      paint();
    };
    var up = function () {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      $('.stageWrap').classList.remove('panning');
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  }

  function onMove(ev) {
    if (!S.image) return;
    var p = canvasXY(ev), w = toWorld(p[0], p[1]);
    $('#cursor').textContent = Math.round(w[0]) + ', ' + Math.round(w[1]);
    if (S.dragCorner >= 0) { S.trimDraft[S.dragCorner] = w; paint(); return; }
    if (S.dragLine) { S.dragLine.x2 = w[0]; S.dragLine.y2 = w[1]; paint(); }
    else if (S.dragPt) {
      var d = S.dragPt;
      if (d.path === 'pairs' || d.path === 'curves') d.layer[d.path][d.i][d.j] = w;
      else if (d.path === 'lines') {
        d.layer.lines[d.i][2 * d.j] = w[0];
        d.layer.lines[d.i][2 * d.j + 1] = w[1];
      } else if (d.path === 'bez') {
        var segs = d.layer.bez[d.i], near = ctrlAt(segs, d.j);
        var adj = d.layer.bezAdj[d.i] = d.layer.bezAdj[d.i] || [];
        adj[d.j] = [w[0] - near[0], w[1] - near[1]];
        // An arch is symmetric about its own upright, so the hand is moving
        // BOTH limbs: the partner's own adjustment is dropped and it follows
        // this one from here on. Letting each keep its own was the earlier
        // rule and it means a four-handle arch stops being an arch -- once
        // all four had been touched, nothing moved together any more.
        adj[2 * segs.length - 1 - d.j] = null;
        // Only the hand's own move is kept. The other limb follows at draw
        // time, in `bezOf`, through the mirror of the plane the arch is in
        // THEN -- see the note there.
      } else d.layer[d.path][d.i] = w;
      if (contactful(d.layer) && (d.path === 'pts' || d.path === 'pairs')) {
        d.layer.markK[d.i] = S.view.k;      // re-placed, so re-dated
      }
      paint();
    }
    else if (S.tracing) {
      // drag-to-trace: sample along the stroke instead of asking for a click
      // per point. Marking a vault limb by clicking is 139 clicks.
      var last = S.pending[S.pending.length - 1];
      if (!last || Math.hypot(w[0] - last[0], w[1] - last[1]) > TRACE_PX / S.view.k) {
        S.pending.push(w); paint();
      }
    }
  }

  function onUp() {
    if (S.dragCorner >= 0) { S.dragCorner = -1; paint(); commit(); return; }
    if (S.tracing) {
      S.tracing = false;
      refreshLayers();
      // a stroke is a finished run; a single click is not, so the next click
      // carries on the same run
      if (S.pending.length > 1) finishRun();
    }
    if (S.dragLine) {
      var L = selected();
      var d = S.dragLine; S.dragLine = null;
      if (L && Math.hypot(d.x2 - d.x1, d.y2 - d.y1) > 12 / S.view.k) {
        var q = [d.x1, d.y1, d.x2, d.y2];
        // the first line of a family is free: there is nothing to be square to
        if (L.kind === 'family' && L.snapSquare && L.lines.length) snapLine(L, q);
        else addLine(L, q);
      } else if (L && L.kind === 'family') {
        S.pending.push([d.x1, d.y1]);        // a click, not a drag: a run of points
      }
      refreshLayers(); paint();
    }
    if (S.dragPt) {
      if (S.dragPt.path === 'lines') {
        if (S.dragPt.layer.kind === 'family') fitFamily(S.dragPt.layer);
        if (S.dragPt.layer.kind === 'ellipse') fitRing(S.dragPt.layer);
      }
      if (S.dragPt.path === 'curves') fitFamily(S.dragPt.layer);
      if (S.dragPt.path === 'cells') fitCells(S.dragPt.layer);
      if (S.dragPt.layer.kind === 'arch' && S.dragPt.path !== 'bez') {
        var AL = S.dragPt.layer; S.dragPt = null; fitArch(AL);
      }
      if (S.dragPt && S.dragPt.path === 'bez') showEditor();
      S.dragPt = null; refreshLayers();
    }
    commit();        // one entry per interaction; a click that changed nothing collapses
  }

  function onWheel(ev) {
    if (!S.image) return;
    ev.preventDefault();
    if (!(ev.ctrlKey || ev.metaKey)) {      // two-finger pan; pinch/ctrl zooms
      S.view.x += ev.deltaX / S.view.k;
      S.view.y += ev.deltaY / S.view.k;
      paint();
      return;
    }
    var p = canvasXY(ev), before = toWorld(p[0], p[1]);
    var k = S.view.k * Math.pow(1.0016, -ev.deltaY);
    S.view.k = Math.max(0.03, Math.min(40, k));
    var after = toWorld(p[0], p[1]);
    S.view.x += before[0] - after[0];
    S.view.y += before[1] - after[1];
    paint();
  }

  // ------------------------------------------------------------- the server

  // `wh` only for the load path: the picture is still being fetched there, so
  // S.image is not set yet, and without a size the server falls back to the
  // 9 px hand mark of a 1556 px picture and judges a 6583 px one far too
  // harshly. The marks file carries the size, so it is passed in.
  function fitFamily(L, wh) {
    // 5-point has no point for a family of straight lines to meet at: its
    // two vanishing points come with its camera, out of Consensus
    if (S.perspective !== 'five' && (L.refs || {}).square_of) { squareFamily(L); return; }
    if (S.perspective === 'five' || !marked(L)) {
      L.vp = null; L.vpInfo = null; refreshLayers();
      if (S.sel === L.id) showEditor();         // else it still shows the old point
      return;
    }
    var size = wh || (S.image ? [S.image.w, S.image.h] : null);
    var sent = JSON.stringify([L.lines, L.curves]);
    var hz = L.on_horizon && horizonVps(L);
    post('/api/fit_vp', { lines: L.lines, curves: L.curves, image: size, tidied: L.tidied || null,
                          horizon: hz ? [hz[0].vp, hz[1].vp] : null,
                          // no second horizontal family to draw one from, so
                          // the server takes the horizon off the camera
                          doc: (L.on_horizon && !hz) ? buildDoc() : null })
      .then(function (r) {
      // the lines moved on while this was in flight: its point belongs to
      // lines that are gone, and the answer for these is already on its way
      if (JSON.stringify([L.lines, L.curves]) !== sent) return;
      L.vpInfo = r;
      // 'check these' still carries a point: one line disagrees with the rest,
      // which is something to say about the fit, not a reason to throw it away.
      L.vp = (r.verdict === 'ok' || r.verdict === 'check these') ? r.value : null;
      if (r.verdict === 'at_infinity') L.vp = null;
      // the fit lands after showEditor() has already drawn the panel, so the
      // readout has to be put there now or it only appears on re-selecting
      if (S.sel === L.id) showEditor(); else refreshLayers();
      paint();
      // a family held on the horizon is reported against the OTHER families,
      // so moving one of those moves its answer. Only families that are not
      // themselves held refit the held ones, so this cannot loop.
      if (!L.on_horizon) {
        families().forEach(function (F) {
          if (F !== L && (F.on_horizon || (F.refs || {}).square_of)) fitFamily(F);
        });
      }
      // a circle is the shape its plane gives it, and this vanishing point may
      // be one of the two that span that plane -- so the drawn ones are stale
      S.layers.forEach(rebuild);
    });
  }

  // The fourth vanishing point: square to another horizontal, which the
  // camera and the horizon fix between them. Asked for again whenever
  // anything under it moves -- the camera included -- and never kept, for the
  // same reason the arch derives its mirrored handle at paint time: a stored
  // one goes on answering for the geometry it was made in.
  function squareFamily(L) {
    var of = layer(L.refs.square_of);
    post('/api/square_vp', { doc: buildDoc(), of: L.refs.square_of,
                             lines: L.lines,
                             // with no marks of its own there is no length on
                             // the page to measure a swing against
                             scale_lines: (of && of.lines) || [] })
      .then(function (r) {
      L.vpInfo = r;
      L.vp = r.verdict === 'ok' ? r.value : null;
      if (S.sel === L.id) showEditor(); else refreshLayers();
      paint();
      S.layers.forEach(rebuild);
    });
  }

  // A ring and an arch are BUILT by the server out of the plane they are
  // told, so a change of reference has to rebuild them, not just repaint:
  // naming an arch's vertical vanishing point left its refusal on screen.
  function rebuild(X) {
    if (X.kind === 'ellipse' && X.lines.length) fitRing(X);
    if (X.kind === 'ellipse' && X.cells.length) fitCells(X);
    if (X.kind === 'arch' && X.lines.length) fitArch(X);
  }

  // Marking is coarser than drawing, so with three lines or more the wobble
  // of the hand can be told from the error of the drawing and absorbed. What
  // it did is stamped on the layer, because afterwards those lines agree by
  // construction and the spread no longer measures the marking.
  function tidyFamily(L) {
    busy(true);
    var hz = L.on_horizon && horizonVps(L);
    post('/api/tidy', {
      lines: L.lines, image: S.image ? [S.image.w, S.image.h] : null,
      horizon: hz ? [hz[0].vp, hz[1].vp] : null
    }).then(function (r) {
      if (r.error) { toast(r.error, true); return; }
      if (!r.n) { toast('nothing to absorb: every line is already within the hand'); return; }
      L.lines = r.lines;
      L.tidied = { n: r.n, worst: r.worst };
      fitFamily(L);
      commit();
      toast(r.n + ' line' + (r.n > 1 ? 's' : '') + ' absorbed' +
            (r.on_horizon ? ' onto the horizon' : '') + ', worst ' + r.worst + ' px');
    }).finally(function () { busy(false); });
  }

  // The row's own version of tidyFamily, and the same bargain: a post that
  // asks for less than a hand mark is the marker's wobble, anything more is
  // the drawing. The server picks the target from the posts that are not
  // strays, so a stray cannot pull the row onto itself.
  function tidyRow(L) {
    var V = rowVp(L);
    if (!V) { toast('name the surface the row stands on, or the family it runs to', true); return; }
    busy(true);
    post('/api/tidy_row', {
      pts: L.pts, vp: V, image: S.image ? [S.image.w, S.image.h] : null
    }).then(function (r) {
      if (r.error) { toast(r.error, true); return; }
      if (!r.n) {
        toast(r.stray.length
          ? 'nothing to absorb: post ' + r.stray.join(', ') + ' asks for more than a hand mark'
          : 'nothing to absorb: every post is already within the hand');
        return;
      }
      L.pts = r.pts;
      L.rowTidied = { n: r.n, worst: r.worst, stray: r.stray };
      showEditor(); paint(); commit();
      toast(r.n + ' post' + (r.n > 1 ? 's' : '') + ' absorbed, worst ' + r.worst + ' px' +
            (r.stray.length ? '; post ' + r.stray.join(', ') + ' left as marked' : ''));
    }).finally(function () { busy(false); });
  }

  function straighten(L) {
    busy(true);
    fetch('/api/straighten', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc: buildDoc(), feature: L.id, trim: S.trim })
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || 'failed'); });
      return r.blob();
    }).then(function (blob) {
      if (S.straight) URL.revokeObjectURL(S.straight.url);
      S.straight = { id: L.id, url: URL.createObjectURL(blob) };
      showEditor();
    }).catch(function (e) { toast(String(e.message || e), true); })
      .finally(function () { busy(false); });
  }

  // -------------------------------------------------------------- detection
  // An accelerator, never the backbone: the server finds straight segments and
  // groups them into vanishing-point families, but nothing enters a layer
  // until the user clicks it. The grouping's conditioning is carried across
  // with it, because a family of short segments cannot locate a distant
  // vanishing point -- that is bug 9, and it was believed once already.

  function runDetect() {
    if (!S.image) { toast('open a picture first', true); return; }
    post('/api/detect', {
      image: S.image.path, trim: S.trim,
      min_len: parseFloat($('#minLen').value) || 40
    }).then(function (r) {
      if (r.error) { toast(r.error, true); return; }
      S.cand = null;
      if (!r.segments || !r.segments.length) {
        toast(r.note || 'no straight edges found', true);
        refreshLayers(); paint(); return;
      }
      // Which family each segment fell into. The ungrouped rest stay grey and
      // stay clickable -- a line RANSAC did not group is often exactly the one
      // that is wanted, and dropping it would hide it.
      var fam = {};
      (r.families || []).forEach(function (f, i) {
        f.segments.forEach(function (q) { fam[q.join(',')] = i; });
      });
      S.cand = {
        segs: r.segments,
        famOf: r.segments.map(function (q) {
          var k = fam[q.join(',')]; return k === undefined ? -1 : k;
        }),
        families: r.families || []
      };
      $('#showCand').checked = true;
      toast(r.segments.length + ' segments, ' + (r.families || []).length +
            ' famil' + ((r.families || []).length === 1 ? 'y' : 'ies'));
      refreshLayers(); paint(); commit();
    });
  }

  // Only a layer that is currently drawing lines can adopt one.
  // a surface is drawn from the vanishing points, not from a detected edge
  function candTakers(L) { return !!L && drawMode(L) === 'line' && L.kind !== 'surface'; }

  function segDist(p, q) {
    var dx = q[2] - q[0], dy = q[3] - q[1], n2 = dx * dx + dy * dy;
    var t = n2 < 1e-12 ? 0
      : Math.max(0, Math.min(1, ((p[0] - q[0]) * dx + (p[1] - q[1]) * dy) / n2));
    return Math.hypot(p[0] - (q[0] + t * dx), p[1] - (q[1] + t * dy));
  }

  function nearestCand(w, tol) {
    if (!S.cand || !$('#showCand').checked) return -1;
    var best = -1, bd = tol;
    S.cand.segs.forEach(function (q, i) {
      var d = segDist(w, q);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  var ONE_LINE = { surface: 1 };

  function addLine(L, q) {
    // a contact layer holds exactly one support edge, not a family of them,
    // and a surface exactly one patch. An ellipse and an arch hold as many as
    // are dragged: a row of rings or an arcade is one family on one plane
    // under one camera, so its members come out consistent with the horizon
    // because they were built that way.
    if (contactful(L) || ONE_LINE[L.kind]) L.lines = [q.slice()];
    else { L.lines.push(q.slice()); }
    if (L.kind === 'family') fitFamily(L);
    if (L.kind === 'ellipse') fitRing(L);
    if (L.kind === 'arch') fitArch(L);
  }

  // The arch a real circular one would cast, standing on its own wall. Same
  // camera and same two vanishing points the arch CHECK reads it on, so the
  // curve drawn here and the numbers reported later cannot disagree about
  // which plane it is in.
  // round | pointed | straight. Marks saved before straight sides existed
  // carry the old boolean, so it is read here and never written again.
  function shapeOf(L) { return L.shape || (L.pointed ? 'pointed' : 'round'); }

  function fitArch(L) {
    L.arches = null; L.archWhy = null; L.riseSpan = null; L.slope = null;
    L.bez = null; L.bezOff = null; L.bezPlane = null; L.vault = null;
    // The range and the notes belong to a fit. Leaving them behind put a
    // confident 'rise/span 0.97 to 1.91' from the last good fit directly
    // above the refusal that replaced it.
    L.leanDeg = null; L.riseSpanRange = null; L.crownNote = null;
    L.depthNote = null;
    // with no crown yet the server puts one where a semicircle's would be --
    // on the WALL, which is not half the span up the page
    if (!L.lines.length) { showEditor(); paint(); return; }
    post('/api/arch', { doc: buildDoc(), feature: L.id })
      .then(function (r) {
        if (r && r.arches) {
          L.arches = r.arches; L.riseSpan = r.rise_over_span;
          L.riseSpanRange = r.rise_over_span_range || null;
          L.crownNote = r.crown_clamped || null;
          L.depthNote = r.depth_unused || null;
          L.bez = r.bez || null; L.bezOff = r.bez_off || null;
          L.bezPlane = r.plane || null; L.vault = r.vault || null;
          L.slope = r.slope || null; L.leanDeg = r.lean_deg || null;
          // the two snaps per arch, sent back so the handles sit where each
          // one really stands -- but not while a hand is still on one of them
          if (!S.dragPt) {
            L.lines = r.feet.map(function (f) {
              return [f[0][0], f[0][1], f[1][0], f[1][1]];
            });
            L.crown = r.crowns;
          }
        } else L.archWhy = (r && r.error) || 'the arch could not be worked out';
        showEditor(); paint();
      })
      .catch(function (e) { L.archWhy = String(e && e.message || e); showEditor(); paint(); });
  }

  // The arch's Bezier path: one cubic for a round arch, two for a pointed
  // one. The server fits it; what is kept here is the HAND's adjustment, and
  // it is kept as an OFFSET from the fitted control point so that nudging a
  // foot, moving a vanishing point or undoing carries the adjustment along
  // with the curve instead of throwing it away.
  // A page point through a 3x3, and a control point's offset from where the
  // camera put it, which is how an adjustment is kept.
  function applyH(m, p) {
    var z = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2];
    if (!(Math.abs(z) > 1e-9)) return null;
    return [(m[0][0] * p[0] + m[0][1] * p[1] + m[0][2]) / z,
            (m[1][0] * p[0] + m[1][1] * p[1] + m[1][2]) / z];
  }

  function ctrlAt(segs, j) {
    return segs[(j / 2) | 0][(j % 2) + 1];
  }

  function bezOf(L, i) {
    var segs = (L.bez || [])[i];
    if (!segs) return null;
    var adj = (L.bezAdj || [])[i] || [], n = 2 * segs.length;
    // A handle the hand has not touched follows its partner on the other
    // limb. What travels is the HAND'S MOVE and not the handle: the curve
    // the camera fitted is the closest drawing of the arch there is and is
    // left where it is, so nothing jumps the first time a handle is touched.
    //
    // The move is CARRIED ACROSS THE ARCH'S PLANE, not mirrored in it: taken
    // off the page onto the plane where the hand is, and put back on where
    // the other handle stands. "I need a different kind of symmetry. the
    // plane of the arch should bend itself." Reflecting it instead is the
    // rigid arch -- both limbs bulge out together -- and on a vault rib,
    // which stands 1.4 spans tall in its own leaning plane, that sent the
    // far handle 0.5x to 6.9x the hand's move and usually the other way
    // across the page. A bend pinned at the two feet and the crown runs one
    // way between the near foot and the top and the other way past it, which
    // is what carrying the move gives, at 0.8x to 1.4x and within a degree
    // of the hand every time -- measured, selfcheck arch_rib_pair_handles.
    //
    // The plane is read HERE, as the arch stands now. Kept the other way
    // round -- mapped once, at drag time, and stored -- a move made while
    // the arch stood upright stayed on the page when the arch was tipped off
    // its wall: the far handle sat 362 px from where the leaning plane puts
    // it. Pairing by `n - 1 - j` also re-reads itself when the arch changes
    // between round (one cubic) and pointed (two).
    var pl = (L.bezPlane || [])[i];
    function off(j) {
      if (adj[j]) return adj[j];
      var p = n - 1 - j;
      if (!pl || p === j || !adj[p]) return null;
      var c = ctrlAt(segs, p), mine = ctrlAt(segs, j);
      var was = applyH(pl[1], c);
      var now = applyH(pl[1], [c[0] + adj[p][0], c[1] + adj[p][1]]);
      var here = applyH(pl[1], mine);
      if (!was || !now || !here) return null;
      var q = applyH(pl[0], [here[0] + now[0] - was[0], here[1] + now[1] - was[1]]);
      return q ? [q[0] - mine[0], q[1] - mine[1]] : null;
    }
    return segs.map(function (P, s) {
      return P.map(function (c, k) {
        var o = (k === 1 || k === 2) && off(2 * s + (k - 1));
        return o ? [c[0] + o[0], c[1] + o[1]] : c;
      });
    });
  }

  // A vault: the same arch slid back along the depth lines. The slide is
  // one homology, so the far limb is the near one -- handles and all -- put
  // through it, and the two cannot drift apart.
  function bezFar(L, i) {
    var V = (L.vault || [])[i], segs = V && bezOf(L, i);
    if (!segs) return null;
    return segs.map(function (P) {
      return P.map(function (c) { return applyH(V.homology, c) || c; });
    });
  }

  function bezAdjusted(L) {
    return (L.bezAdj || []).some(function (a) {
      return a && a.some(function (o) { return !!o; });
    });
  }

  // How far the path drawn runs from the arch the camera really casts. The
  // server sends the number for the curve it fitted; once a handle has been
  // moved that number is stale, so it is measured again here against the
  // same arc, which is the polyline the server also sent.
  function bezOff(L) {
    if (!bezAdjusted(L)) {
      return (L.bezOff || []).length ? Math.max.apply(null, L.bezOff) : null;
    }
    var worst = 0;
    (L.arches || []).forEach(function (arc, i) {
      var segs = bezOf(L, i);
      if (!segs) return;
      segs.forEach(function (P) {
        for (var t = 0; t <= 1.0001; t += 1 / 160) {
          var m = 1 - t, b = [m * m * m, 3 * m * m * t, 3 * m * t * t, t * t * t];
          var x = 0, y = 0, near = Infinity;
          for (var k = 0; k < 4; k++) { x += b[k] * P[k][0]; y += b[k] * P[k][1]; }
          for (var j = 0; j < arc.length; j++) {
            var d = (arc[j][0] - x) * (arc[j][0] - x) + (arc[j][1] - y) * (arc[j][1] - y);
            if (d < near) near = d;
          }
          worst = Math.max(worst, Math.sqrt(near));
        }
      });
    });
    return worst;
  }

  // The two supports. The feet land on a line of one family and the supports
  // run in another: this draws them standing so the construction can be seen,
  // and nothing is measured off them -- only their direction is the drawing's,
  // the length is whatever looks like a leg.
  function archLegs(L) {
    var U = L.refs.vertical_vp && layer(L.refs.vertical_vp);
    if (!U || !U.vp) return [];
    var out = [];
    L.lines.forEach(function (s, k) {
      var c = L.crown[k];
      if (!c) return;
      var len = Math.hypot(s[2] - s[0], s[3] - s[1]) * 0.6;
      [[s[0], s[1]], [s[2], s[3]]].forEach(function (f) {
        var dx = U.vp[0] - f[0], dy = U.vp[1] - f[1], n = Math.hypot(dx, dy);
        if (n < 1e-9) return;
        // down is away from the crown, whichever side of the wall the vertical
        // vanishing point happens to lie on
        var sg = ((f[0] - c[0]) * dx + (f[1] - c[1]) * dy) > 0 ? 1 : -1;
        out.push([f[0], f[1], f[0] + sg * len * dx / n, f[1] + sg * len * dy / n]);
      });
    });
    return out;
  }

  // The circle a real one in that plane would cast. Two vanishing points name
  // two directions without saying the angle between them, so this needs the
  // camera and the camera is solved on the server -- the same reason the
  // vanishing point itself is fitted there and cached on the layer.
  function fitRing(L) {
    L.rings = null; L.ringsTrue = null; L.ringWhy = null;
    // no surface named means the ground -- the server decides and says so,
    // because the fallback is its rule, not the page's
    if (!L.lines.length) { showEditor(); paint(); return; }
    post('/api/ellipse', { doc: buildDoc(), feature: L.id })
      .then(function (r) {
        if (r && r.rings) {
          L.rings = r.rings; L.ringsTrue = r.rings_true || null; L.ringWhy = null;
          // the levelled drags, sent back so the handles sit on the diameter
          // the ring really has -- but not while a hand is still on one
          if (!S.dragPt && r.ends) L.lines = r.ends;
        } else L.ringWhy = (r && r.error) || 'the circle could not be worked out';
        showEditor(); paint();
      })
      .catch(function (e) { L.ringWhy = String(e && e.message || e); showEditor(); paint(); });
  }

  // The circle in a square. The page keeps only where the hand clicked: the
  // cell round that point, the circle in it and how square it is all come
  // back from the server, which has the camera, and are asked for again
  // whenever anything under them moves. `fresh` is a click just made: with no
  // cell under it the click is withdrawn rather than left as a dot.
  // Which answer is the newest, per layer. Not a property of the layer: that
  // would be saved with the marks and carried by undo, and every recount
  // would then look like an edit and cut the redo history off.
  var cellSeq = {};

  function fitCells(L, fresh) {
    var n = cellSeq[L.id] = (cellSeq[L.id] || 0) + 1;
    if (!L.cells.length) { L.cellOut = []; L.cellWhy = null; showEditor(); paint(); return; }
    post('/api/square_cell', { doc: buildDoc(), feature: L.id })
      .then(function (r) {
        if (cellSeq[L.id] !== n) return;           // a later answer is on its way
        var out = (r && r.cells) || [];
        var bad = !r || r.error ? ((r && r.error) || 'no circle could be worked out')
                : (out[out.length - 1] || {}).ring ? null : out[out.length - 1].why;
        if (fresh && bad) {
          // the mouse-up has very likely recorded the click already: it was
          // withdrawn, so it is not a step to undo either
          var held = snapshot();
          L.cells.pop(); toast(bad, true);
          if (out.length > L.cells.length) out.pop();
          if (S.undo[S.undo.length - 1] === held) S.undo.pop();
        }
        // derived, like a ring: recomputed after any restore, never an edit,
        // so it does not commit -- that would cut the redo history off
        L.cellOut = out;
        L.cellWhy = (r && r.error && !fresh) ? r.error : null;
        showEditor(); paint();
      })
      .catch(function (e) { L.cellWhy = String(e && e.message || e); showEditor(); paint(); });
  }

  // The two families a plane is spanned by: a named surface's own two, or the
  // two that give the horizon -- the ground, and anything level with it.
  function planeFamilies(E) {
    var s = E.refs.surface && layer(E.refs.surface);
    if (!s || s.level) return horizonVps(null);
    var a = s.refs._a && layer(s.refs._a), b = s.refs._b && layer(s.refs._b);
    return a && b ? [a, b] : null;
  }

  // A line drawn into a family that snaps. The server looks for the place a
  // line would close a SQUARE with one already in the family and two of
  // another, and the drawn line is put on it if it is that near. What it did
  // is kept on the layer either way, with the reason when it did nothing.
  function snapLine(L, q) {
    post('/api/square_snap', { doc: buildDoc(), into: L.id, line: q })
      .then(function (r) {
        if (r && r.error) r = { verdict: 'undetermined', why: r.error };
        var ok = r && r.verdict === 'ok' && r.value;
        L.snapInfo = r;
        addLine(L, ok ? r.value : q);
        if (ok) fillSquare(L, r.conditioning);
        refreshLayers(); showEditor(); paint(); commit();
      })
      .catch(function (e) {
        L.snapInfo = { verdict: 'undetermined', why: String(e && e.message || e) };
        addLine(L, q); refreshLayers(); showEditor(); paint(); commit();
      });
  }

  // The square a snap closed holds a circle in every ellipse layer on its
  // plane that asked to be filled.
  function fillSquare(L, c) {
    var B = layer(c.against_id), q = c.corners;
    var mid = [0, 1].map(function (k) {
      return q.reduce(function (t, p) { return t + p[k] / 4; }, 0);
    });
    // a circle already has an anchor in this square: a second would draw it twice
    function inside(p) {
      var on = false;
      for (var i = 0, j = q.length - 1; i < q.length; j = i++) {
        if ((q[i][1] > p[1]) !== (q[j][1] > p[1]) &&
            p[0] < (q[j][0] - q[i][0]) * (p[1] - q[i][1]) / (q[j][1] - q[i][1]) + q[i][0]) on = !on;
      }
      return on;
    }
    S.layers.forEach(function (E) {
      var P = E.kind === 'ellipse' && E.autoCells && planeFamilies(E);
      if (P && P.indexOf(L) >= 0 && P.indexOf(B) >= 0 && !E.cells.some(inside)) {
        E.cells.push(mid); fitCells(E);
      }
    });
  }

  function takeCand(L, i) {
    addLine(L, S.cand.segs[i]);
    S.cand.segs.splice(i, 1); S.cand.famOf.splice(i, 1);
    refreshLayers(); paint(); commit();
  }

  function takeFamily(L, fi) {
    // back to front, so the splices do not move the indices still to come
    for (var i = S.cand.segs.length - 1; i >= 0; i--) {
      if (S.cand.famOf[i] !== fi) continue;
      addLine(L, S.cand.segs[i]);
      S.cand.segs.splice(i, 1); S.cand.famOf.splice(i, 1);
    }
    refreshLayers(); showEditor(); paint(); commit();
  }

  function paintCand() {
    var host = $('#candHost');
    if (!host) return;
    host.textContent = '';
    if (!S.cand) return;
    var L = selected();
    var offered = 0;
    S.cand.families.forEach(function (f, fi) {
      var left = S.cand.famOf.filter(function (k) { return k === fi; }).length;
      if (!left) return;                       // all of it has been taken
      offered++;
      var row = el('div', { className: 'row' });
      var b = el('button', {
        className: 'small ghost',
        textContent: 'take ' + left + ' line' + (left === 1 ? '' : 's'),
        disabled: !candTakers(L),
        title: candTakers(L) ? 'Put these into "' + L.name + '"'
                             : 'Select a layer that is drawing lines first'
      });
      b.onclick = function () { takeFamily(L, fi); };
      row.append(b);
      row.append(el('span', {
        className: 'dim',
        textContent: (f.at_infinity ? 'parallel'
                      : 'vp ' + Math.round(f.vp[0]) + ', ' + Math.round(f.vp[1])) +
                     '  median ' + num(f.median_len_px, 0) + ' px'
      }));
      host.append(row);
      if (f.short_segment_warning) {
        host.append(el('div', {
          className: 'why warn',
          textContent: 'these segments are short and the vanishing point is far ' +
            'outside the picture -- short segments cannot locate a distant ' +
            'vanishing point, so treat this one as a hint and raise the minimum ' +
            'length, or mark the long lines by hand'
        }));
      }
    });
    var ungrouped = S.cand.famOf.filter(function (k) { return k < 0; }).length;
    if (ungrouped) {
      host.append(el('div', {
        className: 'hint',
        textContent: ungrouped + ' more segment' + (ungrouped === 1 ? '' : 's') +
          ' fell into no family (grey) -- click any one to take it'
      }));
    }
    if (!offered && !ungrouped) {
      host.append(el('div', { className: 'hint', textContent: 'every detected segment has been taken' }));
    }
  }

  // ------------------------------------------------------- the picture's real size

  // A pixel means nothing at the easel. Given the real size of the picture,
  // every deviation comes back in millimetres beside the pixels -- the server
  // puts an mm twin next to every *_px number it reports.
  var TO_MM = { mm: 1, cm: 10, m: 1000, in: 25.4, ft: 304.8 };

  function pxPerMM() {
    var sc = S.scale;
    if (!sc || !S.image) return null;
    var span = sc.along === 'height' ? S.image.h : S.image.w;
    var mm = Number(sc.value) * TO_MM[sc.unit];
    return (mm > 0 && span > 0) ? span / mm : null;
  }

  function setScale(sc) {
    var was = JSON.stringify(S.scale);
    S.scale = (sc && Number(sc.value) > 0)
      ? { value: Number(sc.value), unit: sc.unit || 'mm', along: sc.along || 'width' }
      : null;
    if ($('#sizeValue')) {
      $('#sizeValue').value = S.scale ? S.scale.value : '';
      $('#sizeUnit').value = S.scale ? S.scale.unit : 'mm';
      $('#sizeAlong').value = S.scale ? S.scale.along : 'width';
    }
    sizeState();
    // The numbers on screen are in the old units, so they are the wrong
    // numbers now. Same reasoning as setPerspective: re-ask, do not re-label.
    if (was !== JSON.stringify(S.scale)) {
      if (S.consensus) {
        S.consensus = null; S.consensusBoth = null;
        $('#results').textContent = '';
        $('#resultsHint').classList.remove('hidden');
      }
      paint();                  // the top view is drawn to it
    }
  }

  function sizeState() {
    var host = $('#sizeState');
    if (!host) return;
    var ppm = pxPerMM();
    if (!ppm) {
      host.textContent = S.image
        ? 'everything is in pixels. Give the real size of the picture as it is ' +
          'now on screen (' + S.image.w + ' x ' + S.image.h + ' px) to see ' +
          'millimetres beside them.'
        : 'open a picture first.';
      return;
    }
    // Say it against the edge that was NOT declared too: that is the cheapest
    // check that the right edge was measured, since a picture declared along
    // its short edge quietly halves every millimetre.
    var other = S.scale.along === 'height'
      ? ['wide', S.image.w] : ['tall', S.image.h];
    host.textContent = num(ppm, 3) + ' px per mm \u2014 1 px is ' +
      num(1 / ppm, 3) + ' mm, and the picture is ' +
      num(other[1] / ppm, 0) + ' mm ' + other[0] + '.';
  }

  // ---------------------------------------------- the perspective being claimed

  // What each type asks of the drawing, and what it will not answer. These are
  // not solver settings: declaring a type IS the check, and the moves reported
  // are how far the drawing sits from that type rather than from any
  // perspective at all.
  // One line each, and they live on the options themselves: the panel used to
  // carry a paragraph nobody rereads after the first time.
  var PERSP_WHY = {
    three: 'All three families may converge. The focal length and the centre ' +
           'of vision are both measured from the marks.',
    two: 'The uprights must come out parallel. Two-point perspective cannot ' +
         'place the centre of vision along the horizon, so a range of focal ' +
         'lengths fits equally well and the range is printed.',
    one: 'The uprights and one horizontal family must both come out parallel, ' +
         'and the vanishing point is the centre of vision. There is no focal ' +
         'length to measure: f enters no vanishing point here.',
    five: 'Curvilinear: every straight edge drawn as an arc, as 5-point is drawn ' +
          'with a compass, and checked as the stereographic projection that is. ' +
          'Mark each line by three points or more. Only Consensus answers here; ' +
          'the other checks are built on straight lines.'
  };

  function setPerspective(mode) {
    var was = S.perspective;
    S.perspective = PERSP_WHY[mode] ? mode : 'three';
    // An answer to a different question must not stay on screen. Undo reaches
    // here too, and leaving the card up had it reading "checked as one-point"
    // beside a selector saying 2-point.
    // An arch is DRAWN from the camera, and which camera there is at all is
    // what the mode decides: switching the selector left the old rib lying on
    // the picture beside the new mode's refusal.
    if (was !== S.perspective) {
      S.layers.forEach(function (L) {
        if (L.kind === 'arch' && L.lines.length) fitArch(L);
        // a family's own vanishing point is linear perspective's alone
        if (L.kind === 'family' && (was === 'five' || S.perspective === 'five')) fitFamily(L);
      });
    }
    if (was !== S.perspective && S.consensus) {
      S.consensus = null; S.consensusBoth = null;
      $('#results').textContent = '';
      $('#resultsHint').classList.remove('hidden');
      paint();
    }
    var sel = $('#perspective');
    if (!sel) return;
    sel.value = S.perspective;
    // the select keeps its own title -- what declaring a type means at all;
    // what each type asks of the drawing hangs on the option
    [].forEach.call(sel.options, function (o) { o.title = PERSP_WHY[o.value] || ''; });
  }

  function buildDoc() {
    var doc = { image: S.image, trim: S.trim, perspective: S.perspective,
                scale: S.scale, vps: {}, vp_ranges: {}, surfaces: {},
                square_of: {}, vp_names: {}, family_lines: {}, family_curves: {},
                features: {} };
    families().forEach(function (L) {
      if (L.vp) doc.vps[L.id] = L.vp;
      doc.family_lines[L.id] = L.lines;
      doc.family_curves[L.id] = L.curves;
      doc.vp_names[L.id] = L.name;
      // declared, so the camera leaves it out: a derived point came OUT of
      // the camera and cannot be evidence for it
      if ((L.refs || {}).square_of) doc.square_of[L.id] = L.refs.square_of;
      // How well a family pins its own point travels WITH the point. f comes
      // out of the distances to two vanishing points, so a point known to
      // within a thousand pixels is a camera known to within a factor, and
      // every readout taken through the camera is known no better.
      var bc = ((L.vpInfo || {}).conditioning || {});
      if (L.vp && bc.bootstrap_x && bc.bootstrap_y)
        doc.vp_ranges[L.id] = [bc.bootstrap_x, bc.bootstrap_y];
      if (L.vertical) doc.vertical_vp = L.id;
    });
    var horiz = horizonVps(null);
    if (horiz) doc.horizon_vps = [horiz[0].id, horiz[1].id];
    S.layers.forEach(function (L) {
      if (L.kind === 'family') return;
      if (L.kind === 'surface') {
        doc.surfaces[L.id] = { name: L.name, level: !!L.level,
                               vps: (L.refs.vps || []).filter(Boolean) };
        return;
      }
      var f = { kind: L.kind, name: L.name };
      var run = rowPts(L);
      if (run !== L.pts) { f.pts = run; f.built = true; }
      else if (L.pts.length) f.pts = L.pts;
      // the row's vanishing point may be derived from its surface rather than
      // named, and then there is no family id to send -- send the point
      if (L.kind === 'posts' && L.refs.surface) {
        var rv = rowVp(L);
        if (rv) f.row_vp = rv;
      }
      if (L.far.length) f.far = L.far;
      if (L.kind === 'figures') {
        f.figures = L.pairs.map(function (p) { return { feet: p[0], head: p[1] }; });
      }
      // Anything drawn as pairs sends its pairs. Naming the one kind that did
      // meant a new pair-drawing kind silently sent none, and the server's
      // honest "mark the base and the top" was answering an empty list.
      else if (KINDS[L.kind].draw === 'pair' || KINDS[L.kind].draw === 'triple') f.pairs = L.pairs;
      Object.keys(L.refs).forEach(function (k) { if (L.refs[k]) f[k] = L.refs[k]; });
      if (L.kind === 'arc' || L.kind === 'ellipse') f.resolve = Number(L.resolve);
      // the drag the circle was built from, so the server can rebuild it
      if (L.kind === 'ellipse' && L.lines.length) f.ring_from = L.lines.slice();
      if (L.kind === 'ellipse' && L.cells.length) f.cells = L.cells.slice();
      // the three marks each arch was built from, so the server can rebuild it
      if (L.kind === 'arch') {
        if (L.lines.length) {
          f.arch_from = L.lines.map(function (s, i) {
            if (!L.crown[i]) return s.slice();
            var m = s.concat(L.crown[i]);
            return (L.through || [])[i] ? m.concat(L.through[i]) : m;
          });
        }
        f.shape = shapeOf(L);
        if (L.lean) f.lean = true;
        if (L.slope_vp) f.slope_vp = L.slope_vp;
      }
      // px_per_mm only when it was typed: sending 1.0 by default meant the
      // picture's own declared size could never win, because 1.0 is truthy.
      if (L.kind === 'posts') {
        if (Number(L.px_per_mm) > 0) f.px_per_mm = Number(L.px_per_mm);
        f.resolve_mm = Number(L.resolve_mm);
      }
      if (L.kind === 'axis') { f.span = L.span; f.unit = L.unit; }
      if (L.kind === 'landmark') {
        f.size = Number(L.size); f.size_unit = L.size_unit;
        f.on_ground = !!L.on_ground;
      }
      doc.features[L.id] = f;
    });
    return doc;
  }

  function runCheck() {
    if (!S.image) { toast('open a picture first', true); return; }
    post('/api/check', { doc: buildDoc() }).then(function (r) {
      S.results = r; showResults();
    });
  }

  // A family needs two lines to say where its lines run. Two families give a
  // perspective; against a reference camera one is enough, because the camera
  // supplies the others.
  function marked(L) { return L.lines.length + L.curves.length >= 2; }

  function consensusReady() {
    return families().filter(marked).length >= (S.ref ? 1 : 2);
  }

  function runConsensus() {
    var f = families().filter(marked);
    if (!consensusReady()) {
      toast(S.ref ? 'one line family at least, to put against the camera'
                  : 'two line families at least', true);
      return;
    }
    var fam = {}, cur = {};
    f.forEach(function (L) { fam[L.id] = L.lines; cur[L.id] = L.curves; });
    // The family flagged as the vertical one names the upright axis, which is
    // the axis two-point perspective singles out. Without it every candidate
    // is tried and the cheapest wins, which is slower and can pick wrong.
    var roles = {};
    f.forEach(function (L) { if (L.vertical) roles[L.id] = 'vertical'; });
    post('/api/consensus', {
      families: fam, curves: cur, image_size: [S.image.w, S.image.h],
      mode: S.perspective, roles: roles, scale: S.scale,
      reference: S.ref ? S.ref.camera : null
    }).then(function (r) {
      if (r.error) { toast(r.error, true); return; }
      S.consensus = r.free && r.free.verdict ? r.free : (r.reference || r);
      S.consensusBoth = r;
      showConsensus(r);
      // tick the overlay only if there is anything on it
      $('#showMoves').checked = overFloor(r.free) || overFloor(r.reference);
      paint();
    });
  }

  // ------------------------------------------------------------ the camera
  //
  // A camera another program solved, held as a claim to be priced. It is kept
  // in the pixels of the picture AS SHOWN, so anything that re-projects the
  // picture (a trim) takes it away: a homography does not carry a pinhole
  // camera across, and a camera in the wrong pixels would be priced as if it
  // were in the right ones.

  function camState() {
    var r = S.ref, b = $('#camState');
    if (!b) return;
    $('#btnCamClear').classList.toggle('hidden', !r);
    var pick = $('#camPick');
    pick.classList.toggle('hidden', !(r && r.names && r.names.length > 1));
    if (r && r.names && r.names.length > 1) {
      pick.textContent = '';
      r.names.forEach(function (n) {
        pick.append(el('option', { value: n, textContent: n }));
      });
      pick.value = r.camera.source && r.camera.source.camera || r.names[0];
    }
    if (!r) {
      b.className = 'why' + (S.refNote ? ' warn' : '');
      b.textContent = S.refNote || 'No camera. Read one from a .blend, .fspy or ' +
        '.json to see where its vanishing points fall on this picture and what ' +
        'the marks would cost against it.';
      return;
    }
    var c = r.camera, d = r.describe, src = c.source || {};
    b.className = 'why';
    var t = (src.camera ? src.camera + ' in ' : '') + (src.file || 'a file') +
      ': f ' + num(c.f, 0) + ' px = ' + num(d.lens_mm_on_36, 1) + ' mm on a 36 mm ' +
      'sensor; principal point (' + num(c.pp[0], 0) + ', ' + num(c.pp[1], 0) + ').';
    t += c.up ? ' Up is ' + c.up + ' (' + (src.up_basis || 'declared') + ').'
              : ' No axis is clearly vertical, so no horizon is drawn.';
    if (src.scaled_by) t += ' Scaled x' + num(src.scaled_by, 3) + ' from a ' +
      src.scaled_from[0] + ' x ' + src.scaled_from[1] + ' frame.';
    if (src.note) t += ' ' + src.note;
    var ref = r.refused || {};
    Object.keys(ref).forEach(function (n) { t += ' Not read, ' + n + ': ' + ref[n] + '.'; });
    b.textContent = t;
  }

  function cameraIn(path, pick) {
    if (!S.image) { toast('open a picture first', true); return; }
    post('/api/camera_in', { path: path, camera: pick || null,
                             image_size: [S.image.w, S.image.h] })
      .then(function (r) {
        if (r.error) { toast(r.error, true); return; }
        var had = S.ref && S.ref.path === path && S.ref.names;
        S.ref = { camera: r.camera, describe: r.describe, refused: r.refused, path: path,
                  names: had ? S.ref.names
                             : [r.camera.source && r.camera.source.camera]
                                 .concat(r.others).filter(Boolean) };
        S.refNote = null; S.consensus = null; S.consensusBoth = null;
        camState(); paint(); commit();
        if (consensusReady()) runConsensus();
        else toast('camera read: ' + num(r.camera.f, 0) + ' px');
      });
  }

  function cameraClear() {
    S.ref = null; S.refNote = null; S.consensus = null; S.consensusBoth = null;
    camState(); paint(); commit();
  }

  // The camera the marks agree on, as a Blender script with the picture
  // behind it. Same request as the readout, because it IS that readout.
  function cameraOut() {
    var f = families().filter(marked);
    if (!S.image) { toast('open a picture first', true); return; }
    if (f.length < 2) { toast('two line families at least: the camera is what they agree on', true); return; }
    var fam = {}, cur = {}, roles = {}, names = {};
    f.forEach(function (L) {
      fam[L.id] = L.lines; cur[L.id] = L.curves; names[L.id] = L.name;
      if (L.vertical) roles[L.id] = 'vertical';
    });
    post('/api/camera_out', {
      families: fam, curves: cur, image_size: [S.image.w, S.image.h], mode: S.perspective,
      roles: roles, names: names, picture: S.image.path, trim: S.trim,
      yaw: Number($('#camYaw').value) || 0
    }).then(function (r) {
      if (r.error) { toast(r.error, true); return; }
      var blob = new Blob([r.script], { type: 'text/x-python' });
      var a = el('a', { href: URL.createObjectURL(blob), download: r.filename });
      a.click();
      URL.revokeObjectURL(a.href);
      var host = $('#results');
      $('#resultsHint').classList.add('hidden');
      var c = r.conditioning || {};
      var card = roCard('camera out  (' + r.filename + ')', {
        verdict: r.verdict, value: r.value, why: r.why, conditioning: c });
      card.append(el('div', { className: 'why', textContent: (r.verdict === 'undetermined' ?
        'lens ' + num(r.value, 2) + ' mm' : 'lens in millimetres') +
        ' on a ' + num(c.sensor_mm, 0) + ' mm sensor' }));
      (c.notes || []).forEach(function (n) {
        card.append(el('div', { className: 'why', textContent: n }));
      });
      host.prepend(card);
    });
  }

  // --------------------------------------------------------------- the panels

  function refreshLayers() {
    var host = $('#layers');
    host.textContent = '';
    S.layers.forEach(function (L) {
      var row = el('div', { className: 'layer' + (L.id === S.sel ? ' on' : '') +
                                       (L.hidden ? ' hid' : '') });
      // the swatch is the switch: hollow means the layer is not drawn
      var sw = el('span', { className: 'swatch', title: L.hidden ? 'show' : 'hide' });
      sw.style.background = L.hidden ? 'transparent' : L.color;
      sw.style.boxShadow = 'inset 0 0 0 1.5px ' + L.color;
      sw.onclick = function (e) {
        e.stopPropagation();
        L.hidden = !L.hidden;
        if (L.hidden && L.id === S.sel) select(L.id);   // nothing drawn, nothing to mark on
        else { refreshLayers(); paint(); }
      };
      var nm = el('span', { className: 'nm', textContent: L.name });
      var ct = el('span', { className: 'ct' });
      var n = markCount(L);
      if (L.kind === 'family') {
        ct.textContent = n + ' line' + (n === 1 ? '' : 's') + (L.vp || S.perspective === 'five' ? '' : n >= 2 ? ' · no VP' : '');
      } else if (L.kind === 'surface') {
        ct.textContent = surfaceQuad(L) ? 'drawn' : n ? 'no VPs yet' : 'not drawn';
      } else if (KINDS[L.kind].draw === 'none') {
        ct.textContent = 'refs';
      } else {
        ct.textContent = n;
      }
      row.append(sw, nm, ct);
      row.onclick = function () { select(L.id); };
      host.append(row);
    });
    if (!S.layers.length) host.append(el('div', { className: 'hint', textContent: 'No layers yet.' }));
    else host.append(el('div', { className: 'hint',
      textContent: 'Click a swatch to hide that layer; hiding only changes what is drawn.' }));
    var cur = selected();
    var nc = cur ? markCount(cur) : 0;
    $('#edCount').textContent = cur ? nc + (nc === 1 ? ' mark' : ' marks') : '';
    paintSupport(); paintCand();
  }

  function select(id) {
    finishRun();
    S.sel = (S.sel === id) ? null : id;
    // selecting a hidden layer shows it again rather than marking blind
    if (S.sel) S.layers.forEach(function (L) { if (L.id === S.sel) L.hidden = false; });
    S.archFar = false; S.supportEdge = false; S.ringMark = false;
    S.inSquare = false; S.archDraw = true;
    refreshLayers(); showEditor(); paint();
  }

  function refRow(L, key, label, title, pool) {
    var r = el('div', { className: 'row' });
    r.append(el('label', { textContent: label, title: title || '' }));
    var sel = el('select', { className: 'grow' });
    sel.append(el('option', { value: '', textContent: '— none —' }));
    (pool || families()).forEach(function (F) {
      var o = el('option', { value: F.id, textContent: F.name });
      if (L.refs[key] === F.id) o.selected = true;
      sel.append(o);
    });
    // the panel and the picture both read these, so both have to be redrawn:
    // setting a row's family or its count changed nothing on screen until
    // something else happened to repaint
    sel.onchange = function () {
      L.refs[key] = sel.value || null;
      // this one IS the family's vanishing point, so it has to be re-asked,
      // and dropping it puts the family back on its own marks
      if (key === 'square_of') fitFamily(L);
      rebuild(L); showEditor(); paint(); commit();
    };
    r.append(sel);
    return r;
  }

  function numRow(L, key, label, step, title) {
    var r = el('div', { className: 'row' });
    r.append(el('label', { textContent: label, title: title || '' }));
    var inp = el('input', { type: 'number', className: 'grow', step: step, value: L[key] });
    inp.onchange = function () { L[key] = Number(inp.value); showEditor(); paint(); commit(); };
    r.append(inp);
    return r;
  }

  function showEditor() {
    var L = selected();
    var box = $('#editor');
    if (!L) { box.classList.add('hidden'); $('#drawHint').textContent = hintFor(null); return; }
    box.classList.remove('hidden');
    $('#editTitle').textContent = KINDS[L.kind].label;
    $('#edName').value = L.name;
    $('#edColor').value = L.color;
    $('#drawHint').textContent = hintFor(L);

    var f = $('#edFields');
    f.textContent = '';

    if (L.kind === 'family') {
      var r = el('div', { className: 'row' });
      var cb = el('input', { type: 'checkbox', checked: !!L.vertical });
      cb.onchange = function () { L.vertical = cb.checked; paint(); };
      var lab = el('label', { style: 'min-width:0', title: 'The upright direction. Marked so no check ever mistakes it for a horizontal one.' });
      lab.append(cb, document.createTextNode(' this is the vertical family'));
      r.append(lab); f.append(r);
      if (!L.vertical) {
        // Two families and the uprights are the camera; a third is often not
        // part of it at all but an object standing askew to the lot -- a table
        // turned off the wall. Its other direction is then not free, and the
        // camera says where it goes, so it needs no marks of its own.
        var pool = families().filter(function (F) {
          return F !== L && !F.vertical && F.vp && !(F.refs || {}).square_of;
        });
        if (pool.length || L.refs.square_of) {
          f.append(refRow(L, 'square_of', 'square to',
            'At right angles to that family IN THE WORLD -- the other way ' +
            'along the same table. The camera and the horizon put the point ' +
            'between them, so no lines need marking here; mark some anyway ' +
            'and the gap between the two is a reading on the camera.', pool));
        }
        var hv = horizonVps(L), ch = camHorizon(L);
        var hr = el('div', { className: 'row' });
        var hb = el('input', { type: 'checkbox', checked: !!L.on_horizon,
                               disabled: !hv && !ch && !L.on_horizon });
        hb.onchange = function () { L.on_horizon = hb.checked; fitFamily(L); commit(); };
        var hl = el('label', { style: 'min-width:0',
          title: 'Horizontals all share one horizon, whichever way they run. ' +
                 'Ticked, the fit is reported held on the horizon the other ' +
                 'families give, and Tidy swings these lines onto it.' });
        hl.append(hb, document.createTextNode(' its vanishing point is on the horizon'));
        hr.append(hl); f.append(hr);
        if (!hv && !ch) f.append(el('div', { className: 'hint',
          textContent: 'Two other horizontal families are what give a horizon ' +
            'to hold it on -- or one of them and the uprights, through the ' +
            'camera.' }));
        else if (L.on_horizon) f.append(el('div', { className: 'hint',
          textContent: hv ? 'The horizon through ' + hv[0].name + ' and ' + hv[1].name + '.'
                          : 'The horizon the camera gives from the uprights.' }));
      }
      var qr = el('div', { className: 'row' });
      var qb = el('input', { type: 'checkbox', checked: !!L.snapSquare });
      qb.onchange = function () { L.snapSquare = qb.checked; commit(); };
      var ql = el('label', { style: 'min-width:0',
        title: 'A line you draw here is moved onto the place that makes a SQUARE ' +
               'with one already in this family and two lines of another, if it ' +
               'is within a hand mark and a half of it. The first line is free. ' +
               'Needs the camera; it says why when there is none.' });
      ql.append(qb, document.createTextNode(' snap new lines to a square'));
      qr.append(ql); f.append(qr);
      if (L.snapInfo) f.append(snapReadout(L.snapInfo));
      if (L.vpInfo) f.append(vpReadout(L));
      // two lines cannot measure each other, but two lines and a horizon can
      if (L.lines.length >= 3 ||
          (L.lines.length >= 2 && L.on_horizon
           && (horizonVps(L) || camHorizon(L)))) {
        var tr = el('div', { className: 'row' });
        var tb = el('button', {
          className: 'small ghost', textContent: 'Tidy marks',
          title: 'Swing the lines that are only hand-shaky onto the point the ' +
            'rest agree on. Only moves a line by as much as a hand mark is ' +
            'worth here; anything further out is the drawing, and is left ' +
            'exactly as marked. Cmd-Z puts it back.'
        });
        tb.onclick = function () { tidyFamily(L); };
        tr.append(tb);
        if (L.tidied) tr.append(el('span', {
          className: 'dim grow',
          textContent: L.tidied.n + ' absorbed, worst ' + num(L.tidied.worst, 1) + ' px'
        }));
        f.append(tr);
      }
    }
    if (L.kind === 'surface') {
      // A table turned away from the wall is still on the floor, so its top
      // is parallel to the floor and its vanishing line IS the horizon. That
      // needs no lines marked on the table at all -- only that a horizon
      // exists, which two horizontal families already give.
      var horiz = horizonVps(null) || [];
      var lr = el('div', { className: 'row' });
      var lb = el('input', { type: 'checkbox', checked: !!L.level,
                             disabled: horiz.length < 2 && !L.level });
      lb.onchange = function () { L.level = lb.checked; showEditor(); paint(); commit(); };
      lr.append(lb);
      lr.append(el('label', { textContent: 'level with the ground',
        title: 'Its vanishing line is the horizon. Nothing here measures whether it really is level.' }));
      f.append(lr);
      if (horiz.length < 2) {
        f.append(el('div', { className: 'hint',
          textContent: 'Two horizontal families are what deduce a horizon; ' +
            horiz.length + ' marked so far' + (L.level ? ', so a level surface ' +
            'cannot be read yet.' : '.') }));
      }
      if (L.level) {
        if (horiz.length >= 2) f.append(el('div', { className: 'hint',
          textContent: 'The horizon through ' + horiz.slice(0, 2).map(function (F) {
            return F.name; }).join(' and ') + '. Declared, not measured: the check ' +
            'will say so.' }));
      } else {
        f.append(refRow(L, '_a', 'spanned by', 'The two vanishing points of directions lying IN this plane'));
        f.append(refRow(L, '_b', 'and'));
        L.refs.vps = [L.refs._a, L.refs._b];
        f.querySelectorAll('select').forEach(function (s) {
          s.addEventListener('change', function () {
            L.refs.vps = [L.refs._a, L.refs._b];
            showEditor(); paint();     // the patch is drawn from these two points
          });
        });
      }
    }
    if (L.kind === 'arc') {
      f.append(surfRow(L));
      f.append(numRow(L, 'resolve', 'resolve', '0.01',
        'The smallest departure from round worth calling; below it the check says so rather than guessing'));
    }
    if (L.kind === 'ellipse') {
      f.append(surfRow(L));
      var er = el('div', { className: 'row' });
      var eb = el('button', {
        className: 'small' + (S.ringMark ? '' : ' ghost'),
        textContent: S.ringMark ? 'marking the RIM' : 'drawing the CIRCLE',
        title: 'Drawing: one drag, right across the circle the long way, and ' +
               'the ellipse a real circle in that plane would cast is drawn ' +
               'for you. Marking: click round the rim of the one already in the ' +
               'picture and it is measured instead.'
      });
      eb.onclick = function () { finishRun(); S.ringMark = !S.ringMark; S.inSquare = false; showEditor(); };
      er.append(eb);
      er.append(el('span', { className: 'dim',
        textContent: L.pts.length + ' rim mark' + (L.pts.length === 1 ? '' : 's') }));
      f.append(er);
      if (L.ringWhy) f.append(el('div', { className: 'why warn', textContent: L.ringWhy }));
      else if ((L.ringsTrue || []).some(Boolean)) f.append(el('div', { className: 'hint',
        textContent: 'The solid ring is the circle seen square on, so your two '
          + 'dots are the ends of its long diameter. The faint one is what this '
          + 'picture\u2019s own camera casts: it leans away from the centre of '
          + 'vision, and it is the one a check measures against.' }));
      if (!L.ringWhy && L.rings && L.pts.length < 5) f.append(el('div', { className: 'hint',
        textContent: 'drawn as the circle its plane and the camera give, so it ' +
          'is round by construction and nothing here is checked. It is a guide to ' +
          'draw against; mark five or more round the rim of what you drew to have ' +
          'the drawing measured.' }));
      f.append(numRow(L, 'resolve', 'resolve', '0.01',
        'The smallest departure from round worth calling; below it the check says so rather than guessing'));
      var sr = el('div', { className: 'row' });
      var sb = el('button', {
        className: 'small' + (S.inSquare ? '' : ' ghost'),
        textContent: S.inSquare ? 'click inside a cell' : 'circle in a square',
        title: 'Click inside a cell \u2014 the space between two lines of each of ' +
               'the two families that span this plane \u2014 and the circle ' +
               'inscribed in it is drawn. Its four corners fix it exactly, so ' +
               'that needs no camera; whether the cell IS a square is what the ' +
               'camera is asked, below. Drag its dot to move it to another cell, ' +
               'cmd-click to delete it.'
      });
      sb.onclick = function () { finishRun(); S.inSquare = !S.inSquare; showEditor(); };
      sr.append(sb);
      sr.append(el('span', { className: 'dim', textContent: L.cells.length +
        ' circle' + (L.cells.length === 1 ? '' : 's') + ' in squares' }));
      f.append(sr);
      var fr = el('div', { className: 'row' });
      var fb = el('input', { type: 'checkbox', checked: !!L.autoCells });
      fb.onchange = function () { L.autoCells = fb.checked; commit(); };
      var fl = el('label', { style: 'min-width:0',
        title: 'A family that snaps new lines to a square puts a circle in the ' +
               'square it closes, here, when the two families are the ones that ' +
               'span this plane.' });
      fl.append(fb, document.createTextNode(' fill the squares a snap closes'));
      fr.append(fl); f.append(fr);
      f.append(cellReadouts(L));
    }
    if (L.kind === 'arch') {
      f.append(refRow(L, 'vertical_vp', 'vertical VP'));
      f.append(refRow(L, 'wall_vp', "wall's VP",
        'The horizontal direction lying in the arch wall. Leave it empty and ' +
        'the arch stands in the vertical plane through its OWN two feet \u2014 ' +
        'drag the springing line diagonally across the bay and you get a ' +
        'diagonal rib, where a named wall would flatten it onto the wall\u2019s face.'));
      f.append(refRow(L, 'normal_vp', 'depth VP', 'Only for a vault that projects forward; leave empty for a flat arch'));
      if (shapeOf(L) === 'straight') {
        f.append(refRow(L, 'slope_vp', 'slope VP',
          'For a roof you MARKED: the vanishing point its sloping edges run to. ' +
          'It must lie on the line from the vertical VP to the wall\u2019s own, ' +
          'and the check says by how much it misses. A roof you draw here has ' +
          'its slope VP read off the drawing instead.'));
      }
      var ar = el('div', { className: 'row' });
      var ab = el('button', {
        className: 'small' + (S.archDraw ? '' : ' ghost'),
        textContent: S.archDraw ? 'drawing the ARCH' : 'marking the ARCH',
        title: 'Drawing: drag the springing line across between the two ' +
               'supports, then move the top \u2014 it stays over the middle of ' +
               'the span, and the arch a real one would cast is drawn for you. ' +
               'Marking: click round the arch already in the picture and it is ' +
               'measured instead.'
      });
      ab.onclick = function () { finishRun(); S.archDraw = !S.archDraw; showEditor(); paint(); };
      ar.append(ab);
      var sg = null;
      if (S.archDraw) {
        sg = el('div', {
          className: 'seg',
          title: 'Round: one centre, on the crown\u2019s own upright, so both ' +
                 'halves leave the top with the same tangent and the top is ' +
                 'blunt. Pointed: two centres, both on the springing line, so ' +
                 'the halves arrive at an angle and the top is sharp. A round ' +
                 'arch is drawn as ONE Bezier curve, a pointed one as a curve ' +
                 'per side meeting at the point. Straight: no curve at all \u2014 ' +
                 'a gable, which slid back along the depth lines is a roof.'
        });
        ['round', 'pointed', 'straight'].forEach(function (o) {
          var b = el('button', { className: 'small' + (shapeOf(L) === o ? ' active' : ''),
                                 textContent: o });
          b.onclick = function () {
            if (shapeOf(L) === o) return;
            L.shape = o; fitArch(L); commit();
          };
          sg.append(b);
        });
        if ((L.crown || []).some(Boolean)) {
          var lnb = el('button', {
            className: 'small' + (L.lean ? ' active' : ' ghost'),
            textContent: L.lean ? 'leaning' : 'upright',
            title: 'Upright: the arch stands in a vertical plane and the top ' +
                   'is pulled onto the upright over the middle of the span. ' +
                   'Leaning: the feet stay put and the whole arch TIPS about ' +
                   'the springing line, as a gothic rib leaves its respond and ' +
                   'goes over the bay \u2014 the top then stays exactly where ' +
                   'you put it, and where that is says both the lean and the ' +
                   'rise. One mark cannot also say a depth, so a leaning arch ' +
                   'has no vault.'
          });
          lnb.onclick = function () { L.lean = !L.lean; fitArch(L); commit(); };
          sg.append(lnb);
        }
        // One more point ON the curve. Two feet and a top leave the arch
        // nothing to decide -- the centre is forced onto the springing line
        // or onto the crown's upright, which is the whole round/pointed
        // choice -- so the toggle is the only say over the shape, and a real
        // drop or stilted arch is not in its reach. Three points name one
        // circle; the mark is put down ON the curve the camera drew, so
        // adding it changes nothing until it is dragged.
        if ((L.crown || []).some(Boolean) && shapeOf(L) !== 'straight') {
          var hasT = (L.through || []).some(Boolean);
          var tb = el('button', {
            className: 'small' + (hasT ? ' active' : ' ghost'),
            textContent: hasT ? 'through a point' : '+ point on the curve',
            title: 'Two feet and a top force the arch\u2019s centre onto the ' +
                   'springing line or onto the crown\u2019s upright, and that ' +
                   'is all the round/pointed choice is. Mark one more point ON ' +
                   'the curve and three points name one circle instead, which ' +
                   'is what a drop or a stilted arch needs: measured, a raised ' +
                   'centre sits 60 px from the forced arch and 1.6 px from ' +
                   'this one. Mark it on either limb \u2014 the other limb is ' +
                   'its mirror. Drag it to say where the arch really goes; ' +
                   'cmd-click drops it. Adding or dropping re-reads the curve, ' +
                   'so handles moved by hand go back.'
          });
          tb.onclick = function () {
            L.bezAdj = [];       // the path is re-cut, so its offsets are stale
            L.through = hasT ? [] : (L.arches || []).map(function (q) {
              return (q && q.length) ? q[(q.length / 4) | 0].slice() : null;
            });
            fitArch(L); commit();
          };
          sg.append(tb);
        }
      }
      f.append(ar);
      if (S.archDraw) f.append(sg);
      if (S.archDraw) {
        var ar2 = el('div', { className: 'row' }), archSpread = null;
        if (L.riseSpan && L.riseSpan.length) {
          ar2.append(el('span', { className: 'dim',
            title: '0.500 is a semicircle: below it segmental, above it ' +
                   'horseshoe for a round arch, a sharper point for a pointed ' +
                   'one. That reading is the toggle\u2019s: with a point marked ' +
                   'on the curve the centre is wherever the three marks put it, ' +
                   'so the rise is still the rise and 0.500 is no longer a ' +
                   'semicircle.',
            textContent: 'rise/span ' + L.riseSpan.map(function (v) {
              return v.toFixed(3); }).join(', ') }));
        }
        // Its own line, in the refusal's colour when it is wide: an arch
        // whose shape the drawing does not pin down has to say so as loudly
        // as the single number says the opposite.
        var rr = (L.riseSpanRange || []).filter(Boolean);
        var wide = rr.some(function (q) { return q[1] > 1.5 * q[0]; });
        if (rr.length) {
          archSpread = el('div', { className: 'why' + (wide ? ' warn' : ''),
            title: 'An arch\u2019s shape is read through the camera, and the '
                 + 'camera is read from the vanishing points. These are the '
                 + 'same three marks under every camera those points still '
                 + 'allow.',
            textContent: (wide ? 'undetermined \u2014 ' : '') + 'rise/span '
              + rr.map(function (q) {
                  return q[0].toFixed(2) + ' to ' + q[1].toFixed(2);
                }).join(', ') + ' over the cameras the marks allow'
              + (wide ? ': the drawing does not say what shape this arch is. '
                      + 'Mark a second horizontal family, or uprights that '
                      + 'are not all nearly parallel on the page.' : '') });
        }
        // What one Bezier costs. A round arch is ONE cubic by request, and a
        // cubic cannot be a circle: the number is how far the curve on the
        // page runs from the arch the camera really casts.
        var off = bezOff(L);
        if (off !== null && off !== undefined) {
          ar2.append(el('span', {
            className: 'dim' + (off > 6 ? ' warn' : ''),
            title: 'A cubic Bezier is not a circle. One curve over a whole ' +
                   'round arch sits about 2 % of the rise off the arch the ' +
                   'camera casts; a pointed arch, drawn as a curve per side, ' +
                   'about 0.7 %. Moving a handle is measured the same way, so ' +
                   'this says how far what you see is from the truth.',
            textContent: (bezAdjusted(L) ? 'adjusted, ' : '') +
                         off.toFixed(2) + ' px off the true arch' }));
        }
        if (bezAdjusted(L)) {
          var rb = el('button', { className: 'small ghost', textContent: 'reset handles',
            title: 'Put every handle back where the camera put it' });
          rb.onclick = function () { L.bezAdj = []; showEditor(); paint(); commit(); };
          ar2.append(rb);
        }
        if (shapeOf(L) === 'straight' && L.riseSpan && L.riseSpan.length) {
          ar2.append(el('span', { className: 'dim',
            title: 'How steeply each side climbs from the eaves to the ridge, '
                 + 'on the wall: atan(2 x rise/span).',
            textContent: 'pitch ' + L.riseSpan.map(function (v) {
              return (Math.atan(2 * v) * 180 / Math.PI).toFixed(0) + '\u00b0';
            }).join(', ') }));
        }
        var sl = ((L.slope || [])[0] || []).filter(Boolean);
        if (sl.length) {
          ar2.append(el('span', { className: 'dim',
            title: 'Each slope and its twin on the far gable are ONE direction '
                 + 'in the world, so they meet there: the slope\u2019s own '
                 + 'vanishing point. Aim the sloping edges at it.',
            textContent: 'slopes meet at ' + sl.map(function (q) {
              return Math.round(q[0]) + ', ' + Math.round(q[1]);
            }).join(' and ') }));
        }
        var lean = (L.leanDeg || []).filter(function (v) { return v != null; });
        if (lean.length) {
          ar2.append(el('span', { className: 'dim',
            title: 'How far the arch\u2019s own plane tips off the vertical, '
                 + 'about the springing line. The top you marked says it: the '
                 + 'line from the middle of the span through it IS the arch\u2019s '
                 + 'upright, and where it crosses the uprights\u2019 vanishing '
                 + 'line gives the plane. 0 is an arch standing in a wall.',
            textContent: 'leans ' + lean.map(function (v) {
              return v.toFixed(0) + '\u00b0'; }).join(', ') + ' off upright' }));
        }
        if ((L.vault || []).some(function (v) { return !!v; })) {
          ar2.append(el('span', { className: 'dim',
            title: 'The crown has left the wall along the depth lines, so the '
                 + 'arch is drawn twice: once where it springs and once slid '
                 + 'back. Put the crown back on the upright over the middle '
                 + 'of the span for a flat arch.',
            textContent: 'a vault \u2014 the arch slid back' }));
        }
        f.append(ar2);
        if (archSpread) f.append(archSpread);
        if (L.crownNote) f.append(el('div', { className: 'why',
                                              textContent: L.crownNote }));
        if (L.depthNote) f.append(el('div', { className: 'why',
                                              textContent: L.depthNote }));
      }
      if (S.archDraw && L.archWhy) {
        f.append(el('div', { className: 'why warn', textContent: L.archWhy }));
      } else if (S.archDraw && L.arches && L.pts.length < 5) {
        f.append(el('div', { className: 'hint',
          textContent: 'drawn as the arch its wall and the camera give, so it is ' +
            'true by construction and nothing here is checked. It is a guide to ' +
            'draw against; mark five or more round the arch you drew to have the ' +
            'drawing measured.' }));
      }
      var tr = el('div', { className: 'row' });
      var tb = el('button', {
        className: 'small' + (S.archFar ? '' : ' ghost'),
        textContent: S.archFar ? 'marking the FAR limb' : 'marking the NEAR limb'
      });
      tb.onclick = function () { finishRun(); S.archFar = !S.archFar; showEditor(); };
      tr.append(tb);
      tr.append(el('span', { className: 'dim', textContent: L.pts.length + ' near, ' + L.far.length + ' far' }));
      f.append(tr);
    }
    // Corners round a shape that lies in a plane. Naming the plane is the
    // only thing that lets the shape be looked at square on, so the row is
    // here and not only on the kinds that are graded against a surface.
    if (L.kind === 'polygon') f.append(surfRow(L));
    if (contactful(L)) f.append(el('div', { id: 'supportHost' }));
    // Any outline declared to lie in a surface can be looked at square on.
    // Not a check: a drawing cannot be judged in perspective, and this hands
    // the eye the shape with the foreshortening taken out.
    if (L.refs.surface && L.pts.length >= 3) {
      var sr2 = el('div', { className: 'row' });
      var sb = el('button', { className: 'small', textContent: 'Straighten',
        title: 'Show this shape as if its surface were facing us. Nothing is ' +
               'graded \u2014 it is there to be looked at.' });
      sb.onclick = function () { straighten(L); };
      sr2.append(sb);
      f.append(sr2);
      var host = el('div', { id: 'straightHost' });
      if (S.straight && S.straight.id === L.id) {
        host.append(el('img', { src: S.straight.url, className: 'flat' }));
      }
      f.append(host);
    }
    if (L.kind === 'posts') {
      f.append(surfRow(L));
      f.append(refRow(L, 'upright', 'objects stand along',
        'The family the objects themselves run along \u2014 the vertical one ' +
        'for lamp posts. With the top of the first one marked, every object ' +
        'is drawn at the height the perspective gives it.'));
      if (!L.refs.surface) {
        f.append(refRow(L, 'vp', 'row runs to',
          'Even spacing is measured toward the vanishing point the row ' +
          'recedes to. Give a surface instead and this is worked out from it.'));
      } else {
        f.append(el('div', { className: 'why',
          textContent: 'the row runs where its feet meet the surface\u2019s own ' +
            'vanishing line, so there is no family to name here' }));
      }
      if (L.refs.upright) {
        var tr3 = el('div', { className: 'row' });
        var tb3 = el('button', {
          className: 'small' + (S.archFar ? '' : ' ghost'),
          textContent: S.archFar ? 'marking the FIRST TOP' : 'marking FEET',
          title: 'Mark the feet in order along the row, then switch and click ' +
                 'the top of the first object once. The rest follow.'
        });
        tb3.onclick = function () { finishRun(); S.archFar = !S.archFar; showEditor(); };
        tr3.append(tb3);
        tr3.append(el('span', { className: 'dim',
          textContent: L.far.length ? 'top marked' : 'no top yet' }));
        f.append(tr3);
      }
      f.append(numRow(L, 'count', 'or: ends + how many', '1',
        'Mark only the first and last post, then say how many there are in ' +
        'all. The row in between is drawn where even spacing puts it \u2014 a ' +
        'guide to draw against, not a measurement, so it is not graded. ' +
        'Leave blank and tick each post instead to have the drawing checked.'));
      if (builtRow(L)) {
        f.append(el('div', { className: 'why',
          textContent: 'built from two ends: ' + Number(L.count) + ' posts, ' +
            'evenly spaced by construction. Nothing here is checked.' }));
      } else if (L.pts.length >= 4) {
        var tr2 = el('div', { className: 'row' });
        var tb2 = el('button', {
          className: 'small', textContent: 'Tidy marks',
          title: 'Pull the posts onto the even row the others agree on, but ' +
                 'only as far as a hand mark. A post asking for more than ' +
                 'that is the drawing, not the marking, and is left alone.'
        });
        tb2.onclick = function () { tidyRow(L); };
        tr2.append(tb2);
        if (L.rowTidied) {
          tr2.append(el('span', { className: 'dim',
            textContent: L.rowTidied.n + ' absorbed, worst ' + L.rowTidied.worst + ' px' }));
        }
        f.append(tr2);
      }
      f.append(numRow(L, 'px_per_mm', 'px per mm', '0.01',
        'Leave blank to use the real size declared for the whole picture. ' +
        'With no size declared anywhere the numbers are plain pixels.'));
      f.append(numRow(L, 'resolve_mm', 'resolve (mm)', '0.1',
        'How far out a post may sit and still count as even. Never finer ' +
        'than the hand that placed the marks -- the check raises it to that ' +
        'floor and says so.'));
    }
    if (L.kind === 'landmark') {
      f.append(refRow(L, 'vp', 'runs along',
        'The world direction this thing lies along. Anything standing up uses the vertical vanishing point; leave it unset and the document\u2019s vertical is used.'));
      var sr = el('div', { className: 'row' });
      sr.append(el('label', { textContent: 'real size', title: 'How big it really is, out there in the scene.' }));
      var si = el('input', { type: 'number', className: 'grow', step: 'any', value: L.size });
      si.onchange = function () { L.size = Number(si.value); };
      var su = el('select', { className: 'small' });
      ['mm', 'cm', 'm', 'in', 'ft'].forEach(function (u) {
        su.append(el('option', { value: u, textContent: u, selected: L.size_unit === u }));
      });
      su.onchange = function () { L.size_unit = su.value; };
      sr.append(si); sr.append(su);
      f.append(sr);
      var gr = el('div', { className: 'row' });
      var gb = el('input', { type: 'checkbox', checked: !!L.on_ground });
      gb.onchange = function () { L.on_ground = gb.checked; };
      gr.append(gb);
      gr.append(el('label', { textContent: 'stands on the viewer\u2019s own ground',
        title: 'Only then does the eye height mean anything. A figure on a step, a balcony or a boat does not.' }));
      f.append(gr);
    }
    if (L.kind === 'axis') {
      f.append(refRow(L, 'vp', "axis's VP",
        'A standing figure uses the vertical vanishing point; a reclining one does not'));
    }
    if (L.kind === 'pairs') {
      f.append(refRow(L, 'axis_vp', 'axis VP'));
      f.append(refRow(L, 'pair_vp', 'pair VP'));
    }
    refreshLayers();   // which fills #supportHost
  }

  // Re-rendered whenever the marks change, not only when the layer is picked:
  // the gap it reports is about the marks, so a stale copy of it is worse than
  // none. The selects around it keep their state because they are not touched.
  // Thin and faint, so an offer is never mistaken for a mark.
  function drawCand(g) {
    if (!S.cand || !$('#showCand').checked) return;
    g.save();
    // Thin and faint against the marks' 1.6-2.4 px. Dashing them instead was
    // tried and reverted: a 58 px segment at 45% zoom is 26 screen px, and a
    // dash pattern there erases it altogether.
    g.globalAlpha = 0.65; g.lineWidth = 1.2; g.setLineDash([]);
    S.cand.segs.forEach(function (q, i) {
      var f = S.cand.famOf[i];
      g.strokeStyle = f < 0 ? '#8a94a6' : COLORS[f % COLORS.length];
      strokeSeg(g, q);
    });
    g.restore();
  }

  function paintSupport() {
    var host = $('#supportHost');
    if (!host) return;
    var L = selected();
    host.textContent = '';
    if (L && contactful(L)) host.append(supportBlock(L));
  }

  function supportBlock(L) {
    var box = document.createDocumentFragment();
    var tr = el('div', { className: 'row' });
    var tb = el('button', {
      className: 'small' + (S.supportEdge ? '' : ' ghost'),
      textContent: S.supportEdge ? 'marking the SUPPORT EDGE' : 'marking CONTACTS',
      title: 'Drag out the edge of the ground, bench or step these contacts ' +
             'rest on. It is drawn for reference and checked against -- a ' +
             'contact that lands on it is almost certainly the edge, not the figure.'
    });
    tb.onclick = function () { finishRun(); S.supportEdge = !S.supportEdge; showEditor(); };
    tr.append(tb);
    tr.append(el('span', {
      className: 'dim',
      textContent: L.lines.length ? 'edge marked' : 'no support edge'
    }));
    box.append(tr);

    var gaps = contactGaps(L);
    if (gaps && gaps.length) {
      var onEdge = gaps.filter(function (g) { return g.screen < MARK_PX; }).length;
      var px = gaps.map(function (g) { return g.px; });
      var d = el('div', { className: 'ro' + (onEdge ? ' undet' : '') });
      d.append(el('h4', { textContent: 'contacts vs the support edge' }));
      d.append(verdictChip(onEdge ? 'check these' : 'ok'));
      d.append(el('div', {
        className: 'val',
        textContent: 'nearest ' + num(Math.min.apply(null, px)) + ' px, ' +
          'furthest ' + num(Math.max.apply(null, px)) + ' px'
      }));
      d.append(el('div', {
        className: 'why',
        textContent: onEdge
          ? onEdge + ' of ' + gaps.length + ' contacts sit within a hand mark of the ' +
            'edge at the zoom they were placed, so they cannot be told from it. The edge is the crisp line the eye ' +
            'goes to, but a figure meets the support further back, at a different ' +
            'depth -- put the mark on the figure\'s own silhouette.'
          : 'every contact is clear of the edge, so none of them is the edge by mistake'
      }));
      box.append(d);
    }
    return box;
  }

  function surfRow(L) {
    var r = el('div', { className: 'row' });
    // an ellipse falls back to the ground; everything else must say which
    // plane it lies in, because nothing shows the reader that it guessed
    var ground = L.kind === 'ellipse';
    r.append(el('label', { textContent: 'surface', title: ground
      ? 'Optional. Empty means the circle lies flat on the ground, whose vanishing line is the horizon. Name a surface for one standing in a wall or on a slope.'
      : 'Required. An arc in a wall measured against the horizon looked plausible for a whole session and was wrong.' }));
    var sel = el('select', { className: 'grow' });
    sel.append(el('option', { value: '',
      textContent: ground ? '— the ground (the horizon) —' : '— none —' }));
    S.layers.filter(function (X) { return X.kind === 'surface'; }).forEach(function (X) {
      var o = el('option', { value: X.id, textContent: X.name });
      if (L.refs.surface === X.id) o.selected = true;
      sel.append(o);
    });
    sel.onchange = function () {
      L.refs.surface = sel.value || null;
      if (L.kind === 'ellipse') { fitRing(L); fitCells(L); }
      showEditor(); paint(); commit();
    };
    r.append(sel);
    return r;
  }

  // What the last snap did. Doing nothing is an answer, so it says why.
  function snapReadout(r) {
    var c = r.conditioning || {};
    var d = el('div', { className: 'ro' + (r.verdict === 'undetermined' ? ' undet' : '') });
    d.append(el('h4', { textContent: 'last snap' }));
    d.append(verdictChip(r.verdict));
    d.append(el('div', { className: 'why', textContent: r.verdict === 'ok'
      ? 'moved ' + num(c.moved_px, 1) + ' px onto the square it makes with ' + c.against +
        ' \u2014 ' + (c.focal_span_assumed ? 'a tenth either side of the fitted focal length shifts'
                                           : 'the focal lengths the marks allow shift') +
        ' that side by ' + num(c.f_moves_it_px, 1) + ' px'
      : (r.why || 'no reason given') }));
    var cb = condBlock(c);
    if (cb) d.append(cb);
    return d;
  }

  // One card per circle in a square: the cell it is in, and whether that cell
  // is a square. The circle is exact either way; the number is the claim.
  function cellReadouts(L) {
    var box = el('div');
    if (L.cellWhy) box.append(el('div', { className: 'why warn', textContent: L.cellWhy }));
    (L.cellOut || []).forEach(function (c, i) {
      var r = c.reading;
      var d = el('div', { className: 'ro' + (!c.ring || (r && r.verdict === 'undetermined') ? ' undet' : '') });
      d.append(el('h4', { textContent: 'circle ' + (i + 1) + ' in a square' }));
      if (!c.ring) {
        d.append(el('div', { className: 'why warn', textContent: c.why }));
      } else if (r) {
        d.append(verdictChip(r.verdict));
        d.append(el('div', { className: 'why', textContent: r.why }));
        var cb = condBlock(r.conditioning);
        if (cb) d.append(cb);
      }
      box.append(d);
    });
    return box;
  }

  function vpReadout(L) {
    var d = el('div', { className: 'ro' });
    var r = L.vpInfo, c = r.conditioning || {};
    d.append(el('h4', { textContent: 'vanishing point' }));
    d.append(verdictChip(r.verdict));
    if (r.verdict === 'at_infinity') {
      d.append(el('div', { className: 'why', textContent: 'these lines are parallel on the page' }));
    } else if (r.value) {
      d.append(el('div', { className: 'val', textContent: num(r.value[0], 1) + ', ' + num(r.value[1], 1) }));
    }
    if (c.spread_unknown) {
      d.append(el('div', { className: 'why', textContent: c.spread_unknown }));
    } else if (c.resid_deg) {
      var m = Math.max.apply(null, c.resid_deg);
      d.append(el('div', { className: 'why', textContent: 'worst line off by ' + num(m, 3) + ' deg' }));
    }
    if (c.stray_why) d.append(el('div', { className: 'why warn', textContent: c.stray_why }));
    if (c.tidied_why) d.append(el('div', { className: 'why', textContent: c.tidied_why }));
    if (c.horizon_why) d.append(el('div', { className: 'why', textContent: c.horizon_why }));
    if (c.square_why) d.append(el('div', { className: 'why', textContent: c.square_why }));
    if (c.camera_undetermined) d.append(el('div', { className: 'why warn',
      textContent: 'and the camera itself is undetermined: ' + c.camera_undetermined }));
    if (c.moves_px !== undefined) {
      // In pixels of POSITION a derived point is hopeless and in pixels of
      // SWING the same slack is worth under one. Both, or it reads as either
      // a precise point or a useless one, and it is neither.
      d.append(el('div', { className: 'why', textContent:
        'over f from ' + num(c.focal_span_px[0], 0) + ' to ' +
        num(c.focal_span_px[1], 0) + ' px' +
        (c.focal_span_assumed ? ' (a tenth either way, the camera quoting no range)' : '') +
        ' it moves ' + num(c.moves_px, 0) + ' px' +
        (c.swing_px === undefined ? ''
          : ', which swings a mark of this length ' + num(c.swing_px, 2) + ' px') +
        ' \u2014 a point to draw to, not a position to quote' }));
    }
    if (c.gap_why) d.append(el('div', { className: 'why', textContent: c.gap_why }));
    if (c.bootstrap_x) {
      d.append(el('div', {
        className: 'why',
        textContent: 'x within ' + num(c.bootstrap_x[0], 0) + ' to ' + num(c.bootstrap_x[1], 0) +
          ', y within ' + num(c.bootstrap_y[0], 0) + ' to ' + num(c.bootstrap_y[1], 0)
      }));
    }
    return d;
  }

  function hintFor(L) {
    if (!S.image) return 'Open a picture, then add a layer and click on it.';
    if (!L) return 'Pick a layer to mark on it. Drag anywhere to pan, scroll to zoom.';
    var how = drawMode(L);
    var tail = L.kind === 'surface'
        ? (surfaceVps(L)
           ? ' Drag across the plane: the two ends are opposite corners, and the '
             + 'edges run to its vanishing points. Drag an end to resize it.'
           : ' Name its two vanishing points below, or tick level with the ground, '
             + 'and then drag across the plane to see it.')
      : how === 'line'
        ? (contactful(L) ? ' Drag out the edge they rest on.'
         : ' Drag out each line; drag a dot on its end to move that end, '
           + 'cmd-click it to delete the line.')
      : how === 'cell' ? ' Click inside a cell \u2014 between two lines of each family \u2014 and the circle inscribed in it is drawn.'
      : how === 'poly' ? ' Drag to trace it, or click point by point; Enter or double-click ends a clicked run.'
      : how === 'triple' ? " Click three for each object: its base on the ground, its top, then its shadow's tip."
      : how === 'pair' ? ' Click the two points of each pair.'
      : ' Nothing to draw -- choose the vanishing points below.';
    return KINDS[L.kind].blurb + tail;
  }

  // ------------------------------------------------------------- the results

  function verdictChip(v) {
    var cls = (v === 'undetermined' || v === 'check these') ? 'v-undetermined'
      : (v === 'ok' || v === 'already consistent') ? 'v-ok'
      : (v === 'rectifiable' || v === 'at_infinity') ? 'v-info' : 'v-off';
    return el('span', { className: 'verdict ' + cls, textContent: v });
  }

  function condBlock(c) {
    if (!c || !Object.keys(c).length) return null;
    var d = el('details');
    d.append(el('summary', { textContent: 'conditioning' }));
    var lines = Object.keys(c).map(function (k) {
      var v = c[k];
      if (v && typeof v === 'object') v = JSON.stringify(v).slice(0, 260);
      else if (typeof v === 'number') v = num(v, 3);
      return k + ': ' + v;
    });
    d.append(el('div', { className: 'cond', textContent: lines.join('\n') }));
    return d;
  }

  function roCard(title, r) {
    var d = el('div', { className: 'ro' + (r.verdict === 'undetermined' ? ' undet' : '') });
    d.append(el('h4', { textContent: title }));
    d.append(verdictChip(r.verdict));
    if (r.verdict === 'undetermined') {
      d.append(el('div', { className: 'why', textContent: r.why || 'no reason given' }));
    } else if (r.value !== null && r.value !== undefined) {
      var v = r.value;
      if (typeof v === 'object') {
        var txt = Object.keys(v).slice(0, 8).map(function (k) {
          var x = v[k];
          if (Array.isArray(x)) x = x.map(function (y) { return num(y, 3); }).join(', ');
          else if (typeof x === 'number') x = num(x, 3);
          return k + ' ' + x;
        }).join('\n');
        d.append(el('div', { className: 'cond', textContent: txt }));
      } else {
        d.append(el('div', { className: 'val', textContent: num(v, 3) }));
      }
    }
    var cb = condBlock(r.conditioning);
    if (cb) d.append(cb);
    return d;
  }

  function showResults() {
    var host = $('#results');
    host.textContent = '';
    var r = S.results;
    $('#resultsHint').classList.add('hidden');
    if (r.camera) host.append(roCard('camera', r.camera));
    Object.keys(r.results || {}).forEach(function (id) {
      var one = r.results[id];
      var card = roCard(one.name + '  (' + one.kind + ')', one);
      // A landmark's answer is one line per thing marked, and the range is
      // the answer as much as the number is -- a distance good to +-30 % read
      // as a bare "26 m" is the confident wrong number this app avoids.
      ((one.conditioning || {}).per_landmark || []).forEach(function (e) {
        var u = one.conditioning.unit;
        card.append(el('div', {
          className: 'why',
          textContent: e.distance === null
            ? '#' + e.n + ': ' + e.why
            : '#' + e.n + ': the viewer is ' + num(e.distance, 2) + ' ' + u + ' away' +
              ' (' + num(e.range[0], 2) + '-' + num(e.range[1], 2) + ' ' + u +
              ', \u00b1' + num(e.plus_minus_pct, 0) + '%, marked ' +
              num(e.height_px, 0) + ' px tall)' +
              (e.eye_height === null ? ''
                : ', and the eye is ' + num(e.eye_height, 2) + ' ' + u + ' above its ground')
        }));
      });
      host.append(card);
    });
    if (r.total) {
      host.append(el('div', {
        className: 'hint',
        textContent: r.undetermined + ' of ' + r.total + ' checks could not be answered from these marks.'
      }));
    }
  }

  function showConsensus(r) {
    var host = $('#results');
    host.textContent = '';
    $('#resultsHint').classList.add('hidden');
    ['free', 'level', 'reference'].forEach(function (key) {
      if (!r[key]) return;
      var pc = (r[key].conditioning || {}).perspective;
      var title = key === 'free'
        ? 'as drawn, checked as ' + (pc || 'three-point')
        : key === 'level' ? 'forced level'
        : 'against the reference camera' +
          (S.ref && S.ref.camera.source ? '  (' + (S.ref.camera.source.file || 'file') + ')' : '');
      var card = roCard(title, r[key]);
      var c = r[key].conditioning || {};
      // Exact by construction is not consistent: when the marks number no more
      // than the camera's unknowns any camera fits them, and the card says
      // that and nothing that would read as agreement
      if (r[key].verdict === 'undetermined') { host.append(card); return; }
      if (c.worst_move_px !== undefined) {
        var n = 0;
        Object.keys(c.per_family || {}).forEach(function (id) {
          (c.per_family[id].move_px || []).forEach(function (m) {
            if (m > (c.mark_accuracy_px || MARK_PX)) n++;
          });
        });
        card.append(el('div', {
          className: 'val',
          textContent: 'worst line moves ' + num(c.worst_move_px) + ' px, median ' +
            num(c.median_move_px) + ' px  (hand marking is ' + num(c.mark_accuracy_px, 0) + ' px)' +
            (c.worst_move_mm === undefined ? '' :
              '   =  ' + num(c.worst_move_mm, 2) + ' mm worst, ' +
              num(c.median_move_mm, 2) + ' mm median (hand marking ' +
              num(c.mark_accuracy_mm, 2) + ' mm)'),
          // on a big scan that figure is scaled, and an unexplained 36 px reads
          // as arbitrary -- the server says what it rests on
          title: c.mark_accuracy_basis || ''
        }));
        card.append(el('div', {
          className: 'why',
          textContent: n ? n + ' line' + (n === 1 ? '' : 's') + ' would have to move further ' +
            'than a hand mark is worth' : 'no line has to move further than a hand mark is worth'
        }));
      }
      if (key === 'reference') {
        referenceNotes(card, c);
      } else if (c.focal_refused) {
        card.append(el('div', { className: 'why', textContent: 'focal length refused: ' + c.focal_refused }));
      } else if (c.focal_px && c.mode !== 'five') {
        var rg = c.focal_range_over_pp || c.focal_range_over_pp_x;
        // In the picture's own units f is the station point: how far the eye
        // belongs from the paper. Say the range in whatever unit the sentence
        // is already speaking -- quoting mm and then a span of pixels reads
        // as one number with two meanings.
        var mm = c.mm_per_px, f = num(c.focal_px, 0) + ' px';
        if (c.viewing_distance_mm !== undefined) {
          f += ' \u2014 the eye belongs ' + num(c.viewing_distance_mm, 0) + ' mm from the picture';
        }
        if (rg) {
          f += ', anywhere from ' + (mm ? num(rg[0] * mm, 0) + ' to ' + num(rg[1] * mm, 0) + ' mm'
                                        : num(rg[0], 0) + ' to ' + num(rg[1], 0) + ' px') +
               ' depending on where the principal point is assumed';
        }
        card.append(el('div', { className: 'why', textContent: f }));
      }
      if (c.focal_note) {
        card.append(el('div', { className: 'why', textContent: c.focal_note }));
      }
      // The principal point can be fitted rather than assumed, but only when
      // the vanishing points can place it. When they cannot, the solver's
      // answer looks BETTER by every residual, so the refusal has to be as
      // loud as the number it replaces.
      if (c.pp_refused) {
        card.append(el('div', {
          className: 'why warn',
          textContent: 'principal point not measured, pinned at ' +
            (c.pp_assumed || 'the image centre') + ': ' + c.pp_refused
        }));
      }
      Object.keys(c.per_family || {}).forEach(function (id) {
        var L = layer(id); if (!L) return;
        var d = c.per_family[id];
        // Which axis a family went to is only a statement against a reference
        // camera. In the free fit the labels are the solver's own order, and
        // printing them would read as a claim about which family is upright.
        var on = '';
        if (key === 'reference' && d.axis) {
          var rms = d.rms_on_axis_px || {};
          var other = Object.keys(rms).filter(function (k) { return k !== d.axis; })
            .map(function (k) { return k + ' ' + num(rms[k], 0) + ' px'; }).join(', ');
          on = '  \u2192 axis ' + d.axis + (other ? ' (rms ' + num(rms[d.axis], 1) +
               ' px there; on the others ' + other + ')' : '');
        }
        card.append(el('div', {
          className: 'why',
          textContent: L.name + ': worst ' + num(d.worst_px) + ' px, median ' + num(d.median_px) + ' px' +
            (d.worst_mm === undefined ? '' :
              '  =  ' + num(d.worst_mm, 2) + ' / ' + num(d.median_mm, 2) + ' mm') + on
        }));
      });
      host.append(card);
    });
  }

  // What the reference card says that the others do not: the camera is
  // somebody else's claim, so the readout is how far the marks are from it, how
  // that compares with how far they are from consistency on their own, and
  // whether the marks are even sure which axis is which.
  function referenceNotes(card, c) {
    var as = c.assignment || {};
    if (as.ambiguous) {
      card.append(el('div', { className: 'why warn', textContent:
        'which axis each family belongs to is not settled by the marks: the ' +
        'next-best assignment costs only ' + num(as.margin_px, 1) + ' px rms more' }));
    }
    if (c.focal_px) {
      var cm = c.camera || {};
      card.append(el('div', { className: 'why', textContent:
        'the camera: f ' + num(c.focal_px, 0) + ' px' +
        (cm.lens_mm_on_36 ? ' = ' + num(cm.lens_mm_on_36, 1) + ' mm on a 36 mm sensor' : '') +
        (c.principal_point ? ', principal point (' + num(c.principal_point[0], 0) + ', ' +
           num(c.principal_point[1], 0) + ')' : '') }));
    }
    if (c.price_of_reference_px !== undefined) {
      var p = c.price_of_reference_px, own = c.worst_move_px - p;
      card.append(el('div', { className: 'why', textContent:
        c.own_redundancy !== undefined && c.own_redundancy <= 0
        ? 'the marks have no line to spare, so the camera they define costs them 0 px ' +
          'by construction; this one asks ' + num(c.worst_move_px) + ' px of them'
        : 'on their own the marks need at most ' + num(own) + ' px to be consistent; ' +
          'this camera asks ' + num(c.worst_move_px) + ' px \u2014 ' +
          (p >= 0 ? num(p) + ' px more' : num(-p) + ' px less') + ' than the marks do' }));
    }
    var v = c.versus_marks;
    if (v) {
      card.append(el('div', { className: 'why', textContent:
        'the camera the marks alone imply is ' + num(v.rotation_deg, 1) +
        '\u00b0 from this one in rotation, its lens is x' + num(v.f_ratio, 2) +
        ' of this one and its principal point is ' + num(v.pp_offset_px, 0) + ' px away' }));
    }
  }

  // ------------------------------------------------------------------- trim
  //
  // A photograph of a drawing is not the drawing. It carries the desk, the
  // tape and the drafting board, and the straightest lines in the frame are
  // usually the sheet's own edges -- on testdata/1.jpg they formed the
  // second-largest vanishing-point family, 25 segments with a longer median
  // than any family in the artwork itself. Worse, shooting a flat sheet
  // off-axis composes a homography onto the perspective the artist drew, so
  // every vanishing point moves and the horizon tilts. Both are fixed by
  // mapping the sheet back to the rectangle it must have been.
  //
  // The trim is a setting, not a file: it travels with each request and the
  // user's original picture is never rewritten.

  function shownTrim() {
    // while the corners are being placed, the whole photograph is on screen
    return S.trimEdit ? null : S.trim;
  }

  function picURL(base, path) {
    var t = shownTrim();
    return base + '?path=' + encodeURIComponent(path) +
           (t ? '&trim=' + encodeURIComponent(JSON.stringify(t)) : '');
  }

  function trimState() {
    var b = $('#trimState'), on = !!S.trim;
    b.className = 'why' + (S.trimEdit ? ' warn' : '');
    $('#btnTrimApply').classList.toggle('hidden', !S.trimEdit);
    $('#btnTrimCancel').classList.toggle('hidden', !S.trimEdit);
    $('#btnTrimEdit').textContent = on ? 'Move corners' : 'Place corners';
    $('#btnTrimClear').classList.toggle('hidden', !on);
    if (S.trimEdit) {
      b.textContent = 'Drag the four corners onto the sheet, then Apply. ' +
                      'They may sit outside the picture if the sheet runs off it.';
    } else if (on) {
      b.textContent = 'Trimmed to ' + S.trim.out[0] + ' \u00d7 ' + S.trim.out[1] +
                      '. ' + (S.trim.note || '');
    } else {
      b.textContent = 'Untrimmed. A flat scan needs no trim; a photograph ' +
                      'usually does.';
    }
  }

  function findSheet() {
    if (!S.image) { toast('open a picture first', true); return; }
    busy(true);
    fetch('/api/findsheet', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ image: S.image.path })
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) { toast(d.error, true); return; }
      // It refuses more often than it answers, and the refusal is the useful
      // half: on an artwork that is already cropped there is no sheet, and a
      // detector that always returns four corners would be inventing one.
      S.trimEdit = true;
      S.trimDraft = (d.corners || d.suggest).map(function (c) { return c.slice(); });
      S.found = d;
      $('#trimFound').textContent = d.note || '';
      $('#trimFound').className = 'why' + (d.corners ? '' : ' warn');
      if (d.edge_strength) {
        $('#trimFound').textContent += '  Sides measure ' +
          d.edge_strength.map(function (v) { return v === null ? 'off-picture' : v + 'x'; })
            .join(', ') + ' the picture\u2019s own texture.';
      }
      showPicture(S.image.path, true, function () { trimState(); commit(); });
    }).catch(function (e) { toast(String(e), true); })
      .finally(function () { busy(false); });
  }

  function enterTrim() {
    if (!S.image) { toast('open a picture first', true); return; }
    S.trimEdit = true;
    S.trimDraft = S.trim ? S.trim.corners.map(function (c) { return c.slice(); }) : null;
    showPicture(S.image.path, true, function () {
      if (!S.trimDraft) {
        var w = S.image.w, h = S.image.h, mx = 0.08 * w, my = 0.08 * h;
        S.trimDraft = [[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]];
      }
      trimState(); paint(); commit();
    });
  }

  function setTrim(next) {
    // Marks are stored against the picture they were placed on, so a trim
    // that changes under them would move every one off the paint. Carry them
    // across rather than silently corrupting them or refusing the trim.
    var prev = S.trim, pts = harvest();
    var finish = function (mapped) {
      if (mapped) plant(mapped);
      if (S.ref) {
        S.ref = null;
        S.refNote = 'The camera was taken away when the trim changed: a trim ' +
          're-projects the picture, and a camera belongs to the pixels it was ' +
          'solved on. Read it again with the trimmed picture on screen.';
        camState();
      }
      S.trim = next; S.trimEdit = false; S.trimDraft = null;
      $('#trimFound').textContent = '';     // that search is answered now
      showPicture(S.image.path, true, function () {
        trimState(); refreshLayers(); paint(); commit();
        if (pts.length) toast(pts.length + ' marks moved with the trim');
      });
    };
    if (!pts.length) { finish(null); return; }
    busy(true);
    fetch('/api/retrim', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ image: S.image.path, from: prev, to: next, points: pts })
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d.error) { toast(d.error, true); finish(null); return; }
      finish(d.points);
    }).catch(function (e) { toast(String(e), true); finish(null); })
      .finally(function () { busy(false); });
  }

  // Every mark coordinate, in one flat list and back again. Listed by field
  // rather than walked generically: `span` and `unit` are two-number arrays
  // too, and a generic walker would happily "move" them.
  function eachMark(fn) {
    S.layers.forEach(function (L) {
      L.lines.forEach(function (ln, i) {
        fn(ln, 0, 1, L, 'lines', i); fn(ln, 2, 3, L, 'lines', i);
      });
      L.pts.forEach(function (q, i) { fn(q, 0, 1, L, 'pts', i); });
      L.far.forEach(function (q, i) { fn(q, 0, 1, L, 'far', i); });
      L.pairs.forEach(function (pr) {
        pr.forEach(function (q) { fn(q, 0, 1, L, 'pairs', 0); });
      });
      (L.curves || []).forEach(function (cv) {
        cv.forEach(function (q) { fn(q, 0, 1, L, 'curves', 0); });
      });
    });
  }

  function harvest() {
    var out = [];
    eachMark(function (a, ix, iy) { out.push([a[ix], a[iy]]); });
    return out;
  }

  function plant(pts) {
    var n = 0;
    eachMark(function (a, ix, iy) {
      if (n < pts.length) { a[ix] = pts[n][0]; a[iy] = pts[n][1]; }
      n++;
    });
    S.layers.forEach(function (L) { if (L.kind === 'family') fitFamily(L); });
  }

  function drawTrim(g, W, H) {
    if (!S.trimEdit || !S.trimDraft) return;
    var q = S.trimDraft.map(function (c) { return toScreen(c); });
    g.save();
    g.beginPath();
    g.rect(0, 0, W, H);
    g.moveTo(q[0][0], q[0][1]);
    for (var i = 1; i < 4; i++) g.lineTo(q[i][0], q[i][1]);
    g.closePath();
    g.fillStyle = 'rgba(8,10,14,0.55)';
    g.fill('evenodd');                      // everything outside the sheet
    g.strokeStyle = '#4ea1ff'; g.lineWidth = 2; g.setLineDash([]);
    g.beginPath();
    g.moveTo(q[0][0], q[0][1]);
    for (i = 1; i < 4; i++) g.lineTo(q[i][0], q[i][1]);
    g.closePath(); g.stroke();
    var names = ['TL', 'TR', 'BR', 'BL'];
    q.forEach(function (c, j) {
      g.beginPath(); g.arc(c[0], c[1], 7, 0, 6.2832);
      g.fillStyle = j === S.dragCorner ? '#ffd14e' : '#4ea1ff'; g.fill();
      g.lineWidth = 2; g.strokeStyle = '#0b0e14'; g.stroke();
      g.fillStyle = '#cfe4ff'; g.font = '11px system-ui';
      g.fillText(names[j], c[0] + 10, c[1] - 8);
    });
    g.restore();
  }

  function nearestCorner(sx, sy, r) {
    if (!S.trimEdit || !S.trimDraft) return -1;
    var best = -1, bd = r;
    S.trimDraft.forEach(function (c, i) {
      var p = toScreen(c), d = Math.hypot(p[0] - sx, p[1] - sy);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  }

  // ---------------------------------------------------------------- opening

  function showPicture(path, keep, after) {
    // `keep` is a trim change: the marks and results stay, only the pixels
    // under them are re-fetched. Opening a different picture keeps nothing.
    busy(true);
    fetch(picURL('/api/imageinfo', path))
      .then(function (r) { return r.json(); })
      .then(function (info) {
        if (info.error) { toast(info.error, true); return; }
        var img = new Image();
        img.onload = function () {
          var had = S.image && keep;
          S.bitmap = img;
          S.image = { path: path, w: info.w, h: info.h };
          // the output rectangle is the server's to decide: it comes out of
          // the keystone, not out of anything the page knows
          if (S.trim && info.trim) {
            S.trim.out = info.trim.out;
            S.trim.note = info.trim.keystone;
          }
          $('#meta').textContent = path.split('/').pop() + '  ' + info.w + ' x ' + info.h +
            (shownTrim() ? '  (trimmed)' : '');
          // The declared size describes the picture as it is NOW, so trimming
          // changes what a millimetre is worth. Resolving it against the live
          // width on every use means trimming to the sheet keeps the same
          // declaration meaning the same edge, and the panel shows it move.
          sizeState();
          $('#btnBeside').classList.toggle('hidden', !info.marks);
          S.cand = null;        // detected in the old picture's pixels
          if (!had) {
            S.results = null; S.consensus = null;
            $('#results').textContent = '';
            $('#resultsHint').classList.remove('hidden');
          } else if (S.results) {
            // the marks moved with the trim, so last run's numbers are about
            // a picture that is no longer on screen
            S.results = null; S.consensus = null;
            $('#results').textContent = '';
            $('#resultsHint').classList.remove('hidden');
          }
          if (!had) fitView();
          showEditor(); paint();
          if (after) after();
        };
        img.onerror = function () { toast('that file did not decode as a picture', true); };
        img.src = picURL('/api/image', path);
      })
      .finally(function () { busy(false); });
  }

  function openImage(path) {
    S.trim = null; S.trimEdit = false; S.trimDraft = null; S.found = null;
    $('#trimFound').textContent = '';
    showPicture(path, false, trimState);
  }

  // A different picture is a different document: the marks, the size and the
  // history belong to the one it replaces. applyMarks sets its own and calls
  // openImage itself, so the reset lives here and not there. There is no
  // autosave, so what is on the page is asked about before it goes.
  function openFresh(path) {
    var same = S.image && S.image.path === path;
    if (!same && S.layers.length &&
        !confirm('Discard the ' + S.layers.length + ' layers marked on ' +
                 (S.image ? S.image.path.split('/').pop() : 'this picture') + '?')) {
      var s = $('#samples');
      s.value = S.image ? S.image.path : '';
      if (s.selectedIndex < 0) s.selectedIndex = 0;
      return;
    }
    if (!same) {
      S.layers = []; S.nextId = 1; S.sel = null; S.pending = [];
      S.undo = []; S.redo = [];
      setPerspective('three'); setScale(null);
      S.ref = null; S.refNote = null; S.consensus = null; S.consensusBoth = null;
      camState();
    }
    openImage(path);
    refreshLayers(); commit();
  }

  function saveMarks() {
    var blob = new Blob([JSON.stringify({
      image: S.image, trim: S.trim, perspective: S.perspective, scale: S.scale,
      reference: S.ref, layers: S.layers, nextId: S.nextId
    }, null, 1)], { type: 'application/json' });
    var a = el('a', {
      href: URL.createObjectURL(blob),
      download: (S.image ? S.image.path.split('/').pop().replace(/\.[^.]+$/, '') : 'marks') + '.marks.json'
    });
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // The page already knows how to work out a surface's quad and which two
  // families give the horizon, so it sends those with the layers rather than
  // having the server derive them a second time in another language.
  function exportPdf() {
    if (!S.image) { toast('open a picture first', true); return; }
    var hz = horizonVps(null);
    var doc = {
      path: S.image.path, w: S.image.w, h: S.image.h,
      horizon: hz ? [hz[0].vp, hz[1].vp] : null,
      horizon_color: $('#colHorizon').value,
      layers: S.layers.map(function (L) {
        var o = { kind: L.kind, name: L.name, color: L.color, hidden: !!L.hidden,
                  lines: L.lines, curves: L.curves, pts: L.pts, far: L.far, pairs: L.pairs,
                  vp: L.vp };
        if (L.kind === 'surface') o.quad = surfaceQuad(L);
        // both rings, as on screen: the export gets annotated in
        // Illustrator, and the lean is the thing to look at
        if (L.kind === 'ellipse') {
          o.rings = (L.rings || []).concat((L.cellOut || []).filter(function (c) {
            return c.ring;
          }).map(function (c) { return c.ring; }));
          o.rings_true = L.ringsTrue;
        }
        if (L.kind === 'arch') {
          o.arches = L.arches;
          o.bez = (L.arches || []).map(function (_, i) { return bezOf(L, i); });
          o.bez_far = (L.arches || []).map(function (_, i) { return bezFar(L, i); });
          o.vault = (L.vault || []).map(function (V) {
            return V && [V.near_crown, V.far_crown];
          });
        }
        return o;
      })
    };
    busy(true);
    fetch('/api/export_pdf', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc: doc, trim: S.trim, with_image: true })
    }).then(function (r) {
      if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || 'export failed'); });
      return r.blob();
    }).then(function (blob) {
      var a = el('a', { href: URL.createObjectURL(blob),
        download: S.image.path.split('/').pop().replace(/\.[^.]+$/, '') + '.lines.pdf' });
      a.click();
      URL.revokeObjectURL(a.href);
    }).catch(function (e) { toast(String(e.message || e), true); })
      .finally(function () { busy(false); });
  }

  function applyMarks(d) {
    setPerspective(d.perspective || 'three');
    setScale(d.scale || null);
    S.ref = d.reference || null; S.refNote = null; camState();
    S.layers = d.layers || [];
    S.nextId = d.nextId || (S.layers.length + 1);
    S.sel = null; S.results = null; S.consensus = null; S.consensusBoth = null;
    S.layers.forEach(function (L) {
      L.lines = L.lines || []; L.curves = L.curves || []; L.pts = L.pts || []; L.far = L.far || [];
      L.pairs = L.pairs || []; L.refs = L.refs || {};
      // the roof slope is the arch layer with straight sides now, and it
      // kept the one reference it had
      if (L.kind === 'roof') { L.kind = 'arch'; L.shape = 'straight'; }
      L.markK = L.markK || [];   // absent in marks saved before this existed
      L.crown = L.crown || [];   // same: an arch saved before it could be drawn
      L.through = L.through || [];
      L.bezAdj = L.bezAdj || [];
      L.cells = L.cells || []; L.cellOut = L.cellOut || [];   // absent before circles in squares
      if (L.size === undefined) { L.size = 180; L.size_unit = 'cm'; L.on_ground = true; }
      if (L.kind === 'family') fitFamily(L, d.image ? [d.image.w, d.image.h] : null);
    });
    if (d.image && (!S.image || S.image.path !== d.image.path)) {
      // the marks are in the trimmed picture's coordinates, so the trim has
      // to be restored with them or every one lands on different paint
      openImage(d.image.path);
      S.trim = d.trim || null;
      if (S.trim) showPicture(d.image.path, true, trimState);
    } else {
      S.trim = d.trim || null;
      S.trimEdit = false; S.trimDraft = null;
      if (S.image) showPicture(S.image.path, true, trimState); else trimState();
    }
    // loading a file is opening a document, not an edit: start its history
    // fresh rather than letting cmd-Z walk back into the previous picture's
    S.undo = []; S.redo = [];
    refreshLayers(); showEditor(); paint(); commit();
    toast(S.layers.length + ' layers loaded');
  }

  function loadMarks(file) {
    var fr = new FileReader();
    fr.onload = function () {
      try { applyMarks(JSON.parse(fr.result)); }
      catch (e) { toast('that is not a marks file', true); }
    };
    fr.readAsText(file);
  }

  // ------------------------------------------------------------------- wiring

  function init() {
    if (window.LK && LK.mountThemeToggle) LK.mountThemeToggle('#themeSlot');

    // The menu says what each mark IS, nothing more; what it is for hangs on
    // the option, as the perspective menu already does. Built from KINDS so
    // the menu, the editor's title and a new layer's name cannot drift apart
    // -- the hand-written menu still offered "arch (near limb, optional far
    // limb)" after the arch started building itself.
    Object.keys(KINDS).forEach(function (k) {
      $('#newKind').append(el('option', { value: k, textContent: KINDS[k].label,
                                          title: KINDS[k].blurb }));
    });

    fetch('/api/testdata').then(function (r) { return r.json(); }).then(function (list) {
      var s = $('#samples');
      (list || []).forEach(function (f) {
        s.append(el('option', { value: f.path, textContent: f.name + '  (' + f.w + 'x' + f.h + ')' }));
      });
    });

    $('#samples').onchange = function (e) { if (e.target.value) openFresh(e.target.value); };
    $('#btnOpen').onclick = function () {
      if (!window.LK || !LK.browse) { toast('file browser unavailable', true); return; }
      LK.browse({
        mode: 'file', exts: ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'],
        title: 'Open a picture'
      }).then(function (p) { if (p) openFresh(Array.isArray(p) ? p[0] : p); });
    };

    trimState();
    $('#btnFindSheet').onclick = findSheet;
    $('#btnTrimEdit').onclick = enterTrim;
    $('#btnTrimCancel').onclick = function () {
      S.trimEdit = false; S.trimDraft = null;
      showPicture(S.image.path, true, function () { trimState(); paint(); commit(); });
    };
    $('#btnTrimApply').onclick = function () {
      if (!S.trimDraft) { toast('place the corners first', true); return; }
      var a = S.trimDraft;
      var area = Math.abs((a[0][0] * a[1][1] - a[1][0] * a[0][1]) +
                          (a[1][0] * a[2][1] - a[2][0] * a[1][1]) +
                          (a[2][0] * a[3][1] - a[3][0] * a[2][1]) +
                          (a[3][0] * a[0][1] - a[0][0] * a[3][1])) / 2;
      if (area < 400) { toast('those four corners enclose almost nothing', true); return; }
      setTrim({ corners: a.map(function (c) { return c.slice(); }),
                out: null, note: S.found && S.found.note || '' });
    };
    $('#btnTrimClear').onclick = function () { setTrim(null); };
    $('#perspective').onchange = function (e) {
      setPerspective(e.target.value);   // which also drops the stale answer
      commit();
      if (consensusReady()) runConsensus();
    };
    setPerspective(S.perspective);
    ['#sizeValue', '#sizeUnit', '#sizeAlong'].forEach(function (q) {
      $(q).onchange = function () {
        setScale({ value: $('#sizeValue').value, unit: $('#sizeUnit').value,
                   along: $('#sizeAlong').value });
        commit();
        if (consensusReady()) runConsensus();
      };
    });
    $('#btnSizeClear').onclick = function () { setScale(null); commit(); };
    setScale(S.scale);

    $('#btnAdd').onclick = function () {
      if (!S.image) { toast('open a picture first', true); return; }
      var L = newLayer($('#newKind').value);
      S.sel = L.id; S.archFar = false; S.supportEdge = false;
      refreshLayers(); showEditor(); paint(); commit();
    };
    $('#edName').oninput = function () {
      var L = selected(); if (L) { L.name = $('#edName').value; refreshLayers(); }
    };
    // live while the picker is dragged, one undo step when it is let go
    $('#edColor').after(LK.hexBox($('#edColor')));
    $('#edColor').oninput = function () {
      var L = selected(); if (L) { L.color = $('#edColor').value; refreshLayers(); paint(); }
    };
    $('#edColor').onchange = commit;
    $('#colHorizon').oninput = $('#colMoves').oninput = paint;
    $('#btnDelLayer').onclick = function () {
      var L = selected(); if (!L) return;
      S.layers = S.layers.filter(function (X) { return X !== L; });
      S.sel = null; S.pending = [];
      refreshLayers(); showEditor(); paint(); commit();
    };
    $('#btnClearMarks').onclick = function () {
      var L = selected(); if (!L) return;
      L.lines = []; L.curves = []; L.pts = []; L.far = []; L.pairs = []; L.vp = null; L.vpInfo = null;
      L.markK = [];          // or the next contacts inherit cleared marks' zooms
      S.pending = [];
      refreshLayers(); showEditor(); paint(); commit();
    };

    camState();
    $('#btnCamIn').onclick = function () {
      if (!S.image) { toast('open a picture first', true); return; }
      if (!window.LK || !LK.browse) { toast('file browser unavailable', true); return; }
      LK.browse({ mode: 'file', exts: ['.blend', '.fspy', '.json'],
                  title: 'Read a camera' })
        .then(function (p) { if (p) cameraIn(Array.isArray(p) ? p[0] : p, null); });
    };
    $('#camPick').onchange = function (e) { if (S.ref) cameraIn(S.ref.path, e.target.value); };
    $('#btnCamClear').onclick = cameraClear;
    $('#btnCamOut').onclick = cameraOut;
    $('#btnCheck').onclick = runCheck;
    $('#btnConsensus').onclick = runConsensus;
    $('#btnSave').onclick = saveMarks;
    $('#btnPdf').onclick = exportPdf;
    $('#btnLoad').onclick = function () { $('#fileLoad').click(); };
    $('#btnBeside').onclick = function () {
      if (!S.image) return;
      if (S.layers.length && !confirm('Replace the ' + S.layers.length +
          ' layers already marked?')) return;
      fetch('/api/marks?path=' + encodeURIComponent(S.image.path))
        .then(function (r) { return r.json(); })
        .then(function (d) { applyMarks(d); })
        .catch(function () { toast('could not read the marks beside this picture', true); });
    };
    $('#fileLoad').onchange = function (e) { if (e.target.files[0]) loadMarks(e.target.files[0]); };
    $('#btnFit').onclick = fitView;
    $('#btnDetect').onclick = runDetect;
    $('#showCand').onchange = paint;
    $('#showFits').onchange = paint;
    $('#showMoves').onchange = paint;
    $('#showPlan').onchange = paint;
    document.addEventListener('labkit:theme', paint);   // the top view's ink is the panel's

    var c = $('#view');
    c.addEventListener('mousedown', onDown);
    c.addEventListener('dblclick', function (e) { e.preventDefault(); finishRun(); });
    c.addEventListener('wheel', onWheel, { passive: false });
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('resize', paint);

    window.addEventListener('keydown', function (e) {
      // in a text field cmd-Z is the field's own undo, which the early
      // return above leaves alone
      if (/^(INPUT|SELECT|TEXTAREA)$/.test((e.target.tagName || ''))) return;
      if ((e.metaKey || e.ctrlKey) && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault(); if (e.shiftKey) redo(); else undo(); return;
      }
      if ((e.metaKey || e.ctrlKey) && (e.key === 'y' || e.key === 'Y')) {
        e.preventDefault(); redo(); return;         // the Windows spelling
      }
      if (e.key === 'Enter') { finishRun(); }
      else if (e.key === 'Escape') { S.pending = []; paint(); commit(); }
      else if (e.key === 'Backspace') {
        e.preventDefault();
        var L = selected(); if (!L) return;
        if (S.pending.length) S.pending.pop();
        else if (L.kind === 'family') {
          // the kind of mark the mode is marked in goes first
          ((L.curves.length && (S.perspective === 'five' || !L.lines.length))
            ? L.curves : L.lines).pop();
          fitFamily(L);
        }
        else if (L.pairs.length && (KINDS[L.kind].draw === 'pair' || KINDS[L.kind].draw === 'triple')) L.pairs.pop();
        else if (L.kind === 'arch' && S.archFar) L.far.pop();
        else L.pts.pop();
        refreshLayers(); paint(); commit();
      } else if (e.key === 'f') fitView();
    });

    refreshLayers();
    paint();
    commit();                       // the empty page is a state you can get back to
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
