// Skeleton runtime for fixed-skeleton characters (charkit.py): one template's bones, a character's part pictures,
// rigid pieces on their bones, soft pieces (torso, skirt) as meshes bent by several bones, two-bone IK for arms and
// legs, and actions (keyframes) layered on top of each other. Used by the character studio (char.html), the
// production editor and the final render, so what is seen while working is what is rendered.
//
//   const R = await Skel.load('<id>')                   template + character + images
//   const pose = Skel.pose(R, [{ action: A, t }])          sample actions (later layers win), solve IK
//   Skel.draw(ctx, R, pose, [a, b, c, d, e, f])            draw with a stage matrix (position, size, flip)
//
// Pose: { bones: { name: { r, x, y, s } }, ik: { chain: [dx, dy] } }  r in degrees added to the rest angle, x/y a
// shift in the parent bone's frame (character pixels), ik targets relative to where the chain's end rests.
// Action: { name, dur, loop, speed, bones: { bone: { r: [[t, v, ease]], x: [...], y: [...] } },
//           ik: { chain: [[t, dx, dy, ease]] }, swap: { slot: [[t, picture]] } }   ease: 'io' (default) | 'l' | 'step'
(function () {
  const rad = d => d * Math.PI / 180;
  const M = {
    mul: (m, n) => [m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1], m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3],
      m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5]],
    inv: m => { const k = m[0] * m[3] - m[1] * m[2];
      return [m[3] / k, -m[1] / k, -m[2] / k, m[0] / k, (m[2] * m[5] - m[3] * m[4]) / k, (m[1] * m[4] - m[0] * m[5]) / k]; },
    T: (x, y) => [1, 0, 0, 1, x, y],
    R: a => [Math.cos(a), Math.sin(a), -Math.sin(a), Math.cos(a), 0, 0],
    S: (x, y) => [x, 0, 0, y === undefined ? x : y, 0, 0],
    ap: (m, x, y) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]],
    ang: m => Math.atan2(m[1], m[0]),
  };

  function loadImage(src) {
    return new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => bad(new Error('no image ' + src)); im.src = src; });
  }

  // ---------- building a rig from the template and the character
  async function load(id, opts = {}) {
    const res = await fetch(`/api/char?id=${encodeURIComponent(id)}`, { cache: 'no-store' });
    if (!res.ok) throw new Error('no character ' + id);
    const d = await res.json();
    const images = {};
    await Promise.all(d.char.parts.map(async p => { images[p.file] = await loadImage(`/file/${d.dir}/${p.file}?v=${d.mtime || 0}`); }));
    const R = build(d.rig, d.char, images, opts);
    R.actions = d.actions || {};
    R.id = id;
    return R;
  }

  function build(rig, chr, images, opts = {}) {
    const B = {}, order = [];
    for (const b of [...rig.bones, ...(chr.bones || [])]) {      // the template's bones + the character's own (a braid)
      const a = Math.atan2(b.ey - b.y, b.ex - b.x);
      B[b.name] = { ...b, len: Math.hypot(b.ex - b.x, b.ey - b.y), W0: M.mul(M.T(b.x, b.y), M.R(a)) };
      order.push(b.name);
    }
    for (const n of order) {
      const b = B[n];
      b.L0 = b.parent ? M.mul(M.inv(B[b.parent].W0), b.W0) : b.W0;
      b.iW0 = M.inv(b.W0);
    }
    const R = { rig, chr, images, B, order, parts: [], deltas: chr.deltas || {}, ik: rig.ik || [] };
    for (const p of chr.parts) {
      const img = images[p.file];
      const part = { ...p, img };
      if (p.kind === 'mesh') part.mesh = makeMesh(R, part, opts.step || 22);
      if (p.kind === 'chain') part.mesh = makeChain(R, part, opts.chainStep || 9);
      R.parts.push(part);
    }
    return R;
  }

  // a grid over the picture, cells with paint kept; each corner bent by the part's bones, closest bones most
  function makeMesh(R, p, step) {
    const c = document.createElement('canvas'); c.width = p.w; c.height = p.h;
    const g = c.getContext('2d'); g.drawImage(p.img, 0, 0);
    const A = g.getImageData(0, 0, p.w, p.h).data;
    const cols = Math.ceil(p.w / step), rows = Math.ceil(p.h / step);
    const filled = (i, j) => {          // any paint in cell (i, j)?
      for (let y = j * step; y < Math.min(p.h, (j + 1) * step); y += 3)
        for (let x = i * step; x < Math.min(p.w, (i + 1) * step); x += 3) if (A[(y * p.w + x) * 4 + 3] > 8) return true;
      return false;
    };
    const vid = {}, verts = [], tris = [];
    const vert = (i, j) => {
      const k = i + ',' + j;
      if (vid[k] === undefined) { vid[k] = verts.length; verts.push({ u: Math.min(p.w, i * step), v: Math.min(p.h, j * step) }); }
      return vid[k];
    };
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      if (!filled(i, j)) continue;
      const a = vert(i, j), b = vert(i + 1, j), cc = vert(i + 1, j + 1), d = vert(i, j + 1);
      tris.push([a, b, cc], [a, cc, d]);
    }
    const names = p.bones && p.bones.length ? p.bones : [p.bone];
    const sigma = p.sigma || 46, pull = p.pull || {}, floor = p.floor || 0;
    for (const vtx of verts) {
      const x = p.x + vtx.u, y = p.y + vtx.v;
      if (p.blend) {           // hanging pieces (a braid): the top follows one bone, lower down another takes over
        const b = p.blend, u = Math.max(0, Math.min(1, (y - b.from) / (b.to - b.from))), k = u * u * (3 - 2 * u);
        vtx.x = x; vtx.y = y;
        vtx.w = [[b.top, 1 - k], [b.bottom, k]].filter(e => e[1] > 0.01);
        continue;
      }
      let ws = names.map(n => { const b = R.B[n]; return (pull[n] === undefined ? 1 : pull[n]) * Math.exp(-((segDist(x, y, b.x, b.y, b.ex, b.ey) / sigma) ** 2)); });
      const main = names.indexOf(p.bone);
      if (main >= 0) ws[main] = Math.max(ws[main], floor);   // the part's own bone always keeps this much
      let sum = ws.reduce((s, w) => s + w, 0);
      if (sum < 1e-6) {   // far from every bone: the nearest one carries it
        let best = 0, bd = 1e9;
        names.forEach((n, k) => { const b = R.B[n]; const dd = segDist(x, y, b.x, b.y, b.ex, b.ey); if (dd < bd) { bd = dd; best = k; } });
        ws = names.map((n, k) => k === best ? 1 : 0); sum = 1;
      }
      vtx.x = x; vtx.y = y;
      vtx.w = names.map((n, k) => [n, ws[k] / sum]).filter(e => e[1] > 0.01);
      if (p.hold) {            // the top stays on one bone (a skirt's waistband on the hips), blending into the rest
        const h = p.hold, u = Math.max(0, Math.min(1, (y - h.from) / (h.to - h.from))), k = u * u * (3 - 2 * u);
        const m = {}; for (const [n, w] of vtx.w) m[n] = w * k;
        m[h.bone] = (m[h.bone] || 0) + (1 - k);
        vtx.w = Object.entries(m).filter(e => e[1] > 0.01);
      }
    }
    return { verts, tris };
  }

  // a limb as one mesh: every corner belongs to the bone it lies along; near a joint it bends with both bones
  // (w: 0 = the bone above, 1 = the bone below), so the drawing curves round the joint instead of hinging
  function makeChain(R, p, step) {
    const base = gridMesh(p, step);
    const bones = [p.root, ...p.chain].map(n => R.B[n]);
    // distance along the chain of each joint (the head of each chain bone)
    const segs = p.chain.map(n => R.B[n]);
    for (const v of base.verts) {
      const x = p.x + v.u, y = p.y + v.v;
      let best = 0, bd = 1e9, bt = 0;
      segs.forEach((b, k) => {
        const dx = b.ex - b.x, dy = b.ey - b.y, L = dx * dx + dy * dy;
        let t = L ? ((x - b.x) * dx + (y - b.y) * dy) / L : 0;
        const tc = Math.max(0, Math.min(1, t));
        const d = Math.hypot(x - (b.x + tc * dx), y - (b.y + tc * dy));
        if (d < bd - 1e-6) { bd = d; best = k; bt = t; }
      });
      // signed distance from the joint at the top of this bone (k) and from the one at its bottom (k + 1)
      const b = segs[best], len = b.len;
      const fromTop = bt * len, fromBot = (1 - bt) * len;
      const sv = p.soft || [], softTop = sv[best] ?? 20, softBot = sv[best + 1] ?? 20;
      v.x = x; v.y = y;
      if (softTop > 0 && fromTop < softTop) {             // near the joint with the bone above (the root bone for the first one)
        const above = best === 0 ? p.root : p.chain[best - 1];
        const w = 0.5 + 0.5 * Math.max(-1, Math.min(1, fromTop / softTop));
        v.a = above; v.b = p.chain[best]; v.w = smooth01(w); v.j = [b.x, b.y];
      } else if (fromBot < softBot && best + 1 < p.chain.length) {
        const below = p.chain[best + 1], nb = segs[best + 1];
        const w = 0.5 - 0.5 * Math.max(-1, Math.min(1, fromBot / softBot));
        v.a = p.chain[best]; v.b = below; v.w = smooth01(w); v.j = [nb.x, nb.y];
      } else {
        v.a = p.chain[best]; v.b = null; v.w = 0; v.j = null;
      }
    }
    return base;
  }
  const smooth01 = w => w * w * (3 - 2 * w);

  // the grid over a picture, cells with paint kept (used by soft parts and limbs)
  function gridMesh(p, step) {
    const c = document.createElement('canvas'); c.width = p.w; c.height = p.h;
    const g = c.getContext('2d'); g.drawImage(p.img, 0, 0);
    const A = g.getImageData(0, 0, p.w, p.h).data;
    const cols = Math.ceil(p.w / step), rows = Math.ceil(p.h / step);
    const filled = (i, j) => {
      for (let y = j * step; y < Math.min(p.h, (j + 1) * step); y += 2)
        for (let x = i * step; x < Math.min(p.w, (i + 1) * step); x += 2) if (A[(y * p.w + x) * 4 + 3] > 8) return true;
      return false;
    };
    const vid = {}, verts = [], tris = [];
    const vert = (i, j) => {
      const k = i + ',' + j;
      if (vid[k] === undefined) { vid[k] = verts.length; verts.push({ u: Math.min(p.w, i * step), v: Math.min(p.h, j * step) }); }
      return vid[k];
    };
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      if (!filled(i, j)) continue;
      const a = vert(i, j), b = vert(i + 1, j), cc = vert(i + 1, j + 1), d = vert(i, j + 1);
      tris.push([a, b, cc], [a, cc, d]);
    }
    return { verts, tris };
  }

  function segDist(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
    let t = L ? ((px - ax) * dx + (py - ay) * dy) / L : 0; t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  // ---------- posing
  function world(R, pose) {
    const W = {};
    for (const n of R.order) {
      const b = R.B[n], q = (pose.bones && pose.bones[n]) || {};
      let L = b.L0;
      if (q.x || q.y) L = M.mul(M.T(q.x || 0, q.y || 0), L);
      if (q.r) L = M.mul(L, M.R(rad(q.r)));
      if (q.s && q.s !== 1) L = M.mul(L, M.S(q.s));
      if (q.sx !== undefined && q.sx !== 1) L = M.mul(L, M.S(q.sx, 1));   // shorter along the bone: it points at the camera
      W[n] = b.parent ? M.mul(W[b.parent], L) : L;
    }
    return W;
  }

  // two-bone IK: the chain's end reaches the target (character pixels); bend +1 / -1 picks the side of the middle joint
  function solveIK(R, pose, chain, target) {
    const [n1, n2] = chain.bones, b1 = R.B[n1], b2 = R.B[n2];
    pose.bones = pose.bones || {};
    const q1 = pose.bones[n1] = { ...(pose.bones[n1] || {}) }, q2 = pose.bones[n2] = { ...(pose.bones[n2] || {}) };
    q1.r = 0; q2.r = 0;
    let W = world(R, pose);
    const P = M.ap(W[n1], 0, 0), L1 = b1.len, L2 = b2.len;
    let dx = target[0] - P[0], dy = target[1] - P[1], d = Math.hypot(dx, dy);
    d = Math.max(Math.abs(L1 - L2) + 1e-3, Math.min(L1 + L2 - 1e-3, d));
    const th = Math.atan2(dy, dx), bend = chain.bend || 1;
    const a = Math.acos(Math.max(-1, Math.min(1, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))));
    const g = Math.acos(Math.max(-1, Math.min(1, (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2))));
    const w1 = th - bend * a, w2 = w1 + bend * (Math.PI - g);
    q1.r = (w1 - M.ang(W[n1])) * 180 / Math.PI;
    W = world(R, pose);
    q2.r = (w2 - M.ang(W[n2])) * 180 / Math.PI;
  }

  // where a chain's end sits in the rest pose (its end bone's head)
  function restEnd(R, chain) { const e = R.B[chain.end]; return [e.x, e.y]; }

  // ---------- actions
  const ease = (k, u) => k === 'l' ? u : k === 'step' ? 0 : u * u * (3 - 2 * u);
  function sampleKeys(keys, t) {
    if (!keys || !keys.length) return undefined;
    if (t <= keys[0][0]) return keys[0].slice(1);
    for (let i = 1; i < keys.length; i++) {
      const k1 = keys[i];
      if (t <= k1[0]) {
        const k0 = keys[i - 1], u = (t - k0[0]) / Math.max(1e-6, k1[0] - k0[0]);
        const e = ease(typeof k1[k1.length - 1] === 'string' ? k1[k1.length - 1] : 'io', u);
        const out = [];
        for (let j = 1; j < k1.length; j++) if (typeof k1[j] === 'number') out.push(k0[j] + (k1[j] - k0[j]) * e);
        return out;
      }
    }
    return keys[keys.length - 1].slice(1).filter(x => typeof x === 'number');
  }

  function localTime(a, t) {
    if (!a.dur) return t;
    return a.loop ? ((t % a.dur) + a.dur) % a.dur : Math.min(t, a.dur);
  }

  // layers: [{ action, t, weight }] -> pose (later layers override what earlier ones set; weight blends toward it)
  function pose(R, layers, extra) {
    const P = sampleLayers(layers, 0);
    if (R.rig.drag && layers.length) {          // follow-through: a bone takes back part of its parent's recent turn
      const dt = 0.12, Q = sampleLayers(layers, dt);
      for (const d of R.rig.drag) {
        const now = (P.bones[d.of] || {}).r || 0, before = (Q.bones[d.of] || {}).r || 0;
        const lag = Math.max(-25, Math.min(25, -(now - before) * d.k));
        if (Math.abs(lag) > 0.01) { const q = P.bones[d.bone] = P.bones[d.bone] || {}; q.r = (q.r || 0) + lag; }
      }
    }
    if (extra) merge(P, extra);
    shrug(R, P);
    for (const sp of R.rig.spread || []) {      // part of a bone's turn goes to the bone under it (head -> neck)
      const q = P.bones[sp.from];
      if (!q || !q.r || !R.B[sp.to]) continue;
      const give = q.r * sp.k;
      P.bones[sp.from] = { ...q, r: q.r - give };
      P.bones[sp.to] = { ...(P.bones[sp.to] || {}), r: ((P.bones[sp.to] || {}).r || 0) + give };
    }
    solve(R, P);
    hang(R, P, layers, extra);
    return P;
  }

  // a braid hangs from the head: its first bone is tied to the head, each one below follows the head less (w) and
  // the back more (it hangs and lies on her back), and a little later (lag, seconds), so it swings after the head
  function hang(R, P, layers, extra) {
    const H = R.chr.hang;
    if (!H || !R.B[H.follow]) return;
    const upto = n => { const out = []; for (let b = n; b; b = R.B[b].parent) out.push(b); return out; };
    const turn = (Q, n) => upto(n).reduce((s, b) => s + ((Q.bones[b] || {}).r || 0), 0);
    const at = lag => {
      if (!lag || !layers.length) return P;
      const Q = sampleLayers(layers, lag);
      if (extra) merge(Q, extra);
      return Q;
    };
    let prev = turn(P, H.follow);
    H.bones.forEach((n, i) => {
      const Q = at(H.lag ? H.lag[i] : 0), w = H.w ? H.w[i] : 1;
      const want = w * turn(Q, H.follow) + (1 - w) * turn(Q, H.rest_on);
      P.bones[n] = { ...(P.bones[n] || {}), r: want - prev };
      prev = want;
    });
  }

  function sampleLayers(layers, back) {
    const P = { bones: {}, ik: {}, swap: {} };
    for (const L of layers) {
      const a = L.action; if (!a) continue;
      const t = localTime(a, L.t - back), w = L.weight === undefined ? 1 : L.weight;
      for (const [bn, ch] of Object.entries(a.bones || {})) {
        const q = P.bones[bn] = P.bones[bn] || {};
        for (const [k, keys] of Object.entries(ch)) {
          const v = sampleKeys(keys, t); if (!v) continue;
          q[k] = w >= 1 || q[k] === undefined ? v[0] * (w >= 1 ? 1 : w) : q[k] + (v[0] - q[k]) * w;
        }
      }
      for (const [cn, keys] of Object.entries(a.ik || {})) {
        const v = sampleKeys(keys, t); if (!v) continue;
        const o = P.ik[cn];
        P.ik[cn] = !o || w >= 1 ? [v[0], v[1]] : [o[0] + (v[0] - o[0]) * w, o[1] + (v[1] - o[1]) * w];
      }
      for (const [slot, keys] of Object.entries(a.swap || {})) {
        if (w < 0.5) continue;                       // a fading action gives its pictures over at half way
        let pic = keys[0][1];
        for (const k of keys) if (t >= k[0]) pic = k[1];
        P.swap[slot] = pic;
      }
    }
    return P;
  }

  // an arm lifted high takes its shoulder with it a little (the soft torso follows the shoulder bones)
  function shrug(R, P) {
    for (const [s, sign] of [['n', 1], ['f', -1]]) {
      const up = P.bones[s + '_upper'], sh = s + '_shoulder';
      if (!up || up.r === undefined || !R.B[sh]) continue;
      const ex = Math.abs(up.r) - 40;
      if (ex <= 0) continue;
      const q = P.bones[sh] = { ...(P.bones[sh] || {}) };
      q.r = (q.r || 0) + sign * Math.min(18, ex * 0.16) * (R.rig.shrug ?? 1);
    }
  }

  function merge(P, x) {
    for (const [bn, q] of Object.entries(x.bones || {})) P.bones[bn] = { ...(P.bones[bn] || {}), ...q };
    Object.assign(P.ik, x.ik || {});
    Object.assign(P.swap, x.swap || {});
  }

  // IK targets are given relative to the chain end's rest place, in the frame of the chain's root parent... kept
  // simple: in character pixels, moved along with the hips/root so a step stays a step when the body bobs
  function solve(R, P) {
    for (const ch of R.ik) {
      const off = P.ik[ch.name];
      if (!off) continue;
      const rest = restEnd(R, ch);
      const root = R.B[ch.bones[0]].parent;
      let base = [0, 0];
      if (root && ch.name.endsWith('_leg')) {   // legs: targets stay on the ground (root frame), not on the hips
        const Wr = world(R, { bones: { root: P.bones.root || {} } }).root, W0 = R.B.root.W0;
        const k = M.mul(Wr, M.inv(W0));
        const p = M.ap(k, rest[0] + off[0], rest[1] + off[1]);
        solveIK(R, P, ch, p);
        // the foot keeps its rest angle on the ground (plus its own r: toes up / heel up)
        const fq = P.bones[ch.end] = { ...(P.bones[ch.end] || {}) }, want = M.ang(M.mul(k, R.B[ch.end].W0)) + rad(fq.r || 0);
        const W = world(R, { ...P, bones: { ...P.bones, [ch.end]: { ...fq, r: 0 } } });
        fq.r = (want - M.ang(W[ch.end])) * 180 / Math.PI;
        continue;
      }
      if (root) {   // arms: targets move with the chest
        const W = world(R, P);
        const k = M.mul(W[root], R.B[root].iW0);
        base = M.ap(k, rest[0] + off[0], rest[1] + off[1]);
        solveIK(R, P, ch, base);
      }
    }
    // bones that follow others part of the way (each half of a skirt follows its thigh)
    for (const f of R.rig.follow || []) {
      if (!R.B[f.bone]) continue;
      const q = P.bones[f.bone] = { ...(P.bones[f.bone] || {}) };
      const a = (P.bones[f.of] || {}).r || 0;
      let v = f.k * a;
      if (f.with) {          // the turn both legs share (leaning the same way: hips pushed aside) counts less
        const b = (P.bones[f.with] || {}).r || 0;
        const shared = a * b > 0 ? Math.sign(a) * Math.min(Math.abs(a), Math.abs(b)) : 0;
        v -= (f.shared || 0) * shared;
      }
      if (f.max) v = Math.max(-f.max, Math.min(f.max, v));
      q.r = (q.r || 0) + v;
    }
    return P;
  }

  // ---------- drawing
  function draw(ctx, R, P, stage, opts = {}) {
    const W = world(R, P);
    const K = {};
    for (const n of R.order) K[n] = M.mul(W[n], R.B[n].iW0);   // rest pixels -> posed pixels
    if (!opts.noGL && glDraw(ctx, R, P, stage, K, opts)) {
      if (opts.bones) drawBones(ctx, R, W, stage);
      return W;
    }
    ctx.save();
    ctx.setTransform(...stage);
    for (const p of R.parts) {
      if (p.hide || (opts.only && !opts.only(p))) continue;
      if (p.slot && ((P.swap && P.swap[p.slot]) || p.slot) !== p.name) continue;   // one picture per slot
      const dl = R.deltas[p.name];
      if (p.kind === 'chain') drawChain(ctx, p, K, stage);
      else if (p.mesh) drawMesh(ctx, p, K, dl, stage);
      else {
        let m = K[p.bone];
        if (dl) m = M.mul(m, deltaMat(R, p, dl));
        ctx.setTransform(...M.mul(stage, m));
        ctx.drawImage(p.img, p.x, p.y);
      }
    }
    ctx.restore();
    if (opts.bones) drawBones(ctx, R, W, stage);
    return W;
  }

  function deltaMat(R, p, d) {
    const b = R.B[p.bone], cx = b.x, cy = b.y;
    return M.mul(M.mul(M.T(cx + (d.x || 0), cy + (d.y || 0)), M.mul(M.R(rad(d.r || 0)), M.S(d.s || 1))), M.T(-cx, -cy));
  }

  function meshPos(p, K, dl) {
    return p.mesh.verts.map(v => {
      let x = 0, y = 0;
      for (const [n, w] of v.w) { const q = M.ap(K[n], v.x + (dl ? dl.x || 0 : 0), v.y + (dl ? dl.y || 0 : 0)); x += q[0] * w; y += q[1] * w; }
      return [x, y];
    });
  }

  function chainPos(p, K) {
    return p.mesh.verts.map(v => {
      const A = K[v.a];
      const q = M.ap(A, v.x, v.y);
      if (!v.b || v.w <= 0) return q;
      const Bm = K[v.b];
      const jp = M.ap(A, v.j[0], v.j[1]);
      let th = Math.atan2(Bm[1], Bm[0]) - Math.atan2(A[1], A[0]);
      th = Math.atan2(Math.sin(th), Math.cos(th));
      const a = th * v.w, ca = Math.cos(a), sa = Math.sin(a);
      const dx = q[0] - jp[0], dy = q[1] - jp[1];
      return [jp[0] + dx * ca - dy * sa, jp[1] + dx * sa + dy * ca];
    });
  }

  // ---------- the GPU path: one textured mesh per part, drawn in order into a WebGL canvas that is then laid on the
  // page's canvas. Each pixel is painted once (no doubled edges or seams between triangles) and pictures shrink with
  // mipmaps (no shimmer when a character is small on screen).
  let GL = null;
  function glInit(w, h) {
    if (GL === false) return null;
    if (!GL) {
      const c = document.createElement('canvas');
      const attrs = { premultipliedAlpha: true, preserveDrawingBuffer: true, antialias: true, alpha: true };
      const gl = c.getContext('webgl2', attrs) || c.getContext('webgl', attrs);
      if (!gl) { GL = false; return null; }
      const vs = 'attribute vec2 p; attribute vec2 t; uniform vec2 size; varying vec2 vt;' +
        'void main(){ vt = t; gl_Position = vec4(p.x / size.x * 2.0 - 1.0, 1.0 - p.y / size.y * 2.0, 0.0, 1.0); }';
      const fs = 'precision mediump float; varying vec2 vt; uniform sampler2D tex; void main(){ gl_FragColor = texture2D(tex, vt); }';
      const sh = (type, src) => { const x = gl.createShader(type); gl.shaderSource(x, src); gl.compileShader(x); return x; };
      const prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { GL = false; return null; }
      gl.useProgram(prog);
      GL = { c, gl, webgl2: typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext,
             pBuf: gl.createBuffer(), tBuf: gl.createBuffer(), iBuf: gl.createBuffer(),
             aP: gl.getAttribLocation(prog, 'p'), aT: gl.getAttribLocation(prog, 't'), uSize: gl.getUniformLocation(prog, 'size'),
             tex: new Map(), u32: gl.getExtension && gl.getExtension('OES_element_index_uint') };
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    }
    if (GL.c.width !== w || GL.c.height !== h) { GL.c.width = w; GL.c.height = h; }
    return GL;
  }

  function glTex(G, img) {
    let t = G.tex.get(img);
    if (t) return t;
    const gl = G.gl;
    t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    if (G.webgl2) { gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); }
    else gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    G.tex.set(img, t);
    return t;
  }

  function glDraw(ctx, R, P, stage, K, opts) {
    const cw = ctx.canvas.width, ch = ctx.canvas.height;
    const G = glInit(cw, ch);
    if (!G) return false;
    const gl = G.gl;
    gl.viewport(0, 0, cw, ch);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(G.uSize, cw, ch);
    for (const p of R.parts) {
      if (p.hide || (opts.only && !opts.only(p))) continue;
      if (p.slot && ((P.swap && P.swap[p.slot]) || p.slot) !== p.name) continue;   // one picture per slot
      const dl = R.deltas[p.name];
      let pts, uv, idx;
      if (p.mesh) {
        pts = p.kind === 'chain' ? chainPos(p, K) : meshPos(p, K, dl);
        if (!p.mesh.uv) {
          p.mesh.uv = new Float32Array(p.mesh.verts.flatMap(v => [v.u / p.w, v.v / p.h]));
          const flat = p.mesh.tris.flat();
          p.mesh.idx = p.mesh.verts.length > 65535 ? new Uint32Array(flat) : new Uint16Array(flat);
        }
        uv = p.mesh.uv; idx = p.mesh.idx;
      } else {
        let m = K[p.bone];
        if (dl) m = M.mul(m, deltaMat(R, p, dl));
        pts = [[p.x, p.y], [p.x + p.w, p.y], [p.x + p.w, p.y + p.h], [p.x, p.y + p.h]].map(q => M.ap(m, q[0], q[1]));
        uv = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]); idx = new Uint16Array([0, 1, 2, 0, 2, 3]);
      }
      const pos = new Float32Array(pts.length * 2);
      for (let i = 0; i < pts.length; i++) { const q = M.ap(stage, pts[i][0], pts[i][1]); pos[2 * i] = q[0]; pos[2 * i + 1] = q[1]; }
      gl.bindTexture(gl.TEXTURE_2D, glTex(G, p.img));
      gl.bindBuffer(gl.ARRAY_BUFFER, G.pBuf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STREAM_DRAW);
      gl.enableVertexAttribArray(G.aP); gl.vertexAttribPointer(G.aP, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, G.tBuf); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STREAM_DRAW);
      gl.enableVertexAttribArray(G.aT); gl.vertexAttribPointer(G.aT, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, G.iBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STREAM_DRAW);
      gl.drawElements(gl.TRIANGLES, idx.length, idx instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT, 0);
    }
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(G.c, 0, 0);
    ctx.restore();
    return true;
  }

  function drawMesh(ctx, p, K, dl, stage) {
    const { verts, tris } = p.mesh;
    const pos = meshPos(p, K, dl);
    ctx.setTransform(...stage);
    for (const t of tris) {
      const [a, b, c] = t;
      tri(ctx, p.img, verts[a], verts[b], verts[c], pos[a], pos[b], pos[c], stage);
    }
  }

  // a limb: each corner moves with its bone; near a joint it moves with the bone above and is then turned round the
  // (moved) joint by part of the angle between the two bones, so the limb keeps its thickness through the bend
  function drawChain(ctx, p, K, stage) {
    const { verts, tris } = p.mesh;
    const pos = chainPos(p, K);
    ctx.setTransform(...stage);
    for (const t of tris) {
      const [a, b, c] = t;
      tri(ctx, p.img, verts[a], verts[b], verts[c], pos[a], pos[b], pos[c], stage);
    }
  }

  // one textured triangle: clip to the (slightly grown) triangle, map the picture with the affine that takes the
  // source corners to the bent corners
  function tri(ctx, img, s0, s1, s2, d0, d1, d2, stage) {
    const cx = (d0[0] + d1[0] + d2[0]) / 3, cy = (d0[1] + d1[1] + d2[1]) / 3;
    const grow = p => { const dx = p[0] - cx, dy = p[1] - cy, l = Math.hypot(dx, dy) || 1; return [p[0] + dx / l * 0.9, p[1] + dy / l * 0.9]; };
    const g0 = grow(d0), g1 = grow(d1), g2 = grow(d2);
    const x0 = s0.u, y0 = s0.v, x1 = s1.u, y1 = s1.v, x2 = s2.u, y2 = s2.v;
    const den = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (Math.abs(den) < 1e-9) return;
    const a = ((d1[0] - d0[0]) * (y2 - y0) - (d2[0] - d0[0]) * (y1 - y0)) / den;
    const b = ((d1[1] - d0[1]) * (y2 - y0) - (d2[1] - d0[1]) * (y1 - y0)) / den;
    const c = ((d2[0] - d0[0]) * (x1 - x0) - (d1[0] - d0[0]) * (x2 - x0)) / den;
    const d = ((d2[1] - d0[1]) * (x1 - x0) - (d1[1] - d0[1]) * (x2 - x0)) / den;
    const e = d0[0] - a * x0 - c * y0, f = d0[1] - b * x0 - d * y0;
    ctx.save();
    ctx.beginPath(); ctx.moveTo(g0[0], g0[1]); ctx.lineTo(g1[0], g1[1]); ctx.lineTo(g2[0], g2[1]); ctx.closePath(); ctx.clip();
    ctx.transform(a, b, c, d, e, f);
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  }

  function drawBones(ctx, R, W, stage) {
    ctx.save();
    ctx.setTransform(...stage);
    for (const n of R.order) {
      const b = R.B[n], h = M.ap(W[n], 0, 0), e = M.ap(W[n], b.len, 0);
      ctx.strokeStyle = 'rgba(255,60,170,.9)'; ctx.lineWidth = 3 / Math.hypot(stage[0], stage[1]);
      ctx.beginPath(); ctx.moveTo(h[0], h[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
      ctx.fillStyle = 'rgba(255,230,0,.95)';
      ctx.beginPath(); ctx.arc(h[0], h[1], 5 / Math.hypot(stage[0], stage[1]), 0, 7); ctx.fill();
    }
    ctx.restore();
  }

  // where a bone's head/tail is now, in character pixels
  function jointAt(R, W, name, tail) { const b = R.B[name]; return M.ap(W[name], tail ? b.len : 0, 0); }

  window.Skel = { load, build, pose, draw, world, solveIK, jointAt, restEnd, M, sampleKeys, localTime };
})();
