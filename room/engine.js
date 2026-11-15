// The episode engine of the production editor: draws any moment of an edit (episodes/<ep>/edit.json) and mixes
// its sound. The editor's preview and the final render (render_edit.js) both use this file and nothing else,
// so what the user sees and hears while editing is what the video will be.
// Pictures and sound are pure functions of the edit and t: any frame, in any order.
(function () {
  'use strict';
  const W = 1920, H = 1080, XF = 0.45, FADE = 0.6, ENV_HZ = 25, CURVE_HZ = 50;
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = x => { x = clamp(x); return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2; };
  const lerp = (a, b, k) => a + (b - a) * k;
  const hash = s => { let h = 7; for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) % 1000; return h / 1000; };
  const url = p => '/file/' + p.split('/').map(encodeURIComponent).join('/');

  const IMG = {}, BUF = {}, PEAK = {}, ENV = {}, WAVE = {};
  let E = null, PUP = {}, kindOf = {}, trackOf = {}, actx = null;

  function audioCtx() { if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 }); return actx; }

  function loadImage(p) {
    if (!p || IMG[p]) return Promise.resolve();
    const im = new Image(); IMG[p] = im; im.src = url(p);
    return im.decode().catch(() => {});
  }
  async function loadSound(p) {
    if (!p || BUF[p] !== undefined) return;
    BUF[p] = null;
    try {
      const data = await (await fetch(url(p))).arrayBuffer();
      const b = await audioCtx().decodeAudioData(data);
      BUF[p] = b;
      const ch = b.getChannelData(0); let pk = 1e-6;
      for (let i = 0; i < ch.length; i += 3) { const v = Math.abs(ch[i]); if (v > pk) pk = v; }
      PEAK[p] = pk;
      // waveform for the timeline: the loudest sample per 1/100 s
      const hop = Math.floor(b.sampleRate / 100), wv = new Float32Array(Math.ceil(ch.length / hop));
      for (let i = 0; i < wv.length; i++) { let m = 0; for (let j = i * hop, e = Math.min(ch.length, j + hop); j < e; j += 4) { const v = Math.abs(ch[j]); if (v > m) m = v; } wv[i] = m; }
      WAVE[p] = wv;
      // mouth open/shut at ENV_HZ from loudness (same rule as film.py's envelope)
      const eh = Math.floor(b.sampleRate / ENV_HZ), n = Math.floor(ch.length / eh), rms = new Float32Array(n);
      let mx = 0;
      for (let i = 0; i < n; i++) { let s = 0; for (let j = i * eh; j < (i + 1) * eh; j++) s += ch[j] * ch[j]; rms[i] = Math.sqrt(s / eh); if (rms[i] > mx) mx = rms[i]; }
      const thr = Math.max(mx * 0.18, 0.01), o = new Uint8Array(n);
      for (let i = 0; i < n; i++) { let loc = 0, c = 0; for (let j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) { loc += rms[j]; c++; } o[i] = rms[i] > thr && rms[i] >= loc / c * 0.95 ? 1 : 0; }
      for (let i = 1; i < n - 1; i++) if (o[i - 1] === o[i + 1] && o[i] !== o[i - 1]) o[i] = o[i - 1];
      ENV[p] = o;
    } catch (e) { console.warn('sound', p, e); }
  }
  async function loadPuppet(id, info) {
    if (PUP[id]) return;
    const L = info.layout, P = { L, dir: info.dir };
    PUP[id] = P;
    await Promise.all(L.parts.map(q => loadImage(info.dir + '/' + q.file)));
    // the figure's box in the rest pose (default drawings only): feet at the bottom
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const q of L.parts) {
      if (q.hide || (q.swap && !/:(open|rest)$/.test(q.swap))) continue;
      const im = IMG[P.dir + '/' + q.file]; if (!im || !im.width) continue;
      const m = partMat(P, q, null);
      for (const [u, v] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const c = m.transformPoint(new DOMPoint(u * im.width / 2, v * im.height / 2));
        x0 = Math.min(x0, c.x); x1 = Math.max(x1, c.x); y0 = Math.min(y0, c.y); y1 = Math.max(y1, c.y);
      }
    }
    P.box = [x0, y0, x1, y1];
  }

  // ---------- the edit
  function index() {
    kindOf = {}; trackOf = {};
    for (const t of E.tracks) { kindOf[t.id] = t.kind; trackOf[t.id] = t; }
  }
  const live = c => trackOf[c.track] && !trackOf[c.track].mute && !c.mute;   // a muted row or a muted clip is silent
  const ofKind = k => E.clips.filter(c => kindOf[c.track] === k && live(c));
  const endOf = c => c.start + c.dur;
  function duration() { return E.clips.reduce((m, c) => Math.max(m, endOf(c)), 0) + 1; }

  async function load(state) {
    E = state.edit; index();
    const jobs = [];
    for (const p of Object.values(E.images || {})) jobs.push(loadImage(p));
    for (const ch of Object.values(E.chars || {})) {
      jobs.push(loadImage(ch.img));
      if (ch.eye_closed) jobs.push(loadImage(ch.eye_closed.img));
      if (ch.mouth_aa) jobs.push(loadImage(ch.mouth_aa.img));
    }
    for (const [id, info] of Object.entries(state.puppets || {})) jobs.push(loadPuppet(id, info));
    for (const c of E.clips) if (c.file && kindOf[c.track] !== 'shot' && kindOf[c.track] !== 'actor') jobs.push(loadSound(c.file));
    await Promise.all(jobs);
  }
  function setEdit(e) { E = e; index(); }

  // ---------- pictures
  function cover(g, im) {
    if (!im || !im.width) { g.fillStyle = '#3a3128'; g.fillRect(0, 0, W, H); return; }
    const s = Math.max(W / im.width, H / im.height);
    g.drawImage(im, (W - im.width * s) / 2, (H - im.height * s) / 2, im.width * s, im.height * s);
  }
  function speaking(who, t) {
    for (const c of ofKind('voice')) {
      if (trackOf[c.track].who !== who || t < c.start || t >= endOf(c) || !ENV[c.file]) continue;
      return ENV[c.file][Math.floor((t - c.start + (c.in || 0)) * ENV_HZ)] === 1;
    }
    return false;
  }
  function boneMat(P, name, pose) {
    const b = P.L.bones.find(x => x.name === name); if (!b) return new DOMMatrix();
    const m = b.parent ? boneMat(P, b.parent, pose) : new DOMMatrix();
    const a = pose && pose[name] || 0;
    return a ? m.multiply(new DOMMatrix().translate(b.x, b.y).rotate(a).translate(-b.x, -b.y)) : m;
  }
  function partMat(P, q, pose) {
    const m = new DOMMatrix().translate(q.x, q.y).rotate(q.rot || 0).scale((q.s || 1) * (q.sx || 1), (q.s || 1) * (q.sy || 1));
    return q.bone && pose ? boneMat(P, q.bone, pose).multiply(m) : m;
  }
  function puppet(g, c, t, k) {
    const P = PUP[c.puppet]; if (!P || !P.box) return;
    const [x0, y0, x1, y1] = P.box, s = (c.h || 0.65) * H / Math.max(1, y1 - y0);
    const x = lerp(c.x ?? 0.5, c.x1 ?? c.x ?? 0.5, c.x1 != null ? ease(k) : 0) * W, y = (c.y || 0.97) * H;
    const ph = hash(c.who) * 10, per = 3 + hash(c.who + 'b') * 2;
    const state = { eye: ((t + ph) % per) < 0.14 ? 'closed' : 'open', mouth: speaking(c.who, t) ? 'aa' : 'rest' };
    g.save(); g.translate(x, y); if (c.flip) g.scale(-1, 1); g.scale(s, s); g.translate(-(x0 + x1) / 2, -y1);
    for (const q of P.L.parts) {
      if (q.hide) continue;
      if (q.swap) {
        const [grp, st] = q.swap.split(':'), want = state[grp] || (grp === 'near_hand' ? 'open' : null);
        const has = P.L.parts.some(o => o.swap === grp + ':' + want);
        if (st !== (has ? want : (grp === 'mouth' ? 'rest' : 'open'))) continue;
      }
      const im = IMG[P.dir + '/' + q.file]; if (!im || !im.width) continue;
      const m = partMat(P, q, c.pose || {});
      g.save(); g.transform(m.a, m.b, m.c, m.d, m.e, m.f); g.drawImage(im, -im.width / 2, -im.height / 2); g.restore();
    }
    g.restore();
  }
  function still(g, c, t, k) {
    const ch = (E.chars || {})[c.who], im = ch && IMG[ch.img];
    if (!im || !im.width) return;
    const fh = (c.h || 0.65) * H, s = fh / ch.h;
    const x = lerp(c.x ?? 0.5, c.x1 ?? c.x ?? 0.5, c.x1 != null ? ease(k) : 0) * W, y = (c.y || 0.97) * H;
    const ph = hash(c.who) * 10, walking = c.x1 != null && c.x1 !== c.x;
    g.save(); g.translate(x, y); if (c.flip) g.scale(-1, 1);
    g.rotate(walking ? Math.sin(t * 5 + ph) * 0.012 : Math.sin(t * 0.7 + ph) * 0.004);
    g.scale(s, s * (1 + Math.sin(t * 1.6 + ph) * 0.006));
    if (walking) g.translate(0, -Math.abs(Math.sin(t * 5 + ph)) * 8 / s);
    g.translate(-ch.w / 2, -ch.h);
    if (c.shadow) { g.filter = 'brightness(0) blur(14px)'; g.globalAlpha *= 0.55; g.drawImage(im, 0, 0); g.restore(); return; }
    g.drawImage(im, 0, 0);
    const per = 3 + hash(c.who + 'b') * 2;
    if (ch.eye_closed && ((t + ph) % per) < 0.14) g.drawImage(IMG[ch.eye_closed.img], ch.eye_closed.at[0], ch.eye_closed.at[1]);
    if (ch.mouth_aa && speaking(c.who, t)) g.drawImage(IMG[ch.mouth_aa.img], ch.mouth_aa.at[0], ch.mouth_aa.at[1]);
    g.restore();
  }
  function shot(g, sh, t, prevPass) {
    const k = clamp((t - sh.start) / Math.max(sh.dur, 0.01));
    if (sh.title) {
      g.fillStyle = '#0b0806'; g.fillRect(0, 0, W, H);
      const a = g.globalAlpha; g.globalAlpha = a * clamp(Math.min((t - sh.start) / 0.8, (endOf(sh) - t) / 0.8));
      g.fillStyle = '#e8c27a'; g.font = '600 96px "Nirmala UI", "Mangal", sans-serif';
      g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(sh.title, W / 2, H / 2);
      g.globalAlpha = a; return;
    }
    const cam = sh.cam || [1.0, 0.5, 0.5, 1.06, 0.5, 0.5];
    const e = ease(k), z = lerp(cam[0], cam[3], e), cx = lerp(cam[1], cam[4], e) * W, cy = lerp(cam[2], cam[5], e) * H;
    g.save();
    let sx = 0, sy = 0;
    if (sh.shake && t - sh.start >= sh.shake[0] && t - sh.start <= sh.shake[1]) {
      const a = 14 * (1 - (t - sh.start - sh.shake[0]) / (sh.shake[1] - sh.shake[0]));
      sx = Math.sin(t * 91) * a; sy = Math.cos(t * 77) * a;
    }
    g.translate(W / 2 + sx, H / 2 + sy); g.scale(z, z);
    const hx = W / 2 / z, hy = H / 2 / z;
    g.translate(-clamp(cx, hx, W - hx), -clamp(cy, hy, H - hy));
    const img = n => IMG[(E.images || {})[n]];
    if (sh.bgs) {
      const n = sh.bgs.length, seg = sh.dur / n, i = Math.min(n - 1, Math.floor((t - sh.start) / seg));
      cover(g, img(sh.bgs[i]));
      const into = (t - sh.start) - i * seg;
      if (i > 0 && into < 0.8) { const a = g.globalAlpha; g.globalAlpha = a * (1 - into / 0.8); cover(g, img(sh.bgs[i - 1])); g.globalAlpha = a; }
    } else cover(g, img(sh.bg));
    // characters on screen during this shot (rows lower in the timeline are drawn in front)
    const rows = E.tracks.filter(tr => tr.kind === 'actor').map(tr => tr.id);
    for (const c of ofKind('actor').sort((a, b) => rows.indexOf(a.track) - rows.indexOf(b.track))) {
      if (c.start >= endOf(sh) - 1e-3 || endOf(c) <= sh.start + 1e-3) continue;
      const on = t >= c.start && t < endOf(c), held = prevPass && Math.abs(endOf(c) - endOf(sh)) < 0.05;
      if (!on && !held) continue;
      const ck = clamp((t - c.start) / Math.max(c.dur, 0.01));
      if (c.puppet) puppet(g, c, t, ck); else still(g, c, t, ck);
    }
    g.restore();
  }
  function frame(g, t) {
    g.setTransform(1, 0, 0, 1, 0, 0); g.globalAlpha = 1; g.filter = 'none';
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    if (!E) return;
    const S = ofKind('shot').sort((a, b) => a.start - b.start);
    const i = S.findIndex(s => t >= s.start && t < endOf(s));
    if (i < 0) return;
    const sh = S[i], prev = S[i - 1];
    const same = prev && prev.scene != null && prev.scene === sh.scene && Math.abs(endOf(prev) - sh.start) < 0.01;
    if (same && t - sh.start < XF) { shot(g, prev, t, true); g.globalAlpha = (t - sh.start) / XF; shot(g, sh, t); g.globalAlpha = 1; }
    else shot(g, sh, t);
    const next = S[i + 1], last = !next || next.scene == null || next.scene !== sh.scene || Math.abs(endOf(sh) - next.start) > 0.01;
    let dark = 0;
    if (!same) dark = Math.max(dark, 1 - (t - sh.start) / FADE);
    if (last) dark = Math.max(dark, 1 - (endOf(sh) - t) / FADE);
    if (dark > 0) { g.globalAlpha = clamp(dark); g.fillStyle = '#000'; g.fillRect(0, 0, W, H); g.globalAlpha = 1; }
  }

  // ---------- sound: every clip on its own, music ducked while someone speaks
  function talkCurve() {
    const n = Math.ceil(duration() * CURVE_HZ) + 2, on = new Float32Array(n);
    for (const c of ofKind('voice')) if (BUF[c.file]) for (let i = Math.floor(c.start * CURVE_HZ); i < Math.min(n, endOf(c) * CURVE_HZ); i++) on[i] = 1;
    const k = Math.round(0.25 * CURVE_HZ), out = new Float32Array(n);
    for (let i = 0; i < n; i++) { let s = 0, m = 0; for (let j = i - (k >> 1); j <= i + (k >> 1); j++) if (j >= 0 && j < n) { s += on[j]; m++; } out[i] = s / k; }
    return out;
  }
  function gainAt(c, lt, talk) {
    const kind = kindOf[c.track];
    let g = (c.gain ?? 1) * (kind === 'voice' ? 1 : 0.9 / (PEAK[c.file] || 1));
    const fi = c.fade_in || 0, fo = c.fade_out || 0;
    if (fi > 0 && lt < fi) g *= lt / fi;
    if (fo > 0 && lt > c.dur - fo) g *= Math.max(0, (c.dur - lt) / fo);
    if (kind === 'music') g *= 1 - (E.duck ?? 0.65) * (talk[Math.round((c.start + lt) * CURVE_HZ)] || 0);
    return g;
  }
  // schedule the mix on ctx (a live AudioContext or an OfflineAudioContext) from timeline second `from`,
  // starting at ctx time `when`; `only` limits it to one clip (to listen to it alone)
  function mix(ctx, from, when, only) {
    const master = ctx.createGain(); master.gain.value = 0.9; master.connect(ctx.destination);
    const talk = talkCurve(), nodes = [];
    for (const c of E.clips) {
      const kind = kindOf[c.track];
      if (!['voice', 'music', 'sfx'].includes(kind) || !BUF[c.file]) continue;
      if (only ? c !== only : !live(c)) continue;
      if (endOf(c) <= from) continue;
      const skip = Math.max(0, from - c.start), len = Math.min(c.dur - skip, BUF[c.file].duration - (c.in || 0) - skip);
      if (len <= 0.01) continue;
      const t0 = when + Math.max(0, c.start - from);
      const src = ctx.createBufferSource(); src.buffer = BUF[c.file];
      const gn = ctx.createGain(); src.connect(gn); gn.connect(master);
      const n = Math.max(2, Math.ceil(len * CURVE_HZ) + 1), curve = new Float32Array(n);
      for (let i = 0; i < n; i++) curve[i] = gainAt(c, skip + Math.min(len, i / CURVE_HZ), talk);
      gn.gain.value = curve[0];
      gn.gain.setValueCurveAtTime(curve, t0, Math.max(0.02, (n - 1) / CURVE_HZ));
      src.start(t0, (c.in || 0) + skip, len);
      nodes.push(src);
    }
    return { stop() { for (const s of nodes) try { s.stop(); } catch (e) { } master.disconnect(); } };
  }
  async function renderAudio() {
    const rate = 48000, off = new OfflineAudioContext(2, Math.ceil(duration() * rate), rate);
    mix(off, 0, 0);
    return off.startRendering();
  }

  window.Engine = { W, H, load, setEdit, frame, mix, renderAudio, duration, audioCtx, loadSound, loadImage, loadPuppet,
    IMG, BUF, WAVE, ENV, url, get edit() { return E; } };
})();
