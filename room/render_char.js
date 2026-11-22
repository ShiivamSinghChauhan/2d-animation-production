// A test video of a fixed-skeleton character, drawn by the character studio's own code (room/skel.js).
//   node room/render_char.js <id> [demo|bend|...] [--plain] [--bones]  ->  out/chars/<id>/<seq>.mp4
// The page's files come straight from disk (the room server does not have to be running).
const fs = require('fs'), path = require('path'), { spawn } = require('child_process');
const puppeteer = require(path.join(__dirname, '..', 'bridge', 'node_modules', 'puppeteer-core'));
const ROOT = path.join(__dirname, '..'), [id, seq = 'demo'] = process.argv.slice(2).filter(a => !a.startsWith('--'));
const flags = process.argv.filter(a => a.startsWith('--'));
const CHROME = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(f => f && fs.existsSync(f));
const OUT = path.join(ROOT, 'out', 'chars', id);
fs.mkdirSync(OUT, { recursive: true });
(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', userDataDir: path.join(ROOT, 'out', 'edit', '.chrome'), protocolTimeout: 600000 });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await serveFromDisk(page);
    const errs = []; page.on('pageerror', e => errs.push(e.message)); page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    const opt = name => { const f = flags.find(a => a.startsWith(`--${name}=`)); return f ? f.split('=')[1] : null; };
    const q = `id=${encodeURIComponent(id)}&render=1&seq=${seq}` + (flags.includes('--plain') ? '&bg=plain' : '') + (flags.includes('--bones') ? '&bones=1' : '')
      + (opt('zoom') ? `&zoom=${opt('zoom')}` : '') + (opt('fy') ? `&fy=${opt('fy')}` : '')
      + (opt('hips') ? `&hips=${opt('hips')}` : '') + (opt('hand') ? `&hand=${opt('hand')}` : '');
    await page.goto(`http://room.local/char?${q}`);
    await page.waitForFunction('window.RENDER_READY === true', { timeout: 120000 });
    const { fps, dur } = await page.evaluate(() => ({ fps: RENDER_API.fps, dur: RENDER_API.dur }));
    if (opt('at')) {   // --at=2.5,5,7.6: only these moments, as stills next to each other in one picture
      const ts = opt('at').split(',').map(Number), files = [];
      for (const t of ts) {
        const f = path.join(OUT, `still_${seq}_${t}.jpg`);
        fs.writeFileSync(f, Buffer.from(await page.evaluate(t => RENDER_API.frame(t), t), 'base64'));
        files.push(f);
      }
      console.log('stills', files.join(' '));
      return;
    }
    const mp4 = path.join(OUT, `${seq}.mp4`);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
      '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', mp4]);
    const done = new Promise(r => ff.on('close', r));
    const n = Math.ceil(dur * fps);
    for (let i = 0; i < n; i++) {
      const jpg = Buffer.from(await page.evaluate(t => RENDER_API.frame(t), i / fps), 'base64');
      if (i === Math.round(n * 0.3) || i === Math.round(n * 0.5) || i === Math.round(n * 0.75)) fs.writeFileSync(path.join(OUT, `${seq}_${i}.jpg`), jpg);
      if (!ff.stdin.write(jpg)) await new Promise(r => ff.stdin.once('drain', r));
    }
    ff.stdin.end();
    await done;
    console.log('done', mp4, errs.length ? 'page errors: ' + errs.slice(0, 3).join(' | ') : '');
  } finally { await browser.close(); }
})();

// what room.py serves to the character studio, answered from disk inside the page's own requests
async function serveFromDisk(page) {
  const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg' };
  const file = f => ({ status: 200, contentType: TYPES[path.extname(f).toLowerCase()] || 'application/octet-stream', body: fs.readFileSync(f) });
  const json = o => ({ status: 200, contentType: 'application/json', body: JSON.stringify(o) });
  const readJ = f => JSON.parse(fs.readFileSync(f, 'utf8'));
  await page.setRequestInterception(true);
  page.on('request', req => {
    const u = new URL(req.url());
    if (u.host !== 'room.local') return req.continue();
    const p = decodeURIComponent(u.pathname);
    try {
      if (p === '/favicon.ico') return req.respond({ status: 204, body: '' });
      if (p === '/char') return req.respond(file(path.join(ROOT, 'room', 'char.html')));
      if (p === '/skel.js') return req.respond(file(path.join(ROOT, 'room', 'skel.js')));
      if (p === '/api/char/list') return req.respond(json(fs.readdirSync(path.join(ROOT, 'assets', 'chars'))
        .filter(c => fs.existsSync(path.join(ROOT, 'assets', 'chars', c, 'char.json'))).sort()));
      if (p === '/api/char') {
        const cid = u.searchParams.get('id'), d = path.join(ROOT, 'assets', 'chars', cid), info = readJ(path.join(d, 'char.json'));
        const actions = {};
        for (const f of fs.readdirSync(path.join(ROOT, 'rigs', 'actions')).filter(f => f.endsWith('.json')).sort())
          actions[f.slice(0, -5)] = readJ(path.join(ROOT, 'rigs', 'actions', f));
        return req.respond(json({ char: info, rig: readJ(path.join(ROOT, 'rigs', info.template, 'rig.json')), actions,
          dir: `assets/chars/${cid}`, mtime: Date.now() }));
      }
      if (p.startsWith('/file/')) {
        const f = path.resolve(ROOT, p.slice(6));
        if (f.startsWith(path.resolve(ROOT)) && fs.existsSync(f)) return req.respond(file(f));
      }
      return req.respond({ status: 404, body: 'not found' });
    } catch (e) { return req.respond({ status: 500, body: String(e) }); }
  });
}
