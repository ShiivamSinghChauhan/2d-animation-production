// The final video of an episode's edit: the editor page in render mode (same engine.js as the preview) draws
// every frame and mixes the sound; ffmpeg encodes them. Needs room.py running.
//   node room/render_edit.js <ep>   ->  out/edit/<ep>/<ep>.mp4 (progress in out/edit/<ep>/render.json)
const fs = require('fs'), path = require('path'), { spawn } = require('child_process');
const puppeteer = require(path.join(__dirname, '..', 'bridge', 'node_modules', 'puppeteer-core'));
const ROOM_PORT = (() => { try { return fs.readFileSync(path.join(__dirname, '..', 'out', 'room_port.txt'), 'utf8').trim(); } catch { return '8790'; } })();
const ROOT = path.join(__dirname, '..'), ep = process.argv[2];
const OUT = path.join(ROOT, 'out', 'edit', ep), STATUS = path.join(OUT, 'render.json');
const CHROME = [process.env.CHROME, 'C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'].find(f => f && fs.existsSync(f));
fs.mkdirSync(OUT, { recursive: true });
const status = s => fs.writeFileSync(STATUS, JSON.stringify(s));

(async () => {
  let browser;
  try {
    status({ state: 'working', done: 0, total: 0, step: 'loading' });
    browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', userDataDir: path.join(ROOT, 'out', 'edit', '.chrome'),
      args: ['--autoplay-policy=no-user-gesture-required'], protocolTimeout: 600000 });
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    const errs = []; page.on('pageerror', e => errs.push(e.message));
    await page.goto(`http://127.0.0.1:${ROOM_PORT}/edit?ep=${encodeURIComponent(ep)}&render=1`);
    await page.waitForFunction('window.RENDER_READY === true', { timeout: 600000 });
    const { fps, dur } = await page.evaluate(() => ({ fps: RENDER_API.fps, dur: RENDER_API.dur }));
    const total = Math.ceil(dur * fps);

    // sound: the whole mix, read out of the page in pieces into a WAV file
    status({ state: 'working', done: 0, total, step: 'mixing sound' });
    const bytes = await page.evaluate(() => RENDER_API.audio());
    const wav = path.join(OUT, 'mix.wav'), fd = fs.openSync(wav, 'w'), hdr = Buffer.alloc(44);
    hdr.write('RIFF', 0); hdr.writeUInt32LE(36 + bytes, 4); hdr.write('WAVEfmt ', 8); hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20);
    hdr.writeUInt16LE(2, 22); hdr.writeUInt32LE(48000, 24); hdr.writeUInt32LE(48000 * 4, 28); hdr.writeUInt16LE(4, 32); hdr.writeUInt16LE(16, 34);
    hdr.write('data', 36); hdr.writeUInt32LE(bytes, 40); fs.writeSync(fd, hdr);
    const CH = 4 << 20;
    for (let off = 0; off < bytes; off += CH) fs.writeSync(fd, Buffer.from(await page.evaluate((o, l) => RENDER_API.chunk(o, l), off, CH), 'base64'));
    fs.closeSync(fd);

    // pictures: every frame through ffmpeg
    const mp4 = path.join(OUT, `${ep}.mp4`);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
      '-i', wav, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', mp4]);
    let fferr = ''; ff.stderr.on('data', d => fferr += d);
    const done = new Promise(r => ff.on('close', r));
    for (let i = 0; i < total; i++) {
      const jpg = Buffer.from(await page.evaluate(t => RENDER_API.frame(t), i / fps), 'base64');
      if (!ff.stdin.write(jpg)) await new Promise(r => ff.stdin.once('drain', r));
      if (i % 48 === 0) status({ state: 'working', done: i, total, step: 'drawing' });
    }
    ff.stdin.end();
    const code = await done;
    if (code !== 0) throw new Error('ffmpeg: ' + fferr.slice(-300));
    status({ state: 'done', done: total, total, file: path.relative(ROOT, mp4).split(path.sep).join('/'), time: new Date().toLocaleString(), page_errors: errs.slice(0, 5) });
    console.log('done', mp4);
  } catch (e) {
    status({ state: 'failed', error: String(e.message || e).slice(0, 400) });
    console.error(e);
    process.exitCode = 1;
  } finally { if (browser) await browser.close(); }
})();
