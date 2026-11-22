// Speech from Google AI Studio's speech playground (free in the browser, no API key).
//   node bridge/aistudio_tts.js <request.json>
// request.json: { "text": "...", "voice": "Rami", "style": "Narration", "out": "E:/.../x.wav" }
// Result (one JSON line): {"ok":true,"file":"...","seconds":12.3} or {"ok":false,"error":"...","limit":true?}
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { connect, sitePage, ownTab, takeTurn, finish, sleep, log, clickText } from './browser.js';
import { insertText } from './image_job.js';

const SEL = {
  newDialog: 'Create new dialog',                       // card on the playground's start page
  voiceButton: 'Open voice settings',
  voiceSearch: 'input[aria-label="Search voices"]',
  closePanel: 'Close panel',
  style: 'Style',
  menuItem: '[role=menuitem],[role=option],[role=menuitemradio],mat-option',
  text: 'textarea[aria-label="Speech block text"]',
  run: 'Run',
  busy: /Cancel generation|^Stop$/m,
  limit: /quota|rate limit|limit reached|too many requests|resource exhausted/i,
};

const startedAt = Date.now();
const req = JSON.parse(fs.readFileSync(process.argv[2], 'utf8').replace(/^\uFEFF/, ''));
const release = await takeTurn('google', `aistudio: ${path.basename(req.out)}`);
const browser = await connect('google');
let page = null, ok = false;
try {
  page = req.collect ? await sitePage(browser, 'aistudio') : await ownTab(browser, 'aistudio');
  if (!req.collect) {
  await sleep(5000);
  await clickText(page, 'button,[role=button]', SEL.newDialog); await sleep(2000);

  if (req.voice) {
    await clickText(page, 'button', SEL.voiceButton); await sleep(1500);
    await page.click(SEL.voiceSearch); await page.type(SEL.voiceSearch, req.voice); await sleep(1500);
    const ok = await page.evaluate(v => {
      const b = [...document.querySelectorAll('button[aria-label]')].find(e => e.getAttribute('aria-label').replace(' (Current)', '') === v);
      if (b) b.click(); return !!b;
    }, req.voice);
    if (!ok) throw new Error(`voice ${req.voice} not found`);
    await sleep(1000);
    await clickText(page, 'button', SEL.closePanel); await sleep(1000);
  }
  if (req.style) {
    await clickText(page, 'button', SEL.style); await sleep(1200);
    await clickText(page, SEL.menuItem, req.style); await sleep(800);
    await page.keyboard.press('Escape').catch(() => {});
  }
  await insertText(page, SEL.text, req.text);
  await clickText(page, 'button', SEL.run);
  log(`aistudio: ${req.text.length} chars sent (${req.voice})`);
  }

  // generation streams while it plays; it is done when Stop/Cancel is gone and a Download button shows
  const t0 = Date.now();
  let done = false;
  while (Date.now() - t0 < 15 * 60 * 1000) {
    await sleep(3000);
    const s = await page.evaluate(() => ({
      text: document.body.innerText,
      dl: !!document.querySelector('[aria-label="Download"]'),
    }));
    if (SEL.limit.test(s.text) && !s.dl) throw Object.assign(new Error('limit: ' + s.text.match(SEL.limit)[0]), { limit: true });
    if (!SEL.busy.test(s.text) && s.dl) { done = true; break; }
  }
  if (!done) throw new Error('timed out waiting for audio');
  await page.evaluate(() => document.querySelectorAll('audio').forEach(a => a.pause()));
  await clickText(page, '[aria-label="Pause"]', 'Pause').catch(() => {});
  // the Download button saves "Generated Audio <date>.wav" to the user's Downloads folder: move it from there
  const dlDir = path.join(os.homedir(), 'Downloads');
  // the button is not ready the moment generation ends: click again every 6 s until the file shows up
  let file = null;
  for (let i = 0; i < 120 && !file; i++) {
    if (i % 6 === 0) { await page.evaluate(() => document.querySelectorAll('audio').forEach(a => a.pause())); await clickText(page, '[aria-label="Download"]', 'Download'); }
    await sleep(1000);
    file = fs.readdirSync(dlDir).filter(n => /^Generated Audio.*\.wav$/.test(n))
      .map(n => path.join(dlDir, n)).find(f => fs.statSync(f).mtimeMs > startedAt);
  }
  if (!file) throw new Error('Download gave no file');
  await sleep(1500);                                  // let Chrome finish writing it
  fs.mkdirSync(path.dirname(req.out), { recursive: true });
  fs.copyFileSync(file, req.out); fs.unlinkSync(file);
  log(`aistudio: saved ${req.out}`);
  await finish({ ok: true, file: req.out, seconds: Math.round((Date.now() - t0) / 1000) });
  ok = true;
} catch (e) {
  await finish({ ok: false, error: String(e.message || e), limit: e.limit || undefined }, page, path.dirname(req.out));
} finally {
  if (ok && page && !req.collect) await page.close().catch(() => {});
  await browser.disconnect();
  release();
}
