// Flow Music (flowmusic.app) used like an API through the signed-in "google" Chrome window.
//   node bridge/flowmusic.js <job.json>
// job.json: { "prompt": "...", "n": 1, "start": 1, "out_dir": "...", "audio_format": "wav", "collect": false }
// Each request starts a new session: type the prompt to the Producer, send, wait for the songs,
// download every new song (Flow Music often makes two takes) as <out_dir>/v<k>.<format>.
// collect: only download the songs the open session already shows.
import fs from 'node:fs';
import path from 'node:path';
import { connect, sitePage, ownTab, takeTurn, catchDownloads, clickText, finish, sleep, log } from './browser.js';
import { LimitReached } from './image_job.js';

// Everything that depends on Flow Music's page layout (checked 2026-11-14).
const START = 'https://www.flowmusic.app/';
const SEL = {
  textbox: 'textarea[aria-label="Chat message"]',
  send: 'button[aria-label="Send message"]',
  card: 'div[role=button][aria-label^="Open details for "]',
  menuItem: '[role=menuitem]',
};
const LIMIT = /out of credits|no credits|credit limit|reached (your|the) .{0,30}limit|upgrade to (continue|keep)/i;

// Song cards in the session, with their titles and the y position of each (to find its menu button).
async function cards(page) {
  return page.evaluate(css => [...document.querySelectorAll(css)].map(c => c.getAttribute('aria-label').replace('Open details for ', '')), SEL.card);
}

async function waitForSongs(page, before, timeoutMs = 600000) {
  const t0 = Date.now();
  let quietSince = Date.now(), last = '';
  while (Date.now() - t0 < timeoutMs) {
    await sleep(3000);
    const s = await page.evaluate(css => {
      const text = (document.querySelector('main') || document.body).innerText;
      return { cards: document.querySelectorAll(css).length, busy: /Generating [^\n]*\.\.\.|\n\d{1,3}%\n|Creating "/.test(text), tail: text.trim().slice(-400) };
    }, SEL.card);
    const sig = s.cards + '|' + s.busy + '|' + s.tail.length;
    if (sig !== last) { last = sig; quietSince = Date.now(); }
    if (s.cards > before && !s.busy && Date.now() - quietSince > 5000) return s.cards;
    if (s.cards === before && !s.busy && Date.now() - quietSince > 45000) {
      if (LIMIT.test(s.tail)) throw new LimitReached(`Flow Music: ${s.tail.slice(-200)}`);
      throw new Error(`Flow Music answered without a song: "${s.tail.slice(-300)}"`);
    }
  }
  throw new Error('timed out waiting for Flow Music');
}

// Song i of the session: its "More options" menu -> Download -> format.
async function downloadSong(page, downloads, i, format) {
  const box = await page.evaluate((css, i) => {
    const card = document.querySelectorAll(css)[i];
    card.scrollIntoView({ block: 'center' });
    const title = card.getAttribute('aria-label').replace('Open details for ', '');
    // the menu button sits on the card's row
    const r = card.getBoundingClientRect();
    const btn = [...document.querySelectorAll('button[aria-label^="More options for "]')]
      .filter(b => b.getAttribute('aria-label') === `More options for ${title}`)
      .find(b => { const q = b.getBoundingClientRect(); return q.y >= r.y - 4 && q.y <= r.bottom + 4; });
    if (!btn) return null;
    const q = btn.getBoundingClientRect();
    return { x: q.x + q.width / 2, y: q.y + q.height / 2 };
  }, SEL.card, i);
  if (!box) throw new Error(`no options button for song ${i + 1}`);
  await page.mouse.click(box.x, box.y);
  await sleep(800);
  // The formats are in a submenu that opens on hover and closes if the pointer jumps: move like a hand.
  const item = label => page.evaluate((css, label) => {
    const e = [...document.querySelectorAll(css)].find(x => x.innerText.trim() === label);
    if (!e) return null;
    const r = e.getBoundingClientRect();
    return { x: r.x + Math.min(30, r.width / 2), y: r.y + r.height / 2 };
  }, SEL.menuItem, label);
  const dl = await item('Download');
  if (!dl) throw new Error('no "Download" in the song menu');
  await page.mouse.move(dl.x, dl.y, { steps: 4 });
  await sleep(500);
  const f = await item(format);
  if (!f) throw new Error(`no "${format}" download option`);
  await page.mouse.move(f.x, f.y, { steps: 12 });
  await sleep(400);
  await page.mouse.down();
  await sleep(80);
  await page.mouse.up();
  return downloads.next(180000);
}

const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8').replace(/^﻿/, ''));
const format = (job.audio_format || 'wav').toUpperCase();
const files = [];
let page = null, ok = false;
const release = await takeTurn('google', `flowmusic: ${path.basename(job.out_dir)}`);
const browser = await connect('google');
const downloads = await catchDownloads(browser, path.join(job.out_dir, '.downloads'));
try {
  page = job.collect ? await sitePage(browser, 'flowmusic') : await ownTab(browser, 'flowmusic');
  for (let k = 0; k < (job.n || 1); k++) {
    let before = 0;
    if (!job.collect) {
      await page.goto(START, { waitUntil: 'domcontentloaded' });
      await page.waitForSelector(SEL.textbox, { visible: true, timeout: 45000 });
      await sleep(1500);
      before = (await cards(page)).length;
      await page.click(SEL.textbox);
      await page.type(SEL.textbox, job.prompt);
      await sleep(400);
      await page.click(SEL.send);
      log(`flowmusic: request ${k + 1}/${job.n || 1} sent, waiting`);
    }
    const count = job.collect ? (await cards(page)).length : await waitForSongs(page, before);
    // stop the new song playing in the window
    await page.evaluate(() => document.querySelectorAll('button[aria-label^="Pause "]').forEach(b => b.click()));
    for (let i = before; i < count; i++) {
      const got = await downloadSong(page, downloads, i, format);
      const dst = path.join(job.out_dir, `v${(job.start || 1) + files.length}${path.extname(got.name || '') || '.' + format.toLowerCase()}`);
      fs.renameSync(got.file, dst);
      files.push(dst);
      log(`flowmusic: saved ${dst}`);
      await page.keyboard.press('Escape');
      await sleep(1000);
    }
  }
  await finish({ ok: true, files });
  ok = true;
} catch (e) {
  await finish({ ok: false, error: String(e.message || e), limit: e instanceof LimitReached || undefined, files }, page, job.out_dir);
} finally {
  await downloads.release();
  if (ok && page && !job.collect) await page.close().catch(() => {});
  await browser.disconnect();
  release();
}
