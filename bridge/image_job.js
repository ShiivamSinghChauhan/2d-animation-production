// The part every image site shares: read the job, make n variants, save them, report.
// A site file gives the steps (newChat, useImageTool, attach, typePrompt, send, waitForImages,
// download) and calls runImageJob(site, steps).
//
// job.json: { "prompt": "...", "refs": ["E:/.../ref.png"], "n": 2, "start": 1, "out_dir": "E:/.../jobs/x", "mode": null }
// Result (one JSON line): {"ok":true,"files":[...]} or {"ok":false,"error":"...","limit":true?,"files":[...]}
import fs from 'node:fs';
import path from 'node:path';
import { connect, sitePage, catchDownloads, finish, sleep, log, CONFIG } from './browser.js';

// Thrown when the site says the plan's quota is used up: the desk stops using the site until it resets.
export class LimitReached extends Error {}

export async function runImageJob(site, steps) {
  const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8').replace(/^﻿/, ''));
  const files = [];
  let page = null;
  const browser = await connect(CONFIG.sites[site].profile);
  const downloads = await catchDownloads(browser, path.join(job.out_dir, '.downloads'));
  try {
    page = await sitePage(browser, site);
    for (let k = 0; k < (job.n || 1); k++) {
      if (!job.collect) {   // collect: only save what the open chat already shows (after a timeout)
        await steps.newChat(page, job);
        await steps.useImageTool(page, job);
        if (job.refs?.length) await steps.attach(page, job.refs);
        await steps.typePrompt(page, job.prompt);
        await steps.send(page);
        log(`${site}: variant ${k + 1}/${job.n || 1} sent, waiting`);
      }
      const count = await steps.waitForImages(page);
      for (let i = 0; i < count; i++) {
        const got = await steps.download(page, downloads, i);
        const ext = (path.extname(got.name || '') || '.png').toLowerCase();
        const dst = path.join(job.out_dir, `v${(job.start || 1) + files.length}${ext}`);
        if (got.file) fs.renameSync(got.file, dst); else fs.writeFileSync(dst, got.bytes);
        files.push(dst);
        log(`${site}: saved ${dst}`);
      }
      await sleep(3000);
    }
    await finish({ ok: true, files });
  } catch (e) {
    await finish({ ok: false, error: String(e.message || e), limit: e instanceof LimitReached || undefined, files }, page, job.out_dir);
  } finally {
    await downloads.release();
    await browser.disconnect();
  }
}

// Put text into a rich text box (Gemini's and ChatGPT's prompt boxes) in one go.
export async function insertText(page, css, text) {
  await page.click(css);
  await page.evaluate((css, t) => { document.querySelector(css).focus(); document.execCommand('insertText', false, t); }, css, text);
  await sleep(400);
}

// Fetch an image the page shows (with the page's own sign-in) when there is no download button.
export async function fetchInPage(page, src) {
  const b64 = await page.evaluate(async src => {
    const blob = await (await fetch(src, { credentials: 'include' })).blob();
    return new Promise(res => { const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(blob); });
  }, src);
  const [head, data] = b64.split(',');
  return { bytes: Buffer.from(data, 'base64'), name: head.includes('jpeg') ? 'image.jpg' : head.includes('webp') ? 'image.webp' : 'image.png' };
}
