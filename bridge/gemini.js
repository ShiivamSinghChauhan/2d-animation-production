// Gemini (gemini.google.com) used like an API through the signed-in "google" Chrome window.
//   node bridge/gemini.js <job.json>          (job format: see image_job.js)
// Each variant: new chat, "Create image" tool on, attach references, paste prompt, send, wait,
// download the full-size image with the page's own download button.
import { runImageJob, insertText, LimitReached } from './image_job.js';
import { clickText, sleep, log } from './browser.js';

// Everything that depends on Gemini's page layout (checked 2026-11-14).
const START = 'https://gemini.google.com/app';
const SEL = {
  textbox: 'div[role=textbox][contenteditable=true]',
  tools: 'button[aria-label="Upload & tools"]',
  imageInput: 'input[type=file][accept^="image"]',
  send: 'button[aria-label*="Send" i]',
  stop: 'button[aria-label*="Stop" i]',
  modeButton: '[data-test-id=bard-mode-menu-button]',     // modes: "Quick answers", "Everyday help", "Detailed research"
  menuItem: '[role=menuitem], [role=menuitemradio], [role=menuitemcheckbox], [role=option]',
  response: 'model-response',
  download: 'button[aria-label*="Download" i]',
};

// The plan's quota banner above the prompt box: "Limit reached · Resets 3:41 AM"
async function checkLimit(page) {
  const msg = await page.evaluate(() => {
    const m = document.body.innerText.match(/Limit reached[^\n]*/);
    return m ? m[0] : null;
  });
  if (msg) throw new LimitReached(`Gemini: ${msg.replace(/\s+/g, ' ')}`);
}

const steps = {
  async newChat(page, job) {
    await page.goto(START, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector(SEL.textbox, { visible: true, timeout: 45000 });
    await sleep(1500);
    if (await page.evaluate(() => [...document.querySelectorAll('button, a')].some(b => b.innerText.trim() === 'Sign in')))
      throw new Error('the Gemini window is not signed in: run  python prod.py login  and sign in');
    await checkLimit(page);
    if (job.mode) {
      const btn = await page.$(SEL.modeButton);
      if (!(await btn.evaluate(b => b.getAttribute('aria-label') + b.innerText)).toLowerCase().includes(job.mode.toLowerCase())) {
        await btn.click();
        await sleep(700);
        if (!(await clickText(page, SEL.menuItem, job.mode))) throw new Error(`mode "${job.mode}" is not in Gemini's mode picker`);
        await sleep(700);
      }
    }
  },

  // "Create image" makes the answer a picture, not a description of one.
  async useImageTool(page) {
    await page.click(SEL.tools);
    await sleep(800);
    const state = await page.evaluate(css => {
      const el = [...document.querySelectorAll(css)].find(e => /create image/i.test(e.innerText));
      if (!el) return 'missing';
      if (el.disabled || el.getAttribute('aria-disabled') === 'true') return 'disabled';
      if (el.getAttribute('aria-checked') === 'true') return 'on';
      el.click();
      return 'switched on';
    }, SEL.menuItem);
    if (state !== 'switched on') await page.keyboard.press('Escape');
    if (state === 'disabled') throw new LimitReached('Gemini: "Create image" is switched off (plan limit reached?)');
    if (state === 'missing') log('gemini: no "Create image" tool in the menu, relying on the prompt');
    await sleep(500);
  },

  async attach(page, refs) {
    const input = await page.$(SEL.imageInput);
    if (!input) throw new Error('Gemini image upload input not found');
    await input.uploadFile(...refs);
    for (let i = 0; i < 120; i++) {      // wait until no preview is still uploading
      await sleep(500);
      const busy = await page.evaluate(() => document.querySelectorAll('[role=progressbar], mat-progress-spinner, mat-spinner').length);
      if (!busy && i > 3) break;
    }
  },

  typePrompt: (page, text) => insertText(page, SEL.textbox, text),

  async send(page) {
    const btn = await page.waitForSelector(`${SEL.send}:not([disabled]):not([aria-disabled=true])`, { visible: true, timeout: 90000 });
    await btn.click();
  },

  // Marks the last answer with data-prod-last; returns how many finished images it holds.
  async waitForImages(page, timeoutMs = 360000) {
    const t0 = Date.now();
    let lastText = '', since = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      await sleep(2000);
      const s = await page.evaluate(SEL => {
        document.querySelectorAll('[data-prod-last]').forEach(e => e.removeAttribute('data-prod-last'));
        const all = document.querySelectorAll(SEL.response);
        const last = all[all.length - 1];
        if (!last) return null;
        last.setAttribute('data-prod-last', '1');
        const imgs = [...last.querySelectorAll('img')].filter(i => i.complete && i.naturalWidth >= 256);
        return { imgs: imgs.length, busy: !!document.querySelector(SEL.stop), text: last.innerText.trim().slice(0, 400) };
      }, SEL);
      if (!s) continue;
      if (s.imgs && !s.busy) return s.imgs;
      if (s.text !== lastText) { lastText = s.text; since = Date.now(); }
      if (!s.busy && s.text && Date.now() - since > 20000) {
        await checkLimit(page);
        throw new Error(`Gemini answered without an image: "${s.text.slice(0, 300)}"`);
      }
    }
    throw new Error('timed out waiting for Gemini\'s image');
  },

  async download(page, downloads, i) {
    const imgs = [];
    for (const h of await page.$$('[data-prod-last] img'))
      if (await h.evaluate(e => e.complete && e.naturalWidth >= 256)) imgs.push(h);
    await imgs[i].scrollIntoView();
    await imgs[i].hover();
    await sleep(800);
    const clicked = await imgs[i].evaluate((e, css) => {
      for (let box = e.parentElement; box && !box.matches('[data-prod-last]'); box = box.parentElement) {
        const b = box.querySelector(css);
        if (b) { b.click(); return true; }
      }
      return false;
    }, SEL.download);
    if (!clicked) throw new Error('no download button next to Gemini\'s image');
    return downloads.next(120000);
  },
};

await runImageJob('gemini', steps);
