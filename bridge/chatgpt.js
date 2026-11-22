// ChatGPT (chatgpt.com) used like an API through the signed-in "openai" Chrome window.
//   node bridge/chatgpt.js <job.json>          (job format: see image_job.js)
// Each variant: new chat, "Create image" on, attach references, paste prompt, send, wait, download.
import { runImageJob, insertText, fetchInPage, LimitReached } from './image_job.js';
import { clickText, sleep, log } from './browser.js';

// Everything that depends on ChatGPT's page layout (checked 2026-11-14; Stop check fixed 2026-11-20).
const START = 'https://chatgpt.com/';
const SEL = {
  textbox: '#prompt-textarea, div[role=textbox][contenteditable=true]',
  plus: 'button[aria-label="Add files and more"]',
  imageInput: 'input[type=file][aria-label="Attach photos"], input[type=file][accept="image/*"]:not([capture])',
  send: 'button[data-testid="send-button"], button[aria-label*="Send" i]',
  stop: 'button[data-testid="stop-button"], button[aria-label*="Stop" i]',
  gallery: '[data-testid="generated-image-gallery"]',      // its <img>s have blob: sources at full size
};
const LIMIT = /(hit|reached) (the|your) .{0,40}limit|limit (resets|will reset)|try again (after|in|later)/i;

const steps = {
  async newChat(page) {
    await page.goto(START, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector(SEL.textbox, { visible: true, timeout: 45000 });
    await sleep(1500);
    if (await page.evaluate(() => [...document.querySelectorAll('button')].some(b => /^log in$/i.test(b.innerText.trim()))))
      throw new Error('the ChatGPT window is not signed in: run  python prod.py login  and sign in');
  },

  async useImageTool(page) {
    await page.click(SEL.plus);
    await sleep(800);
    if (!(await clickText(page, 'button, [role=menuitem], [role=menuitemradio]', 'create image'))) {
      await page.keyboard.press('Escape');
      log('chatgpt: no "Create image" in the menu, relying on the prompt');
    }
    await sleep(600);
  },

  async attach(page, refs) {
    const input = await page.$(SEL.imageInput);
    if (!input) throw new Error('ChatGPT image upload input not found');
    await input.uploadFile(...refs);
    // previews show a spinner while the files upload
    for (let i = 0; i < 240; i++) {
      await sleep(500);
      const busy = await page.evaluate(() => document.querySelectorAll('[role=progressbar], .animate-spin, circle[stroke-dasharray]').length);
      if (!busy && i > 3) break;
    }
  },

  typePrompt: (page, text) => insertText(page, SEL.textbox, text),

  async send(page) {
    const ready = SEL.send.split(',').map(s => `${s.trim()}:not([disabled])`).join(', ');
    const btn = await page.waitForSelector(ready, { visible: true, timeout: 120000 });
    await btn.click();
  },

  // Done when nothing is streaming and the last image gallery has stopped changing for a while
  // (ChatGPT paints its image in stages). Marks that gallery with data-prod-last.
  async waitForImages(page, timeoutMs = 420000) {
    const t0 = Date.now();
    let sig = '', since = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      await sleep(2500);
      const s = await page.evaluate(SEL => {
        document.querySelectorAll('[data-prod-last]').forEach(e => e.removeAttribute('data-prod-last'));
        const all = document.querySelectorAll(SEL.gallery);
        const last = all[all.length - 1];
        if (last) last.setAttribute('data-prod-last', '1');
        const imgs = last ? [...last.querySelectorAll('img')].filter(i => i.complete && i.naturalWidth >= 256) : [];
        const text = (document.querySelector('main') || document.body).innerText;
        return {
          // the page keeps a hidden Stop button after the answer is done: only a visible one means busy (2026-11-20)
          imgs: imgs.length, busy: [...document.querySelectorAll(SEL.stop)].some(e => e.getBoundingClientRect().width > 0),
          creating: /creating image|getting started/i.test(text),
          sig: imgs.map(i => i.currentSrc + i.naturalWidth).join('|') + text.length, tail: text.trim().slice(-400),
        };
      }, SEL);
      if (s.sig !== sig) { sig = s.sig; since = Date.now(); }
      const quiet = Date.now() - since;
      if (s.imgs && !s.busy && !s.creating && quiet > 8000) return s.imgs;
      if (!s.imgs && !s.busy && !s.creating && quiet > 30000 && Date.now() - t0 > 45000) {
        if (LIMIT.test(s.tail)) throw new LimitReached(`ChatGPT: ${s.tail.slice(-200)}`);
        throw new Error(`ChatGPT answered without an image: "${s.tail.slice(-300)}"`);
      }
    }
    throw new Error('timed out waiting for ChatGPT\'s image');
  },

  // The gallery image is a blob: URL holding the full-size picture; read it straight from the page.
  async download(page, downloads, i) {
    const src = await page.evaluate(i => {
      const imgs = [...document.querySelectorAll('[data-prod-last] img')].filter(e => e.complete && e.naturalWidth >= 256);
      return imgs[i].currentSrc || imgs[i].src;
    }, i);
    return fetchInPage(page, src);
  },
};

await runImageJob('chatgpt', steps);
