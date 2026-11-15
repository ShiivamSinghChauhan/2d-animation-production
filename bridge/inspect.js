// Look at a site's tab the way a driver needs to: every visible control, one line each, plus a
// small screenshot. Used to write and repair the site drivers without reading whole pages.
//
//   node bridge/inspect.js gemini                    controls of the gemini tab
//   node bridge/inspect.js gemini --goto             reload the site's start page first
//   node bridge/inspect.js gemini --click "Tools"    click the control with this label/text, then list
//   node bridge/inspect.js gemini --eval "<js>"      run js in the page, print the result
//   node bridge/inspect.js gemini --all              include links and images
import fs from 'node:fs';
import path from 'node:path';
import { connect, sitePage, clickText, CONFIG, ROOT, sleep } from './browser.js';

const CLICKABLE = 'button, [role=button], [role=menuitem], [role=menuitemradio], [role=menuitemcheckbox], [role=option], [role=tab], [role=switch], a';

const args = process.argv.slice(2);
const site = args[0];
const opt = f => { const i = args.indexOf(f); return i >= 0 ? args[i + 1] : null; };
if (!CONFIG.sites[site]) { console.log('usage: node bridge/inspect.js <site> [--goto] [--click label] [--eval js] [--all]'); process.exit(1); }

const browser = await connect(CONFIG.sites[site].profile);
try {
  const page = await sitePage(browser, site, { fresh: args.includes('--goto') });
  if (args.includes('--goto')) await sleep(4000);
  for (let i = 0; i < args.length; i++) {           // --click may be given several times: clicked in order
    if (args[i] !== '--click') continue;
    const label = args[i + 1];
    const ok = await clickText(page, CLICKABLE, label);
    console.log(ok ? `clicked "${label}"` : `no control labelled "${label}"`);
    await sleep(1500);
  }
  const js = opt('--eval');
  if (js) console.log(JSON.stringify(await page.evaluate(js), null, 1));
  console.log(page.url());
  for (const line of await page.evaluate(listControls, args.includes('--all'))) console.log(line);
  const dir = path.join(ROOT, 'out', 'bridge');
  fs.mkdirSync(dir, { recursive: true });
  const shot = path.join(dir, `${site}.jpg`);
  await page.screenshot({ path: shot, type: 'jpeg', quality: 60 });
  console.log('screenshot', shot);
} finally {
  await browser.disconnect();
}

// --- these two run inside the page ---

function listControls(all) {
  const sel = 'button, input, textarea, select, [contenteditable="true"], [role=button], [role=menuitem], [role=menuitemradio], '
    + '[role=option], [role=tab], [role=switch], [role=checkbox], [role=radio], [role=textbox], [role=combobox]' + (all ? ', a[href], img' : '');
  const found = [];
  const walk = root => {
    root.querySelectorAll(sel).forEach(el => found.push(el));
    root.querySelectorAll('*').forEach(el => el.shadowRoot && walk(el.shadowRoot));
  };
  walk(document);
  const out = [];
  for (const el of found) {
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    const hiddenInput = el.tagName === 'INPUT' && el.type === 'file';
    if (!hiddenInput && (r.width < 2 || r.height < 2 || st.visibility === 'hidden' || st.display === 'none')) continue;
    const text = (el.innerText || el.value || '').replace(/\s+/g, ' ').trim().slice(0, 50);
    const bits = [el.tagName.toLowerCase() + (el.type && el.tagName === 'INPUT' ? `[${el.type}]` : '')];
    const lab = el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('placeholder') || el.getAttribute('alt');
    if (lab) bits.push(`"${lab.slice(0, 50)}"`);
    if (text && text !== lab) bits.push(`'${text}'`);
    for (const a of ['data-test-id', 'data-testid', 'id', 'role', 'accept', 'aria-expanded', 'aria-checked', 'aria-pressed', 'disabled'])
      if (el.hasAttribute(a)) bits.push(`${a}=${(el.getAttribute(a) || '1').slice(0, 40)}`);
    if (el.tagName === 'IMG') bits.push(`src=${(el.currentSrc || el.src).slice(0, 70)} nat=${el.naturalWidth}x${el.naturalHeight}`);
    bits.push(`@${Math.round(r.x)},${Math.round(r.y)} ${Math.round(r.width)}x${Math.round(r.height)}`);
    out.push(bits.join(' '));
  }
  return out;
}
