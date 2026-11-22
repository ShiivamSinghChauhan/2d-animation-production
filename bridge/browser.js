// The Chrome windows the production scripts drive: one window per account, each with its own
// profile folder in .browser/ (so its sign-in is kept) and a local debugging port.
//
//   node bridge/browser.js open google openai   -> open the windows (sign in by hand, once)
//   node bridge/browser.js close google
import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const CONFIG = JSON.parse(fs.readFileSync(path.join(ROOT, 'bridge', 'sites.json'), 'utf8'));
const CHROME = process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';

export const sleep = ms => new Promise(r => setTimeout(r, ms));
export const log = (...a) => console.error(...a);   // stdout is kept for the JSON result

async function alive(port) {
  try { return (await fetch(`http://127.0.0.1:${port}/json/version`)).ok; } catch { return false; }
}

// Start the profile's Chrome if it isn't running. Returns its debugging port.
export async function launch(profile, urls = []) {
  const { port } = CONFIG.profiles[profile];
  if (await alive(port)) return port;
  const dir = path.join(ROOT, '.browser', profile);
  fs.mkdirSync(dir, { recursive: true });
  spawn(CHROME, [`--user-data-dir=${dir}`, `--remote-debugging-port=${port}`, '--no-first-run', '--no-default-browser-check',
    // keep pages working while the window is in the background or minimised
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    ...urls], { detached: true, stdio: 'ignore' }).unref();
  for (let i = 0; i < 80; i++) { if (await alive(port)) return port; await sleep(250); }
  throw new Error(`Chrome for profile "${profile}" did not start`);
}

export async function connect(profile) {
  const port = await launch(profile);
  return puppeteer.connect({ browserURL: `http://127.0.0.1:${port}`, defaultViewport: null });
}

// The tab showing this site (opened if there is none).
export async function sitePage(browser, site, { fresh = false } = {}) {
  const { url } = CONFIG.sites[site];
  const host = new URL(url).host.replace(/^www\./, '');
  let page = (await browser.pages()).find(p => p.url().includes(host));
  if (!page) page = await browser.newPage();
  if (fresh || !page.url().includes(host)) await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.bringToFront();
  return page;
}

// One job at a time per window. A driver waits while another of our drivers holds the window
// (out/bridge/turn_<profile>.json, freed when that process ends), so two jobs never fight over a page.
// Returns release(). Automations that are not ours don't take turns: work in a tab of your own (ownTab).
const TURNS = path.join(ROOT, 'out', 'bridge');
const alivePid = pid => { try { process.kill(pid, 0); return true; } catch { return false; } };
export async function takeTurn(profile, what, maxWaitMs = 3600000) {
  fs.mkdirSync(TURNS, { recursive: true });
  const f = path.join(TURNS, `turn_${profile}.json`);
  const read = () => { try { return JSON.parse(fs.readFileSync(f, 'utf8')); } catch { return null; } };
  for (const t0 = Date.now(); ; await sleep(5000)) {
    let cur = read();
    if (!cur || !alivePid(cur.pid) || Date.now() - cur.since > 45 * 60000) {
      fs.writeFileSync(f, JSON.stringify({ pid: process.pid, since: Date.now(), what }));
      await sleep(300);
      cur = read();
      if (cur && cur.pid === process.pid)
        return () => { const c = read(); if (c && c.pid === process.pid) fs.rmSync(f, { force: true }); };
    }
    if (Date.now() - t0 > maxWaitMs) throw new Error(`waited an hour for the ${profile} window (${cur && cur.what})`);
    log(`${profile}: waiting for "${cur && cur.what}" to finish`);
  }
}

// A new tab on the site for this job, so a tab someone else is using is never taken over.
export async function ownTab(browser, site) {
  const page = await browser.newPage();
  await page.goto(CONFIG.sites[site].url, { waitUntil: 'domcontentloaded' });
  return page;
}

// Send the browser's downloads to dir while a job runs.
//   next(timeout) waits for the next finished download -> { file, name }
//   release() gives downloads back to the normal Downloads folder
export async function catchDownloads(browser, dir) {
  fs.mkdirSync(dir, { recursive: true });
  const cdp = await browser.target().createCDPSession();
  await cdp.send('Browser.setDownloadBehavior', { behavior: 'allowAndName', downloadPath: dir, eventsEnabled: true });
  const names = {}, done = [];
  cdp.on('Browser.downloadWillBegin', e => { names[e.guid] = e.suggestedFilename; });
  cdp.on('Browser.downloadProgress', e => {
    if (e.state === 'completed') done.push({ file: path.join(dir, e.guid), name: names[e.guid] });
  });
  return {
    async next(timeoutMs = 90000) {
      for (const t0 = Date.now(); !done.length; await sleep(300))
        if (Date.now() - t0 > timeoutMs) throw new Error('download did not finish');
      return done.shift();
    },
    async release() { await cdp.send('Browser.setDownloadBehavior', { behavior: 'default' }).catch(() => {}); },
  };
}

// Click (with the real mouse, which menus that open on pointer-down need) the first visible element
// matching css whose label or text contains `text`, case-insensitive. Exact matches win. False if none.
export async function clickText(page, css, text) {
  const box = await page.evaluate((css, want) => {
    const name = e => ((e.getAttribute('aria-label') || '') + ' ' + e.innerText).replace(/\s+/g, ' ').trim().toLowerCase();
    const els = [...document.querySelectorAll(css)].filter(e => e.getBoundingClientRect().width > 0);
    const el = els.find(e => name(e) === want || (e.getAttribute('aria-label') || '').toLowerCase() === want) || els.find(e => name(e).includes(want));
    if (!el) return null;
    el.scrollIntoView({ block: 'center' });
    const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  }, css, text.toLowerCase());
  if (box) await page.mouse.click(box.x, box.y);
  return !!box;
}

// Write one JSON line for the caller and leave a screenshot when something went wrong.
export async function finish(result, page, dir) {
  if (!result.ok && page && dir) {
    const shot = path.join(dir, `error_${Date.now()}.jpg`);
    await page.screenshot({ path: shot, type: 'jpeg', quality: 60 }).catch(() => {});
    result.screenshot = shot;
  }
  console.log(JSON.stringify(result));
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const [cmd, ...profiles] = process.argv.slice(2);
  for (const profile of profiles) {
    if (cmd === 'open') {   // the window with a tab for each of the profile's sites
      const b = await connect(profile);
      for (const [site, s] of Object.entries(CONFIG.sites))
        if (s.profile === profile) await sitePage(b, site).catch(e => console.log(`${site}: ${e.message}`));
      await b.disconnect();
      console.log(`${profile}: window open with ${Object.entries(CONFIG.sites).filter(([, s]) => s.profile === profile).map(([n]) => n).join(', ')}`);
    } else if (cmd === 'close') {
      const b = await connect(profile); await b.close(); console.log(`${profile}: closed`);
    } else {
      console.log('usage: node bridge/browser.js open|close <profile...>'); process.exit(1);
    }
  }
}
