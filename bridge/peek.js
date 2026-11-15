// node bridge/peek.js <site> <url> : open url in the site's tab, print the page text, save out/bridge/<site>.jpg
import { connect, sitePage, sleep, ROOT } from './browser.js';
import path from 'node:path';
const [site, url] = process.argv.slice(2);
const b = await connect((await import('./browser.js')).CONFIG.sites[site].profile);
const p = await sitePage(b, site);
if (url !== "-") { await p.goto(url, { waitUntil: "domcontentloaded" }); await sleep(6000); }
for (const t of process.argv.slice(4)) { await (await import("./browser.js")).clickText(p, "button, [role=button], a, [role=option], [role=tab]", t); await sleep(2500); }
console.log((await p.evaluate(() => document.body.innerText)).replace(/\n{2,}/g, '\n').slice(0, 4000));
await p.screenshot({ path: path.join(ROOT, 'out', 'bridge', `${site}.jpg`), type: 'jpeg' });
b.disconnect();
