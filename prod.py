"""Production desk for the stories channel: one command for each repetitive step.

  python prod.py login                            open the Chrome windows; sign in by hand (once)
  python prod.py plan   <assets.yaml> [--full]    what would be generated and the prompts; nothing is sent
  python prod.py gen    <assets.yaml> [--only a,b] [--dry]
                                                  generate what is missing, clean it, make a contact sheet
  python prod.py sheet  [id-prefix ...]           contact sheet of every variant
  python prod.py pick   <id> <k>                  variant k becomes the asset: assets/<id>.png
  python prod.py redo   <id> "<correction>" [--n 2] [--dry]
                                                  new variants with a correction added to the prompt
  python prod.py status [id-prefix]               one line per asset
  python prod.py collect <id>                     save the image the site's open chat shows (after a timeout)
  python prod.py notes [--done <id>]              the user's open notes from the asset room (python room.py)

A request (one item of an assets.yaml) names a recipe from bible/recipes.yaml and fills its blanks.
The browser drivers in bridge/ do the generating; every try is kept in jobs/<id>/ (v1.png as made,
v1_clean.png processed) and the picked one is copied to assets/.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'blender'))
import prep_png  # noqa: E402

BIBLE, JOBS, ASSETS, REVIEW = ROOT / 'bible', ROOT / 'jobs', ROOT / 'assets', ROOT / 'out' / 'review'
DRIVERS = {'gemini': 'bridge/gemini.js', 'chatgpt': 'bridge/chatgpt.js', 'flowmusic': 'bridge/flowmusic.js'}
AUDIO = {'.wav', '.mp3', '.m4a'}
LIMITED = {}   # site -> the site's "limit reached" message; the site is skipped for the rest of the run
FORMATS = {   # sentence added to the prompt, and the size the dry run fakes
    'tall': ('Make it a tall portrait image (2:3).', (1696, 2528)),
    'wide': ('Make it a wide landscape image (16:9).', (2752, 1536)),
    'landscape': ('Make it a landscape image (4:3).', (2400, 1792)),
}


class Waiting(Exception):
    """A request that needs another asset to be picked first."""


class Fields(dict):
    def __missing__(self, key):
        raise KeyError(f'the prompt needs "{key}", but the request does not give it')


def load_yaml(p):
    return yaml.safe_load(Path(p).read_text(encoding='utf-8')) or {}


# ---------- requests -> jobs ----------

def read_requests(path):
    """The requests of an assets.yaml, with the rig layers of 'layers: true' requests added."""
    layers = load_yaml(BIBLE / 'recipes.yaml')['layers']
    out = []
    for req in load_yaml(path):
        out.append(req)
        want = req.get('layers')            # true = all layers, or a list of layer names
        if want:
            out += [{'id': f"{req['id']}_{name}", 'recipe': 'edit', 'from': req['id'], 'change': change}
                    for name, change in layers.items() if want is True or name in want]
    return out


def picked_file(ref, which):
    """Image of a picked asset: 'raw' (as generated, background still on) or 'clean' (assets/<id>.png)."""
    if ref.endswith('.png'):                       # a file made before the desk existed
        return ROOT / ref
    job = load_job(ref)
    if not job or not job['picked']:
        raise Waiting(ref)
    return job_dir(ref) / job['variants'][job['picked'] - 1]['raw'] if which == 'raw' else ASSETS / f'{ref}.png'


def build(req):
    """Request -> everything a job needs: the full prompt, references, site, clean-up steps."""
    recipes = load_yaml(BIBLE / 'recipes.yaml')
    if req['recipe'] not in recipes:
        raise KeyError(f"no recipe \"{req['recipe']}\" in bible/recipes.yaml")
    r = recipes[req['recipe']]
    f = Fields(load_yaml(BIBLE / 'style.yaml'))
    f.update(r.get('defaults') or {})
    char = load_yaml(BIBLE / 'characters' / f"{req['character']}.yaml") if req.get('character') else {}
    f.update({k: v for k, v in char.items() if isinstance(v, str)})
    f.update({k: v for k, v in req.items() if isinstance(v, str)})
    refs = []
    for ref in r.get('refs', []):
        refs.append(ROOT / char['ref'] if ref == 'character' else picked_file(req['from'], 'raw'))
    prompt = ' '.join(r['prompt'].format_map(f).split())
    fmt = req.get('format', r.get('format'))
    if fmt:
        prompt += ' ' + FORMATS[fmt][0]
    clean = dict(r.get('clean') or {})
    align_to = None
    if clean.get('align'):
        align_to = str(ROOT / req['align_to']) if req.get('align_to') else str(picked_file(req['from'], 'clean'))
    return {'id': req['id'], 'recipe': req['recipe'], 'site': req.get('site', r.get('site', 'chatgpt')),
            'mode': req.get('mode', r.get('mode')), 'format': fmt, 'prompt': prompt,
            'refs': [str(p) for p in refs], 'clean': clean, 'align_to': align_to,
            'audio': r.get('audio'), 'n': int(req.get('n', r.get('n', 1)))}


def job_dir(asset_id):
    return JOBS / asset_id.replace('/', '__')


def load_job(asset_id):
    p = job_dir(asset_id) / 'job.json'
    return json.loads(p.read_text('utf-8')) if p.exists() else None


def save_job(job):
    d = job_dir(job['id'])
    d.mkdir(parents=True, exist_ok=True)
    (d / 'job.json').write_text(json.dumps(job, ensure_ascii=False, indent=1), 'utf-8')


# ---------- generating ----------

def run_driver(site, spec_path):
    """Run a bridge driver; its progress goes to the console, its JSON result comes back."""
    p = subprocess.run(['node', str(ROOT / DRIVERS[site]), str(spec_path)], cwd=ROOT,
                       stdout=subprocess.PIPE, text=True, encoding='utf-8')
    lines = [ln for ln in p.stdout.splitlines() if ln.strip().startswith('{')]
    return json.loads(lines[-1]) if lines else {'ok': False, 'error': f'driver gave no result (exit {p.returncode})', 'files': []}


def fake_images(job, spec):
    """--dry: stand-in images instead of the browser, to test the desk without using the sites."""
    files = []
    for i in range(spec['n']):
        out = Path(spec['out_dir']) / f"v{spec['start'] + i}.png"
        if job.get('audio'):               # one second of silence
            out = out.with_suffix('.wav')
            with wave.open(str(out), 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(b'\0\0' * 44100)
            files.append(str(out))
            continue
        if job['recipe'] == 'edit':        # the image it edits, nudged a little, as the generators do
            im = Image.open(job['refs'][0]).convert('RGB')
            im = im.transform(im.size, Image.AFFINE, (1, 0, -7, 0, 1, 5), fillcolor=im.getpixel((0, 0)))
        else:
            size = FORMATS[job['format'] or 'landscape'][1]
            bg = (0, 255, 0) if '#00FF00' in spec['prompt'] else (255, 0, 255) if '#FF00FF' in spec['prompt'] else (205, 190, 150)
            im = Image.new('RGB', size, bg)
            d = ImageDraw.Draw(im)
            w, h = size
            d.ellipse((w * .3, h * .15, w * .7, h * .85), fill=(170, 110, 40), outline=(60, 30, 10), width=12)
            d.text((w * .33, h * .45), job['id'], fill=(255, 255, 255), font=ImageFont.load_default(size=w // 25))
        im.save(out)
        files.append(str(out))
    return files


def process(raw, job, out):
    """Clean one generated image as its recipe says. Returns the processed file (sounds are kept as they are)."""
    if Path(raw).suffix.lower() in AUDIO:
        return Path(raw)
    c = job['clean']
    rgb = np.asarray(Image.open(raw).convert('RGB')).astype(np.float32)
    if not c.get('key'):
        Image.fromarray(prep_png.remove_star(rgb).astype(np.uint8), 'RGB').save(out)
        return out
    rgba, bg = prep_png.clean(rgb, c.get('tol', 40), c.get('bg_top', False), c.get('largest', False))
    if (rgba[..., 3] > 128).mean() > 0.97:
        print(f'  ! {job["id"]}: {Path(raw).name} background (colour {bg.round().astype(int).tolist()}) was not removed')
    if job.get('align_to'):
        try:
            rgba = prep_png.align_to(rgba, np.asarray(Image.open(job['align_to']).convert('RGBA')).astype(np.float32))
        except Exception as e:   # noqa: BLE001 - keep the unaligned image and say so
            print(f'  ! {job["id"]}: could not line it up with {Path(job["align_to"]).name} ({e})')
    Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), 'RGBA').save(out)
    return out


def generate(job, n, dry=False, collect=False):
    """Make n more variants of a job, clean them, save the job. Returns the new variants.
    collect: don't ask again, only save the image the site's open chat already shows (after a timeout)."""
    d = job_dir(job['id'])
    d.mkdir(parents=True, exist_ok=True)
    prompt = ' '.join([job['prompt']] + job['notes'])
    spec = {'prompt': prompt, 'refs': job['refs'], 'n': n, 'start': len(job['variants']) + 1,
            'out_dir': str(d), 'mode': job.get('mode'), 'audio_format': job.get('audio'), 'collect': collect}
    t0 = time.time()
    if dry:
        files = fake_images(job, spec)
    else:
        (d / 'request.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1), 'utf-8')
        res = run_driver(job['site'], d / 'request.json')
        files = res['files']
        if res.get('limit'):
            LIMITED[job['site']] = res['error']
        if not res['ok']:
            print(f"  ! {job['id']}: {res['error']}" + (f"  [screenshot {res['screenshot']}]" if res.get('screenshot') else ''))
    new = []
    for f in files:
        k = len(job['variants']) + 1
        clean = process(Path(f), job, d / f'v{k}_clean.png')
        v = {'k': k, 'raw': Path(f).name, 'clean': clean.name, 'prompt': prompt, 'time': time.strftime('%Y-%m-%d %H:%M')}
        job['variants'].append(v)
        new.append(v)
    save_job(job)
    print(f"  {job['id']}: {len(new)} new variant(s) in {time.time() - t0:.0f} s")
    return new


def remember(job, who, action, text=''):
    """Keep a short history on the job: who picked, asked for a redo or left a note."""
    job.setdefault('log', []).append({'time': time.strftime('%Y-%m-%d %H:%M'), 'who': who, 'action': action, 'text': text})


def add_note(asset_id, text, who='user'):
    """A note for the other side (the user's notes are what Claude reads with `prod.py notes`)."""
    job = load_job(asset_id)
    job.setdefault('inbox', []).append({'time': time.strftime('%Y-%m-%d %H:%M'), 'who': who, 'text': text, 'done': False})
    remember(job, who, 'note', text)
    save_job(job)


def pick(asset_id, k, who='claude'):
    job = load_job(asset_id)
    v = job['variants'][k - 1]
    job['picked'] = k
    remember(job, who, 'pick', f'v{k}')
    save_job(job)
    src = job_dir(asset_id) / v['clean']
    dst = ASSETS / f'{asset_id}{src.suffix}'
    dst.parent.mkdir(parents=True, exist_ok=True)
    if job['clean'].get('split'):
        prep_png.split(np.asarray(Image.open(src).convert('RGBA')).astype(np.float32), str(dst.parent), dst.stem)
    else:
        shutil.copyfile(src, dst)
        print(f'  {asset_id}: v{k} -> {dst.relative_to(ROOT)}')


# ---------- looking at results ----------

def contact_sheet(items, name):
    """items: [(label, image path)] -> one small JPEG to look at many results at once.
    Sounds can't go on a sheet: they are listed instead."""
    for label, p in items:
        if Path(p).suffix.lower() in AUDIO:
            print(f'  sound {label}: {p}')
    items = [(label, p) for label, p in items if Path(p).suffix.lower() not in AUDIO]
    if not items:
        return None
    cell, cols = 300, min(5, len(items))
    rows = math.ceil(len(items) / cols)
    sheet = Image.new('RGB', (cols * cell, rows * (cell + 26)), (34, 34, 34))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    check = Image.new('RGB', (cell - 6, cell - 6), (200, 200, 200))
    cd = ImageDraw.Draw(check)
    for y in range(0, cell, 12):
        for x in range((y // 12) % 2 * 12, cell, 24):
            cd.rectangle((x, y, x + 11, y + 11), fill=(165, 165, 165))
    for i, (label, p) in enumerate(items):
        im = Image.open(p).convert('RGBA')
        im.thumbnail((cell - 6, cell - 6))
        tile = check.copy()
        tile.paste(im, ((tile.width - im.width) // 2, (tile.height - im.height) // 2), im)
        x, y = (i % cols) * cell, (i // cols) * (cell + 26)
        sheet.paste(tile, (x + 3, y + 3))
        draw.text((x + 6, y + cell + 3), label, fill=(235, 235, 235), font=font)
    REVIEW.mkdir(parents=True, exist_ok=True)
    out = REVIEW / f'{name}.jpg'
    sheet.save(out, quality=85)
    return out


def all_jobs(prefixes=()):
    jobs = [json.loads(p.read_text('utf-8')) for p in sorted(JOBS.glob('*/job.json'))]
    return [j for j in jobs if not prefixes or any(j['id'].startswith(x) for x in prefixes)]


def variant_items(job, variants):
    return [(f"{job['id']} v{v['k']}" + (' *' if job['picked'] == v['k'] else ''), job_dir(job['id']) / v['clean'])
            for v in variants]


# ---------- commands ----------

def cmd_gen(a):
    only = set(a.only.split(',')) if a.only else None
    shown = []
    for req in read_requests(a.file):
        if only and req['id'] not in only:
            continue
        job = load_job(req['id'])
        if job and job['picked']:
            continue
        try:
            spec = build(req)
        except Waiting as w:
            print(f"  {req['id']}: waits until {w} is picked")
            continue
        if job and len(job['variants']) >= spec['n']:
            print(f"  {req['id']}: {len(job['variants'])} variants made, waiting for a pick")
            continue
        if spec['site'] in LIMITED and not a.dry:
            print(f"  {req['id']}: skipped, {LIMITED[spec['site']]}")
            continue
        if not job:
            job = dict(spec, variants=[], picked=None, notes=[])
        else:
            job.update({k: spec[k] for k in ('prompt', 'refs', 'clean', 'align_to', 'site', 'mode', 'format', 'audio')})
        new = generate(job, spec['n'] - len(job['variants']), a.dry)
        if spec['n'] == 1 and len(new) == 1:
            pick(job['id'], new[0]['k'])
            job = load_job(job['id'])
        shown += variant_items(job, new)
    out = contact_sheet(shown, time.strftime('gen_%Y%m%d_%H%M%S'))
    print(f'contact sheet: {out}' if out else '' if shown else 'nothing new to make')


def redo(asset_id, note, n=None, who='claude', dry=False):
    """New variants with a correction added to the prompt. Returns the new variants."""
    job = load_job(asset_id)
    if note:
        job['notes'].append(note)
    job['picked'] = None
    remember(job, who, 'redo', note)
    new = generate(job, n or job['n'], dry)
    if (n or job['n']) == 1 and len(new) == 1:
        pick(asset_id, new[0]['k'], who)
    return new


def cmd_redo(a):
    new = redo(a.id, a.note, a.n, dry=a.dry)
    out = contact_sheet(variant_items(load_job(a.id), new), time.strftime('redo_%Y%m%d_%H%M%S'))
    if out:
        print(f'contact sheet: {out}')


def cmd_notes(a):
    """The user's open notes from the asset room; --done <id> marks that asset's notes as handled."""
    for j in all_jobs():
        open_notes = [n for n in j.get('inbox', []) if not n['done'] and n['who'] == 'user']
        if a.done and j['id'] == a.done:
            for n in open_notes:
                n['done'] = True
            save_job(j)
            print(f"{j['id']}: {len(open_notes)} note(s) marked done")
        elif not a.done:
            for n in open_notes:
                print(f"{j['id']} [{n['time']}]: {n['text']}")


def cmd_plan(a):
    for req in read_requests(a.file):
        job = load_job(req['id'])
        if job and job['picked']:
            print(f"{req['id']}: done (v{job['picked']})")
            continue
        try:
            s = build(req)
        except Waiting as w:
            print(f"{req['id']}: waits until {w} is picked")
            continue
        refs = ', '.join(Path(r).name for r in s['refs']) or 'none'
        print(f"{s['id']}: {s['recipe']} on {s['site']}, {s['n']} variant(s), refs: {refs}, {len(s['prompt'])} chars")
        print('   ' + (s['prompt'] if a.full else s['prompt'][:150] + '...'))


def cmd_status(a):
    for j in all_jobs(a.prefix):
        state = f"picked v{j['picked']}" if j['picked'] else 'waiting for a pick' if j['variants'] else 'nothing made yet'
        notes = f", {len(j['notes'])} correction(s)" if j['notes'] else ''
        print(f"{j['id']:<34} {len(j['variants'])} variant(s)  {state}{notes}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('login')
    p = sub.add_parser('plan'); p.add_argument('file'); p.add_argument('--full', action='store_true')
    p = sub.add_parser('gen'); p.add_argument('file'); p.add_argument('--only'); p.add_argument('--dry', action='store_true')
    p = sub.add_parser('sheet'); p.add_argument('prefix', nargs='*')
    p = sub.add_parser('pick'); p.add_argument('id'); p.add_argument('k', type=int)
    p = sub.add_parser('redo'); p.add_argument('id'); p.add_argument('note'); p.add_argument('--n', type=int)
    p.add_argument('--dry', action='store_true')
    p = sub.add_parser('status'); p.add_argument('prefix', nargs='*')
    p = sub.add_parser('collect'); p.add_argument('id')
    p = sub.add_parser('notes'); p.add_argument('--done')
    a = ap.parse_args()
    if a.cmd == 'login':
        subprocess.run(['node', str(ROOT / 'bridge' / 'browser.js'), 'open', 'google', 'openai'], cwd=ROOT)
        print('Sign in inside each window (Gemini + Flow Music with the Google account, ChatGPT with the OpenAI one).')
    elif a.cmd == 'gen':
        cmd_gen(a)
    elif a.cmd == 'plan':
        cmd_plan(a)
    elif a.cmd == 'sheet':
        items = [it for j in all_jobs(a.prefix) for it in variant_items(j, j['variants'])]
        print(f"contact sheet: {contact_sheet(items, time.strftime('sheet_%Y%m%d_%H%M%S'))}")
    elif a.cmd == 'pick':
        pick(a.id, a.k)
    elif a.cmd == 'redo':
        cmd_redo(a)
    elif a.cmd == 'collect':
        job = load_job(a.id)
        new = generate(job, 1, collect=True)
        if job['n'] == 1 and len(new) == 1:
            pick(a.id, new[0]['k'])
        out = contact_sheet(variant_items(load_job(a.id), new), time.strftime('collect_%Y%m%d_%H%M%S'))
        if out:
            print(f'contact sheet: {out}')
    elif a.cmd == 'status':
        cmd_status(a)
    elif a.cmd == 'notes':
        cmd_notes(a)


if __name__ == '__main__':
    main()
