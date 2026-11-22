"""Asset room: the page where the user and Claude go through what the sites made.

    python room.py          -> http://127.0.0.1:8790 (opens in the browser)

Every asset with its variants: look at the pictures, listen to the music and sound effects, pick one, ask for a
redo with a correction, or leave a note for Claude (Claude reads them with `python prod.py notes`).
Redos run one at a time in the background through the same browser bridge as prod.py.
Only this computer can open the page.
"""
import contextlib
import json
import mimetypes
import queue
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

import prod

PORT = 8790   # 7790 fell inside a port range Windows reserves (netsh int ipv4 show excludedportrange protocol=tcp)
ROOT = prod.ROOT
SERVABLE = (ROOT / 'jobs', ROOT / 'assets', ROOT / 'episodes', ROOT / 'out' / 'edit', ROOT / 'rigs')
LOG = deque(maxlen=300)
TASKS = queue.Queue()
BUSY = {}                      # asset id -> 'waiting' | 'working'


class Tee:
    """Everything the desk prints during a redo goes to the console and to the page's log."""
    def write(self, s):
        sys.__stdout__.write(s)
        for line in s.splitlines():
            if line.strip():
                LOG.append(f"{time.strftime('%H:%M')} {line.strip()}")
        return len(s)

    def flush(self):
        sys.__stdout__.flush()


def worker():
    while True:
        args = TASKS.get()
        aid = args['id']
        BUSY[aid] = 'working'
        try:
            with contextlib.redirect_stdout(Tee()):
                if args.get('kind') == 'fix':
                    print(f'fix {aid}: {args.get("note") or "(edited prompt)"}')
                    made = {v['k'] for v in prod.generate(prod.load_job(aid), 1)}
                    job = prod.load_job(aid)
                    for v in job['variants']:
                        if v['k'] in made:
                            v['fit'] = args['fit']
                    prod.save_job(job)
                    continue
                print(f'redo {aid}: {args.get("note") or "(no correction)"}')
                prod.redo(aid, args.get('note', ''), args.get('n'), who='user')
        except Exception as e:     # noqa: BLE001 - show it on the page, keep the worker alive
            LOG.append(f"{time.strftime('%H:%M')} ! {aid}: {e}")
        finally:
            BUSY.pop(aid, None)


def url_of(p):
    return '/file/' + Path(p).resolve().relative_to(ROOT).as_posix()


def job_view(j):
    d = prod.job_dir(j['id'])
    variants = []
    for v in j['variants']:
        f = d / v['clean']
        if not f.exists():
            continue
        sound = f.suffix.lower() in prod.AUDIO
        variants.append({'k': v['k'], 'kind': 'sound' if sound else 'image', 'url': url_of(f),
                         'thumb': None if sound else '/thumb/' + f.relative_to(ROOT).as_posix(), 'time': v['time']})
    return {'id': j['id'], 'recipe': j['recipe'], 'site': j['site'], 'prompt': j['prompt'], 'corrections': j['notes'],
            'picked': j['picked'], 'n': j['n'], 'inbox': j.get('inbox', []), 'log': j.get('log', [])[-6:],
            'busy': BUSY.get(j['id']), 'variants': variants}


def thumb(rel):
    src = (ROOT / rel).resolve()
    out = src.parent / '.thumbs' / (src.stem + '.png')
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        out.parent.mkdir(exist_ok=True)
        im = Image.open(src)
        if im.mode == 'RGBA':      # crop to the painted area so a part on a big empty canvas is not a dot
            box = im.getchannel('A').point(lambda a: 255 if a > 20 else 0).getbbox()
            if box:
                im = im.crop(box)
        im.thumbnail((520, 520))
        im.save(out)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def local_only(self):
        host = (self.headers.get('Host') or '').split(':')[0]
        if host not in ('127.0.0.1', 'localhost'):
            self.send_error(403)
            return False
        return True

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, f):
        size = f.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):      # audio players seek with byte ranges
            a, _, b = rng[6:].partition('-')
            start = int(a) if a else max(0, size - int(b))
            end = int(b) if a and b else size - 1
        self.send_response(206 if rng else 200)
        self.send_header('Content-Type', mimetypes.guess_type(f.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if rng:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with open(f, 'rb') as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def do_GET(self):
        if not self.local_only():
            return
        path = unquote(urlparse(self.path).path)
        if path == '/':
            return self.send_file(ROOT / 'room' / 'index.html')
        if path == '/favicon.ico':
            self.send_response(204)
            return self.end_headers()
        if path == '/puppet':
            return self.send_file(ROOT / 'room' / 'puppet.html')
        if path == '/api/puppet':          # a cut-out puppet: its parts and the user's layout
            pid = parse_qs(urlparse(self.path).query).get('id', [''])[0]
            d = ROOT / 'assets' / 'puppets' / pid
            if not (d / 'parts' / 'parts.json').exists():
                return self.send_error(404)
            pinfo = json.loads((d / 'parts' / 'parts.json').read_text(encoding='utf-8'))
            parts = pinfo['parts']
            lay = json.loads((d / 'layout.json').read_text(encoding='utf-8')) if (d / 'layout.json').exists() else {}
            for pp in parts:
                f = d / 'parts' / pp['file']
                pp['mtime'] = f.stat().st_mtime if f.exists() else 0
                j = prod.load_job(pp['job']) if pp.get('job') else None
                if j:
                    v = job_view(j)
                    orig = prod.job_dir(j['id']) / 'original.png'
                    pp['review'] = {'variants': v['variants'], 'picked': pp.get('picked', v['picked']), 'busy': v['busy'],
                                    'prompt': j['prompt'], 'corrections': v['corrections'],
                                    'asked': [e['text'] for e in j.get('log', []) if e['action'] == 'fix' and e['text']][-4:],
                                    'original': '/thumb/' + orig.relative_to(ROOT).as_posix() if orig.exists() else None}
            reqs = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((d / 'requests').glob('r*/request.json'),
                                                                               key=lambda f: int(f.parent.name[1:]))]
            for pp in parts:
                pp['requests'] = [{k: q.get(k) for k in ('n', 'note', 'status', 'reply', 'time')}
                                  for q in reqs if q['file'] == pp['file']][-3:]
            ups = json.loads((d / 'updates.json').read_text(encoding='utf-8')) if (d / 'updates.json').exists() else []
            return self.send_json({'dir': f'assets/puppets/{pid}/parts', 'parts': parts, 'layout': lay,
                                   'updates': [u for u in ups if not u.get('applied')],
                                   'reference': pinfo.get('reference')})
        if path == '/char':                # character studio for fixed-skeleton characters (charkit.py)
            return self.send_file(ROOT / 'room' / 'char.html')
        if path == '/skel.js':
            return self.send_file(ROOT / 'room' / 'skel.js')
        if path == '/api/char':            # a character: its template's rig, its parts, the actions for its skeleton
            cid = parse_qs(urlparse(self.path).query).get('id', [''])[0]
            r = char_state(cid)
            return self.send_json(r) if r else self.send_error(404)
        if path == '/api/char/list':
            return self.send_json(sorted(f.parent.name for f in (ROOT / 'assets' / 'chars').glob('*/char.json')))
        if path == '/edit':
            return self.send_file(ROOT / 'room' / 'edit.html')
        if path == '/engine.js':
            return self.send_file(ROOT / 'room' / 'engine.js')
        if path == '/api/edit':            # an episode's edit, the sounds Claude asked for, the user's notes
            ep = parse_qs(urlparse(self.path).query).get('ep', [''])[0]
            r = edit_state(ep)
            return self.send_json(r) if r else self.send_error(404)
        if path == '/api/edit/list':
            return self.send_json(sorted(f.parent.name for f in (ROOT / 'episodes').glob('*/edit.json')))
        if path == '/rig':
            return self.send_file(ROOT / 'room' / 'rig.html')
        if path == '/api/rig/list':        # picked character drawings that can be rigged
            return self.send_json([j['id'] for j in prod.all_jobs()
                                   if j.get('recipe') in ('character', 'character_design') and j.get('picked')])
        if path == '/api/rig':
            aid = parse_qs(urlparse(self.path).query).get('id', [''])[0]
            f = rig_file(aid)
            return self.send_json({'image': f'assets/{aid}.png',
                                   'points': json.loads(f.read_text(encoding='utf-8'))['points'] if f.exists() else {}})
        if path == '/api/state':
            jobs = [job_view(j) for j in prod.all_jobs()]
            return self.send_json({'jobs': jobs, 'log': list(LOG)[-40:], 'queue': TASKS.qsize()})
        if path.startswith('/file/') or path.startswith('/thumb/'):
            rel = path.split('/', 2)[2]
            f = (ROOT / rel).resolve()
            if not any(f.is_relative_to(s) for s in SERVABLE) or not f.is_file():
                return self.send_error(404)
            return self.send_file(thumb(rel) if path.startswith('/thumb/') else f)
        self.send_error(404)

    def do_POST(self):
        if not self.local_only():
            return
        if self.headers.get('X-Room') != '1':      # only the room page itself sends this header
            return self.send_error(403)
        if urlparse(self.path).path == '/api/edit/upload':     # the user's own sound file, dropped on the editor
            q = parse_qs(urlparse(self.path).query)
            ep, name = q.get('ep', [''])[0], Path(q.get('name', [''])[0]).name
            d = ROOT / 'episodes' / ep
            ok = name and Path(name).suffix.lower() in ('.wav', '.mp3', '.ogg', '.m4a', '.flac', '.png', '.jpg', '.jpeg')
            if '..' in ep or not (d / 'edit.json').exists() or not ok:
                return self.send_error(404)
            (d / 'media').mkdir(exist_ok=True)
            f = d / 'media' / name
            i = 1
            while f.exists():
                i += 1
                f = d / 'media' / f'{Path(name).stem}_{i}{Path(name).suffix}'
            f.write_bytes(self.rfile.read(int(self.headers.get('Content-Length') or 0)))
            LOG.append(f"{time.strftime('%H:%M')} {ep}: added {f.name}")
            return self.send_json({'ok': True, 'file': f.relative_to(ROOT).as_posix()})
        data = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if urlparse(self.path).path == '/api/char':            # the user's nudges to a character's parts
            d = ROOT / 'assets' / 'chars' / str(data.get('id', ''))
            if '..' in str(data.get('id', '')) or not (d / 'char.json').exists():
                return self.send_error(404)
            info = json.loads((d / 'char.json').read_text(encoding='utf-8'))
            info['deltas'] = {k: {kk: float(vv) for kk, vv in v.items() if kk in ('x', 'y', 'r', 's')}
                              for k, v in (data.get('deltas') or {}).items()}
            (d / 'char.json').write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
            return self.send_json({'ok': True})
        if urlparse(self.path).path == '/api/puppet/ask':      # a fix for Claude: marks + words on one part
            return self.send_json(ask_claude(data))
        if urlparse(self.path).path == '/api/puppet/applied':  # the editor has applied Claude's changes
            d = ROOT / 'assets' / 'puppets' / data['id']
            if '..' in data['id'] or not (d / 'updates.json').exists():
                return self.send_error(404)
            ups = json.loads((d / 'updates.json').read_text(encoding='utf-8'))
            for u in ups:
                if u['id'] in data['ids']:
                    u['applied'] = True
            (d / 'updates.json').write_text(json.dumps(ups, indent=1), encoding='utf-8')
            return self.send_json({'ok': True})
        if urlparse(self.path).path == '/api/puppet/fix':      # redraw one part with ChatGPT from the user's marks
            r = fix_job(data)
            return self.send_json(r, 200 if r.get('ok') else 409)
        if urlparse(self.path).path == '/api/puppet/use':      # take variant k of a part's ChatGPT job
            d = ROOT / 'assets' / 'puppets' / data['id'] / 'parts'
            info = json.loads((d / 'parts.json').read_text(encoding='utf-8'))
            pp = next((q for q in info['parts'] if q['file'] == data['file']), None)
            if not pp or not pp.get('job') or '..' in data['id']:
                return self.send_error(404)
            job = prod.load_job(pp['job'])
            if job.get('recipe') == 'part_fix':
                return self.send_json(use_fix(d, info, pp, job, int(data['k'])))
            with contextlib.redirect_stdout(Tee()):
                prod.pick(pp['job'], int(data['k']), who='user')
            import numpy as np
            ref = Image.open(ROOT / info['reference'])
            im = Image.open(ROOT / 'assets' / f"{pp['job']}.png").convert('RGBA')
            if im.size != ref.size:
                im = im.resize(ref.size, Image.LANCZOS)
            ys, xs = np.nonzero(np.asarray(im.getchannel('A')) > 20)
            box = [int(xs.min()) - 2, int(ys.min()) - 2, int(xs.max()) + 3, int(ys.max()) + 3]
            im.crop(box).save(d / pp['file'])
            pp['box'] = box
            (d / 'parts.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
            return self.send_json({'ok': True, 'box': box})
        if urlparse(self.path).path == '/api/puppet/part':     # a new part made in the editor (a cut)
            import base64
            d = ROOT / 'assets' / 'puppets' / data['id'] / 'parts'
            name = ''.join(c for c in data['name'] if c.isalnum() or c in '_-')
            if not d.is_dir() or '..' in data['id'] or not name:
                return self.send_error(404)
            (d / f'{name}.png').write_bytes(base64.b64decode(data['png'].split(',', 1)[1]))
            info = json.loads((d / 'parts.json').read_text(encoding='utf-8'))
            if not any(p['file'] == f'{name}.png' for p in info['parts']):
                info['parts'].append({'file': f'{name}.png', 'name': name, 'box': data.get('box', [0, 0, 1, 1])})
            (d / 'parts.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
            return self.send_json({'ok': True, 'file': f'{name}.png'})
        if urlparse(self.path).path == '/api/edit':            # the user saved the edit
            f = ROOT / 'episodes' / data['ep'] / 'edit.json'
            if '..' in data['ep'] or not f.exists():
                return self.send_error(404)
            f.write_text(json.dumps(data['edit'], ensure_ascii=False, indent=1), encoding='utf-8')
            return self.send_json({'ok': True})
        if urlparse(self.path).path == '/api/edit/note':       # a note to Claude on a clip
            f = ROOT / 'episodes' / data['ep'] / 'notes.json'
            if '..' in data['ep'] or not f.parent.is_dir():
                return self.send_error(404)
            ns = json.loads(f.read_text(encoding='utf-8')) if f.exists() else []
            n = {'n': max([x['n'] for x in ns] + [0]) + 1, 'clip': data.get('clip'), 'what': data.get('what', ''),
                 't': data.get('t', 0), 'text': data['text'].strip(), 'status': 'new', 'reply': '', 'time': time.strftime('%Y-%m-%d %H:%M')}
            ns.append(n)
            f.write_text(json.dumps(ns, ensure_ascii=False, indent=1), encoding='utf-8')
            LOG.append(f"{time.strftime('%H:%M')} note #{n['n']} for Claude on {data['ep']}: {n['text']}")
            return self.send_json({'ok': True, 'n': n['n']})
        if urlparse(self.path).path == '/api/edit/render':     # the final video, in the background
            ep = data['ep']
            if '..' in ep or not (ROOT / 'episodes' / ep / 'edit.json').exists():
                return self.send_error(404)
            st = ROOT / 'out' / 'edit' / ep / 'render.json'
            if st.exists() and json.loads(st.read_text(encoding='utf-8')).get('state') == 'working':
                return self.send_json({'error': 'already rendering'}, 409)
            st.parent.mkdir(parents=True, exist_ok=True)
            st.write_text(json.dumps({'state': 'working', 'done': 0, 'total': 0}), encoding='utf-8')
            import subprocess
            subprocess.Popen(['node', str(ROOT / 'room' / 'render_edit.js'), ep], cwd=ROOT,
                             stdout=subprocess.DEVNULL, stderr=open(st.parent / 'render_log.txt', 'w'))
            return self.send_json({'ok': True})
        if urlparse(self.path).path == '/api/puppet':
            d = ROOT / 'assets' / 'puppets' / data['id']
            if not d.is_dir() or '..' in data['id']:
                return self.send_error(404)
            (d / 'layout.json').write_text(json.dumps(data['layout'], indent=1), encoding='utf-8')
            LOG.append(f"{time.strftime('%H:%M')} puppet layout saved for {data['id']}")
            return self.send_json({'ok': True})
        aid = data.get('id', '')
        if not prod.load_job(aid):
            return self.send_json({'error': f'no asset {aid}'}, 404)
        if aid in BUSY:
            return self.send_json({'error': f'{aid} is being made right now'}, 409)
        path = urlparse(self.path).path
        if path == '/api/rig':
            f = rig_file(aid)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({'id': aid, 'points': data['points']}, indent=1), encoding='utf-8')
            LOG.append(f"{time.strftime('%H:%M')} rig points saved for {aid} ({len(data['points'])})")
        elif path == '/api/pick':
            with contextlib.redirect_stdout(Tee()):
                prod.pick(aid, int(data['k']), who='user')
        elif path == '/api/note':
            prod.add_note(aid, data['text'].strip())
            LOG.append(f"{time.strftime('%H:%M')} note on {aid}: {data['text'].strip()}")
        elif path == '/api/redo':
            BUSY[aid] = 'waiting'
            TASKS.put({'id': aid, 'note': data.get('note', '').strip(), 'n': data.get('n')})
        else:
            return self.send_error(404)
        self.send_json({'ok': True})


def char_state(cid):
    """A fixed-skeleton character for the studio and the editors: rig + parts + the actions for its skeleton."""
    d = ROOT / 'assets' / 'chars' / cid
    if not cid or '..' in cid or not (d / 'char.json').exists():
        return None
    info = json.loads((d / 'char.json').read_text(encoding='utf-8'))
    if not info.get('parts'):
        return None
    rig = json.loads((ROOT / 'rigs' / info['template'] / 'rig.json').read_text(encoding='utf-8'))
    actions = {}
    for f in sorted((ROOT / 'rigs' / 'actions').glob('*.json')):
        actions[f.stem] = json.loads(f.read_text(encoding='utf-8'))
    mt = max((d / p['file']).stat().st_mtime for p in info['parts'] if (d / p['file']).exists())
    return {'char': info, 'rig': rig, 'actions': actions, 'dir': f'assets/chars/{cid}', 'mtime': int(mt)}


def edit_state(ep):
    """Everything the production editor shows for one episode."""
    import yaml
    d = ROOT / 'episodes' / ep
    if '..' in ep or not (d / 'edit.json').exists():
        return None
    E = json.loads((d / 'edit.json').read_text(encoding='utf-8'))
    prefix = ep.split('_')[0]
    library = []          # the music and effects Claude asked for (prompt, length) and every take made so far
    for s in (yaml.safe_load(open(d / 'sound.yaml', encoding='utf-8')) if (d / 'sound.yaml').exists() else []):
        j = prod.load_job(s['id'])
        jd = prod.job_dir(s['id'])
        takes = [{'file': (jd / v['clean']).relative_to(ROOT).as_posix(), 'label': f"try {v['k']}"}
                 for v in (j or {}).get('variants', []) if (jd / v['clean']).exists()]
        library.append({'id': s['id'], 'name': s['id'].split('/')[-1], 'kind': 'music' if s.get('recipe') == 'music' else 'sfx',
                        'prompt': (j or {}).get('prompt') or s.get('subject', ''), 'length': s.get('length', ''), 'takes': takes})
    media = [f.relative_to(ROOT).as_posix() for f in sorted((d / 'media').glob('*'))] if (d / 'media').is_dir() else []
    notes = json.loads((d / 'notes.json').read_text(encoding='utf-8')) if (d / 'notes.json').exists() else []
    st = ROOT / 'out' / 'edit' / ep / 'render.json'
    puppets = {}
    for c in E['clips']:
        if c.get('puppet') and c['puppet'] not in puppets:
            pd = ROOT / 'assets' / 'puppets' / c['puppet']
            if (pd / 'layout.json').exists():
                puppets[c['puppet']] = {'dir': f"assets/puppets/{c['puppet']}/parts",
                                        'layout': json.loads((pd / 'layout.json').read_text(encoding='utf-8'))}
    return {'edit': E, 'library': library, 'media': media, 'notes': notes, 'puppets': puppets,
            'render': json.loads(st.read_text(encoding='utf-8')) if st.exists() else None, 'prefix': prefix}


PART_WHAT = {'head': 'the head with all its hair', 'chest': 'the upper body (torso), without arms',
             'hips': 'the hips with the lower garment', 'upper': 'upper arm', 'fore': 'forearm', 'hand': 'hand',
             'thigh': 'thigh', 'shin': 'lower leg (shin)', 'foot': 'foot'}


def part_what(name):
    side, _, rest = name.partition('_')
    if side in ('n', 'f') and rest in PART_WHAT:
        return f"the {'near' if side == 'n' else 'far'} {PART_WHAT[rest]}"
    return PART_WHAT.get(name, 'the ' + name.replace('_', ' '))


def fix_job(data):
    """Start a ChatGPT redraw of one puppet part: the part with the user's marks (image 1), the whole character
    with the same marks (image 2), the user's words in the part_fix recipe, or a prompt the user edited."""
    import base64
    pid = data['id']
    d = ROOT / 'assets' / 'puppets' / pid / 'parts'
    if '..' in pid or not (d / 'parts.json').exists():
        return {'error': 'no such puppet'}
    info = json.loads((d / 'parts.json').read_text(encoding='utf-8'))
    pp = next((q for q in info['parts'] if q['file'] == data['file']), None)
    if not pp:
        return {'error': 'no such part'}
    name = pp.get('name') or Path(pp['file']).stem
    jid = f'puppets/{pid}/{Path(pp["file"]).stem}'
    if jid in BUSY:
        return {'error': f'ChatGPT is still working on {name}'}
    jd = prod.job_dir(jid)
    jd.mkdir(parents=True, exist_ok=True)
    if not (jd / 'original.png').exists():
        (jd / 'original.png').write_bytes((d / pp['file']).read_bytes())
    n = len(list(jd.glob('fix*_part.png'))) + 1
    files = []
    for key in ('part_png', 'context_png'):
        f = jd / f"fix{n}_{key.split('_')[0]}.png"
        f.write_bytes(base64.b64decode(data[key].split(',', 1)[1]))
        files.append(str(f))
    note = data.get('note', '').strip()
    prompt = data.get('prompt', '').strip()
    if not prompt:
        r = prod.load_yaml(prod.BIBLE / 'recipes.yaml')['part_fix']
        f = prod.Fields(prod.load_yaml(prod.BIBLE / 'style.yaml'))
        f.update(what=part_what(name), note=note)
        prompt = ' '.join(r['prompt'].format_map(f).split())
    job = prod.load_job(jid) or {'id': jid, 'recipe': 'part_fix', 'site': 'chatgpt', 'mode': None, 'format': None,
                                 'align_to': None, 'audio': None, 'n': 1, 'variants': [], 'picked': None, 'notes': []}
    job.update(prompt=prompt, refs=files, clean={'key': True, 'largest': True}, notes=[])
    prod.remember(job, 'user', 'fix', note)
    prod.save_job(job)
    if pp.get('job') != jid:
        pp['job'] = jid
        (d / 'parts.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    BUSY[jid] = 'waiting'
    TASKS.put({'kind': 'fix', 'id': jid, 'note': note,
               'fit': {'k': data['k'], 'W': data['W'], 'H': data['H'], 'base': pp.get('offset', [0, 0])}})
    LOG.append(f"{time.strftime('%H:%M')} fix {jid}: {note or '(edited prompt)'}")
    return {'ok': True, 'job': jid}


def ask_claude(data):
    """Keep the user's request for Claude: assets/puppets/<id>/requests/r<n>/ with request.json, part.png (the part
    with the marks, on green), clean.png (the same without marks), context.png (the whole character, marks, box)."""
    import base64
    pid = data['id']
    d = ROOT / 'assets' / 'puppets' / pid
    if '..' in pid or not (d / 'parts' / 'parts.json').exists():
        return {'error': 'no such puppet'}
    info = json.loads((d / 'parts' / 'parts.json').read_text(encoding='utf-8'))
    pp = next((q for q in info['parts'] if q['file'] == data['file']), None)
    if not pp:
        return {'error': 'no such part'}
    rd = d / 'requests'
    n = max([int(f.name[1:]) for f in rd.glob('r*')] + [0]) + 1
    out = rd / f'r{n}'
    out.mkdir(parents=True)
    for key in ('part_png', 'clean_png', 'context_png'):
        (out / f"{key.split('_')[0]}.png").write_bytes(base64.b64decode(data[key].split(',', 1)[1]))
    req = {'n': n, 'puppet': pid, 'file': pp['file'], 'name': pp.get('name'), 'note': data.get('note', '').strip(),
           'status': 'new', 'reply': '', 'time': time.strftime('%Y-%m-%d %H:%M'),
           'fit': {'k': data['k'], 'W': data['W'], 'H': data['H'], 'base': pp.get('offset', [0, 0])}}
    (out / 'request.json').write_text(json.dumps(req, ensure_ascii=False, indent=1), encoding='utf-8')
    LOG.append(f"{time.strftime('%H:%M')} request r{n} for Claude on {pid}/{pp['file']}: {req['note']}")
    return {'ok': True, 'n': n}


def use_fix(d, info, pp, job, k):
    """Put try k (0 = the picture from before ChatGPT) in place of the part. Returns how far the new picture's
    centre is from the old one, in the part's own pixels, so the editor keeps it where it was drawn."""
    import numpy as np
    jd = prod.job_dir(job['id'])
    if k == 0:
        im, off = Image.open(jd / 'original.png').convert('RGBA'), [0.0, 0.0]
    else:
        v = job['variants'][k - 1]
        fit = v.get('fit')
        if not fit:
            return {'error': 'this try has no placement'}
        im = Image.open(jd / v['clean']).convert('RGBA')
        W, H, s = fit['W'], fit['H'], fit['k']
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        ys, xs = np.nonzero(np.asarray(im.getchannel('A')) > 20)
        if not len(xs):
            return {'error': 'this try is empty'}
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        im = im.crop(box)
        im = im.resize((max(1, round(im.width / s)), max(1, round(im.height / s))), Image.LANCZOS)
        off = [fit['base'][0] + ((box[0] + box[2]) / 2 - W / 2) / s, fit['base'][1] + ((box[1] + box[3]) / 2 - H / 2) / s]
    im.save(d / pp['file'])
    cur = pp.get('offset', [0.0, 0.0])
    pp['offset'], pp['picked'] = off, k
    (d / 'parts.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    job['picked'] = k or None
    prod.remember(job, 'user', 'pick', f'v{k}' if k else 'original')
    prod.save_job(job)
    return {'ok': True, 'delta': [off[0] - cur[0], off[1] - cur[1]]}


def rig_file(aid):
    """The user's rig marks for a character drawing (room/rig.html)."""
    return ROOT / 'bible' / 'rigs' / (aid.replace('/', '__') + '.json')


def main():
    threading.Thread(target=worker, daemon=True).start()
    # Windows (Hyper-V / WSL) reserves blocks of 100 ports that move when they start: if 8790 is taken, the next
    # free one of these; the port in use is written to out/room_port.txt for the render scripts
    server = None
    for port in (PORT, 9790, 6790, 5790, 4790):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            break
        except OSError as e:
            print(f'  port {port} not free ({e.strerror or e})')
    if server is None:
        sys.exit('no free port for the asset room')
    port = server.server_address[1]
    (ROOT / 'out').mkdir(exist_ok=True)
    (ROOT / 'out' / 'room_port.txt').write_text(str(port), encoding='utf-8')
    url = f'http://127.0.0.1:{port}/'
    print(f'asset room: {url}  (Ctrl+C to stop)')
    if '--no-open' not in sys.argv:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == '__main__':
    main()
