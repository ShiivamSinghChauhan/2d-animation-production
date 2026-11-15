"""Claude's side of the character studio: the fixes the user sends with "Send to Claude".

  python fixdesk.py wait                       wait for a new request (run it in the background), print it
  python fixdesk.py list [--all]               open requests (all with --all)
  python fixdesk.py redraw <puppet> <n> "<prompt>" [--keep]
        ChatGPT redraws the request's part from Claude's prompt (images: the part with the user's marks, the whole
        character); the result is keyed and replaces the part in the editor (--keep: only add it to the tries)
  python fixdesk.py add <puppet> <n> <new name> <swap> "<prompt>" [--from clean|part]
        ChatGPT draws a new piece IN PLACE on the request's part picture (default: the clean one); it is keyed,
        cropped and added in the editor as a layer in front of the part, where it was drawn. swap: '' or e.g. eye:half
  python fixdesk.py use <puppet> <part file> <k>      put try k (0 = before ChatGPT) in place
  python fixdesk.py reply <puppet> <n> "<message>" [--done]   the message shows under the request in the editor

A request lives in assets/puppets/<puppet>/requests/r<n>/: request.json (note, status, fit), part.png (marks),
clean.png (no marks), context.png (the whole character). Changes for the open editor go to updates.json; the
editor applies them within a few seconds (or when it is opened next).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import prod
import room

ROOT = prod.ROOT
PUP = ROOT / 'assets' / 'puppets'


def req_dir(pid, n):
    return PUP / pid / 'requests' / f'r{n}'


def load_req(pid, n):
    return json.loads((req_dir(pid, n) / 'request.json').read_text(encoding='utf-8'))


def save_req(r):
    (req_dir(r['puppet'], r['n']) / 'request.json').write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')


def all_reqs():
    return [json.loads(f.read_text(encoding='utf-8')) for f in sorted(PUP.glob('*/requests/r*/request.json'))]


def show(r):
    d = req_dir(r['puppet'], r['n'])
    print(f"{r['puppet']} r{r['n']} [{r['status']}] part {r['file']} ({r['time']}): {r['note']}")
    print(f"  marked part: {d / 'part.png'}\n  whole character: {d / 'context.png'}\n  clean part: {d / 'clean.png'}")


def parts_info(pid):
    f = PUP / pid / 'parts' / 'parts.json'
    return f, json.loads(f.read_text(encoding='utf-8'))


def update(pid, op):
    f = PUP / pid / 'updates.json'
    ups = json.loads(f.read_text(encoding='utf-8')) if f.exists() else []
    op['id'] = max([u['id'] for u in ups] + [0]) + 1
    ups.append(op)
    f.write_text(json.dumps(ups, indent=1), encoding='utf-8')


def make(jid, prompt, refs, fit):
    """One ChatGPT image for job jid, keyed; returns the new variant number (None if nothing came back)."""
    job = prod.load_job(jid) or {'id': jid, 'recipe': 'part_fix', 'site': 'chatgpt', 'mode': None, 'format': None,
                                 'align_to': None, 'audio': None, 'n': 1, 'variants': [], 'picked': None, 'notes': []}
    job.update(prompt=prompt, refs=[str(r) for r in refs], clean={'key': True, 'largest': True}, notes=[])
    prod.remember(job, 'claude', 'fix', prompt[:120])
    prod.save_job(job)
    new = prod.generate(job, 1)
    if not new:
        return None
    job = prod.load_job(jid)
    job['variants'][-1]['fit'] = fit
    prod.save_job(job)
    return new[-1]['k']


def cmd_wait(a):
    while True:
        for r in all_reqs():
            if r['status'] == 'new':
                r['status'] = 'working'
                save_req(r)
                show(r)
                return
        time.sleep(3)


def cmd_list(a):
    for r in all_reqs():
        if a.all or r['status'] != 'done':
            show(r)


def cmd_redraw(a):
    r = load_req(a.puppet, a.n)
    d = req_dir(a.puppet, a.n)
    jid = f"puppets/{a.puppet}/{Path(r['file']).stem}"
    jd = prod.job_dir(jid)
    jd.mkdir(parents=True, exist_ok=True)
    pf = PUP / a.puppet / 'parts' / r['file']
    if not (jd / 'original.png').exists():
        (jd / 'original.png').write_bytes(pf.read_bytes())
    k = make(jid, a.prompt, [d / 'part.png', d / 'context.png'], r['fit'])
    if k is None:
        sys.exit('nothing came back from ChatGPT')
    print(f'try {k}: {jd / prod.load_job(jid)["variants"][k - 1]["clean"]}')
    f, info = parts_info(a.puppet)
    pp = next(q for q in info['parts'] if q['file'] == r['file'])
    if pp.get('job') != jid:
        pp['job'] = jid
        f.write_text(json.dumps(info, indent=1), encoding='utf-8')
    if not a.keep:
        use(a.puppet, r['file'], k)


def use(pid, file, k):
    f, info = parts_info(pid)
    pp = next(q for q in info['parts'] if q['file'] == file)
    res = room.use_fix(f.parent, info, pp, prod.load_job(pp['job']), k)
    if not res.get('ok'):
        sys.exit(res.get('error'))
    update(pid, {'op': 'replace', 'file': file, 'delta': res['delta']})
    print(f'{file}: try {k} in place (moved {res["delta"][0]:.0f}, {res["delta"][1]:.0f} px)')


def cmd_use(a):
    use(a.puppet, a.file, a.k)


def cmd_add(a):
    r = load_req(a.puppet, a.n)
    d = req_dir(a.puppet, a.n)
    name = ''.join(c for c in a.name if c.isalnum() or c in '_-')
    jid = f'puppets/{a.puppet}/{name}'
    fit = dict(r['fit'])
    k = make(jid, a.prompt, [d / f'{a.src}.png'], fit)
    if k is None:
        sys.exit('nothing came back from ChatGPT')
    job = prod.load_job(jid)
    v = job['variants'][k - 1]
    im = Image.open(prod.job_dir(jid) / v['clean']).convert('RGBA')
    W, H, s = fit['W'], fit['H'], fit['k']
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    ys, xs = np.nonzero(np.asarray(im.getchannel('A')) > 20)
    if not len(xs):
        sys.exit('the new piece is empty')
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    cx, cy = ((box[0] + box[2]) / 2 - W / 2) / s, ((box[1] + box[3]) / 2 - H / 2) / s
    piece = im.crop(box)
    piece = piece.resize((max(1, round(piece.width / s)), max(1, round(piece.height / s))), Image.LANCZOS)
    f, info = parts_info(a.puppet)
    piece.save(f.parent / f'{name}.png')
    # the new piece's own frame is centred on this first picture: later tries line up through fit.base
    job['variants'][k - 1]['fit'] = dict(fit, base=[-cx, -cy])
    job['picked'] = k
    prod.save_job(job)
    par = next(q for q in info['parts'] if q['file'] == r['file'])
    base, now = fit['base'], par.get('offset', [0, 0])
    local = [base[0] + cx - now[0], base[1] + cy - now[1]]
    entry = next((q for q in info['parts'] if q['file'] == f'{name}.png'), None)
    if not entry:
        entry = {'file': f'{name}.png', 'name': name}
        info['parts'].append(entry)
    entry.update(box=[0, 0, piece.width, piece.height], job=jid, offset=[0, 0], picked=k)
    f.write_text(json.dumps(info, indent=1), encoding='utf-8')
    update(a.puppet, {'op': 'add', 'file': f'{name}.png', 'name': name, 'parent': r['file'], 'local': local,
                      'swap': a.swap, 'order': a.order})
    print(f'{name}.png added on {r["file"]} at {local[0]:.0f}, {local[1]:.0f} ({piece.width}x{piece.height}): '
          f'{prod.job_dir(jid) / v["clean"]}')


def cmd_reply(a):
    r = load_req(a.puppet, a.n)
    r['reply'] = a.message
    if a.done:
        r['status'] = 'done'
    save_req(r)
    print(f"r{a.n}: {r['status']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('wait').set_defaults(fn=cmd_wait)
    s = sub.add_parser('list'); s.add_argument('--all', action='store_true'); s.set_defaults(fn=cmd_list)
    s = sub.add_parser('redraw'); s.add_argument('puppet'); s.add_argument('n', type=int); s.add_argument('prompt')
    s.add_argument('--keep', action='store_true'); s.set_defaults(fn=cmd_redraw)
    s = sub.add_parser('add'); s.add_argument('puppet'); s.add_argument('n', type=int); s.add_argument('name')
    s.add_argument('swap'); s.add_argument('prompt'); s.add_argument('--from', dest='src', default='clean', choices=['clean', 'part'])
    s.add_argument('--order', type=int, default=0); s.set_defaults(fn=cmd_add)
    s = sub.add_parser('use'); s.add_argument('puppet'); s.add_argument('file'); s.add_argument('k', type=int); s.set_defaults(fn=cmd_use)
    s = sub.add_parser('reply'); s.add_argument('puppet'); s.add_argument('n', type=int); s.add_argument('message')
    s.add_argument('--done', action='store_true'); s.set_defaults(fn=cmd_reply)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
