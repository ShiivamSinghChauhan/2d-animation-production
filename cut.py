"""Editor 2 (production editor) on Claude's side: the episode's edit, episodes/<ep>/edit.json.

  python cut.py first <ep> [--force]     Claude's first cut from lines.yaml + vo/ + shots.yaml + cues.yaml + sound.yaml
  python cut.py show <ep>                the edit as text: tracks, clips, times (what the user changed is in here)
  python cut.py notes <ep> [--all]       the user's notes on clips (open ones; all with --all)
  python cut.py done <ep> <n> "<reply>"  answer note n; the reply shows under the clip in the editor
  python cut.py render <ep>              the final video, drawn by the same code as the editor's preview

The edit: {ep, size, fps, tracks: [{id, kind, name, who?}], clips: [{id, track, start, dur, ...}]}.
Track kinds: shot (background, camera, title), actor (a character on screen), voice (one per speaker), music, sfx.
Sound clips: file, in (seconds cut from the file's start), gain, fade_in, fade_out, prompt, takes [{file, label}].
The editor is http://127.0.0.1:8790/edit?ep=<ep> (room/edit.html, drawing and mixing in room/engine.js).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

import film
import prod

ROOT = Path(__file__).parent


def ep_dir(ep):
    return ROOT / 'episodes' / ep


def edit_file(ep):
    return ep_dir(ep) / 'edit.json'


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def wav_len(f):
    return len(film.read_wav(f)) / film.RATE


def job_takes(jid):
    """Every try of a desk job as clip takes; the picked one first."""
    j = prod.load_job(jid)
    if not j:
        return None, []
    d = prod.job_dir(jid)
    takes = [{'file': rel(d / v['clean']), 'label': f"try {v['k']}" + (' (picked)' if v['k'] == j.get('picked') else '')}
             for v in j['variants'] if (d / v['clean']).exists()]
    return j, takes


def cmd_first(a):
    ep = a.ep
    if edit_file(ep).exists() and not a.force:
        sys.exit(f'{edit_file(ep)} exists (the user may have edited it); --force makes a new first cut')
    d = ep_dir(ep)
    lines = yaml.safe_load(open(d / 'lines.yaml', encoding='utf-8'))
    shots = yaml.safe_load(open(d / 'shots.yaml', encoding='utf-8'))
    voices = yaml.safe_load(open(ROOT / 'bible/voices.yaml', encoding='utf-8'))
    prefix = ep.split('_')[0]
    tracks, clips = [{'id': 'shots', 'kind': 'shot', 'name': 'Shots'}], []
    n = [0]

    def clip(track, start, dur, **kw):
        n[0] += 1
        c = {'id': f'c{n[0]}', 'track': track, 'start': round(start, 3), 'dur': round(dur, 3), **kw}
        clips.append(c)
        return c

    def track(tid, kind, name, **kw):
        if not any(t['id'] == tid for t in tracks):
            tracks.append({'id': tid, 'kind': kind, 'name': name, **kw})

    # same timing as film.py: every shot takes its lines, a gap between lines, a pause between scenes
    by_scene = {}
    for x in lines:
        by_scene.setdefault(x['scene'], []).append(x)
    t, prev, shot_t = 0.0, None, []
    for sh in shots:
        if sh['scene'] != prev and prev is not None:
            t += film.SCENE_GAP
        prev = sh['scene']
        t0, cur = t, t + film.LEAD if sh.get('lines') else t
        mine = [by_scene[sh['scene']].pop(0) for _ in range(sh.get('lines', 0))]
        for x in mine:
            f = d / 'vo' / f"{x['id']}.wav"
            ln = wav_len(f) if f.exists() else max(1.5, len(x['text']) / 11)
            v = voices.get(x['speaker'], {})
            track(f"vo_{x['speaker']}", 'voice', f"Voice · {x['speaker']}", who=x['speaker'])
            takes = [{'file': rel(f), 'label': 'take 1'}] if f.exists() else []
            clip(f"vo_{x['speaker']}", cur, ln, file=rel(f) if f.exists() else None, line=x['id'], text=x['text'],
                 how=x.get('how', ''), voice=f"{v.get('voice', '?')} ({v.get('style', '')})", gain=1.0, fade_in=0, fade_out=0,
                 prompt=f"{x['text']}\n[delivery: {x.get('how') or 'normal'}; voice {v.get('voice', '?')}, {v.get('style', '')}]",
                 takes=takes, **{'in': 0})
            cur += ln + film.GAP
        t = (cur - film.GAP + film.TAIL if mine else t) + sh.get('hold', 0)
        t = max(t, t0 + 1.5)
        shot_t.append((t0, t))
        keep = {k: v for k, v in sh.items() if k in ('scene', 'bg', 'bgs', 'title', 'cam', 'shake')}
        sc = clip('shots', t0, t - t0, **keep)
        jid = f"{prefix}/{sh['bg']}" if sh.get('bg') else None
        j, takes = job_takes(jid) if jid else (None, [])
        if j:
            sc.update(job=jid, prompt=j['prompt'], takes=takes)
        for c in sh.get('chars', []):
            track(f"act_{c['who']}", 'actor', c['who'])
            clip(f"act_{c['who']}", t0, t - t0, who=c['who'],
                 **{k: v for k, v in c.items() if k in ('x', 'x1', 'y', 'h', 'flip', 'shadow', 'puppet')})

    # music and effects: cues.yaml (placed by shot) with their prompts from sound.yaml and tries from the desk
    sound = {s['id'].split('/')[-1]: s for s in (yaml.safe_load(open(d / 'sound.yaml', encoding='utf-8'))
                                                   if (d / 'sound.yaml').exists() else [])}
    track('music', 'music', 'Music')
    nsfx = 0
    for c in (yaml.safe_load(open(d / 'cues.yaml', encoding='utf-8')) if (d / 'cues.yaml').exists() else []):
        src = ROOT / c['file']
        if not src.exists():
            print(f"  missing {c['file']}")
            continue
        start = (shot_t[c['shot']][0] if 'shot' in c else c.get('t', 0)) + c.get('offset', 0)
        end = shot_t[c['until']][1] if 'until' in c else start + c.get('len', wav_len(src) - c.get('from', 0))
        kind = c.get('kind', 'music')
        if kind == 'music':
            tid = 'music'
        else:      # effects that overlap go on separate rows
            busy = {x['track'] for x in clips if x['track'].startswith('sfx') and x['start'] < end and x['start'] + x['dur'] > start}
            tid = next(f'sfx{i}' for i in range(1, 20) if f'sfx{i}' not in busy)
            if not any(tr['id'] == tid for tr in tracks):
                nsfx += 1
                track(tid, 'sfx', f'Effects {nsfx}')
        name = src.stem
        s = sound.get(name, {})
        j, takes = job_takes(f'{prefix}/{name}')
        cur = next((tk['file'] for tk in takes if 'picked' in tk['label']), c['file'])   # the desk's try, so Tries shows which is in use
        clip(tid, start, min(end - start, wav_len(src) - c.get('from', 0)), file=cur, gain=c.get('gain', 0.5),
             fade_in=c.get('fade_in', 1.5), fade_out=c.get('fade_out', 2.5), name=name,
             prompt=(j or {}).get('prompt') or s.get('subject', ''), length=s.get('length', ''),
             takes=takes or [{'file': c['file'], 'label': 'file'}], **{'in': c.get('from', 0)})
    left = {s: v for s, v in by_scene.items() if v}
    if left:
        print('lines not used by any shot: ' + ', '.join(f'scene {s}: {len(v)}' for s, v in left.items()))
    order = {'shot': 0, 'actor': 1, 'voice': 2, 'music': 3, 'sfx': 4}
    tracks.sort(key=lambda tr: order[tr['kind']])
    # pictures: backgrounds by name; still characters cut to their figure, with eye / mouth patches (as film.py)
    assets = ROOT / 'assets' / prefix
    images = {}
    for c in clips:
        for name in ([c['bg']] if c.get('bg') else c.get('bgs', [])):
            if (assets / f'{name}.png').exists():
                images[name] = rel(assets / f'{name}.png')
    chars, cd = {}, d / 'chars'
    for who in sorted({c['who'] for c in clips if c.get('who') and not c.get('puppet')}):
        src = assets / f'{who}.png'
        if not src.exists():
            print(f'  missing character {who}')
            continue
        cd.mkdir(exist_ok=True)
        box, im = film.crop_figure(src, cd / f'{who}.png')
        ch = {'img': rel(cd / f'{who}.png'), 'w': im.width, 'h': im.height}
        for layer in ('eye_closed', 'mouth_aa'):
            ls = assets / f'{who}_{layer}.png'
            if ls.exists():
                at = film.face_patch(src, ls, box, cd / f'{who}_{layer}.png', layer)
                ch[layer] = {'img': rel(cd / f'{who}_{layer}.png'), 'at': at}
        chars[who] = ch
    E = {'ep': ep, 'size': [1920, 1080], 'fps': 24, 'duck': 0.65, 'tracks': tracks, 'clips': clips,
         'images': images, 'chars': chars, 'made': time.strftime('%Y-%m-%d %H:%M')}
    edit_file(ep).write_text(json.dumps(E, ensure_ascii=False, indent=1), encoding='utf-8')
    end = max(c['start'] + c['dur'] for c in clips)
    print(f'{edit_file(ep)}: {end / 60:.1f} min, {len(tracks)} tracks, {len(clips)} clips')


def cmd_show(a):
    E = json.loads(edit_file(a.ep).read_text(encoding='utf-8'))
    for tr in E['tracks']:
        cs = sorted((c for c in E['clips'] if c['track'] == tr['id']), key=lambda c: c['start'])
        print(f"== {tr['name']} ({tr['kind']}, {len(cs)} clips){' MUTED' if tr.get('mute') else ''}")
        for c in cs:
            what = c.get('text') or c.get('title') or c.get('bg') or c.get('name') or c.get('who') or c.get('file') or ''
            extra = f" gain {c['gain']}" if 'gain' in c else ''
            print(f"  {c['id']:>5} {c['start']:8.2f} +{c['dur']:6.2f}{extra}  {str(what)[:70]}")


def notes_file(ep):
    return ep_dir(ep) / 'notes.json'


def cmd_notes(a):
    f = notes_file(a.ep)
    for n in (json.loads(f.read_text(encoding='utf-8')) if f.exists() else []):
        if a.all or n['status'] != 'done':
            print(f"#{n['n']} [{n['status']}] {n['time']} clip {n.get('clip')} ({n.get('what', '')}) at {n.get('t', 0):.1f}s: {n['text']}")
            if n.get('reply'):
                print(f"    Claude: {n['reply']}")


def cmd_done(a):
    f = notes_file(a.ep)
    ns = json.loads(f.read_text(encoding='utf-8'))
    for n in ns:
        if n['n'] == a.n:
            n['status'], n['reply'] = 'done', a.reply
    f.write_text(json.dumps(ns, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'note #{a.n} done')


def cmd_render(a):
    r = subprocess.run(['node', str(ROOT / 'room' / 'render_edit.js'), a.ep], cwd=ROOT)
    sys.exit(r.returncode)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('first'); s.add_argument('ep'); s.add_argument('--force', action='store_true'); s.set_defaults(fn=cmd_first)
    s = sub.add_parser('show'); s.add_argument('ep'); s.set_defaults(fn=cmd_show)
    s = sub.add_parser('notes'); s.add_argument('ep'); s.add_argument('--all', action='store_true'); s.set_defaults(fn=cmd_notes)
    s = sub.add_parser('done'); s.add_argument('ep'); s.add_argument('n', type=int); s.add_argument('reply'); s.set_defaults(fn=cmd_done)
    s = sub.add_parser('render'); s.add_argument('ep'); s.set_defaults(fn=cmd_render)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
