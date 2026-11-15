"""Turn an episode (lines.yaml + vo/*.wav + shots.yaml + picked assets) into a VideoHub film.

  python film.py build <ep>      films/<ep>/ in the VideoHub workspace: index.html, film.js, timeline.js, img/, audio/
  python film.py mix   <ep>      audio/mix.wav = voice + music cues + effects (cues.yaml), music ducked under speech

The film is drawn by film_engine.js (one generic engine for every episode); render with VideoHub or
`node render.js` in the film folder (DUR is set in .videohub/project.json).
"""
import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

ROOT = Path(__file__).parent
HUB = ROOT.parent / 'VideoHub' / 'films'
TEMPLATE = HUB / 'hello-film'
RATE = 48000
LEAD, GAP, TAIL, SCENE_GAP = 0.5, 0.35, 0.7, 0.6     # seconds around spoken lines
ENV_HZ = 25


def ep_dir(ep):
    return ROOT / 'episodes' / ep


def film_dir(ep):
    return HUB / ep.replace('_', '-')


def read_wav(path):
    with wave.open(str(path)) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    a = a.reshape(-1, ch).mean(1) if ch > 1 else a
    if rate != RATE:                                   # linear resample is fine for speech
        a = np.interp(np.arange(int(len(a) * RATE / rate)) * rate / RATE, np.arange(len(a)), a)
    return a


def write_wav(path, a):
    a = np.atleast_2d(a.T).T if a.ndim == 1 else a
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(a.shape[1] if a.ndim == 2 else 1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def envelope(a):
    """Mouth open (1) / shut (0) at ENV_HZ from the loudness of a line."""
    hop = RATE // ENV_HZ
    fr = a[:len(a) // hop * hop].reshape(-1, hop)
    rms = np.sqrt((fr ** 2).mean(1))
    if not len(rms):
        return ''
    thr = max(rms.max() * 0.18, 0.01)
    local = np.convolve(rms, np.ones(5) / 5, 'same')
    o = ((rms > thr) & (rms >= local * 0.95)).astype(int)
    for i in range(1, len(o) - 1):                     # no one-frame flaps
        if o[i - 1] == o[i + 1] != o[i]:
            o[i] = o[i - 1]
    return ''.join(map(str, o))


def crop_figure(src, dst):
    im = Image.open(src).convert('RGBA')
    box = im.getchannel('A').point(lambda v: 255 if v > 20 else 0).getbbox()
    im = im.crop(box)
    im.save(dst)
    return box, im


def face_patch(base, layer, box, dst, part):
    """The eye or mouth of an edited layer, cut from where the face is: the figures face right in profile, so
    the nose tip is the rightmost point of the head. The edit redraws texture everywhere, so a fixed feathered
    window is taken instead of the pixels that differ."""
    a = np.asarray(Image.open(base).convert('RGBA').crop(box))
    b = Image.open(layer).convert('RGBA').crop(box)
    h = a.shape[0]
    rows = range(int(h * 0.04), int(h * 0.17))
    right = [(np.nonzero(a[y, :, 3] > 128)[0].max() if (a[y, :, 3] > 128).any() else 0, y) for y in rows]
    nx, ny = max(right)
    if part == 'mouth_aa':
        x0, y0, x1, y1 = nx - h * 0.07, ny + h * 0.005, nx + 4, ny + h * 0.05
    else:
        x0, y0, x1, y1 = nx - h * 0.075, ny - h * 0.045, nx - h * 0.005, ny - h * 0.005
    x0, y0, x1, y1 = (int(v) for v in (x0, y0, x1, y1))
    w, hh = x1 - x0, y1 - y0
    m = Image.new('L', (w, hh), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(m).ellipse([w * 0.12, hh * 0.12, w * 0.88, hh * 0.88], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(min(w, hh) * 0.12))
    patch = b.crop((x0, y0, x1, y1))
    alpha = np.minimum(np.asarray(m), np.asarray(patch.getchannel('A')))
    patch.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    patch.save(dst)
    return [x0, y0]


def cmd_build(ep):
    lines = yaml.safe_load(open(ep_dir(ep) / 'lines.yaml', encoding='utf-8'))
    shots = yaml.safe_load(open(ep_dir(ep) / 'shots.yaml', encoding='utf-8'))
    prefix = ep.split('_')[0]
    assets = ROOT / 'assets' / prefix
    fd = film_dir(ep)
    (fd / 'img').mkdir(parents=True, exist_ok=True)
    (fd / 'audio').mkdir(exist_ok=True)
    for f in ('render.js', 'package.json'):
        if not (fd / f).exists():
            shutil.copy(TEMPLATE / f, fd / f)
    shutil.copy(ROOT / 'film_engine.js', fd / 'film.js')

    # timing: every shot takes its lines; the voice track is laid out on the way
    by_scene = {}
    for x in lines:
        by_scene.setdefault(x['scene'], []).append(x)
    t, prev_scene, out_shots, out_lines, scenes = 0.0, None, [], [], []
    vo_parts = []
    for sh in shots:
        sc = sh['scene']
        if sc != prev_scene:
            if prev_scene is not None:
                t += SCENE_GAP
            scenes.append({'name': f'{sc}', 't0': round(t, 2)})
            prev_scene = sc
        t0 = t
        cur = t + LEAD if sh.get('lines') else t
        mine = [by_scene[sc].pop(0) for _ in range(sh.get('lines', 0))]
        for x in mine:
            wav = ep_dir(ep) / 'vo' / f"{x['id']}.wav"
            a = read_wav(wav) if wav.exists() else np.zeros(int(RATE * max(1.5, len(x['text']) / 11)))
            vo_parts.append((cur, a))
            out_lines.append({'who': x['speaker'], 't0': round(cur, 3), 't1': round(cur + len(a) / RATE, 3),
                              'env': envelope(a), 'text': x['text']})
            cur += len(a) / RATE + GAP
        t = (cur - GAP + TAIL if mine else t) + sh.get('hold', 0)
        if t - t0 < 1.5:
            t = t0 + 1.5
        out_shots.append({**{k: v for k, v in sh.items() if k not in ('lines',)}, 't0': round(t0, 3), 't1': round(t, 3)})
    left = {s: v for s, v in by_scene.items() if v}
    if left:
        sys.exit(f'lines not used by any shot: ' + ', '.join(f'scene {s}: {len(v)}' for s, v in left.items()))
    dur = round(t + 1.0, 2)
    for i, s in enumerate(scenes):
        s['t1'] = scenes[i + 1]['t0'] if i + 1 < len(scenes) else dur

    # images
    used = set()
    for sh in out_shots:
        used.update([sh['bg']] if sh.get('bg') else sh.get('bgs', []))
    imgs = {}
    for name in sorted(used):
        src = assets / f'{name}.png'
        if not src.exists():
            print(f'  missing {src.name}: drawn as a grey card'); continue
        dst = fd / 'img' / f'{name}.jpg'
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            im = Image.open(src).convert('RGB'); im.thumbnail((2400, 2400)); im.save(dst, quality=90)
        imgs[name] = f'img/{name}.jpg'
    chars = {}
    for who in sorted({c['who'] for sh in out_shots for c in sh.get('chars', [])}):
        src = assets / f'{who}.png'
        if not src.exists():
            print(f'  missing character {who}'); continue
        box, im = crop_figure(src, fd / 'img' / f'{who}.png')
        ch = {'img': f'img/{who}.png', 'w': im.width, 'h': im.height}
        for layer in ('eye_closed', 'mouth_aa'):
            ls = assets / f'{who}_{layer}.png'
            if ls.exists():
                at = face_patch(src, ls, box, fd / 'img' / f'{who}_{layer}.png', layer)
                if at:
                    ch[layer] = {'img': f'img/{who}_{layer}.png', 'at': at}
        chars[who] = ch

    tl = {'dur': dur, 'shots': out_shots, 'lines': out_lines, 'scenes': scenes, 'imgs': imgs, 'chars': chars, 'env_hz': ENV_HZ}
    (fd / 'timeline.js').write_text('window.TL = ' + json.dumps(tl, ensure_ascii=False) + ';\n', encoding='utf-8')
    (fd / 'index.html').write_text(INDEX, encoding='utf-8')
    vo = np.zeros(int(dur * RATE) + RATE)
    for t0, a in vo_parts:
        i = int(t0 * RATE); vo[i:i + len(a)] += a
    write_wav(fd / 'audio' / 'vo.wav', vo)
    if not (fd / 'audio' / 'mix.wav').exists():
        shutil.copy(fd / 'audio' / 'vo.wav', fd / 'audio' / 'mix.wav')
    pj = fd / '.videohub' / 'project.json'
    pj.parent.mkdir(exist_ok=True)
    meta = json.loads(pj.read_text(encoding='utf-8')) if pj.exists() else {'title': ep, 'status': 'in-progress'}
    meta['render'] = {'workers': 2, 'env': {'DUR': str(dur), 'FPS': '24'}}
    pj.write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f'{fd}: {dur / 60:.1f} min, {len(out_shots)} shots, {len(out_lines)} lines, {len(imgs)} images, {len(chars)} characters')


def cmd_mix(ep):
    """vo.wav + music and effect cues (cues.yaml: file, at shot/scene or seconds, gain, fade) -> mix.wav (stereo)."""
    fd = film_dir(ep)
    vo = read_wav(fd / 'audio' / 'vo.wav')
    n = len(vo)
    music, sfx = np.zeros(n), np.zeros(n)
    cues_file = ep_dir(ep) / 'cues.yaml'
    tl = json.loads((fd / 'timeline.js').read_text(encoding='utf-8')[len('window.TL = '):-2])
    for c in (yaml.safe_load(open(cues_file, encoding='utf-8')) if cues_file.exists() else []):
        src = ROOT / c['file']
        if not src.exists():
            print(f"  missing {c['file']}"); continue
        a = read_wav(src) if src.suffix == '.wav' else decode(src)
        start = (tl['shots'][c['shot']]['t0'] if 'shot' in c else c.get('t', 0)) + c.get('offset', 0)
        end = tl['shots'][c['until']]['t1'] if 'until' in c else start + c.get('len', len(a) / RATE)
        a = a[int(c.get('from', 0) * RATE):][:int((end - start) * RATE)]
        fi, fo = int(c.get('fade_in', 1.5) * RATE), int(c.get('fade_out', 2.5) * RATE)
        env = np.ones(len(a))
        env[:fi] = np.linspace(0, 1, min(fi, len(a)))[:len(env[:fi])]
        if fo and len(a) > fo:
            env[-fo:] = np.linspace(1, 0, fo)
        a = a / max(np.abs(a).max(), 1e-6) * 0.9 * c.get('gain', 0.5) * env
        i = int(start * RATE)
        (music if c.get('kind', 'music') == 'music' else sfx)[i:i + len(a)] += a[:max(0, n - i)]
    # duck music under speech: -9 dB while someone talks, smoothed
    talk = np.zeros(n)
    for l in tl['lines']:
        talk[int(l['t0'] * RATE):int(l['t1'] * RATE)] = 1
    k = int(0.25 * RATE)
    c = np.concatenate([[0], np.cumsum(talk)])           # moving average (np.convolve is far too slow here)
    idx = np.arange(n)
    talk = (c[np.minimum(idx + k // 2, n)] - c[np.maximum(idx - k // 2, 0)]) / k
    mix = vo * 1.0 + music * (1 - 0.65 * talk) + sfx
    mix = mix / max(np.abs(mix).max() / 0.95, 1)
    write_wav(fd / 'audio' / 'mix.wav', np.stack([mix, mix], 1))
    print(f"mix.wav: {n / RATE / 60:.1f} min")


def decode(src):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(src), '-ac', '1', '-ar', str(RATE), '-f', 's16le', '-'], capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768


INDEX = '''<!doctype html>
<html lang="hi"><head><meta charset="utf-8"><title>film</title>
<style>html,body{margin:0;background:#000;overflow:hidden}
canvas{display:block;width:100vw;height:56.25vw;max-height:100vh;max-width:177.78vh;margin:auto}</style></head>
<body><canvas id="c" width="1920" height="1080"></canvas>
<script src="timeline.js"></script><script src="film.js"></script></body></html>
'''

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'mix'])
    ap.add_argument('ep')
    a = ap.parse_args()
    {'build': cmd_build, 'mix': cmd_mix}[a.cmd](a.ep)
