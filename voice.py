"""Voice-over for an episode, made in Google AI Studio's speech playground (bridge/aistudio_tts.js).

  python voice.py lines <ep>            script.md -> lines.yaml (one item per spoken line)
  python voice.py make  <ep> [--only sutradhar,nati]
                                         one run per speaker with all their lines, cut into vo/<line id>.wav
  python voice.py check <ep>            line lengths vs text lengths (finds bad cuts)

Speakers, voices and delivery are in bible/voices.yaml. A speaker's lines go in one request (few runs, the
free quota is small) separated by long pauses; the result is cut at the longest silences.
"""
import argparse
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent
SPEAKER = re.compile(r'^([ऀ-ॿ]+)(?: \(([^)]*)\))?: (.+)$')


def ep_dir(ep):
    return ROOT / 'episodes' / ep


def cmd_lines(ep):
    voices = yaml.safe_load(open(ROOT / 'bible/voices.yaml', encoding='utf-8'))
    names = {v['name']: k for k, v in voices.items()}
    out, scene, cur = [], 0, None
    for raw in open(ep_dir(ep) / 'script.md', encoding='utf-8'):
        line = raw.strip()
        m = re.match(r'^## (\d+)\.', line)
        if m:
            scene, cur = int(m.group(1)), None
            continue
        m = SPEAKER.match(line)
        if m and m.group(1) in names:
            cur = {'id': f's{scene}_{len(out) + 1:03d}', 'scene': scene, 'speaker': names[m.group(1)],
                   'how': m.group(2) or '', 'text': m.group(3)}
            out.append(cur)
        elif cur and line and re.match(r'^[ऀ-ॿ"“]', line):      # verse / continued line
            cur['text'] += ' ' + line
        else:
            cur = None
    for x in out:   # stage directions inside a line: (beat) becomes a pause, the rest is dropped
        x['text'] = re.sub(r'\s*\((beat|pause)\)\s*', ' … ', x['text'])
        x['text'] = re.sub(r'\s*\([A-Za-z ,]+\)\s*', ' ', x['text']).strip()
    yaml.safe_dump(out, open(ep_dir(ep) / 'lines.yaml', 'w', encoding='utf-8'), allow_unicode=True, sort_keys=False, width=200)
    print(f'{len(out)} lines, speakers: ' + ', '.join(f"{s} {sum(x['speaker'] == s for x in out)}" for s in voices))


def read_wav(path):
    with wave.open(str(path)) as w:
        rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    return (a.reshape(-1, ch).mean(1) if ch > 1 else a), rate


def write_wav(path, a, rate):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.clip(a, -32768, 32767).astype(np.int16).tobytes())


def split(a, rate, texts):
    """Cut a take into one piece per text. Cuts go in silences; which silences is chosen so piece lengths
    match the text lengths best (a pause inside a line is shorter than the pause between lines, but not always)."""
    n = len(texts)
    hop = rate // 100                                    # 10 ms frames
    frames = a[:len(a) // hop * hop].reshape(-1, hop).astype(np.float64)
    env = np.convolve(np.sqrt((frames ** 2).mean(1)), np.ones(5) / 5, 'same')
    quiet = env < max(env.max() * 0.02, 60)
    runs, i = [], 0                                     # silent runs: (start, end) in frames
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if i > 0 and j < len(quiet) and j - i >= 20:
                runs.append((i, j))
            i = j
        else:
            i += 1
    loud = np.nonzero(~quiet)[0]
    first, last = (loud[0], loud[-1] + 1) if len(loud) else (0, len(quiet))
    if len(runs) < n - 1:
        raise ValueError(f'found {len(runs) + 1} pieces, expected {n}')
    chars = np.array([max(len(t), 4) for t in texts], float)
    speech = (last - first) - sum(e - s for s, e in runs)
    per = speech / chars.sum()                           # frames per character
    # dp[k][j]: best cost with k cuts, the k-th cut at run j
    pts = [first] + [(s + e) // 2 for s, e in runs] + [last]
    sil = [0] + [e - s for s, e in runs] + [0]
    def cost(i, j, k):                                   # piece k from point i to point j
        d = max(pts[j] - pts[i] - sil[i] / 2 - sil[j] / 2, 1)
        return np.log(d / (chars[k] * per)) ** 2 - 0.6 * min(sil[j], 200) / 100
    m = len(pts)
    INF = 1e18
    dp = np.full((n, m), INF); back = np.zeros((n, m), int)
    for j in range(1, m):
        dp[0][j] = cost(0, j, 0)
    for k in range(1, n):
        for j in range(k + 1, m):
            best = min(range(k, j), key=lambda i: dp[k - 1][i] + cost(i, j, k))
            dp[k][j] = dp[k - 1][best] + cost(best, j, k); back[k][j] = best
    idx, j = [], m - 1
    for k in range(n - 1, 0, -1):
        j = back[k][j]; idx.append(j)
    edges = [first] + [pts[i] for i in reversed(idx)] + [last]
    pieces = []
    for s, e in zip(edges, edges[1:]):
        lo = np.nonzero(~quiet[s:e])[0]
        s2, e2 = (s + lo[0], s + lo[-1] + 1) if len(lo) else (s, e)
        pieces.append((max(0, s2 - 8) * hop, min(len(a), (e2 + 15) * hop)))
    return pieces, [sil[i] / 100 for i in idx]


def cmd_make(ep, only):
    voices = yaml.safe_load(open(ROOT / 'bible/voices.yaml', encoding='utf-8'))
    lines = yaml.safe_load(open(ep_dir(ep) / 'lines.yaml', encoding='utf-8'))
    vo = ep_dir(ep) / 'vo'
    (vo / 'takes').mkdir(parents=True, exist_ok=True)
    for spk, v in voices.items():
        mine = [x for x in lines if x['speaker'] == spk]
        if not mine or (only and spk not in only):
            continue
        if all((vo / f"{x['id']}.wav").exists() for x in mine):
            continue
        take = vo / 'takes' / f'{spk}.wav'
        if not take.exists():
            text = '\n\n[long pause]\n\n'.join(x['text'] for x in mine)
            req = vo / 'takes' / f'{spk}.json'
            json.dump({'text': text, 'voice': v['voice'], 'style': v.get('style'), 'out': str(take)}, open(req, 'w', encoding='utf-8'), ensure_ascii=False)
            print(f'{spk}: {len(mine)} lines, {len(text)} chars -> {v["voice"]}', flush=True)
            p = subprocess.run(['node', str(ROOT / 'bridge/aistudio_tts.js'), str(req)], capture_output=True, text=True, encoding='utf-8')
            res = json.loads([l for l in p.stdout.splitlines() if l.startswith('{')][-1])
            if not res['ok']:
                print(f'  failed: {res}')
                if res.get('limit'):
                    break
                continue
        a, rate = read_wav(take)
        try:
            pieces, gaps = split(a, rate, [x['text'] for x in mine])
        except ValueError as e:
            print(f'  {spk}: {e}; take kept at {take}')
            continue
        for x, (s, e) in zip(mine, pieces):
            write_wav(vo / f"{x['id']}.wav", a[s:e], rate)
        print(f'  {spk}: {len(a) / rate:.0f} s cut into {len(mine)}, shortest cut gap {min(gaps, default=0):.2f} s')
    cmd_check(ep)


def cmd_check(ep):
    lines = yaml.safe_load(open(ep_dir(ep) / 'lines.yaml', encoding='utf-8'))
    total, rows = 0, []
    for x in lines:
        f = ep_dir(ep) / 'vo' / f"{x['id']}.wav"
        if not f.exists():
            continue
        with wave.open(str(f)) as w:
            sec = w.getnframes() / w.getframerate()
        total += sec
        rows.append((sec / max(len(x['text']), 1), x['id'], sec, x['text'][:40]))
    if not rows:
        return print('no voice yet')
    med = sorted(r[0] for r in rows)[len(rows) // 2]
    odd = [r for r in rows if r[0] > med * 1.8 or r[0] < med * 0.5]
    print(f'{len(rows)}/{len(lines)} lines voiced, {total / 60:.1f} min of speech; odd lengths: {len(odd)}')
    for r in odd:
        print(f'  {r[1]} {r[2]:.1f}s  {r[3]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['lines', 'make', 'check'])
    ap.add_argument('ep')
    ap.add_argument('--only')
    a = ap.parse_args()
    {'lines': lambda: cmd_lines(a.ep), 'check': lambda: cmd_check(a.ep),
     'make': lambda: cmd_make(a.ep, set(a.only.split(',')) if a.only else None)}[a.cmd]()
