"""Cut a body-parts sheet (ChatGPT) into one PNG per part (system Python).

  python parts.py <sheet.png> <out dir>

The background is whatever colour touches the border (white or flat green); it is removed by flood fill
from the edges, so a white vest or a green sari inside an outline stays. Each separate island becomes a
part, cropped with a margin; out/index.jpg shows their numbers so they can be named in the puppet editor.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

DILATE = 4   # grow before finding islands so a curl or finger gap does not split a part (0 for exploded views)


def background(rgb, tol=38):
    """Pixels connected to the border whose colour is close to the border colour."""
    h, w, _ = rgb.shape
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    near = np.abs(rgb.astype(int) - bg).sum(-1) < tol * 3
    lab, _ = ndimage.label(near)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return np.isin(lab, list(edge)), bg


def run(sheet, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    src = Image.open(sheet)
    rgb = np.asarray(src.convert('RGB'))
    if src.mode == 'RGBA' and np.asarray(src)[..., 3][0].max() < 10:      # already keyed: transparency is the background
        fg = np.asarray(src)[..., 3] > 40
    else:
        bg, bgc = background(rgb)
        fg = ~bg
    fg = ndimage.binary_opening(fg, iterations=1)
    # soft edge: one pixel of partial alpha where the outline meets the background
    alpha = (fg * 255).astype(np.uint8)
    alpha = np.asarray(Image.fromarray(alpha).filter(__import__('PIL.ImageFilter', fromlist=['x']).GaussianBlur(0.7)))
    alpha = np.where(fg, np.maximum(alpha, 200), alpha // 2).astype(np.uint8)
    # islands: grow a little first so a hair curl or a finger gap does not split a part
    lab, n = ndimage.label(ndimage.binary_dilation(fg, iterations=DILATE))
    parts = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = (lab[sl] == i) & fg[sl]
        if m.sum() < 1500:
            continue
        y0, x0 = sl[0].start, sl[1].start
        part = np.dstack([rgb[sl], np.where(m, alpha[sl], 0)])
        parts.append((y0, x0, part, (x0, y0, sl[1].stop, sl[0].stop)))
    # reading order: rows top to bottom, then left to right
    parts.sort(key=lambda p: (round(p[0] / 120), p[1]))
    index = Image.fromarray(rgb).convert('RGB')
    d = ImageDraw.Draw(index)
    meta = []
    for k, (_, _, part, box) in enumerate(parts, 1):
        name = f'p{k:02d}.png'
        Image.fromarray(part, 'RGBA').save(out / name)
        d.rectangle(box, outline=(255, 0, 0), width=3)
        d.text((box[0] + 6, box[1] + 4), str(k), fill=(255, 0, 0))
        meta.append({'file': name, 'box': list(map(int, box))})
    index.save(out / 'index.jpg', quality=85)
    (out / 'parts.json').write_text(json.dumps({'sheet': str(sheet), 'parts': meta}, indent=1), encoding='utf-8')
    print(f'{len(parts)} parts -> {out} (numbers on {out / "index.jpg"})')


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
