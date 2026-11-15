"""Turn an AI image painted on a flat background into a clean transparent PNG.

Only background that touches the image border is removed, so whites and greens inside the
figure (eyes, tilak, leaves) survive. Edge pixels get soft alpha and the background colour
is un-mixed from them, so no green or magenta fringe is left.

    python prep_png.py in.png out.png [--tol 40] [--align base.png]

--align moves/scales the result to sit exactly on base.png (for layers made by editing the
base image, which AI tools tend to shift by a few pixels).
"""
import argparse

import numpy as np
from PIL import Image
from scipy import ndimage


def key(rgb, tol, top_only=False):
    h, w, _ = rgb.shape
    border = rgb[0] if top_only else np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(-1))
    lab, _ = ndimage.label(dist < tol)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bgmask = np.isin(lab, edge[edge > 0])
    # background colour trapped inside the object (between fronds, under branches)
    hl, hn = ndimage.label(dist < tol * 0.8)
    if hn:
        sizes = ndimage.sum(np.ones_like(dist), hl, range(1, hn + 1))
        bgmask |= np.isin(hl, np.nonzero(sizes > 25)[0] + 1)
    # Edge band: pixels that are part figure, part key colour. The key colour's excess
    # ("spill") tells how much background is mixed in: take it out and use it as transparency.
    band = ndimage.binary_dilation(bgmask, iterations=4) & ~ndimage.binary_erosion(bgmask, iterations=1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if bg[1] > bg[0] + 80 and bg[1] > bg[2] + 80:            # green key
        spill = np.maximum(0, g - np.maximum(r, b))
        full = max(bg[1] - max(bg[0], bg[2]), 1)
        clean = np.dstack([r, g - spill, b])
    elif min(bg[0], bg[2]) > bg[1] + 80:                      # magenta key
        spill = np.maximum(0, np.minimum(r, b) - g)
        full = max(min(bg[0], bg[2]) - bg[1], 1)
        clean = np.dstack([r - spill, g, b - spill])
    else:                                                      # any other flat colour
        spill = np.clip(1 - dist / (tol * 2.5), 0, 1) * 255
        full, clean = 255, rgb
    a = np.clip(1 - spill / full, 0, 1)
    alpha = np.where(band, a, np.where(bgmask, 0.0, 1.0))
    out = np.where(band[..., None], clean, rgb)
    if min(bg[0], bg[2]) > bg[1] + 80:
        out = clean   # nothing in our world is magenta, so take the tint out everywhere (thin roots, fronds)
    if bg[1] > bg[0] + 80 and bg[1] > bg[2] + 80:
        # green light bleeds a few pixels into the outline: no green excess near the edge, and the edge pulled in
        # by a pixel, so cut-out parts laid over each other show no green seam
        near = ndimage.binary_dilation(bgmask, iterations=8) & ~bgmask
        g2 = np.minimum(out[..., 1], np.maximum(out[..., 0], out[..., 2]))
        out = np.dstack([out[..., 0], np.where(near, g2, out[..., 1]), out[..., 2]])
        alpha = np.minimum(alpha, ndimage.grey_erosion(alpha, size=(3, 3)) * 0.5 + alpha * 0.5)
    # drop islands (e.g. the generator's watermark): keep pieces at least 3% of the largest
    lab, n = ndimage.label(alpha > 0.5)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(alpha), lab, index=np.arange(1, n + 1))
        small = np.isin(lab, np.nonzero(sizes < sizes.max() * 0.03)[0] + 1)
        alpha = np.where(ndimage.binary_dilation(small, iterations=3), 0.0, alpha)
    return np.clip(out, 0, 255), alpha, bg


def align_to(rgba, base_rgba):
    """Similarity transform (shift, scale, small rotation) that lays the layer on the base."""
    import cv2
    a = (rgba[..., 3] > 0.5).astype(np.uint8) * 255
    b = (base_rgba[..., 3] > 0.5).astype(np.uint8) * 255
    ga = cv2.cvtColor(rgba[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(base_rgba[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(ga, a)
    kb, db = orb.detectAndCompute(gb, b)
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda x: x.distance)[:600]
    src = np.float32([ka[x.queryIdx].pt for x in m])
    dst = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    h, w = base_rgba.shape[:2]
    out = cv2.warpAffine(rgba.astype(np.float32), M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    print(f'aligned with {int(inl.sum())} matching points; scale {np.hypot(M[0, 0], M[1, 0]):.4f}, shift {M[0, 2]:.1f},{M[1, 2]:.1f}')
    return out


def remove_star(rgb):
    """Paint out the generator's star logo in the bottom-right corner (for images not keyed)."""
    import cv2
    h, w, _ = rgb.shape
    box = rgb[h - 260:, w - 260:].astype(np.float32)
    lum = box.mean(-1)
    m = (lum - ndimage.gaussian_filter(lum, 45)) > 22       # the logo is a pale shape brighter than its surroundings
    lab, n = ndimage.label(m)
    if n:                                                   # the logo is the biggest such blob
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)
    m = ndimage.binary_dilation(ndimage.binary_fill_holes(m), iterations=7)
    full = np.zeros((h, w), np.uint8)
    full[h - 260:, w - 260:] = m * 255
    return cv2.inpaint(rgb.astype(np.uint8), full, 9, cv2.INPAINT_TELEA).astype(np.float32)


def split(rgba, outdir, stem):
    """Write each separate object of a sheet as its own trimmed PNG."""
    import os
    os.makedirs(outdir, exist_ok=True)
    lab, n = ndimage.label(ndimage.binary_dilation(rgba[..., 3] > 20, iterations=4))
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if min(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < 150:
            continue    # crumbs and the logo
        piece = rgba[sl].copy()
        piece[..., 3] *= (lab[sl] == k)
        p = f'{outdir}/{stem}_{k}.png'
        Image.fromarray(np.clip(piece, 0, 255).astype(np.uint8), 'RGBA').save(p)
        print('piece', p, piece.shape[1], 'x', piece.shape[0])


def clean(rgb, tol=40, bg_top=False, largest=False):
    """Logo out, flat background keyed out -> float RGBA (0..255)."""
    rgb = remove_star(rgb)
    col, alpha, bg = key(rgb, tol, bg_top)
    if largest:
        lab, n = ndimage.label(alpha > 0.5)
        if n > 1:
            keep = ndimage.binary_dilation(lab == np.argmax(ndimage.sum(alpha > 0.5, lab, range(1, n + 1))) + 1, iterations=3)
            alpha = alpha * keep
    return np.dstack([col, alpha * 255]), bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--tol', type=float, default=40)
    ap.add_argument('--align')
    ap.add_argument('--largest', action='store_true', help='keep only the biggest object (drops ghost copies Gemini sometimes paints)')
    ap.add_argument('--no-key', action='store_true', help='backdrop: keep everything, only paint out the logo')
    ap.add_argument('--split', help='sheet of objects: write each one to this folder')
    ap.add_argument('--bg-top', action='store_true', help='strips: the background colour is read from the top edge only')
    o = ap.parse_args()
    rgb = np.asarray(Image.open(o.src).convert('RGB')).astype(np.float32)
    if o.no_key:
        Image.fromarray(remove_star(rgb).astype(np.uint8), 'RGB').save(o.dst)
        print('logo painted out ->', o.dst)
        return
    rgba, bg = clean(rgb, o.tol, o.bg_top, o.largest)
    alpha = rgba[..., 3] / 255
    if o.split:
        split(rgba, o.split, o.dst)
        return
    if o.align:
        rgba = align_to(rgba, np.asarray(Image.open(o.align).convert('RGBA')).astype(np.float32))
    Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), 'RGBA').save(o.dst)
    print(f'background {bg.round().astype(int).tolist()} removed; {100 * (alpha > 0.5).mean():.1f}% of pixels kept -> {o.dst}')


if __name__ == '__main__':
    main()
