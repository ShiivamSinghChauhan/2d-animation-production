"""Character kit: fixed-skeleton characters. An image model paints a template's grey mannequin as the character; the
painting is cut into the template's parts, so every character shares the skeleton and its actions.

  python charkit.py template <tid>                      build a template: skeleton, grey mannequin, part masks
  python charkit.py dress <char> <tid> <ref.png> "<look>" [--n 1]
                                                         ChatGPT paints the template's mannequin as the character
  python charkit.py cut <char> [--k K] [--from file.png] cut the painting into the template's parts
  python charkit.py reveal <char>                        ChatGPT paints what the near arm hides (arm taken away);
                                                         the next cut uses it for the bodice side, braid and skirt
  python charkit.py hands <char>                         ChatGPT paints the template's sheet of hands (open palm, fist,
                                                         pointing; near and far) as the character's; the next cut makes
                                                         them arm pictures that actions swap in (a wave shows the palm)
  python charkit.py sheet <char>                         contact sheet: painting, parts, re-assembled check
  python charkit.py check <char>                         the risky moments (arms up, head tilt, knee lift, walk, wave)
                                                         in close-up on one sheet: out/chars/<char>/check.jpg

A template (rigs/<tid>/) is one body type + costume + view: rig.json (bones as joint points in the mannequin's
pixels, parts in draw order, which bones bend each soft part, IK chains), mannequin.png (what ChatGPT paints),
masks/<part>.png (each part's whole shape, also the bits hidden under other parts), owner.png (which part is on top).
Every character made from a template keeps its bones, so every action made for the skeleton plays on it.
A character (assets/chars/<id>/) is char.json (template, part pictures and where they sit, per-part nudges) + parts/.
Limbs are capsules whose round ends are centred on the joints, so a turning joint never opens a gap; the lower piece
is drawn over the upper one at elbows, wrists, knees and ankles. Torso and skirt bend as soft meshes (room/skel.js).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
RIGS = ROOT / 'rigs'
CHARS = ROOT / 'assets' / 'chars'
SS = 3                                   # masks are drawn this many times larger, then shrunk (soft edges)
GREEN = (0, 255, 0)
TONE = {'skin': (190, 188, 186), 'cloth': (122, 125, 134), 'hair': (62, 60, 60)}
LINE = (40, 38, 38)


# ---------------------------------------------------------------- shapes (canvas pixels)

def v(p):
    return np.asarray(p, float)


def spline(points, n=24):
    """Closed Catmull-Rom curve through the points -> many points."""
    P = [v(p) for p in points]
    out = []
    for i in range(len(P)):
        p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % len(P)], P[(i + 2) % len(P)]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return out


def circle(c, r):
    return ('circle', v(c), r)


def capsule(a, b, ra, rb):
    """Tapered capsule: round ends of radius ra at a and rb at b."""
    return ('capsule', v(a), v(b), ra, rb)


def ellipse(c, ra, rb, toward):
    """Ellipse centred on c, radius ra along the direction to point `toward`, rb across."""
    return ('ellipse', v(c), ra, rb, v(toward))


def poly(points, smooth=True):
    return ('poly', [v(p) for p in (spline(points) if smooth else points)])


def draw_shape(d, s):
    k = SS
    if s[0] == 'circle':
        c, r = s[1] * k, s[2] * k
        d.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=255)
    elif s[0] == 'capsule':
        a, b, ra, rb = s[1], s[2], s[3], s[4]
        u = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        nrm = np.array([-u[1], u[0]])
        pts = [a + nrm * ra, b + nrm * rb, b - nrm * rb, a - nrm * ra]
        d.polygon([tuple(p * k) for p in pts], fill=255)
        draw_shape(d, circle(a, ra))
        draw_shape(d, circle(b, rb))
    elif s[0] == 'ellipse':
        c, ra, rb, to = s[1], s[2], s[3], s[4]
        u = (to - c) / (np.linalg.norm(to - c) + 1e-9)
        nrm = np.array([-u[1], u[0]])
        pts = [c + u * ra * math.cos(t) + nrm * rb * math.sin(t) for t in np.linspace(0, 2 * math.pi, 120, endpoint=False)]
        d.polygon([tuple(p * k) for p in pts], fill=255)
    elif s[0] == 'poly':
        d.polygon([tuple(p * k) for p in s[1]], fill=255)


def raster(shapes, size):
    """Union of shapes -> boolean mask at SS scale."""
    im = Image.new('L', (size[0] * SS, size[1] * SS), 0)
    d = ImageDraw.Draw(im)
    for s in shapes:
        draw_shape(d, s)
    return np.asarray(im) > 127


def shrink(m):
    """SS-scale boolean mask -> soft 0..1 mask at canvas size."""
    h, w = m.shape
    return m.reshape(h // SS, SS, w // SS, SS).mean(axis=(1, 3)).astype(np.float32)


# ---------------------------------------------------------------- templates

def hand_shapes(wr, tip):
    """A relaxed open hand hanging down, seen from the side: palm, four fingers a little apart, the thumb in front.
    Drawn as separate shapes so the mannequin shows fingers (ChatGPT paints what the mannequin shows)."""
    L = np.linalg.norm(tip - wr)
    u = (tip - wr) / L
    f = np.array([u[1], -u[0]])                  # across the hand, toward the front (+x, the way she faces)
    if f[0] < 0:
        f = -f
    at = lambda a, b: wr + u * a * L + f * b
    palm = poly([at(0.0, -18), at(0.06, -22), at(0.52, -21), at(0.55, 19), at(0.30, 22), at(0.0, 17)])
    fingers = []
    for off, ln, r in ((12.5, 0.40, 7.0), (4.2, 0.45, 7.2), (-4.2, 0.42, 7.0), (-12, 0.34, 6.2)):   # index .. little
        fingers.append(capsule(at(0.48, off), at(0.48 + ln, off * 1.08), r, r * 0.9))
    fingers.append(capsule(at(0.14, 18), at(0.44, 25), 8.5, 7.0))                                  # thumb
    return [circle(wr, 17), palm] + fingers, fingers


def girl_frock_3q():
    """A girl of about 8 in a knee-length frock with short puff sleeves, three-quarter view facing right,
    arms a little away from the body (A pose) so ChatGPT can paint every piece. 1024 x 1536 like ChatGPT's tall images."""
    J = dict(
        root=(512, 1440), root_t=(562, 1440),
        pelvis=(510, 760), waist=(510, 700), belly=(508, 600), neck_base=(506, 470), neck_top=(510, 395), crown=(515, 140),
        # the arm turns at the centre of its round puff sleeve (inside the shoulder, where a real arm turns): the
        # puff turns in place on the shoulder at any angle and the arm comes out of it. (Turning at the top of the
        # shoulder swung the whole puff up into the air above the bodice when the arm went up.)
        n_sh_in=(498, 480), n_shoulder=(437, 483), n_puff=(432, 512), n_elbow=(361, 688), n_wrist=(295, 850), n_tip=(256, 947),
        # the far puff is a ball like the near one, half behind the far side of the bodice: painters show the outer half
        f_sh_in=(516, 478), f_shoulder=(599, 480), f_puff=(608, 505), f_elbow=(673, 684), f_wrist=(739, 846), f_tip=(778, 943),
        n_hip=(470, 770), n_knee=(458, 1030), n_ankle=(452, 1305), n_toe=(552, 1386),
        n_sk0=(470, 735), n_sk1=(430, 1005), f_sk0=(552, 735), f_sk1=(610, 1000),
        f_hip=(552, 770), f_knee=(566, 1028), f_ankle=(574, 1300), f_toe=(674, 1381),
    )
    bones = [
        ('root', None, 'root', 'root_t'), ('hips', 'root', 'pelvis', 'waist'), ('belly', 'hips', 'waist', 'belly'),
        ('chest', 'belly', 'belly', 'neck_base'), ('neck', 'chest', 'neck_base', 'neck_top'), ('head', 'neck', 'neck_top', 'crown'),
    ]
    for s in 'nf':
        bones += [(f'{s}_shoulder', 'chest', f'{s}_sh_in', f'{s}_puff'), (f'{s}_upper', f'{s}_shoulder', f'{s}_puff', f'{s}_elbow'),
                  (f'{s}_fore', f'{s}_upper', f'{s}_elbow', f'{s}_wrist'), (f'{s}_hand', f'{s}_fore', f'{s}_wrist', f'{s}_tip'),
                  (f'{s}_thigh', 'hips', f'{s}_hip', f'{s}_knee'), (f'{s}_shin', f'{s}_thigh', f'{s}_knee', f'{s}_ankle'),
                  (f'{s}_foot', f'{s}_shin', f'{s}_ankle', f'{s}_toe'), (f'skirt_{s}', 'hips', f'{s}_sk0', f'{s}_sk1')]

    def arm(s):
        sh, el, wr, tip = v(J[f'{s}_puff']), v(J[f'{s}_elbow']), v(J[f'{s}_wrist']), v(J[f'{s}_tip'])
        # the puff is round and centred ON the shoulder joint, like every other joint's round end: when the arm turns
        # it turns in place and never uncovers the shoulder; the sleeve's tube goes on down the arm with it
        puff = circle(sh, 42)
        tube = capsule(sh, sh + (el - sh) * 0.40, 33, 31)
        hand, fingers = hand_shapes(wr, tip)
        if s == 'f':
            puff = circle(sh, 40)
        return {
            f'{s}_upper': dict(shape=[capsule(sh, el, 30, 22), puff, tube], zones=[('cloth', [puff, tube])]),
            f'{s}_fore': dict(shape=[capsule(el, wr, 22, 17)]),
            f'{s}_hand': dict(shape=hand, zones=[('skin', [f_]) for f_ in fingers]),
        }

    def leg(s):
        hp, kn, an = v(J[f'{s}_hip']), v(J[f'{s}_knee']), v(J[f'{s}_ankle'])
        ax, ay = an
        foot = poly([(ax - 20, ay - 8), (ax - 27, ay + 38), (ax - 19, ay + 76), (ax + 20, ay + 86), (ax + 70, ay + 88),
                     (ax + 99, ay + 81), (ax + 104, ay + 66), (ax + 72, ay + 47), (ax + 26, ay + 18), (ax + 18, ay - 6)])
        return {
            f'{s}_thigh': dict(shape=[capsule(hp, kn, 46, 33)]),
            f'{s}_shin': dict(shape=[capsule(kn, an, 33, 21)]),
            f'{s}_foot': dict(shape=[circle(an, 21), foot]),
        }

    head = poly([(500, 136), (575, 152), (622, 196), (642, 250), (647, 292), (643, 318), (661, 342), (646, 357),
                 (649, 373), (639, 393), (609, 410), (560, 413), (502, 393), (462, 362), (418, 322), (398, 252),
                 (420, 180), (455, 150)])
    hair = poly([(624, 204), (590, 160), (500, 132), (418, 176), (394, 252), (414, 326), (458, 366), (486, 338),
                 (500, 296), (530, 258), (572, 236), (612, 230)])
    neck = poly([(482, 372), (538, 380), (544, 486), (476, 486)], smooth=False)
    # the near side under the sleeve is never seen in the mannequin: a slim armhole and side seam, so the bodice looks
    # right when the arm lifts and uncovers it. The far side turns away: it ends where the far sleeve's seam is
    bodice = poly([(478, 462), (446, 474), (430, 503), (420, 543), (418, 585), (424, 640), (432, 704), (512, 712),
                   (598, 704), (603, 640), (606, 570), (605, 528), (598, 494), (584, 472), (544, 462), (512, 486)])
    skirt = poly([(432, 682), (512, 676), (598, 682), (640, 790), (690, 900), (728, 985), (690, 1003), (620, 1014),
                  (520, 1018), (420, 1012), (350, 1000), (316, 984), (350, 890), (395, 785)])
    P = {}
    P.update(arm('f'))
    P.update(leg('f'))
    P.update(leg('n'))
    P['skirt'] = dict(shape=[skirt], zones=[('cloth', [skirt])])
    P['torso'] = dict(shape=[neck, bodice], zones=[('cloth', [bodice])])
    P['head'] = dict(shape=[head], zones=[('hair', [hair])])
    P.update(arm('n'))
    order = ['f_upper', 'f_fore', 'f_hand', 'f_thigh', 'f_shin', 'f_foot', 'n_thigh', 'n_shin', 'n_foot',
             'skirt', 'torso', 'head', 'n_upper', 'n_fore', 'n_hand']
    bone_of = {p: p for p in order}
    bone_of.update(skirt='hips', torso='chest', head='head')
    cont = {}
    for s in 'nf':
        for a, b in (('upper', 'fore'), ('fore', 'hand'), ('thigh', 'shin'), ('shin', 'foot')):
            cont.setdefault(f'{s}_{a}', []).append(f'{s}_{b}')
            cont.setdefault(f'{s}_{b}', []).append(f'{s}_{a}')
    # soft parts: which bones bend them, how strongly each pulls, and how much the part's own bone always keeps
    # (a skirt swings a little with the knees but mostly hangs from the hips)
    # the skirt's waistband stays on the hips (under the bodice's edge); lower down the skirt bones take over
    soft = {'skirt': {'bones': {'hips': 1, 'skirt_n': 1, 'skirt_f': 1}, 'floor': 0, 'sigma': 110,
                      'hold': {'bone': 'hips', 'from': 715, 'to': 800}},
            # the neck bends with the head (rig 'spread'): above the collar the bodice's neck follows the neck bone,
            # more and more up to the jaw; the collar itself stays on the chest
            'torso': {'bones': {'hips': 1, 'belly': 1, 'chest': 1, 'neck': 0}, 'floor': 0,
                      'hold': {'bone': 'neck', 'from': 410, 'to': 466}}}
    parts = []
    for name in order:
        d = P[name]
        sp = soft.get(name, {})
        parts.append({'name': name, 'bone': bone_of[name], 'kind': 'mesh' if name in soft else 'rigid',
                      'bones': list(sp.get('bones', {})), 'pull': sp.get('bones', {}), 'floor': sp.get('floor', 0),
                      **({k_: sp[k_] for k_ in ('sigma', 'hold') if k_ in sp}),
                      'cont': cont.get(name, []), 'shape': d['shape'], 'zones': d.get('zones', [])})
    ik = [{'name': f'{s}_leg', 'bones': [f'{s}_thigh', f'{s}_shin'], 'end': f'{s}_foot', 'bend': 1} for s in 'nf']
    ik += [{'name': f'{s}_arm', 'bones': [f'{s}_upper', f'{s}_fore'], 'end': f'{s}_hand', 'bend': -1} for s in 'nf']
    # bones that move by themselves when others move: each half of the skirt swings 45% as far as its thigh
    # a skirt hangs: each half follows its thigh ('k': a step, a knee lifted), but the turn both thighs share (the
    # hips pushed aside, both legs leaning the same way) counts less ('shared' is taken off); never more than 'max'
    follow = [{'bone': f'skirt_{s}', 'of': f'{s}_thigh', 'with': f'{o}_thigh', 'k': 0.45, 'shared': 0.3, 'max': 32}
              for s, o in (('n', 'f'), ('f', 'n'))]
    # each limb is ONE picture that bends smoothly at its joints (not separate pieces pinned together): the painted
    # arm from the shoulder puff to the fingertips, the leg from the hip to the toes. 'root' is the bone the limb
    # hangs from; 'soft' = how far each joint bends (0 at the shoulder: the whole sleeve turns with the arm)
    layers = {
        f'{s}_arm': {'parts': [f'{s}_upper', f'{s}_fore', f'{s}_hand'], 'root': f'{s}_shoulder', 'soft': [0, 22, 16]}
        for s in 'nf'}
    layers.update({
        f'{s}_leg': {'parts': [f'{s}_thigh', f'{s}_shin', f'{s}_foot'], 'root': 'hips', 'soft': [30, 26, 18]}
        for s in 'nf'})
    # follow-through: a bone lags behind its parent's turning (k = how much of the parent's recent turn it takes
    # back), so a hand trails the forearm, a forearm the upper arm, the head the chest
    drag = [{'bone': f'{s}_hand', 'of': f'{s}_fore', 'k': 0.55} for s in 'nf']
    drag += [{'bone': f'{s}_fore', 'of': f'{s}_upper', 'k': 0.3} for s in 'nf']
    drag += [{'bone': 'head', 'of': 'chest', 'k': 0.5}]
    # no shrug: the bodice does not follow the shoulder bones (a lifted shoulder stretched it), so a lifted shoulder
    # would only pull the sleeve off the bodice
    return {'id': 'girl_frock_3q', 'view': '3q', 'facing': 'right', 'size': [1024, 1536], 'tones': 'skin/cloth/hair',
            'follow': follow, 'layers': layers, 'drag': drag, 'shrug': 0,
            # a head turn is shared with the neck (it bends from its base), so the head does not hinge on a stiff neck
            'spread': [{'from': 'head', 'to': 'neck', 'k': 0.45}],
            'about': 'girl of about 8, knee-length frock with short puff sleeves, three-quarter view facing right, A pose',
            'joints': J, 'bones': bones, 'parts': parts, 'ik': ik}


TEMPLATES = {'girl_frock_3q': girl_frock_3q}

# hands the painting does not have: a hanging hand turned up for a wave shows its back and its thumb points the wrong
# way, so an animator draws another hand. Each pose is drawn upright (fingers up), the wrist at the bottom with a short
# piece of wrist below it that fades into the forearm. 'b' runs across the hand toward the thumb.
HAND_POSES = ('open', 'fist', 'point')
HAND_SCALE = 2.4                     # drawn this much bigger than on the character (ChatGPT paints finer detail)


def hand_pose_shapes(pose, w0, L, thumb_dx):
    """Shapes of one mannequin hand, fingers up: (all shapes, zones that get their own outline)."""
    up, across = np.array([0.0, -1.0]), np.array([thumb_dx, 0.0])
    P = lambda a, b: w0 + up * a * L + across * b * L
    stub = capsule(P(-0.32, 0), P(0, 0), 0.165 * L, 0.165 * L)
    zones = []
    if pose == 'open':                                   # palm toward us, fingers a little apart, thumb out
        body = poly([P(0.0, -0.17), P(0.06, -0.20), P(0.50, -0.21), P(0.53, 0.0), P(0.50, 0.20), P(0.20, 0.21),
                     P(0.0, 0.17)])
        for b, ang, ln, r in ((-0.15, -12, 0.33, 0.058), (-0.05, -4, 0.43, 0.064), (0.05, 2, 0.47, 0.066),
                              (0.15, 8, 0.42, 0.064)):                       # little, ring, middle, index
            t = math.radians(ang)
            base = P(0.48, b)
            zones.append(capsule(base, base + (up * math.cos(t) + across * math.sin(t)) * ln * L, r * L, r * L * 0.92))
        zones.append(capsule(P(0.12, 0.17), P(0.40, 0.40), 0.08 * L, 0.068 * L))                        # thumb
    else:                                                # a fist seen from the front: curled fingers, thumb across
        body = poly([P(0, -0.17), P(0.08, -0.21), P(0.42, -0.22), P(0.52, -0.16), P(0.54, 0), P(0.52, 0.16),
                     P(0.42, 0.22), P(0.08, 0.21), P(0, 0.17)])
        curled = (-0.15, -0.05, 0.05) if pose == 'point' else (-0.15, -0.05, 0.05, 0.15)
        for b in curled:
            zones.append(capsule(P(0.28, b), P(0.46, b), 0.052 * L, 0.052 * L))
        if pose == 'point':
            zones.append(capsule(P(0.45, 0.15), P(0.95, 0.17), 0.064 * L, 0.058 * L))                   # index up
        zones.append(capsule(P(0.14, 0.20), P(0.26, -0.06) if pose == 'fist' else P(0.28, 0.0), 0.075 * L, 0.065 * L))
    return [stub, body] + zones, zones


def build_hand_sheet(tid, spec):
    """rigs/<tid>/hands/: sheet.png (grey hands on green for ChatGPT), hands.json (where each one's wrist is), masks."""
    J = spec['joints']
    out = RIGS / tid / 'hands'
    out.mkdir(parents=True, exist_ok=True)
    W, H = spec['size']
    L0 = float(np.linalg.norm(v(J['n_tip']) - v(J['n_wrist'])))
    L = L0 * HAND_SCALE
    canvas = np.zeros((H, W, 3), np.float32) + GREEN
    hands = []
    for row, side in enumerate('nf'):
        # the near hand is her right hand: its thumb toward her face when the arm is up (to the right); the far one
        # her left hand: thumb to the left
        thumb_dx = 1.0 if side == 'n' else -1.0
        for col, pose in enumerate(HAND_POSES):
            w0 = np.array([W * (col + 0.5) / 3, 620 + row * 700], float)
            shapes, zones = hand_pose_shapes(pose, w0, L, thumb_dx)
            m = raster(shapes, (W, H))
            lines = m & ~ndimage.binary_erosion(m, iterations=2 * SS)
            for z in zones:
                zm = raster([z], (W, H)) & m
                lines |= zm & ~ndimage.binary_erosion(zm, iterations=2 * SS)
            a = shrink(m)
            dist = ndimage.distance_transform_edt(a > 0.5)
            sh = 0.74 + 0.26 * np.clip(dist / max(6.0, float(np.percentile(dist[dist > 0], 97))), 0, 1) ** 0.6
            ln = shrink(lines)
            colr = np.array(TONE['skin'], np.float32) * sh[..., None]
            colr = colr * (1 - ln[..., None]) + np.array(LINE, np.float32) * ln[..., None]
            canvas = canvas * (1 - a[..., None]) + colr * a[..., None]
            name = f'{side}_{pose}'
            Image.fromarray((a * 255).astype(np.uint8), 'L').save(out / f'{name}.png')
            hands.append({'name': name, 'side': side, 'pose': pose, 'wrist': w0.tolist(), 'dir': [0, -1], 'len': L})
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), 'RGB').save(out / 'sheet.png')
    (out / 'hands.json').write_text(json.dumps({'scale': HAND_SCALE, 'hands': hands}, indent=1), encoding='utf-8')
    print(f'{out}: {len(hands)} hands -> sheet.png')


def build_template(tid):
    spec = TEMPLATES[tid]()
    W, H = spec['size']
    out = RIGS / tid
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    J = spec['joints']
    full, soft = {}, {}
    for p in spec['parts']:
        full[p['name']] = raster(p['shape'], (W, H))
        soft[p['name']] = shrink(full[p['name']])
    # owner: the part on top at each pixel (0 = background)
    owner = np.zeros((H, W), np.uint8)
    for i, p in enumerate(spec['parts'], 1):
        owner[soft[p['name']] > 0.5] = i
    # the mannequin: each part a grey tone (skin / cloth / hair), darker toward its edges, with outlines on real
    # edges only (none where a limb continues into the next piece)
    canvas = np.zeros((H, W, 3), np.float32) + GREEN
    group = {}
    for p in spec['parts']:
        g = p['name'].split('_')[0] + ('_arm' if p['name'].endswith(('upper', 'fore', 'hand', 'sleeve')) else
                                       '_leg' if p['name'].endswith(('thigh', 'shin', 'foot')) else p['name'])
        group.setdefault(g, []).append(p['name'])
    shade_of, limb_edge = {}, {}
    for g, names_ in group.items():
        u = np.zeros((H, W), bool)
        uf = np.zeros(full[names_[0]].shape, bool)
        for n_ in names_:
            u |= soft[n_] > 0.5
            uf |= full[n_]
        ue = uf & ~ndimage.binary_erosion(uf, iterations=2 * SS)      # the limb's own outline, unbroken at joints
        for n_ in names_:
            limb_edge[n_] = ue
        dist = ndimage.distance_transform_edt(u)
        hw = max(6.0, float(np.percentile(dist[dist > 0], 97))) if (dist > 0).any() else 6.0
        sh = 0.74 + 0.26 * np.clip(dist / hw, 0, 1) ** 0.6
        for n_ in names_:
            shade_of[n_] = sh
    for p in spec['parts']:
        m = full[p['name']]
        tone = np.zeros(m.shape + (3,), np.float32) + TONE['skin']
        lines = np.zeros(m.shape, bool)
        for zone, shapes in p['zones']:
            z = raster(shapes, (W, H)) & m
            tone[z] = TONE[zone]
            lines |= z & ~ndimage.binary_erosion(z, iterations=2 * SS)
        edge = m & ~ndimage.binary_erosion(m, iterations=2 * SS)
        for c in p['cont']:                       # no line where the limb goes on into the next piece
            edge &= ~ndimage.binary_dilation(full[c], iterations=SS)
        lines |= edge | (limb_edge[p['name']] & m)
        t = tone.reshape(H, SS, W, SS, 3).mean(axis=(1, 3))
        ln = shrink(lines)
        a = soft[p['name']]
        col = t * shade_of[p['name']][..., None]
        col = col * (1 - ln[..., None]) + np.array(LINE, np.float32) * ln[..., None]
        canvas = canvas * (1 - a[..., None]) + col * a[..., None]
        Image.fromarray((a * 255).astype(np.uint8), 'L').save(out / 'masks' / f"{p['name']}.png")
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), 'RGB').save(out / 'mannequin.png')
    Image.fromarray(owner, 'L').save(out / 'owner.png')
    # rig.json: bones as head/tail joint points (rest pose), parts in draw order (back to front)
    rig = {k: spec[k] for k in ('id', 'view', 'facing', 'size', 'about', 'ik', 'follow', 'layers', 'drag', 'shrug', 'spread')
           if k in spec}
    rig['bones'] = [{'name': n, 'parent': par, 'x': J[a][0], 'y': J[a][1], 'ex': J[b][0], 'ey': J[b][1]}
                    for n, par, a, b in spec['bones']]
    rig['parts'] = [{k: p[k] for k in ('name', 'bone', 'kind', 'bones', 'pull', 'floor', 'sigma', 'hold', 'cont') if k in p}
                    for p in spec['parts']]
    (out / 'rig.json').write_text(json.dumps(rig, indent=1), encoding='utf-8')
    # preview: the mannequin with its bones
    im = Image.open(out / 'mannequin.png').convert('RGB')
    d = ImageDraw.Draw(im)
    for b in rig['bones']:
        d.line((b['x'], b['y'], b['ex'], b['ey']), fill=(255, 40, 160), width=3)
        d.ellipse((b['x'] - 5, b['y'] - 5, b['x'] + 5, b['y'] + 5), outline=(255, 255, 0), width=2)
    im.save(out / 'preview.png')
    print(f'{out}: {len(rig["bones"])} bones, {len(rig["parts"])} parts -> mannequin.png, preview.png, masks/')
    if all(f'{s_}_{q}' in J for s_ in 'nf' for q in ('wrist', 'tip')):
        build_hand_sheet(tid, spec)


# ---------------------------------------------------------------- dress: ChatGPT paints the mannequin

DRESS_PROMPT = (
    "Image 1 is a grey mannequin for a cut-out animation puppet: {about}. Image 2 is the approved design of the "
    "character: {look}. Paint the mannequin as this character, in exactly the painting style, colours, texture and "
    "line work of image 2: her face (three-quarter view, turned to the right, looking to the right), her hair, her "
    "skin, her clothes and her bare feet. Keep the mannequin's exact outline, pose, proportions, size and position: "
    "every arm, hand, leg and foot, the head and the clothes stay exactly where they are, the same shape and the same "
    "size; nothing moves, bends, turns or changes length. The light grey areas are skin, the medium grey areas are "
    "her clothes, the dark grey area is her hair. Hair that hangs down, like a braid, goes behind her back. Do not add "
    "anything that is not part of the character: no ground, no shadow, no other objects. The background stays one "
    "flat solid pure green colour (#00FF00) filling the whole image. Keep the image the same size and shape as image 1."
)


def cmd_dress(a):
    import prod
    tpl = json.loads((RIGS / a.tid / 'rig.json').read_text(encoding='utf-8'))
    jid = f'chars/{a.char}/dress'
    job = prod.load_job(jid) or {'id': jid, 'recipe': 'dress', 'site': 'chatgpt', 'mode': None, 'format': None,
                                 'align_to': None, 'audio': None, 'n': a.n, 'variants': [], 'picked': None, 'notes': []}
    prompt = DRESS_PROMPT.format(about=tpl['about'], look=a.look.strip().rstrip('.'))
    job.update(prompt=prompt, refs=[str(RIGS / a.tid / 'mannequin.png'), str(Path(a.ref).resolve())],
               clean={'key': True}, notes=[], template=a.tid)
    prod.remember(job, 'claude', 'dress', a.look[:120])
    prod.save_job(job)
    new = prod.generate(job, a.n)
    if not new:          # the site's page sometimes reloads mid-step: one more try before giving up
        print('  nothing came back; trying once more')
        new = prod.generate(prod.load_job(jid), a.n)
    for v_ in new:
        print(f"try {v_['k']}: {prod.job_dir(jid) / v_['clean']}")
    d = CHARS / a.char
    d.mkdir(parents=True, exist_ok=True)
    cj = d / 'char.json'
    info = json.loads(cj.read_text(encoding='utf-8')) if cj.exists() else {}
    info.update(id=a.char, template=a.tid, look=a.look, job=jid)
    cj.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')


REVEAL_PROMPT = (
    "Edit this picture. Remove the girl's arm on the left side of the picture completely: the near arm, with its puff "
    "sleeve, the arm and the hand. Paint what that arm was hiding, exactly as it would look: the left side of her "
    "frock (bodice and skirt) and her braid with the red ribbon bow hanging down her back, continuing naturally in the "
    "same painting style. Keep everything else exactly the same: the same pose, size and position, the same face, "
    "the other arm, the legs and the rest of the frock, the same colours, texture and line work, and the same flat "
    "solid pure green background (#00FF00). Keep the image the same size and shape."
)


def cmd_reveal(a):
    """ChatGPT edits the character's painting: the near arm taken away and what it hid painted in. The cut uses it
    for the parts behind the arm (bodice side, braid, skirt), so they look painted when the arm moves away."""
    import prod
    info = json.loads((CHARS / a.char / 'char.json').read_text(encoding='utf-8'))
    dj = prod.load_job(info['job'])
    k = dj.get('picked') or len(dj['variants'])
    src = prod.job_dir(info['job']) / dj['variants'][k - 1]['raw']
    jid = f'chars/{a.char}/reveal'
    job = prod.load_job(jid) or {'id': jid, 'recipe': 'reveal', 'site': 'chatgpt', 'mode': None, 'format': None,
                                 'align_to': None, 'audio': None, 'n': 1, 'variants': [], 'picked': None, 'notes': []}
    job.update(prompt=REVEAL_PROMPT, refs=[str(src)], clean={'key': True}, notes=[], of=f'{info["job"]} try {k}')
    prod.remember(job, 'claude', 'reveal', f'near arm removed from try {k}')
    prod.save_job(job)
    new = prod.generate(job, 1)
    if not new:
        print('  nothing came back; trying once more')
        new = prod.generate(prod.load_job(jid), 1)
    for v_ in new:
        print(f"reveal try {v_['k']}: {prod.job_dir(jid) / v_['clean']}")
    if new:
        info['reveal'] = jid
        (CHARS / a.char / 'char.json').write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')


HANDS_PROMPT = (
    "Image 1 is a sheet of six grey mannequin hands for a cut-out animation puppet. Image 2 is the character: {look}. "
    "Paint every grey hand as this girl's hand, in exactly the painting style, skin colour, texture and line work of "
    "her hands in image 2. Keep each hand's exact outline, pose, size and position: nothing moves, bends or changes "
    "size, every finger and thumb stays exactly where it is, nothing is added. The top row is her right hand, the "
    "bottom row her left hand. In each row, from left to right: an open hand with the palm facing us (paint the palm "
    "side, with soft palm creases and no fingernails), a closed fist seen from the front (the curled fingers and the "
    "thumb across them), and a fist with the index finger pointing up. Each hand ends at the bottom in a short piece "
    "of wrist; do not draw a cuff, sleeve or bracelet. The background stays one flat solid pure green colour (#00FF00) "
    "filling the whole image. Keep the image the same size and shape as image 1."
)


def cmd_hands(a):
    """ChatGPT paints the template's sheet of hands as the character's hands (the character's painting as the
    reference). The next cut turns them into arm pictures that actions swap in."""
    import prod
    info = json.loads((CHARS / a.char / 'char.json').read_text(encoding='utf-8'))
    sheet = RIGS / info['template'] / 'hands' / 'sheet.png'
    if not sheet.exists():
        sys.exit(f'no hand sheet for {info["template"]}: run  python charkit.py template {info["template"]}')
    dj = prod.load_job(info['job'])
    k = dj.get('picked') or len(dj['variants'])
    ref = prod.job_dir(info['job']) / dj['variants'][k - 1]['raw']
    jid = f'chars/{a.char}/hands'
    job = prod.load_job(jid) or {'id': jid, 'recipe': 'hands', 'site': 'chatgpt', 'mode': None, 'format': None,
                                 'align_to': None, 'audio': None, 'n': 1, 'variants': [], 'picked': None, 'notes': []}
    job.update(prompt=HANDS_PROMPT.format(look=info.get('look', 'the girl').strip().rstrip('.')),
               refs=[str(sheet), str(ref)], clean={'key': True}, notes=[], of=f'{info["job"]} try {k}')
    prod.remember(job, 'claude', 'hands', 'hand sheet painted as the character')
    prod.save_job(job)
    new = prod.generate(job, 1)
    if not new:
        print('  nothing came back; trying once more')
        new = prod.generate(prod.load_job(jid), 1)
    for v_ in new:
        print(f"hands try {v_['k']}: {prod.job_dir(jid) / v_['clean']}")
    if new:
        info['hands'] = jid
        (CHARS / a.char / 'char.json').write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')


def load_hands(info, tid, bone, W, H):
    """The painted hands (charkit.py hands), each placed where the character's own hand is at rest: {side: {pose:
    (rgb, alpha)}} in the painting's pixels, colours matched to the painted hand later. Empty if there are none."""
    if not info.get('hands'):
        return {}
    import prod
    import cv2
    j = prod.load_job(info['hands'])
    if not j or not j['variants']:
        return {}
    spec = json.loads((RIGS / tid / 'hands' / 'hands.json').read_text(encoding='utf-8'))
    k = j.get('picked') or len(j['variants'])
    im = Image.open(prod.job_dir(info['hands']) / j['variants'][k - 1]['clean']).convert('RGBA')
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    masks = {h['name']: np.asarray(Image.open(RIGS / tid / 'hands' / f"{h['name']}.png")).astype(np.float32) / 255
             for h in spec['hands']}
    sil = np.zeros((H, W), np.float32)
    for m_ in masks.values():
        sil = np.maximum(sil, m_)
    rgba = align(np.asarray(im).astype(np.float32), sil)
    r_, g_, b_ = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    rgba[..., 3] = np.where((g_ > 170) & (g_ - np.maximum(r_, b_) > 90), 0, rgba[..., 3])
    out = {}
    for h in spec['hands']:
        hm = ndimage.binary_dilation(masks[h['name']] > 0.3, iterations=8)
        a_ = rgba[..., 3] / 255 * hm
        # the piece of wrist below the hand fades out (it lies over the end of the painted forearm)
        w0 = np.asarray(h['wrist'], float)
        L = float(h['len'])
        yy, xx = np.mgrid[0:H, 0:W]
        below = ((yy - w0[1]) * -h['dir'][1] + (xx - w0[0]) * -h['dir'][0]) / L       # how far down the wrist
        a_ = a_ * np.clip(1 - below / 0.26, 0, 1)
        ys, xs = np.nonzero(a_ > 0.01)
        if not len(xs):
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        pm = np.dstack([rgba[y0:y1, x0:x1, :3] * a_[y0:y1, x0:x1, None], a_[y0:y1, x0:x1]]).astype(np.float32)
        # onto the character: the sheet's wrist on the rest wrist, its fingers along the rest hand bone, its size
        hb = bone[f"{h['side']}_hand"]
        wr = np.array([hb['x'], hb['y']], float)
        d = np.array([hb['ex'] - hb['x'], hb['ey'] - hb['y']], float)
        Lr = float(np.linalg.norm(d))
        sc = Lr / L
        th = math.atan2(d[1], d[0]) - math.atan2(h['dir'][1], h['dir'][0])
        small = cv2.resize(pm, (max(1, round(pm.shape[1] * sc)), max(1, round(pm.shape[0] * sc))), interpolation=cv2.INTER_AREA)
        c_, s_ = math.cos(th), math.sin(th)
        # small's pixel (u, v) <- sheet (x0 + u / sc, y0 + v / sc); painting = wr + R (sheet - w0) * sc
        ox, oy = (x0 - w0[0]) * sc, (y0 - w0[1]) * sc
        M = np.array([[c_, -s_, wr[0] + c_ * ox - s_ * oy], [s_, c_, wr[1] + s_ * ox + c_ * oy]], np.float32)
        warped = cv2.warpAffine(small, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        al = np.clip(warped[..., 3], 0, 1)
        col = np.where(al[..., None] > 1e-3, warped[..., :3] / np.maximum(al[..., None], 1e-3), 0)
        out.setdefault(h['side'], {})[h['pose']] = (col, al)
    print(f"  hands: {info['hands']} try {k} ({', '.join(sorted(f'{s_}_{p_}' for s_ in out for p_ in out[s_]))})")
    return out


def clean_limb(col, a):
    """An arm picture: only the arm itself (bits of line or ribbon lying loose beside it are dropped), with an
    unbroken outline: where the painting's line along its edge stayed with the piece behind it (a dark line on dark
    hair can't be told apart), the line is drawn on in the painting's own line colour."""
    # thin slivers along the edge (a few pixels wide, touching it here and there) and loose bits go: opening keeps the
    # arm and its own outline (a layer on its edge), not strips lying beside it
    reg = ndimage.binary_opening(a > 0.5, iterations=2)
    lab, n = ndimage.label(reg, structure=np.ones((3, 3)))
    if n > 1:
        sizes = ndimage.sum(reg, lab, range(1, n + 1))
        reg = np.isin(lab, np.nonzero(sizes >= 0.05 * sizes.max())[0] + 1)
    a = a * ndimage.binary_dilation(reg, iterations=1)
    lum = col @ np.array([0.3, 0.59, 0.11])
    ring = reg & ~ndimage.binary_erosion(reg, iterations=4) & (lum < 85)
    if ring.sum() < 30:
        return col, a
    oc = np.median(col[ring], axis=0)
    band = reg & ~ndimage.binary_erosion(reg, iterations=2)
    draw = band & ~ndimage.binary_dilation(reg & (lum < 100), iterations=2)
    k = (np.clip(ndimage.gaussian_filter(draw.astype(np.float32), 0.7) * 1.3, 0, 0.9) * reg)[..., None]
    return col * (1 - k) + oc * k, a


def outline_seams(col, a, seam, line_rgb, width=2.2):
    """A sleeve's edge along its seam (where it was sewn onto the bodice in the painting: a thin line or none) gets
    the painting's full outline, along a smoothed edge: when the arm turns, that edge is in the open."""
    reg = a > 0.5
    if not seam.any():
        return col, a
    zone = ndimage.binary_dilation(seam, iterations=4) & ndimage.binary_dilation(reg, iterations=4)
    sm = ndimage.gaussian_filter(reg.astype(np.float32), 3.0)
    soft = np.clip((sm - 0.5) * 4 + 0.5, 0, 1)               # a smooth edge, about a pixel soft
    a = np.where(zone, np.minimum(np.maximum(a, (sm > 0.5) * 1.0), soft), a)
    inner = ndimage.distance_transform_edt(sm > 0.5)
    k = np.clip(width + 0.5 - inner, 0, 1) * (sm > 0.5) * zone
    col = col * (1 - k[..., None]) + np.asarray(line_rgb, np.float32) * k[..., None]
    return col, a


def extend_neck(C, O, A, idx, bone, masks):
    """The neck under the head, for when the head turns: the neck's own outline lines are carried on up as straight
    lines (fitted to the neck seen below the jaw), the skin between copied row by row from the neck seen below,
    stretched to the width. (The template's blocky neck showed a step at the front and an outline across its top.)
    Returns (mask, colours)."""
    H, W = O.shape
    hb = bone['head']
    hy = int(round(hb['y']))
    band = ndimage.binary_dilation((masks['torso'] > 0.5) & (np.arange(H)[:, None] < bone['neck']['y'] - 20),
                                   structure=np.ones((1, 15), bool))
    torso = (O == idx['torso']) & (A > 0.5) & band
    src = [y for y in range(hy + 14, hy + 44) if torso[y].sum() > 20]
    if len(src) < 8:
        return None, None
    xl = np.array([np.nonzero(torso[y])[0].min() for y in src], float)
    xr = np.array([np.nonzero(torso[y])[0].max() for y in src], float)
    fl, fr = np.polyfit(src, xl, 1), np.polyfit(src, xr, 1)
    mask = np.zeros((H, W), bool)
    col = np.zeros((H, W, 3), np.float32)
    lo, hi = min(src), max(src)
    for y in range(hy - 55, hy + 16):
        a, b = np.polyval(fl, y), np.polyval(fr, y)
        x0, x1 = int(np.floor(a)), int(np.ceil(b))
        if x1 - x0 < 8:
            continue
        k = (hy + 16 - y) % (2 * (hi - lo))                     # back and forth over the rows seen below
        ys = lo + (k if k <= hi - lo else 2 * (hi - lo) - k)
        sa, sb = np.polyval(fl, ys), np.polyval(fr, ys)
        xs = np.arange(x0, x1 + 1)
        xsrc = np.clip(np.rint(sa + (xs - a) * (sb - sa) / max(1.0, b - a)).astype(int), 0, W - 1)
        mask[y, xs] = True
        col[y, xs] = C[ys, xsrc]
    return mask, col


def neck_col_alpha(mask, bone):
    """The carried-on neck fades out at its top (always under the head: no edge to see if it peeks out)."""
    H = mask.shape[0]
    top = int(round(bone['head']['y'])) - 55
    return np.broadcast_to(np.clip((np.arange(H)[:, None] - top) / 14.0, 0, 1), mask.shape).astype(np.float32)


def match_colour(col, al, ref_col, ref_al):
    """The new hand's skin in the painted hand's colours (ChatGPT paints a sheet a little lighter or darker)."""
    def skin(c, a):
        lum = c @ np.array([0.3, 0.59, 0.11])
        return c[(a > 0.95) & (lum > 110)]
    s_, t_ = skin(col, al), skin(ref_col, ref_al)
    if len(s_) < 200 or len(t_) < 200:
        return col
    mu_s, sd_s, mu_t, sd_t = s_.mean(0), s_.std(0) + 1, t_.mean(0), t_.std(0) + 1
    return np.clip((col - mu_s) / sd_s * sd_t + mu_t, 0, 255)


# ---------------------------------------------------------------- cut: painting -> template parts

def align(rgba, sil):
    """Line the painting up with the mannequin by its outline. Only the body is compared (a braid or a bow outside
    the mannequin must not pull it), and the result is used only if it fits better than the painting as it came."""
    import cv2
    near = ndimage.binary_dilation(sil > 0.5, iterations=24)
    a = (rgba[..., 3] > 127) & near
    b = sil > 0.5
    iou = lambda m: (m & b).sum() / max(1, (m | b).sum())
    before = iou(a)
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(cv2.GaussianBlur(b.astype(np.float32), (0, 0), 4), cv2.GaussianBlur(a.astype(np.float32), (0, 0), 4),
                                    M, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        print(f'  outline fit {before:.3f}; kept as it came')
        return rgba
    h, w = sil.shape
    out = cv2.warpAffine(rgba.astype(np.float32), M, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=(0, 0, 0, 0))
    after = iou((out[..., 3] > 127) & near)
    if after > before + 0.004:
        print(f'  lined up: outline fit {before:.3f} -> {after:.3f} (scale {math.hypot(M[0, 0], M[1, 0]):.3f}, shift {M[0, 2]:.1f},{M[1, 2]:.1f})')
        return out
    print(f'  outline fit {before:.3f}; kept as it came (moving it gave {after:.3f})')
    return rgba


def fill_hidden(rgb, known, todo, band=12, axis=None):
    """Colours for the bits of a piece the painting doesn't show (under another piece): grown out from what it does
    show, a band at a time, each band a mirror image of the painted texture next to it, so it reads as the same cloth or
    skin instead of a smear."""
    todo = todo & ~known
    if not todo.any() or not known.any():
        return rgb
    if axis is not None:
        a0, u = np.asarray(axis[0], float), np.asarray(axis[1], float)
        py, px = np.nonzero(todo)
        dx, dy = px - a0[0], py - a0[1]
        t = dx * u[0] + dy * u[1]
        mx = np.rint(a0[0] + 2 * t * u[0] - dx).astype(int)
        my = np.rint(a0[1] + 2 * t * u[1] - dy).astype(int)
        H_, W_ = known.shape
        ok = (mx >= 0) & (mx < W_) & (my >= 0) & (my < H_)
        ok[ok] &= known[my[ok], mx[ok]]
        rgb = rgb.copy()
        rgb[py[ok], px[ok]] = rgb[my[ok], mx[ok]]
        known = known.copy()
        known[py[ok], px[ok]] = True
        todo = todo.copy()
        todo[py[ok], px[ok]] = False
        if not todo.any():
            return rgb
    ys, xs = np.nonzero(todo | known)
    y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
    out = rgb[y0:y1, x0:x1].copy()
    kn, td = known[y0:y1, x0:x1].copy(), todo[y0:y1, x0:x1].copy()
    first = td.copy()
    H, W = kn.shape
    while td.any():
        dist, (iy, ix) = ndimage.distance_transform_edt(~kn, return_indices=True)
        bnd = td & (dist <= dist[td].min() + band)
        py, px = np.nonzero(bnd)
        qy, qx = iy[py, px], ix[py, px]
        sy, sx = np.clip(2 * qy - py, 0, H - 1), np.clip(2 * qx - px, 0, W - 1)
        ok = kn[sy, sx]
        out[py, px] = np.where(ok[:, None], out[sy, sx], out[qy, qx])
        kn |= bnd
        td &= ~bnd
    soft = np.stack([ndimage.gaussian_filter(out[..., c], 0.8) for c in range(3)], -1)
    out[first] = soft[first]
    res = rgb.copy()
    res[y0:y1, x0:x1] = out
    return res


def outline_hidden(col, rgb, m, vis, hidden, cont_masks):
    """Where a filled-in bit reaches the piece's own edge, draw the painting's line there, so a thigh or the side of
    the bodice that comes into view has an outline like the rest of the drawing (not at joints: they have none)."""
    if not hidden.any():
        return col
    inside = m > 0.5
    band = inside & ~ndimage.binary_erosion(inside, iterations=3)
    draw = band & ndimage.binary_dilation(hidden, iterations=2) & ~vis
    for cm in cont_masks:
        draw &= ~ndimage.binary_dilation(cm > 0.5, iterations=3)
    if not draw.any():
        return col
    luma = rgb @ np.array([0.3, 0.59, 0.11])
    ring = vis & (luma < 85) & ~ndimage.binary_erosion(vis, iterations=5)
    oc = np.median(rgb[ring], axis=0) if ring.sum() > 20 else np.array([45.0, 30.0, 20.0])
    a = np.clip(ndimage.gaussian_filter(draw.astype(np.float32), 0.7) * 1.4, 0, 0.9)[..., None]
    return col * (1 - a) + oc * a


def load_reveal(info, size, base):
    """The reveal painting (charkit.py reveal) as RGBA lined up with the painting, or None."""
    if not info.get('reveal'):
        return None
    import prod
    j = prod.load_job(info['reveal'])
    if not j or not j['variants']:
        return None
    k = j.get('picked') or len(j['variants'])
    im = Image.open(prod.job_dir(info['reveal']) / j['variants'][k - 1]['clean']).convert('RGBA')
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    r = np.asarray(im).astype(np.float32)
    sys.path.insert(0, str(ROOT / 'blender'))
    import prep_png
    try:
        r = prep_png.align_to(r, base)            # an edit of the same picture: its features line up
    except Exception as e:                        # noqa: BLE001
        print(f'  ! could not line up the reveal ({e}); using it as it came')
    r_, g_, b_ = r[..., 0], r[..., 1], r[..., 2]
    r[..., 3] = np.where((g_ > 170) & (g_ - np.maximum(r_, b_) > 90), 0, r[..., 3])
    print(f'  using the reveal: {info["reveal"]} try {k}')
    return r


def complete_rope(region, rgb, fg, owner, me):
    """A hanging braid seen only in part. Row by row: where both its edges are seen, its centre and half-width are
    known; where one edge is hidden (under the sleeve or arm), the seen edge and the half-width from nearby rows
    give the centre; rows hidden altogether take both from the rows around them. Centre and half-width are
    smoothed along the braid, and the braid is rebuilt from them, only where something covered it in the
    painting (never into empty background). Returns (completed mask, hair mask)."""
    luma = rgb @ np.array([0.3, 0.59, 0.11])
    ribbon = (rgb[..., 0] > rgb[..., 1] + 60) & (rgb[..., 0] > rgb[..., 2] + 60)      # the red bow is not hair
    hair = ndimage.binary_closing(region & ~ribbon, iterations=2) & region & ~ribbon
    rows = np.nonzero(hair.any(1))[0]
    if len(rows) < 10:
        return region, hair, None
    H, W = region.shape
    covered = fg & (owner > 0) & (owner != me)
    y0, y1 = int(rows.min()), int(rows.max())
    n = y1 - y0 + 1
    cx = np.full(n, np.nan); hw = np.full(n, np.nan)
    edge_l = np.full(n, np.nan); edge_r = np.full(n, np.nan)
    for y in rows:
        xs = np.nonzero(hair[y])[0]
        xl, xr = int(xs.min()), int(xs.max())
        # an edge is seen only where empty background is next to it (next to another part or the ribbon, the
        # braid goes on underneath)
        lo = xl > 2 and not fg[y, max(0, xl - 3):xl].any()
        ro = xr < W - 3 and not fg[y, xr + 1:xr + 4].any()
        i = y - y0
        if lo and ro:
            cx[i], hw[i] = (xl + xr) / 2, (xr - xl) / 2
        if lo:
            edge_l[i] = xl
        if ro:
            edge_r[i] = xr
    idx_ = np.arange(n)

    def fill_nan(v):
        ok = ~np.isnan(v)
        if not ok.any():
            return None
        return np.interp(idx_, idx_[ok], v[ok])
    hw_f = fill_nan(hw)
    if hw_f is None:
        return region, hair, None
    hw_f = ndimage.gaussian_filter1d(hw_f, 8)
    # centre: from both edges where seen, else from one edge and the half-width
    c = cx.copy()
    only_l = np.isnan(c) & ~np.isnan(edge_l)
    only_r = np.isnan(c) & ~np.isnan(edge_r)
    c[only_l] = edge_l[only_l] + hw_f[only_l]
    c[only_r] = edge_r[only_r] - hw_f[only_r]
    c = fill_nan(c)
    c = ndimage.gaussian_filter1d(c, 6)
    # below the ribbon the braid ends in a tip: its width runs down from the width at the ribbon to a point
    rib_rows = np.nonzero((region & ribbon).any(1))[0]
    if len(rib_rows):
        yb = int(np.percentile(rib_rows, 50))
        if y0 < yb < y1:
            ib = yb - y0
            t = np.clip((np.arange(n) - ib) / max(1, n - 1 - ib), 0, 1)
            hw_f = np.where(np.arange(n) > ib, hw_f[ib] * (1 - t) + 3 * t, hw_f)
    out = region.copy()
    X = np.arange(W)
    for i in range(n):
        y = y0 + i
        seg = np.abs(X - c[i]) <= hw_f[i]
        out[y] |= seg & covered[y]
    out = ndimage.binary_opening(out, iterations=1) | region
    return out, hair, (y0, c, hw_f)


def braid_texture(rgb, known, todo, axis):
    """Paint the hidden stretches of a braid with its own plaits: a braid repeats along its length, so each hidden
    pixel takes the colour from the same place across the braid one or more plait-lengths higher up (or lower down),
    where the braid is painted. Pixels with no painted twin are left for the plain fill."""
    if axis is None or not todo.any():
        return rgb
    y0, c, hw = axis
    n = len(c)
    out = rgb.copy()
    H, W = known.shape
    period = max(12, int(np.median(hw) * 1.6))
    ys, xs = np.nonzero(todo)
    done = np.zeros(len(ys), bool)
    for k in (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
        left = ~done
        if not left.any():
            break
        y, x = ys[left], xs[left]
        i = np.clip(y - y0, 0, n - 1)
        t = (x - c[i]) / np.maximum(hw[i], 1)
        y2 = y - k * period
        ok = (y2 >= y0) & (y2 < y0 + n) & (y2 >= 0) & (y2 < H)
        i2 = np.clip(y2 - y0, 0, n - 1)
        x2 = np.rint(c[i2] + t * hw[i2]).astype(int)
        ok &= (x2 >= 0) & (x2 < W)
        ok[ok] &= known[y2[ok], x2[ok]]
        sel = np.nonzero(left)[0][ok]
        out[ys[sel], xs[sel]] = rgb[y2[ok], x2[ok]]
        done[sel] = True
    return out


def skin_model(rgb, where):
    """Mean and inverse covariance of the skin colour (smoothed) at the given pixels."""
    px = rgb[where].reshape(-1, 3)
    if len(px) < 50:
        return None
    mu = px.mean(0)
    cov = np.cov((px - mu).T) + np.eye(3) * 4
    return mu, np.linalg.inv(cov)


def is_skin(rgb, model, limit=3.2):
    mu, ic = model
    d = rgb - mu
    return np.einsum('...i,ij,...j->...', d, ic, d) < limit * limit


def snap_to_lines(rgb, fg, owner, masks, names, idx, BH, owner_t=None, free=()):
    import cv2
    H, W = fg.shape
    BG = BH + 1
    markers = np.zeros((H, W), np.int32)
    markers[~ndimage.binary_dilation(fg, iterations=1)] = BG
    to_bg = ndimage.distance_transform_edt(fg)
    for lab_ in np.unique(owner[fg]):
        if lab_ == 0:
            continue
        reg = (owner == lab_) & fg
        # a seed must surely be this part: far from the other parts (where the painting may differ from the template)
        # but it may come close to the figure's own outer edge. Template parts: measured on the template itself, so a
        # narrow arm still gets a seed inside its round puff; the big torso keeps further away from the sleeves.
        if owner_t is not None and lab_ <= len(names):
            tpl = owner_t == lab_
            to_other = ndimage.distance_transform_edt(~((owner_t > 0) & ~tpl))
            first = 34 if names[lab_ - 1] == 'torso' else 14
            # also well inside the template's own outline: right at its edge the painting may show something else
            # (a ribbon hanging where the template's arm is a little wider than the painted one)
            reg = reg & tpl & (ndimage.distance_transform_edt(tpl) > 10)
        else:
            to_other = ndimage.distance_transform_edt(~(fg & ~reg))
            first = 34
        for margin in (first, 24, 16, 10, 5, 0):
            if margin > first:
                continue
            core = reg & (to_other > margin) & (to_bg > 3)
            if core.sum() > 40:
                break
        markers[core] = lab_
    img = np.clip(rgb, 0, 255).astype(np.uint8).copy()
    img[~fg] = 0
    cv2.watershed(img, markers)
    lines = markers == -1
    if lines.any():
        _, (iy, ix) = ndimage.distance_transform_edt(lines, return_indices=True)
        markers = np.where(lines, markers[iy, ix], markers)
    new = np.where(fg & (markers > 0) & (markers != BG), markers, owner)
    # a part may not wander far from where the template has it
    for n_ in names:
        i = idx[n_]
        far = (new == i) & ~ndimage.binary_dilation(masks[n_] > 0.5, iterations=30)
        new[far] = owner[far]
    # an outline on a border belongs to the piece on top (the sleeve's edge stays on the sleeve, not on the dress)
    luma = rgb @ np.array([0.3, 0.59, 0.11])
    dark = fg & (luma < 95)
    grey = (rgb.max(-1) - rgb.min(-1)) < 70          # line art is dark brown-black; a red ribbon is dark but vivid
    for i, n_ in enumerate(names):
        top = new == idx[n_]
        below = np.isin(new, [idx[q] for q in names[:i]]) | ((new == BH) & grey)   # hanging hair is behind every part
        inside = ndimage.binary_dilation(masks[n_] > 0.5, iterations=2)
        if n_ in free:
            # a soft piece's edge line may lie a few pixels outside its template shape (the far side of the bodice in
            # front of the far sleeve's ball: the line is the bodice's edge, and must not turn with the sleeve)
            inside = ndimage.binary_dilation(masks[n_] > 0.5, iterations=6) | ((new == BH) & grey)
        grab = ndimage.binary_dilation(top, iterations=4) & dark & below & inside
        new[grab] = idx[n_]
    return new


def own_parts(rgb, alpha, masks, names, idx, BH, owner_t, bone, rig):
    """Which part each painted pixel belongs to: the template's part where the mannequin is; paint outside it joined to
    the part it hangs from (hair hanging below the nape: the back hair, BH); borders snapped to the painting's own
    lines. Used on the painting and on the reveal (the painting with the near arm taken away). Returns (owner, extras):
    a part number per pixel, and how much paint lay outside the mannequin."""
    H, W = alpha.shape
    fg = alpha > 0.5
    sil = np.zeros((H, W), np.float32)
    for m_ in masks.values():
        sil = np.maximum(sil, m_)
    silb = sil > 0.5
    smooth = np.stack([ndimage.gaussian_filter(rgb[..., c], 1.2) for c in range(3)], -1)
    # parts that may take their outline from the back hair wherever it lies (a bodice's side, the nape); limbs only
    # inside their template shape (else an arm would take a strip of the braid along its whole length)
    free = {q['name'] for q in rig['parts'] if q['kind'] == 'mesh' or q['name'] == 'head'}
    # 1. inside the mannequin's outline: the part on top there
    owner = np.where(fg & silb, owner_t, 0)
    # 2. paint outside it: a thin rim is the painting being a little wider (nearest part); a big piece hanging off is
    #    extra: hair hanging from the head goes behind the body, anything else joins the part it hangs from
    dist_out, (iy, ix) = ndimage.distance_transform_edt(~silb, return_indices=True)
    nearest = owner_t[iy, ix]
    out_fg = fg & ~silb
    lab, n = ndimage.label(out_fg, structure=np.ones((3, 3)))
    neck_y = bone['head']['y']                      # the nape: where the head turns on the neck
    extras = {}
    hem = None
    if 'skirt' in idx:
        sk = masks['skirt'] > 0.5
        hem = np.where(sk.any(0), H - 1 - np.argmax(sk[::-1], axis=0), -1)
    for k_ in range(1, n + 1):
        comp = lab == k_
        if dist_out[comp].max() <= 16:
            owner[comp] = nearest[comp]
            continue
        # thin strands hanging off a big piece are outlines drawn a little outside the mannequin: they stay with the
        # part they outline; only the thick body of the piece (and its rim) is extra
        core = ndimage.binary_dilation(ndimage.binary_opening(comp, iterations=3), iterations=3) & comp
        owner[comp & ~core] = nearest[comp & ~core]
        comp = core
        if not comp.any():
            continue
        ring = ndimage.binary_dilation(comp, iterations=6) & silb
        touch = np.bincount(owner_t[ring], minlength=len(names) + 1)[1:]
        if not touch.any():
            continue
        if touch[idx['head'] - 1] > 20:
            # hair: what is above the nape stays on the head, what hangs below it goes behind the body
            below = comp & (np.arange(H)[:, None] > neck_y)
            owner[comp & ~below] = idx['head']
            owner[below] = BH
            extras[idx['head']] = extras.get(idx['head'], 0) + int((comp & ~below).sum())
            if below.any():
                extras[BH] = extras.get(BH, 0) + int(below.sum())
            continue
        elif 'skirt' in idx and touch[idx['skirt'] - 1] > 20:
            # a frill below the hem: only as far down as a frill goes, the rest are the legs' outlines
            band = comp & (np.arange(H)[:, None] <= np.maximum(hem, 0)[None, :] + 42) & (hem >= 0)[None, :]
            owner[comp & ~band] = nearest[comp & ~band]
            owner[band] = idx['skirt']
            extras[idx['skirt']] = extras.get(idx['skirt'], 0) + int(band.sum())
            continue
        else:
            tgt = int(np.argmax(touch)) + 1
        owner[comp] = tgt
        extras[tgt] = extras.get(tgt, 0) + int(comp.sum())
    # 3. hair hanging behind the neck and shoulders, inside the outline: thick dark paint joined to the back hair
    if BH in extras:
        luma = smooth @ np.array([0.3, 0.59, 0.11])
        dark = ndimage.binary_opening(fg & (luma < 80), iterations=3)
        lab2, n2 = ndimage.label(dark | (owner == BH))
        keep = set(np.unique(lab2[owner == BH]).tolist()) - {0}
        movable = np.isin(owner, [idx[q] for q in names if q != 'head'])
        owner[np.isin(lab2, list(keep)) & movable & dark] = BH
    # 4. the skirt's hem: the painted frill may hang lower than the mannequin's hem, over the legs. Going down each
    #    column from the mannequin's hem, everything until the skin of the leg starts belongs to the skirt.
    if 'skirt' in idx:
        sk = masks['skirt'] > 0.5
        legs = [q for q in names if q.endswith(('thigh', 'shin'))]
        lower = np.isin(owner_t, [idx[q] for q in legs]) & fg
        hem_y = np.where(sk.any(0), H - 1 - np.argmax(sk[::-1], axis=0), -1)
        under = lower & (np.arange(H)[:, None] > hem_y[None, :] + 70)
        model = skin_model(smooth, ndimage.binary_erosion(under, iterations=6))
        if model:
            skin = is_skin(smooth, model)
            claim = np.zeros((H, W), bool)
            for x in np.nonzero(hem_y >= 0)[0]:
                y, run = hem_y[x] - 3, 0
                while y < min(H, hem_y[x] + 70):
                    if not fg[y, x]:
                        break
                    if skin[y, x]:
                        run += 1
                        if run >= 5:
                            break
                    else:
                        run = 0
                    y += 1
                stop = y - run + 1 if run >= 5 else y
                ys_ = np.arange(max(0, hem_y[x] - 3), stop)
                claim[ys_[fg[ys_, x]], x] = True
            # thin strands are the legs' own outlines, not frill: keep only the thick part (and its edge)
            body = ndimage.binary_opening(claim, iterations=3)
            claim &= ndimage.binary_dilation(body, iterations=3)
            owner[claim & (owner != idx['skirt'])] = idx['skirt']

    # 5. snap the borders between parts to the painting's own lines: every part grows from its core (well inside the
    #    guess above) until it meets a line; it may not wander far from its place in the template
    owner = snap_to_lines(rgb, fg, owner, masks, names, idx, BH, owner_t, free)
    # pieces of one limb (upper arm / forearm / hand, thigh / shin / foot) meet skin to skin with no line between
    # them, so the border is not taken from the painting: it is the straight line through the joint, across the
    # limb. (A ragged border left a chunk of upper arm on the forearm, which swung out when the elbow bent.)
    for p_ in rig['parts']:
        for c_ in p_['cont']:
            cb, pb = bone.get(c_), bone.get(p_['name'])
            if not cb or not pb or cb.get('parent') != p_['name']:
                continue                          # only parent -> child pairs, each once
            jx, jy = cb['x'], cb['y']
            u = np.array([cb['ex'] - cb['x'], cb['ey'] - cb['y']], float)
            u /= np.linalg.norm(u) + 1e-9
            pair = np.isin(owner, [idx[p_['name']], idx[c_]])
            ys9, xs9 = np.nonzero(pair)
            side = (xs9 - jx) * u[0] + (ys9 - jy) * u[1]
            owner[ys9, xs9] = np.where(side > 0, idx[c_], idx[p_['name']])
    # the neck: below the joint the head turns on it is the torso's, above it the head's, cut straight through the
    # joint (a head piece reaching down the neck in a ragged edge leaves a step in the neck when the head turns)
    if 'head' in idx and 'torso' in idx and 'neck' in bone:
        hb = bone['head']
        u = np.array([hb['ex'] - hb['x'], hb['ey'] - hb['y']], float)
        u /= np.linalg.norm(u) + 1e-9
        col_ = (masks['torso'] > 0.5) & (np.arange(H)[:, None] < bone['neck']['y'] - 20)   # the template's neck
        col_ = ndimage.binary_dilation(col_, structure=np.ones((1, 11), bool))
        sel = col_ & np.isin(owner, [idx['head'], idx['torso']]) & fg
        ys9, xs9 = np.nonzero(sel)
        side = (xs9 - hb['x']) * u[0] + (ys9 - hb['y']) * u[1]
        owner[ys9, xs9] = np.where(side > 0, idx['head'], idx['torso'])
        # the jaw: in a column where a short dark line (the jaw's outline) crosses just under the cut, the head keeps
        # everything down to that line, so its jaw is whole (cut at the joint, the jaw's line was left on the neck as
        # a speck, or hung off the head as a strand). Where a line runs on down, it is the neck's own outline: it stays
        luma9 = smooth @ np.array([0.3, 0.59, 0.11])
        dark9 = (luma9 < 95) & ((smooth.max(-1) - smooth.min(-1)) < 90)
        yc = hb['y'] - (np.arange(W) - hb['x']) * u[0] / u[1]          # the cut's row in each column
        for x in np.nonzero(col_.any(0))[0]:
            y0 = int(np.ceil(yc[x]))
            if y0 < 0 or y0 + 20 > H:
                continue
            seg = dark9[y0:y0 + 20, x] & fg[y0:y0 + 20, x] & (owner[y0:y0 + 20, x] == idx['torso'])
            if not seg.any():
                continue
            i = int(np.argmax(seg))
            r = 0
            while i + r < len(seg) and seg[i + r]:
                r += 1
            if i <= 12 and r <= 7 and i + r < len(seg):
                rows = np.arange(y0, y0 + i + r + 1)
                ok = fg[rows, x] & (owner[rows, x] == idx['torso'])
                owner[rows[ok], x] = idx['head']
    # thin strands sticking out of a part along another part (a leg's outline caught by the skirt's hem line) belong
    # to that other part; outlines hugging the part's own body stay, and strands into empty air (a hair clip) too
    for lab_ in [idx[n_] for n_ in names]:
        reg = owner == lab_
        if not reg.any():
            continue
        thin = reg & ~ndimage.binary_dilation(ndimage.binary_opening(reg, iterations=2), iterations=2)
        if not thin.any():
            continue
        others = fg & (owner > 0) & (owner != lab_)
        near_other = thin & ndimage.binary_dilation(others, iterations=3)
        lab3, n3 = ndimage.label(thin)
        touching = np.unique(lab3[near_other])
        move = np.isin(lab3, touching[touching > 0])
        if move.any() and others.any():
            _, (iy3, ix3) = ndimage.distance_transform_edt(~others, return_indices=True)
            owner[move] = owner[iy3[move], ix3[move]]
    # the back hair is hair and its bow: anything else it took at its borders (the edge of a sleeve, a bit of skin)
    # goes back to the nearest other part
    bh = owner == BH
    if bh.any():
        luma_ = smooth @ np.array([0.3, 0.59, 0.11])
        r_, g_, b_ = smooth[..., 0], smooth[..., 1], smooth[..., 2]
        hairish = (luma_ < 100) | ((r_ > g_ + 90) & (r_ > b_ + 90))          # dark hair, or the red ribbon
        stray = bh & ~hairish
        stray &= ~ndimage.binary_erosion(bh, iterations=4)                    # only at its borders
        others = fg & (owner > 0) & (owner != BH)
        if stray.any() and others.any():
            _, (iy5, ix5) = ndimage.distance_transform_edt(~others, return_indices=True)
            owner[stray] = owner[iy5[stray], ix5[stray]]
    # thin strands on the back hair are other parts' outlines: they go to the nearest other part, not away
    bh = owner == BH
    if bh.any():
        strands = bh & ~ndimage.binary_dilation(ndimage.binary_opening(bh, iterations=3), iterations=3)
        others = fg & (owner > 0) & (owner != BH)
        if strands.any() and others.any():
            _, (iy2, ix2) = ndimage.distance_transform_edt(~others, return_indices=True)
            owner[strands] = owner[iy2[strands], ix2[strands]]
    # whole patches of cloth or skin inside the back hair (the side of a bodice painted right next to the braid) go
    # to the part they touch, however deep they reach
    bh = owner == BH
    if bh.any():
        luma_ = smooth @ np.array([0.3, 0.59, 0.11])
        sat_ = smooth.max(-1) - smooth.min(-1)
        r_, g_, b_ = smooth[..., 0], smooth[..., 1], smooth[..., 2]
        ribbon_ = (r_ > g_ + 90) & (r_ > b_ + 90)
        cloth = bh & (sat_ > 90) & (luma_ > 110) & ~ribbon_
        lab9, n9 = ndimage.label(cloth)
        others = fg & (owner > 0) & (owner != BH)
        if n9 and others.any():
            touch9 = np.unique(lab9[ndimage.binary_dilation(others, iterations=2) & cloth])
            move = np.isin(lab9, touch9[touch9 > 0])
            if move.any():
                _, (iy9, ix9) = ndimage.distance_transform_edt(~others, return_indices=True)
                owner[move] = owner[iy9[move], ix9[move]]
    return owner, extras


def cmd_cut(a):
    d = CHARS / a.char
    info = json.loads((d / 'char.json').read_text(encoding='utf-8'))
    tid = info['template']
    T = RIGS / tid
    rig = json.loads((T / 'rig.json').read_text(encoding='utf-8'))
    W, H = rig['size']
    if a.src:
        src = Path(a.src)
    else:
        import prod
        job = prod.load_job(info['job'])
        k = a.k or job.get('picked') or len(job['variants'])
        src = prod.job_dir(info['job']) / job['variants'][k - 1]['clean']
        info['painted_from'] = f'{info["job"]} try {k}'
    im = Image.open(src).convert('RGBA')
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    names = [p['name'] for p in rig['parts']]
    idx = {n: i + 1 for i, n in enumerate(names)}
    BH = len(names) + 1                                   # back hair: paint hanging behind the body
    masks = {n: np.asarray(Image.open(T / 'masks' / f'{n}.png')).astype(np.float32) / 255 for n in names}
    sil = np.zeros((H, W), np.float32)
    for m in masks.values():
        sil = np.maximum(sil, m)
    silb = sil > 0.5
    rgba = align(np.asarray(im).astype(np.float32), sil)
    r_, g_, b_ = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    keyg = (g_ > 170) & (g_ - np.maximum(r_, b_) > 90)          # bits of the green background left in hair gaps
    rgba[..., 3] = np.where(keyg, 0, rgba[..., 3])
    fringe = (rgba[..., 3] > 0) & ndimage.binary_dilation(rgba[..., 3] < 128, iterations=3) & (g_ > np.maximum(r_, b_))
    rgba[..., 1] = np.where(fringe, np.maximum(r_, b_), g_)
    Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), 'RGBA').save(d / 'painted.png')
    rgb, alpha = rgba[..., :3], rgba[..., 3] / 255
    fg = alpha > 0.5
    owner_t = np.asarray(Image.open(T / 'owner.png')).astype(np.int32)
    smooth = np.stack([ndimage.gaussian_filter(rgb[..., c], 1.2) for c in range(3)], -1)
    bone = {b['name']: b for b in rig['bones']}

    owner, extras = own_parts(rgb, alpha, masks, names, idx, BH, owner_t, bone, rig)
    if getattr(a, 'debug', False):
        np.save(d / 'owner_debug.npy', owner)

    smooth = np.stack([ndimage.gaussian_filter(rgb[..., c], 1.2) for c in range(3)], -1)
    # what the near arm hides: ChatGPT's painting of the character with that arm taken away (charkit.py reveal), cut
    # into parts just like the painting. The pieces behind the arm take their pixels round the arm from it; the back
    # hair takes ALL of its pixels from it (the reveal may draw a bow it no longer has to hide a little differently:
    # half of one bow and half of the other would not fit)
    near_arm = [n_ for n_ in names if n_.startswith('n_') and n_.endswith(('upper', 'fore', 'hand'))]
    arm_ids = [idx[n_] for n_ in near_arm]
    rev = load_reveal(info, (W, H), rgba) if near_arm else None
    O_b, C_b, A_b = owner, rgb, alpha                    # what the pieces behind the near arm are cut from
    bh_src = None
    if rev is not None:
        rgb_r, alpha_r = rev[..., :3], rev[..., 3] / 255
        masks_r = {n_: (np.zeros_like(m_) if n_ in near_arm else m_) for n_, m_ in masks.items()}
        under = np.isin(owner_t, arm_ids)
        owner_t_r = np.where(under, 0, owner_t)
        for n_ in names:                                 # back to front: the part on top there without the arm
            if n_ not in near_arm:
                owner_t_r[under & (masks[n_] > 0.5)] = idx[n_]
        owner_r, _ = own_parts(rgb_r, alpha_r, masks_r, names, idx, BH, owner_t_r, bone, rig)
        # the arm's own outline: dark line paint right next to it that went away with it in the reveal
        arm = np.isin(owner, arm_ids)
        sm_ = lambda x: np.stack([ndimage.gaussian_filter(x[..., c], 1.0) for c in range(3)], -1)
        diff = np.abs(sm_(rgb) - sm_(rgb_r)).mean(-1)
        luma = rgb @ np.array([0.3, 0.59, 0.11])
        grey = (rgb.max(-1) - rgb.min(-1)) < 70
        edge = ndimage.binary_dilation(arm, iterations=4) & ~arm & fg & (diff > 30) & (luma < 110) & grey
        if edge.any():
            _, (iy_, ix_) = ndimage.distance_transform_edt(~arm, return_indices=True)
            owner[edge] = owner[iy_[edge], ix_[edge]]
        zone = ndimage.binary_dilation(np.isin(owner, arm_ids), iterations=3)
        band = ndimage.binary_dilation(zone, iterations=10)
        O_b = np.where(band, owner_r, owner)
        C_b = np.where(band[..., None], rgb_r, rgb)
        A_b = np.where(band, alpha_r, alpha)
        # where the two cuts disagree about the braid but the two paintings agree (the same hair at the nape, cut a
        # little differently), the braid keeps that hair too: else it would be in no piece at all
        agree = ndimage.gaussian_filter(diff, 2.0) < 18
        bh_extra = (owner == BH) & (owner_r != BH) & agree & (alpha_r > 0.5)
        bh_r = (owner_r == BH) | bh_extra
        # hair the painting shows beside the arm where the reveal drew the braid a little narrower: kept as painted
        # (else a gap opens between the braid and the sleeve when the arm is down); not the bow (two bows don't mix)
        lum_p = smooth @ np.array([0.3, 0.59, 0.11])
        hairy_p = (lum_p < 100) & ((smooth.max(-1) - smooth.min(-1)) < 80)
        rib_p = (smooth[..., 0] > smooth[..., 1] + 60) & (smooth[..., 0] > smooth[..., 2] + 60)
        gap = (owner == BH) & ~bh_r & ~zone & hairy_p & fg & ~ndimage.binary_dilation(rib_p, iterations=6)
        gap &= ndimage.binary_dilation(ndimage.binary_opening(gap, iterations=2), iterations=2)   # hair, not a line
        lab_g, n_g = ndimage.label(gap)
        if n_g:
            touch_g = np.unique(lab_g[ndimage.binary_dilation(bh_r, iterations=2) & gap])
            gap = np.isin(lab_g, touch_g[touch_g > 0])
        s_rgb = np.where(gap[..., None], rgb, rgb_r)
        s_alpha = np.where(gap, alpha, alpha_r)
        bh_src = (bh_r | gap, s_rgb, s_alpha, np.where(bh_extra | gap, BH, owner_r))
        # the painting's own braid where it lay outside the mannequin and was joined to a part (its tail peeking out
        # between the arm and the skirt went with the skirt and swung with it): the braid now comes whole from the
        # reveal, so that paint belongs to no part
        loose = fg & ~silb & ndimage.binary_opening(hairy_p, iterations=2) & ~gap
        loose &= np.isin(O_b, [idx[q] for q in names if q != 'head' and q not in near_arm])
        lab_l, n_l = ndimage.label(loose)
        if n_l:
            sz_l = ndimage.sum(loose, lab_l, range(1, n_l + 1))
            loose = np.isin(lab_l, np.nonzero(sz_l > 120)[0] + 1)
            loose = ndimage.binary_dilation(loose, iterations=2) & fg & ~silb & ~np.isin(O_b, [BH, 0])
            O_b = np.where(loose, 0, O_b)
            print(f'  {int(loose.sum())} px of the painting braid left out of the other parts')
        print(f'  back hair: {int(gap.sum())} px kept from the painting beside the arm')
        print(f'  near arm: {int(edge.sum())} px of its outline taken back; behind it: cut from the reveal '
              f'(back hair {int((owner_r == BH).sum())} px)')

    # the hanging braid, rebuilt where it was hidden (see complete_rope). Hair paint lying inside the rebuilt braid
    # (thin slivers of its tail seen between the arm and the skirt) is the braid's, not the skirt's or the arm's
    rope = None
    if rev is None and (owner == BH).any():
        # the braid's tail can show apart from it (between the arm and the skirt): thick blobs painted in the
        # braid's own hair colours, near the braid, are the braid, whichever part's area they lie in
        bhm = owner == BH
        lum7 = smooth @ np.array([0.3, 0.59, 0.11])
        sat7 = smooth.max(-1) - smooth.min(-1)
        hc = smooth[bhm & (lum7 < 90) & (sat7 < 80)]
        if len(hc) > 200:
            mu7 = hc.mean(0)
            ic7 = np.linalg.inv(np.cov((hc - mu7).T) + np.eye(3) * 9)
            dv = smooth - mu7
            like = fg & (np.einsum('...i,ij,...j->...', dv, ic7, dv) < 3.5 ** 2) & (lum7 < 100)
            like = ndimage.binary_opening(like, iterations=2)
            near7 = ndimage.distance_transform_edt(~bhm) < 140
            cand = like & near7 & (owner > 0) & (owner != BH) & (owner != idx['head'])
            lab7, n7 = ndimage.label(cand)
            if n7:
                sz7 = ndimage.sum(cand, lab7, range(1, n7 + 1))
                owner[np.isin(lab7, np.nonzero(sz7 > 150)[0] + 1)] = BH
        rope, _, _ = complete_rope(owner == BH, smooth, fg, owner, BH)
        luma6 = smooth @ np.array([0.3, 0.59, 0.11])
        sat6 = smooth.max(-1) - smooth.min(-1)
        hairy = fg & (luma6 < 90) & (sat6 < 80)                 # dark grey-brown, not a red trim
        hairy = ndimage.binary_dilation(ndimage.binary_opening(hairy, iterations=2), iterations=1) & hairy   # not a line
        claim = rope & hairy & (owner != BH) & (owner != idx['head'])
        owner[claim] = BH

    pd = d / 'parts'
    pd.mkdir(exist_ok=True)
    for f in pd.glob('*.png'):
        f.unlink()
    order = {n_: i for i, n_ in enumerate(names)}
    outer = ndimage.binary_dilation(silb & ~ndimage.binary_erosion(silb, iterations=3), iterations=3)
    pieces = {}
    for p in rig['parts']:
        name, i = p['name'], idx[p['name']]
        m = masks[name]
        # the near arm is cut from the painting; the pieces behind it from the reveal round it (O_b, C_b, A_b)
        O, C, A = (owner, rgb, alpha) if name in near_arm else (O_b, C_b, A_b)
        own = O == i
        cont_px = np.isin(O, [idx[c] for c in p['cont']]) & (m > 0.02) & (A > 0.5)
        vis = own | cont_px
        # covered by a part drawn over this one: the thigh under the skirt, the neck under the head
        above = [idx[q] for q in names[order[name] + 1:] if q not in p['cont']]
        covered = np.isin(O, above)
        hidden = (m > 0.5) & ~vis & covered
        neck_ext = None
        if p['kind'] == 'mesh':
            # a soft piece (bodice, skirt) is completed where something covers it by smoothly closing the 'bite' that
            # the cover took out of its painted outline, not by the template's outline (the painted bodice may be
            # narrower: the template's edge would stick out as a flap when an arm lifts). Up at the neck the
            # template's shape stays (it shows when the head nods).
            import cv2
            neck = np.zeros_like(hidden)
            if 'neck' in bone and name == 'torso':
                neck[:int(bone['neck']['y']) + 15] = True
                neck_ext, neck_col = extend_neck(C, O, A, idx, bone, masks)
            body = vis & ~neck
            hull = np.zeros(hidden.shape, np.uint8)
            ys_, xs_ = np.nonzero(body)
            if len(xs_) > 20:
                pts = np.stack([xs_, ys_], 1)[::7].astype(np.int32)
                cv2.fillPoly(hull, [cv2.convexHull(pts)], 1)
            neck_shape = neck_ext if neck_ext is not None else (m > 0.5)
            hidden = covered & ~vis & (((hull > 0) & ~neck) | (neck_shape & neck))
        # what the part owns keeps the painting's alpha, softened by a pixel where it meets another part (the border
        # follows the painting's lines, so it may lie a little outside the template's shape: never cut it there);
        # a joint's round end under the next piece takes the template's round shape
        own_soft = np.clip(ndimage.gaussian_filter(own.astype(np.float32), 0.7) * 2, 0, 1) * own
        # where a piece drawn over this one meets it: no soft edge here, and the piece goes on a few pixels under that
        # one (only the top piece's edge is soft; two soft edges over each other let the background through as a pale
        # seam, and a seam opens when the two move a little apart)
        near_cover = ndimage.binary_dilation(covered, iterations=2)
        own_soft = np.where(own & near_cover, 1.0, own_soft)
        hidden = hidden | (covered & ~vis & ndimage.binary_dilation(own, iterations=5))
        if p['kind'] == 'rigid' and name != 'head':
            # a limb's paint right along the edge of a piece over it is that piece's seam line (the side of the bodice
            # over the far sleeve's ball): it is painted over like the rest of what that piece hides, or it would
            # turn round with the limb as a line across it
            hidden = hidden | (own & ndimage.binary_dilation(covered, iterations=3))
        a_vis = A * np.where(own, own_soft, np.where(cont_px, m, 0))
        if name == 'head' and 'neck' in bone and 'torso' in idx:
            # the head's piece fades out down the neck over a few pixels below the joint it turns on (the neck under
            # it is the torso's): when the head turns, skin slides softly over skin instead of showing a cut edge
            hb = bone['head']
            u = np.array([hb['ex'] - hb['x'], hb['ey'] - hb['y']], float)
            u /= np.linalg.norm(u) + 1e-9
            col_ = (masks['torso'] > 0.5) & (np.arange(H)[:, None] < bone['neck']['y'] - 20)
            col_ = ndimage.binary_dilation(col_, structure=np.ones((1, 11), bool))
            yy9, xx9 = np.mgrid[0:H, 0:W]
            side2 = (xx9 - hb['x']) * u[0] + (yy9 - hb['y']) * u[1]
            nb = col_ & (O == idx['torso']) & (side2 <= 0) & (side2 > -12)
            # not where the head ends at its jaw's line (the neck below it is not the head's), and no lines (the
            # neck's own outline would turn with the head as a faint second line)
            jaw_cols = (col_ & (O == idx['head']) & (side2 < -2)).any(0)
            lum_c = C @ np.array([0.3, 0.59, 0.11])
            nb &= ~jaw_cols[None, :] & ~ndimage.binary_dilation(lum_c < 100, iterations=1)
            a_vis = np.where(nb, A * np.clip(1 + side2 / 12, 0, 1), a_vis)
        known = vis & (A > 0.9) & ~ndimage.binary_dilation(covered, iterations=4)
        if known.sum() < 30:
            known = vis & (A > 0.9)
        rgb_k = C
        # limbs are about the same on both sides of their bone: mirror across it first. Not the torso, skirt or head:
        # their features (a neckline, a patch, a face) must not be copied to the other side
        axis = None
        if p['kind'] == 'rigid' and name != 'head':
            b_ = bone[p['bone']]
            u_ = np.array([b_['ex'] - b_['x'], b_['ey'] - b_['y']], float)
            axis = ((b_['x'], b_['y']), u_ / (np.linalg.norm(u_) + 1e-9))
        # a far limb's hidden side is painted like its twin on the near side (same sleeve, same arm): take the near
        # piece's painted pixel at the same place along and across its bone, mirrored
        twin = 'n_' + name[2:] if name.startswith('f_') else None
        if name.endswith('_upper'):
            # the far sleeve's ball: its hidden half is its own painted half mirrored across the arm (the cuff runs
            # on without a step; the near sleeve sits a little higher or lower on its arm)
            twin = None
        if twin in idx and hidden.any():
            bf, bn = bone[name], bone[twin]
            uf = np.array([bf['ex'] - bf['x'], bf['ey'] - bf['y']], float); Lf = np.linalg.norm(uf); uf /= Lf
            un = np.array([bn['ex'] - bn['x'], bn['ey'] - bn['y']], float); Ln = np.linalg.norm(un); un /= Ln
            nf_, nn_ = np.array([-uf[1], uf[0]]), np.array([-un[1], un[0]])
            def twin_at(yy, xx, oa=0, oc=0):
                along = ((xx - bf['x']) * uf[0] + (yy - bf['y']) * uf[1]) * (Ln / Lf) + oa
                across = (xx - bf['x']) * nf_[0] + (yy - bf['y']) * nf_[1] + oc
                tx_ = np.rint(bn['x'] + along * un[0] - across * nn_[0]).astype(int)
                ty_ = np.rint(bn['y'] + along * un[1] - across * nn_[1]).astype(int)
                ok_ = (tx_ >= 0) & (tx_ < W) & (ty_ >= 0) & (ty_ < H)
                ok_[ok_] &= (owner[ty_[ok_], tx_[ok_]] == idx[twin]) & (alpha[ty_[ok_], tx_[ok_]] > 0.9)
                return ty_, tx_, ok_
            # line the twin up on what both show: the far piece's own paint right next to its hidden part must match
            # the twin's paint mapped there (else a sleeve's trim steps where the two meet)
            sy, sx = np.nonzero(own & (A > 0.9) & ndimage.binary_dilation(hidden, iterations=14) & ~hidden)
            if len(sy) > 3000:
                pick = np.random.default_rng(0).choice(len(sy), 3000, replace=False)
                sy, sx = sy[pick], sx[pick]
            best = (np.inf, 0, 0)
            def err_at(oa, oc):
                ty_, tx_, ok_ = twin_at(sy, sx, oa, oc)
                if ok_.sum() < 0.4 * len(sy):
                    return np.inf
                return float(np.abs(smooth[sy[ok_], sx[ok_]] - smooth[ty_[ok_], tx_[ok_]]).mean())
            if len(sy) > 50:
                for oa in range(-12, 13, 3):
                    for oc in range(-12, 13, 3):
                        best = min(best, (err_at(oa, oc), oa, oc))
                b0 = best
                for oa in range(b0[1] - 2, b0[1] + 3):
                    for oc in range(b0[2] - 2, b0[2] + 3):
                        best = min(best, (err_at(oa, oc), oa, oc))
                if not np.isfinite(best[0]):
                    best = (np.inf, 0, 0)
            hy, hx = np.nonzero(hidden)
            ty, tx, ok = twin_at(hy, hx, best[1], best[2])
            if ok.any():
                rgb_k = rgb_k.copy()
                rgb_k[hy[ok], hx[ok]] = rgb[ty[ok], tx[ok]]
                known = known.copy()
                known[hy[ok], hx[ok]] = True
                print(f'  {name}: {int(ok.sum())} hidden px painted like {twin} (lined up by {best[1]}, {best[2]} px)')
        col = fill_hidden(rgb_k, known, hidden, axis=axis)
        col = outline_hidden(col, C, m, vis, hidden, [masks[c] for c in p['cont']])
        a_all = np.maximum(a_vis, np.maximum(m, ndimage.binary_dilation(own, iterations=5)) * hidden)
        if neck_ext is not None:
            ext = neck_ext & hidden
            col = np.where(ext[..., None], neck_col, col)
            a_all = np.where(ext, neck_col_alpha(neck_ext, bone), a_all)
        # crumbs: filled-in bits cut off from everything the painting shows of this piece are dropped (painted bits
        # always stay)
        lab4, n4 = ndimage.label(a_all > 0.3)
        if n4 > 1:
            painted_labs = np.unique(lab4[(a_vis > 0.3)])
            crumbs = (lab4 > 0) & ~np.isin(lab4, painted_labs)
            a_all = np.where(crumbs, 0, a_all)
        pieces[name] = (col, a_all)
    if bh_src is None and (owner == BH).any():
        bh_src = (owner == BH, rgb, alpha, owner)
    if bh_src is not None:
        # hanging hair is behind everything, so a painting shows only part of it. Where the body hides it, rebuild it
        # like a rope: row by row, its seen outer edge plus the braid's real width (measured where it is fully seen);
        # fill the new bits with the braid's own plaits
        bh, s_rgb, s_alpha, s_owner = bh_src
        s_smooth = np.stack([ndimage.gaussian_filter(s_rgb[..., c], 1.2) for c in range(3)], -1)
        s_fg = s_alpha > 0.5
        complete, hair_px, axis_ = complete_rope(bh, s_smooth, s_fg, s_owner, BH)
        complete |= rope if rope is not None else False
        hid = complete & ~bh & s_fg & (s_owner > 0)
        print(f'  back hair: {int(bh.sum())} px painted, {int(hid.sum())} px rebuilt where the body hides it')
        col = braid_texture(s_rgb, hair_px & (s_alpha > 0.9), hid, axis_)
        col = fill_hidden(col, (hair_px & (s_alpha > 0.9)) | (hid & (col != s_rgb).any(-1)), hid)
        shape = ndimage.gaussian_filter((bh | hid).astype(np.float32), 1.0)
        col = outline_hidden(col, s_rgb, shape, bh, hid, [])
        a_bh = np.maximum(s_alpha * ndimage.binary_dilation(bh, iterations=1), shape * hid)
        # thin strips of other colours along the braid's edge (a bodice's or a trim's edge line that came with it) go:
        # they show as a stray line when the braid moves; the bow is thick and stays
        lum_b, sat_b = col @ np.array([0.3, 0.59, 0.11]), col.max(-1) - col.min(-1)
        colored = (a_bh > 0.3) & (sat_b > 120) & (lum_b > 70)
        bow = ndimage.binary_dilation(ndimage.binary_opening(colored, iterations=3), iterations=8)
        rim = (a_bh > 0.3) & ndimage.binary_dilation(a_bh <= 0.3, iterations=4)
        stray = colored & rim & ~bow
        a_bh = np.where(ndimage.binary_dilation(stray, iterations=1) & rim & ~bow, 0, a_bh)
        # inside, thin orange-red lines away from the bow (the bodice's edge glow painted onto the braid's side) are
        # painted over with the hair round them
        warm = (a_bh > 0.3) & (sat_b > 60) & (col[..., 0] > 1.6 * col[..., 1] + 10) & ~bow
        warm &= ~ndimage.binary_opening(warm, iterations=2)
        if warm.any():
            import cv2
            box_ = np.nonzero(ndimage.binary_dilation(warm, iterations=6))
            y0_, y1_, x0_, x1_ = box_[0].min(), box_[0].max() + 1, box_[1].min(), box_[1].max() + 1
            sub = np.clip(col[y0_:y1_, x0_:x1_], 0, 255).astype(np.uint8)
            msk = ndimage.binary_dilation(warm, iterations=1)[y0_:y1_, x0_:x1_].astype(np.uint8)
            col = col.copy()
            col[y0_:y1_, x0_:x1_] = cv2.inpaint(sub, msk, 3, cv2.INPAINT_TELEA).astype(np.float32)
            stray = stray | warm
        print(f'  back hair: {int(stray.sum())} px of thin stray colour along its edge dropped')
        pieces['back_hair'] = (col, a_bh)
        if axis_ is not None:
            # the braid gets bones of its own along its middle: the first hangs from the head (it is tied there), the
            # ones below follow the head less and less and the back more (it hangs and lies on her back), each a
            # little later than the one above (skel.js 'hang')
            ry0, rc, _ = axis_
            ys_ = np.nonzero((pieces['back_hair'][1] > 0.5).any(1))[0]
            top, bot = max(int(ys_.min()) + 6, ry0), min(int(ys_.max()) - 4, ry0 + len(rc) - 1)
            nseg = 6
            # the first bone is the braid's root inside the hair (down to just below the nape): it turns with the head
            nape_y = min(max(top + 20, bone['head']['y'] + 25), bot - 200)
            ys7 = [top] + list(np.linspace(nape_y, bot, nseg))
            pts = [(round(float(rc[int(round(y)) - ry0]), 1), round(float(y), 1)) for y in ys7]
            names_h = [f'hair{i}' for i in range(nseg)]
            info['bones'] = [{'name': n_, 'parent': 'head' if i == 0 else names_h[i - 1], 'x': pts[i][0], 'y': pts[i][1],
                              'ex': pts[i + 1][0], 'ey': pts[i + 1][1]} for i, n_ in enumerate(names_h)]
            info['hang'] = {'bones': names_h, 'follow': 'head', 'rest_on': 'chest',
                            'w': [1.0, 0.4, 0.15, 0.05, 0, 0], 'lag': [0, 0.05, 0.1, 0.15, 0.2, 0.25]}
        else:
            info.pop('bones', None)
            info.pop('hang', None)
    # limbs: their pieces laid together into one picture each (one painting, so they fit pixel for pixel)
    order_out = (['back_hair'] if 'back_hair' in pieces else []) + names
    layer_of = {}
    hands = load_hands(info, tid, bone, W, H)
    variants = {}
    for lname, spec_ in (rig.get('layers') or {}).items():
        have = [q for q in spec_['parts'] if q in pieces]
        if not have:
            continue
        # the same arm with another hand (charkit.py hands): the forearm stops at the wrist, the painted hand's
        # piece of wrist fades in over its end
        side, hand_part = lname.split('_')[0], spec_['parts'][-1]
        if side in hands and hand_part.endswith('_hand') and hand_part in have and len(have) >= 2:
            hb = bone[hand_part]
            u = np.array([hb['ex'] - hb['x'], hb['ey'] - hb['y']], float)
            u /= np.linalg.norm(u) + 1e-9
            yy, xx = np.mgrid[0:H, 0:W]
            beyond = ((xx - hb['x']) * u[0] + (yy - hb['y']) * u[1]) > 0
            ref_c, ref_a = pieces[hand_part]
            for pose_, (hc, ha) in hands[side].items():
                hc = match_colour(hc, ha, ref_c, ref_a)
                col = np.zeros((H, W, 3), np.float32)
                acc = np.zeros((H, W), np.float32)
                for q in have[:-1]:
                    c_, a_ = pieces[q]
                    if q == have[-2]:
                        a_ = np.where(beyond, 0, a_)
                    col = col * (1 - a_[..., None]) + c_ * a_[..., None]
                    acc = acc * (1 - a_) + a_
                col = col * (1 - ha[..., None]) + hc * ha[..., None]
                acc = acc * (1 - ha) + ha
                col = np.where(acc[..., None] > 1e-3, col / np.maximum(acc[..., None], 1e-3), 0)
                variants.setdefault(lname, []).append((f'{lname}@{pose_}', col, acc))
        col = np.zeros((H, W, 3), np.float32)
        acc = np.zeros((H, W), np.float32)
        for q in have:                                       # upper, fore, hand: each over the one before
            c_, a_ = pieces.pop(q)
            col = col * (1 - a_[..., None]) + c_ * a_[..., None]
            acc = acc * (1 - a_) + a_
        col = np.where(acc[..., None] > 1e-3, col / np.maximum(acc[..., None], 1e-3), 0)
        pieces[lname] = (col, acc)
        at = order_out.index(have[0])
        order_out = [q for q in order_out if q not in have]
        order_out.insert(at, lname)
        layer_of[lname] = spec_
    for lname, vs in variants.items():                       # each other-hand arm right after the arm itself
        at = order_out.index(lname) + 1
        for vname, col, acc in vs:
            pieces[vname] = (col, acc)
            layer_of[vname] = dict(layer_of[lname], slot=lname)
            order_out.insert(at, vname)
            at += 1
        layer_of[lname] = dict(layer_of[lname], slot=lname)
    out_parts = []
    # the painting's line colour (dark pixels along the figure's outer edge)
    lum_ = smooth @ np.array([0.3, 0.59, 0.11])
    ring_ = fg & ~ndimage.binary_erosion(fg, iterations=4) & (lum_ < 85) & ((smooth.max(-1) - smooth.min(-1)) < 90)
    line_rgb = np.median(rgb[ring_], axis=0) if ring_.sum() > 50 else np.array([45.0, 30.0, 20.0])
    under_arm = np.isin(owner, [idx[q] for q in ('torso', 'skirt') if q in idx] + [BH]) & fg
    for name in order_out:
        col, a_all = pieces[name]
        if name in layer_of and layer_of[name]['parts'][-1].endswith('_hand'):
            col, a_all = clean_limb(col, a_all)
        if name == 'torso':
            # the bodice's edge over the far sleeve: when that arm turns, the edge is in the open, so it gets the
            # drawing's outline (the painting had only the sleeve's seam there)
            limbs_below = [idx[q] for q in names[:order['torso']] if q.endswith(('upper', 'fore', 'hand'))]
            seam = np.isin(owner, limbs_below) & ndimage.binary_dilation(a_all > 0.5, iterations=3) & ~(a_all > 0.5)
            col, a_all = outline_seams(col, a_all, seam, line_rgb)
        if name in layer_of and name.split('@')[0].endswith(('_arm', '_sleeve')):
            # where the sleeve lay on the bodice (or the braid), an unbroken outline like the rest of the drawing
            seam = under_arm & ndimage.binary_dilation(a_all > 0.5, iterations=3) & ~(a_all > 0.5)
            col, a_all = outline_seams(col, a_all, seam, line_rgb)
            if getattr(a, 'debug', False):
                print(f'  {name}: seam {int(seam.sum())} px, line {line_rgb.round()}')
                np.save(d / f'seam_{name}.npy', seam)
        ys, xs = np.nonzero(a_all > 0.01)
        if not len(xs):
            print(f'  ! {name}: nothing painted for it')
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        img = np.dstack([col[y0:y1, x0:x1], a_all[y0:y1, x0:x1] * 255])
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGBA').save(pd / f'{name}.png')
        nape = bone['head']['y']
        if name in layer_of:                                 # a limb: one mesh bent along its chain of bones
            sp_ = layer_of[name]
            out_parts.append({'name': name, 'file': f'parts/{name}.png', 'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0,
                              'bone': sp_.get('chain', sp_['parts'])[0], 'kind': 'chain', 'chain': sp_.get('chain', sp_['parts']),
                              'root': sp_['root'], 'soft': sp_['soft'], 'bones': [sp_['root']] + sp_.get('chain', sp_['parts']),
                              **({'slot': sp_['slot']} if sp_.get('slot') else {})})
            continue
        if name == 'back_hair' and info.get('hang'):           # the braid: one picture bent along its own bones
            hb_ = info['hang']['bones']
            out_parts.append({'name': name, 'file': f'parts/{name}.png', 'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0,
                              'bone': hb_[0], 'kind': 'chain', 'chain': hb_, 'root': 'head', 'soft': [0] + [36] * (len(hb_) - 1),
                              'bones': ['head'] + hb_})
            continue
        tp = next((q for q in rig['parts'] if q['name'] == name),
                  {'bone': 'head', 'kind': 'mesh', 'bones': ['head', 'chest'],      # back hair: top turns with the head,
                   'blend': {'top': 'head', 'bottom': 'chest', 'from': nape, 'to': nape + 170}})   # the rest hangs
        out_parts.append({'name': name, 'file': f'parts/{name}.png', 'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0,
                          'bone': tp['bone'], 'kind': tp['kind'], 'bones': tp['bones'],
                          **({k_: tp[k_] for k_ in ('pull', 'floor', 'blend', 'sigma', 'hold') if k_ in tp} if tp['kind'] == 'mesh' else {})})
    info['parts'] = out_parts
    info.setdefault('deltas', {})
    info['cut'] = time.strftime('%Y-%m-%d %H:%M')
    (d / 'char.json').write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding='utf-8')
    ex = ', '.join(('back hair' if t == BH else names[t - 1]) + f' {px} px' for t, px in extras.items())
    print(f'{d}: {len(out_parts)} parts' + (f' (paint outside the mannequin: {ex})' if ex else ''))
    cmd_sheet(a)


def cmd_sheet(a):
    d = CHARS / a.char
    info = json.loads((d / 'char.json').read_text(encoding='utf-8'))
    rig = json.loads((RIGS / info['template'] / 'rig.json').read_text(encoding='utf-8'))
    W, H = rig['size']
    # the parts put back together in draw order: must look like the painting
    asm = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    for p in info['parts']:
        if p.get('slot') and p['slot'] != p['name']:
            continue                                      # another hand: swapped in by actions, not at rest
        asm.alpha_composite(Image.open(d / p['file']).convert('RGBA'), (p['x'], p['y']))
    asm.save(d / 'assembled.png')
    tiles = [Image.open(d / 'painted.png').convert('RGBA'), asm]
    cell = 330
    cols = 6
    parts = info['parts']
    rows = 1 + math.ceil(len(parts) / cols)
    sheet = Image.new('RGB', (cols * cell, rows * cell), (40, 40, 44))
    chk = Image.new('RGB', (cell - 8, cell - 8), (150, 150, 150))
    for i, t in enumerate(tiles):
        t = t.copy()
        t.thumbnail((cell * 3 - 8, cell - 8))
        bg = Image.new('RGB', t.size, (110, 110, 116))
        bg.paste(t, (0, 0), t)
        sheet.paste(bg, (i * cell * 3 + 4, 4))
    for i, p in enumerate(parts):
        t = Image.open(d / p['file']).convert('RGBA')
        t.thumbnail((cell - 8, cell - 28))
        bg = chk.copy()
        bg.paste(t, ((bg.width - t.width) // 2, (bg.height - 20 - t.height) // 2), t)
        ImageDraw.Draw(bg).text((6, bg.height - 18), p['name'], fill=(20, 20, 20))
        r, c = 1 + i // cols, i % cols
        sheet.paste(bg, (c * cell + 4, r * cell + 4))
    sheet.save(d / 'sheet.jpg', quality=88)
    print(f'sheet: {d / "sheet.jpg"}')


CHECKS = [   # (sequence, zoom, floor line, times): upper body, legs, then moments of the demo
    ('bend', 2.3, 1.37, [0, 2.5, 5.0, 9.5, 10.2]),
    ('bend', 2.3, 0.86, [0, 7.6]),
    ('demo', 1.5, 1.02, [3.6, 7.0, 7.5, 13.6, 16.0]),
    ('poses', 2.6, 1.75, [1.5, 2.5, 3.5, 4.5, 5.5]),   # arms held where users drag them: up, out, forward, crossed
]


def cmd_check(a):
    """Render the moments where cut-out characters break (joints far bent, head tilted, skirt and legs moving) close
    up, and put them on one sheet to look at after every change."""
    import subprocess
    out = ROOT / 'out' / 'chars' / a.char
    tiles = []
    for seq, zoom, fy, ts in CHECKS:
        r = subprocess.run(['node', str(ROOT / 'room' / 'render_char.js'), a.char, seq, '--plain', f'--zoom={zoom}',
                            f'--fy={fy}', '--at=' + ','.join(str(t) for t in ts)], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[-500:])
        for t in ts:
            f = out / f'still_{seq}_{t:g}.jpg'
            im = Image.open(f).convert('RGB')
            px = np.asarray(im).astype(int)
            m = np.abs(px - px[:, 3:4]).sum(-1) > 40          # differs from the background of its own row
            xs = np.nonzero(m.any(0))[0]
            if len(xs):
                im = im.crop((max(0, xs.min() - 20), 0, min(im.width, xs.max() + 20), im.height))
            tiles.append((f'{seq} {t:g}s', im))
            f.unlink()
    Hh, rows, row, w = 540, [], [], 0
    for name, im in tiles:
        im = im.resize((max(1, round(im.width * Hh / im.height)), Hh))
        ImageDraw.Draw(im).text((8, 8), name, fill=(255, 255, 255))
        if w + im.width > 2600 and row:
            rows.append(row)
            row, w = [], 0
        row.append(im)
        w += im.width
    rows.append(row)
    sheet = Image.new('RGB', (max(sum(i.width for i in r) for r in rows), Hh * len(rows)), (30, 30, 34))
    for ri, r in enumerate(rows):
        x = 0
        for im in r:
            sheet.paste(im, (x, ri * Hh))
            x += im.width
    sheet.save(out / 'check.jpg', quality=88)
    print(f'check: {out / "check.jpg"}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('template'); s.add_argument('tid'); s.set_defaults(fn=lambda a: build_template(a.tid))
    s = sub.add_parser('dress'); s.add_argument('char'); s.add_argument('tid'); s.add_argument('ref'); s.add_argument('look')
    s.add_argument('--n', type=int, default=1); s.set_defaults(fn=cmd_dress)
    s = sub.add_parser('cut'); s.add_argument('char'); s.add_argument('--k', type=int); s.add_argument('--from', dest='src')
    s.add_argument('--debug', action='store_true')
    s.set_defaults(fn=cmd_cut)
    s = sub.add_parser('sheet'); s.add_argument('char'); s.set_defaults(fn=cmd_sheet)
    s = sub.add_parser('check'); s.add_argument('char'); s.set_defaults(fn=cmd_check)
    s = sub.add_parser('reveal'); s.add_argument('char'); s.set_defaults(fn=cmd_reveal)
    s = sub.add_parser('hands'); s.add_argument('char'); s.set_defaults(fn=cmd_hands)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
