"""Parallel Earth template engine.
One JSON config -> Kokoro voice-over + motion-graphic scenes + ambient music -> finished 9:16 MP4.
Usage: python3 engine.py config.json   (or import and call build(cfg_dict))
Env: PE_MODEL_DIR (kokoro-v1.0.onnx, voices-v1.0.bin), PE_FONT_DIR (InterDisplay-*.otf), PE_OUT_DIR
"""
import os, sys, json, math, random, re, subprocess, hashlib
import numpy as np

W, H, FPS = 1080, 1920, 30
SR = 24000
MODEL_DIR = os.environ.get('PE_MODEL_DIR', '/content/pe_models')
FONT_DIR = os.environ.get('PE_FONT_DIR', '/content/pe_fonts')
OUT_DIR = os.environ.get('PE_OUT_DIR', '/content')

BG = 0xFF0A0F24
WHITE = 0xFFF4F7FF
MUTED = 0xFF8A93B8
BLUE_L, BLUE_D = 0xFF5AB0FF, 0xFF1446A0
ACCENTS = {'orange': 0xFFFFB862, 'red': 0xFFFF5A4E, 'green': 0xFF5EE6A8, 'purple': 0xFFB79CFF,
           'cyan': 0xFF5FE3F5, 'yellow': 0xFFFFD84D}
BODIES = {'earth': (0xFF5AB0FF, 0xFF1446A0), 'mars': (0xFFFF8A5B, 0xFF8E2A12), 'ice': (0xFFE6F7FF, 0xFF7FB6D6),
          'lava': (0xFFFFB347, 0xFFB3200E), 'sun': (0xFFFFF1A8, 0xFFFF8A00), 'venus': (0xFFFFE0A3, 0xFFC98A2E),
          'parallel': (0xFFFFB862, 0xFFC2410C)}

S = {}  # per-video state: accent, style, fonts


def skia_mod():
    import skia
    return skia


# ------------------------------------------------------------ helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x); return 1 - (1 - x) ** 3


def ease_io(x):
    x = clamp(x); return 3 * x * x - 2 * x * x * x


def acol(c, a):
    a = clamp(a); return ((int(((c >> 24) & 255) * a)) << 24) | (c & 0xFFFFFF)


def lerp_col(c1, c2, k):
    k = clamp(k)
    out = 0xFF000000
    for sh in (16, 8, 0):
        a, b = (c1 >> sh) & 255, (c2 >> sh) & 255
        out |= int(a + (b - a) * k) << sh
    return out


def paint(c, a=1.0, stroke=0):
    skia = S['skia']
    p = skia.Paint(AntiAlias=True, Color=acol(c, a))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style); p.setStrokeWidth(stroke); p.setStrokeCap(skia.Paint.kRound_Cap)
    return p


def font(size, weight='Black'):
    skia = S['skia']
    f = skia.Font(S['tf'][weight], size)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    return f


def text(cv, s, x, y, size, color=WHITE, a=1.0, weight='Black', align='center', spacing=0.0, scale=1.0):
    if a <= 0 or not s:
        return
    f = font(size, weight)
    w = f.measureText(s) + spacing * max(0, len(s) - 1)
    cv.save(); cv.translate(x, y); cv.scale(scale, scale)
    ox = -w / 2 if align == 'center' else 0
    p = paint(color, a)
    if spacing == 0:
        cv.drawString(s, ox, 0, f, p)
    else:
        cx = ox
        for ch in s:
            cv.drawString(ch, cx, 0, f, p); cx += f.measureText(ch) + spacing
    cv.restore()


def wrap(s, size, maxw, weight='Black'):
    f = font(size, weight)
    words, lines, cur = str(s).split(), [], ''
    for wd in words:
        t = (cur + ' ' + wd).strip()
        if f.measureText(t) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur); cur = wd
    if cur:
        lines.append(cur)
    return lines


def fit_size(lines, maxw, start, weight='Black', minsize=48):
    size = start
    while size > minsize and max(font(size, weight).measureText(l) for l in lines) > maxw:
        size -= 4
    return size


def parse_head(head):
    """head: list of lines or a string with '|' breaks; *word* marks accent lines."""
    if isinstance(head, str):
        head = head.split('|')
    out = []
    for l in head or []:
        l = str(l).strip()
        if not l:
            continue
        acc = l.startswith('*') and l.endswith('*') and len(l) > 2
        out.append((l.strip('*').upper(), S['accent'] if acc else WHITE))
    return out


def headline(cv, head, t, y0=430, maxsize=108):
    lines = parse_head(head)
    if not lines:
        return
    size = fit_size([l for l, _ in lines], 960, maxsize)
    gap = size * 1.14
    for i, (s, c) in enumerate(lines):
        k = ease((t - i * 0.12) / 0.45)
        text(cv, s, W / 2, y0 + i * gap + (1 - k) * 40, size, c, k)


# ------------------------------------------------------------ backgrounds / bodies
def has_bg():
    return S.get('bgimg') is not None


def bg(cv, t, dim=1.0, tint=None, tint_a=0.0):
    skia = S['skia']
    cv.clear(BG)
    if has_bg():
        cv.drawImage(S['bgimg'], 0, 0)
        shade = skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)],
                                               [acol(BG, 0.82), acol(BG, 0.35), acol(BG, 0.45), acol(BG, 0.85)], [0, 0.35, 0.65, 1])
        cv.drawRect(skia.Rect(0, 0, W, H), skia.Paint(Shader=shade))
        if tint is not None and tint_a > 0:
            cv.drawRect(skia.Rect(0, 0, W, H), paint(tint, tint_a))
        return
    if tint is not None and tint_a > 0:
        cv.drawRect(skia.Rect(0, 0, W, H), paint(tint, tint_a))
    style = S['style']
    if style == 'grid':
        gp = paint(BLUE_L, 0.07 * dim, stroke=2)
        off = (t * 30) % 120
        for y in range(0, H + 120, 120):
            cv.drawLine(0, y + off, W, y + off, gp)
        for x in range(0, W + 120, 120):
            cv.drawLine(x, 0, x, H, gp)
    for (x, y, r, ph, vx) in S['stars']:
        if style == 'dust':
            yy = (y - t * vx * 40) % H
            cv.drawCircle(x, yy, r * 1.3, paint(S['accent'] if ph > 5 else WHITE, dim * 0.4))
        else:
            cv.drawCircle(x, y, r, paint(WHITE, dim * (0.35 + 0.35 * math.sin(t * 1.7 + ph))))


def globe(cv, cx, cy, r, light, dark, phase=0.0, a=1.0, grid_a=0.28):
    skia = S['skia']
    if a <= 0 or r <= 1:
        return
    sh = skia.GradientShader.MakeRadial(skia.Point(cx - r * 0.3, cy - r * 0.4), r * 1.5, [light, dark])
    p = skia.Paint(AntiAlias=True, Shader=sh); p.setAlphaf(clamp(a))
    cv.drawCircle(cx, cy, r, p)
    cv.save()
    path = skia.Path(); path.addCircle(cx, cy, r)
    cv.clipPath(path, doAntiAlias=True)
    gp = paint(WHITE, grid_a * a, stroke=max(2, r * 0.018))
    for k in range(4):
        ph = (phase + k / 4.0) % 1.0
        rx = abs(math.cos(ph * math.pi)) * r
        cv.drawOval(skia.Rect(cx - rx, cy - r, cx + rx, cy + r), gp)
    for lat in (-0.5, 0, 0.5):
        yy = cy + lat * r
        hw = math.sqrt(max(0, r * r - (lat * r) ** 2))
        cv.drawOval(skia.Rect(cx - hw, yy - r * 0.12, cx + hw, yy + r * 0.12), gp)
    cv.restore()
    cv.drawCircle(cx, cy, r, paint(WHITE, 0.18 * a, stroke=3))


def moon(cv, cx, cy, r, a=1.0):
    skia = S['skia']
    if a <= 0 or r <= 1:
        return
    sh = skia.GradientShader.MakeRadial(skia.Point(cx - r * 0.3, cy - r * 0.35), r * 1.5, [0xFFE9ECF5, 0xFF6B7290])
    p = skia.Paint(AntiAlias=True, Shader=sh); p.setAlphaf(clamp(a))
    cv.drawCircle(cx, cy, r, p)
    for (dx, dy, rr) in [(-0.3, -0.2, 0.18), (0.25, 0.15, 0.22), (-0.05, 0.42, 0.12), (0.35, -0.35, 0.1)]:
        cv.drawCircle(cx + dx * r, cy + dy * r, rr * r, paint(0xFF5A6080, 0.35 * a))


def sun(cv, cx, cy, r, a=1.0, t=0.0):
    skia = S['skia']
    glow = skia.GradientShader.MakeRadial(skia.Point(cx, cy), r * 2.2, [acol(0xFFFFB347, 0.45 * a), 0x00000000])
    cv.drawCircle(cx, cy, r * 2.2, skia.Paint(AntiAlias=True, Shader=glow))
    sh = skia.GradientShader.MakeRadial(skia.Point(cx - r * 0.2, cy - r * 0.2), r * 1.3, [0xFFFFF6C8, 0xFFFF8A00])
    p = skia.Paint(AntiAlias=True, Shader=sh); p.setAlphaf(clamp(a))
    cv.drawCircle(cx, cy, r * (1 + 0.015 * math.sin(t * 4)), p)


def body(cv, kind, cx, cy, r, phase=0.0, a=1.0, t=0.0, colors=None):
    kind = (kind or 'earth').lower()
    if kind == 'moon':
        moon(cv, cx, cy, r, a)
    elif kind == 'sun':
        sun(cv, cx, cy, r, a, t)
    elif kind == 'twin':
        twin_mark(cv, cx, cy, r / 100.0, a)
    elif kind == 'none':
        return
    else:
        l, d = colors or BODIES.get(kind, BODIES['earth'])
        globe(cv, cx, cy, r, l, d, phase, a)


def twin_mark(cv, cx, cy, s, a=1.0):
    skia = S['skia']
    r, d = 100 * s, 40 * s
    globe(cv, cx + d, cy, r, *BODIES['parallel'], 0.1, a, 0.22)
    globe(cv, cx - d, cy, r, *BODIES['earth'], 0.1, a, 0.22)
    cv.save()
    pth = skia.Path(); pth.addCircle(cx - d, cy, r)
    cv.clipPath(pth, doAntiAlias=True)
    cv.drawCircle(cx + d, cy, r, paint(WHITE, 0.95 * a))
    cv.restore()


def brand_tag(cv, a=1.0):
    twin_mark(cv, 412, 150, 0.16, a)
    text(cv, 'PARALLEL EARTH', 448, 162, 34, WHITE, 0.85 * a, 'Bold', 'left', 4)


def fade_in(cv, t, dur=0.22):
    skia = S['skia']
    if t < dur:
        cv.drawRect(skia.Rect(0, 0, W, H), paint(BG, 1 - t / dur))


def labels(cv, t, sc, y=1330, start=0.6):
    k = ease((t - start) / 0.5)
    lab, sub = str(sc.get('label', '') or ''), str(sc.get('sublabel', '') or '')
    if lab:
        size = fit_size([lab.upper()], 960, 96)
        text(cv, lab.upper(), W / 2, y, size, S['accent'], k)
    if sub:
        text(cv, sub, W / 2, y + 62, 42, 0xFFC9D0EA if has_bg() else MUTED, k, 'SemiBold')


# ------------------------------------------------------------ scene types
def sc_title(cv, t, D, sc):
    skia = S['skia']
    bg(cv, t)
    vis = sc.get('visual', 'earth')
    eff = sc.get('effect', 'spin')
    cx, cy, r = 540, 1080, 235
    if has_bg():
        pass
    elif vis == 'moon' or eff == 'vanish':
        # planet with an orbiting object that vanishes
        ang = -0.6 + t * 0.9
        mx, my = cx + 400 * math.cos(ang), cy + 150 * math.sin(ang)
        van = ease((t - D * 0.45) / 0.9) if eff == 'vanish' else 0
        cv.drawOval(skia.Rect(cx - 400, cy - 150, cx + 400, cy + 150), paint(WHITE, 0.16 * (1 - van), stroke=3))
        behind = math.sin(ang) < 0
        if behind: moon(cv, mx, my, 62, 1 - van)
        body(cv, 'earth' if vis == 'moon' else vis, cx, cy, r, t * 0.15, 1, t)
        if not behind: moon(cv, mx, my, 62, 1 - van)
    else:
        sc2 = 1 + (0.05 * math.sin(t * 3) if eff == 'pulse' else 0)
        body(cv, vis, cx, cy, r * sc2, t * 0.15, ease(t / 0.6), t)
    headline(cv, sc.get('head'), t * 1.6, 400, 112)
    brand_tag(cv)


def sc_globe(cv, t, D, sc):
    skia = S['skia']
    eff = (sc.get('effect') or 'spin').lower()
    kind = (sc.get('body') or 'earth').lower()
    k = ease_io((t - 0.4) / max(0.5, D * 0.75))
    tint, ta = None, 0
    if eff == 'heat': tint, ta = 0xFFFF5A1E, 0.18 * k
    if eff == 'freeze': tint, ta = 0xFF7FD3FF, 0.14 * k
    bg(cv, t, 1.0, tint, ta)
    cx, cy, r = 540, 960, 225
    phase = t * 0.25
    a = 1.0
    colors = BODIES.get(kind, BODIES['earth'])
    if eff == 'stop':
        phase = 0.25 * (D * 0.75) * (1 - (1 - k) ** 2) / 2
    elif eff == 'shrink':
        r *= 1 - 0.6 * k
    elif eff == 'grow':
        r *= 1 + 0.45 * k
    elif eff == 'fade':
        a = 1 - 0.92 * k
    elif eff in ('freeze', 'heat'):
        tgt = BODIES['ice'] if eff == 'freeze' else BODIES['lava']
        colors = (lerp_col(colors[0], tgt[0], k), lerp_col(colors[1], tgt[1], k))
    if has_bg():
        pass
    elif eff == 'wobble':
        ang = 23.4 + k * (34 * math.sin(t * 2.3) + 18 * math.sin(t * 5.1 + 1))
        cv.save(); cv.rotate(ang, cx, cy)
        cv.drawLine(cx, cy - r - 130, cx, cy + r + 130, paint(WHITE, 0.7, stroke=6))
        body(cv, kind, cx, cy, r, phase, 1, t, colors)
        cv.restore()
    elif eff == 'split':
        off = 90 * k
        for side, dx in ((-1, -off), (1, off)):
            cv.save()
            cv.clipRect(skia.Rect(0 if side < 0 else cx, 0, cx if side < 0 else W, H))
            body(cv, kind, cx + dx, cy + (30 * k * side), r, phase, 1, t, colors if side < 0 else BODIES['parallel'])
            cv.restore()
    else:
        body(cv, kind, cx, cy, r, phase, a, t, colors)
        if eff == 'crack':
            pts = [(cx - r * 0.9, cy - r * 0.2), (cx - r * 0.4, cy + r * 0.05), (cx - r * 0.1, cy - r * 0.25), (cx + r * 0.3, cy + r * 0.1), (cx + r * 0.85, cy - r * 0.05)]
            n = 1 + int(k * (len(pts) - 1) + 0.999)
            p = skia.Path(); p.moveTo(*pts[0])
            for q in pts[1:n]: p.lineTo(*q)
            cv.drawPath(p, paint(0xFFFFE6A0, 0.95, stroke=9))
        if eff == 'fade':
            cv.drawCircle(cx, cy, r, paint(WHITE, 0.25 * k, stroke=3))
        if eff == 'heat':
            glow = skia.GradientShader.MakeRadial(skia.Point(cx, cy), r * 1.8, [acol(0xFFFF6A00, 0.35 * k), 0x00000000])
            cv.drawCircle(cx, cy, r * 1.8, skia.Paint(AntiAlias=True, Shader=glow))
    labels(cv, t, sc)
    headline(cv, sc.get('head'), t)
    brand_tag(cv)
    fade_in(cv, t)


def fmt_num(v, dec):
    s = f'{v:,.{dec}f}'
    return s


def sc_counter(cv, t, D, sc):
    bg(cv, t)
    try:
        a0, a1 = float(sc.get('from', 0)), float(sc.get('to', 100))
    except Exception:
        a0, a1 = 0.0, 100.0
    dec = int(sc.get('decimals', 0) or 0)
    k = ease_io((t - 0.5) / max(0.5, D * 0.6))
    v = a0 + (a1 - a0) * k
    s = f"{sc.get('prefix', '')}{fmt_num(v, dec)}{sc.get('suffix', '')}"
    size = fit_size([s], 640, 170)
    text(cv, s, 540, 960 + size * 0.36, size, lerp_col(WHITE, S['accent'], k), 1, 'Black')
    lab = str(sc.get('label', '') or '').upper()
    if lab:
        lsz = fit_size([lab], 560, 40, 'Bold', 26)
        text(cv, lab, 540, 960 + size * 0.36 + 70, lsz, 0xFFC9D0EA if has_bg() else MUTED, 1, 'Bold', spacing=4)
    # progress ring
    skia = S['skia']
    frac = clamp(abs(v - a0) / max(1e-9, abs(a1 - a0)))
    cv.drawCircle(540, 960, 330, paint(WHITE, 0.07, stroke=10))
    rect = skia.Rect(210, 630, 870, 1290)
    p = paint(S['accent'], 0.8, stroke=10)
    cv.drawArc(rect, -90, 360 * frac, False, p)
    if sc.get('sublabel'):
        text(cv, str(sc['sublabel']), 540, 1385, 44, MUTED, ease((t - 1) / 0.5), 'SemiBold')
    headline(cv, sc.get('head'), t)
    brand_tag(cv)
    fade_in(cv, t)


def sc_bars(cv, t, D, sc):
    skia = S['skia']
    bg(cv, t, 0.6)
    n = int(clamp(int(sc.get('n', 7) or 7), 3, 9))
    bw = 760 / n * 0.72
    gap = (760 - bw * n) / max(1, n - 1)
    x0 = 160
    base = 1290
    rnd = random.Random(11)
    hs = [rnd.uniform(330, 520) for _ in range(n)]
    up = str(sc.get('trend', 'down')).lower() == 'up'
    for i in range(n):
        kk = ease_io((t - 0.4 - i * 0.15) / 1.4)
        h = hs[i] * ((0.18 + 0.82 * kk) if up else (1 - 0.82 * kk))
        col = lerp_col(BLUE_L, S['accent'], kk)
        r = skia.RRect.MakeRectXY(skia.Rect(x0 + i * (bw + gap), base - h, x0 + i * (bw + gap) + bw, base), 16, 16)
        cv.drawRRect(r, paint(col, 0.92))
    cv.drawLine(70, base + 4, W - 70, base + 4, paint(WHITE, 0.4, stroke=4))
    if sc.get('label'):
        text(cv, str(sc['label']), 540, base + 70, 42, MUTED, 1, 'SemiBold')
    headline(cv, sc.get('head'), t)
    brand_tag(cv)
    fade_in(cv, t)


def sc_words(cv, t, D, sc):
    words = [str(w) for w in (sc.get('words') or ['WHAT IF'])][:3]
    n = len(words)
    seg = D / n
    i = min(n - 1, int(t // seg))
    lt = t - i * seg
    tints = [0xFF4FB3FF, S['accent'], 0xFF8A5CFF]
    cols = [BLUE_L, S['accent'], WHITE]
    bg(cv, t, 0.7, tints[i % 3], 0.16)
    k = ease(lt / 0.3)
    sc_ = 1.35 - 0.35 * k
    lines = wrap(words[i].upper(), 150, 960)
    size = fit_size(lines, 960, 150)
    jit = (math.sin(t * 60) * 5, math.cos(t * 47) * 5) if i == n - 1 and n > 1 else (0, 0)
    y0 = 1040 - (len(lines) - 1) * size * 0.55
    for j, l in enumerate(lines):
        text(cv, l, 540 + jit[0], y0 + j * size * 1.1 + jit[1], size, cols[i % 3], k, 'Black', scale=sc_)
    for j in range(n):
        cv.drawCircle(540 + (j - (n - 1) / 2) * 60, 1360, 12, paint(WHITE, 0.9 if j == i else 0.25))
    brand_tag(cv)
    fade_in(cv, lt, 0.12)


def sc_compare(cv, t, D, sc):
    skia = S['skia']
    bg(cv, t)
    L = sc.get('left') or {}
    R = sc.get('right') or {}
    cards = [(str(L.get('name', 'OUR EARTH')).upper(), str(L.get('value', '')), BLUE_L, 820),
             (str(R.get('name', 'PARALLEL EARTH')).upper(), str(R.get('value', '')), S['accent'], 1160)]
    for i, (nm, val, col, y) in enumerate(cards):
        k = ease((t - 0.4 - i * 0.6) / 0.5)
        rr = skia.RRect.MakeRectXY(skia.Rect(100, y - 150 + (1 - k) * 60, 980, y + 150 + (1 - k) * 60), 36, 36)
        cv.drawRRect(rr, paint(col, 0.12 * k))
        cv.drawRRect(rr, paint(col, 0.7 * k, stroke=4))
        text(cv, nm, 540, y - 50 + (1 - k) * 60, 44, col, k, 'Bold', spacing=5)
        vs = fit_size([val], 820, 120)
        text(cv, val, 540, y + 80 + (1 - k) * 60, vs, WHITE, k, 'Black')
    text(cv, 'VS', 540, 1008, 54, MUTED, ease((t - 0.8) / 0.4), 'Black')
    labels(cv, t, sc, 1410, 1.4)
    headline(cv, sc.get('head'), t, 400, 100)
    brand_tag(cv)
    fade_in(cv, t)


def sc_distance(cv, t, D, sc):
    skia = S['skia']
    bg(cv, t)
    k = ease_io(t / D)
    closer = str(sc.get('motion', 'apart')).lower() == 'closer'
    kk = 1 - k if closer else k
    ex, ey = 300, 1150
    bx, by = 700 + 140 * kk, 800 - 180 * kk
    p = paint(WHITE, 0.45, stroke=4)
    p.setPathEffect(skia.DashPathEffect.Make([18, 16], -t * 40))
    if not has_bg():
        cv.drawLine(ex, ey, bx, by, p)
        body(cv, sc.get('a', 'earth'), ex, ey, 150, t * 0.2, 1, t)
        b = sc.get('b', 'moon')
        body(cv, b, bx, by, 85 if b in ('sun',) else 65, t * 0.2, 1, t)
    labels(cv, t, sc, 1390, 1.0)
    headline(cv, sc.get('head'), t)
    brand_tag(cv)
    fade_in(cv, t)


def sc_wave(cv, t, D, sc):
    skia = S['skia']
    bg(cv, t, 0.6)
    k = ease_io((t - 0.6) / max(0.5, D * 0.6))
    up = str(sc.get('trend', 'down')).lower() == 'up'
    amp = (55 + 130 * k) if up else (170 - 115 * k)
    base = 1230 if not up else 1260 - 120 * k

    def wpath(y0, am, tt, fr):
        pth = skia.Path(); pth.moveTo(0, H)
        for x in range(0, W + 12, 12):
            pth.lineTo(x, y0 + am * math.sin(x / W * fr * 2 * math.pi + tt * 1.6))
        pth.lineTo(W, H); pth.close(); return pth
    if not has_bg():
        cv.drawPath(wpath(base - 40, amp * 0.8, t + 1.3, 1.6), paint(BLUE_D, 0.55))
        cv.drawPath(wpath(base, amp, t, 2.2), paint(0xFF2F7BE0, 0.9))
    lab = str(sc.get('label', '') or '')
    if lab:
        size = fit_size([lab], 960, 180)
        text(cv, lab, 540, 850, size, lerp_col(WHITE, S['accent'], k), 1, 'Black')
    if sc.get('sublabel'):
        text(cv, str(sc['sublabel']).upper(), 540, 920, 40, MUTED, 1, 'Bold', spacing=5)
    headline(cv, sc.get('head'), t)
    brand_tag(cv)
    fade_in(cv, t)


def sc_timeline(cv, t, D, sc):
    bg(cv, t)
    pts = (sc.get('points') or [])[:4] or [{'year': '????', 'text': ''}]
    n = len(pts)
    top, bot = 700, 1360
    x = 190
    k = ease_io(t / (D * 0.85))
    cv.drawLine(x, top, x, top + (bot - top) * k, paint(WHITE, 0.35, stroke=6))
    step = (bot - top) / max(1, n - 1) if n > 1 else 0
    for i, p in enumerate(pts):
        y = top + i * step if n > 1 else (top + bot) / 2
        appear = ease((t - (i / n) * D * 0.8) / 0.4)
        last = i == n - 1
        cv.drawCircle(x, y, 22 * appear, paint(S['accent'] if last else BLUE_L, 1))
        text(cv, str(p.get('year', '')), x + 60, y + 22, 76, S['accent'] if last else WHITE, appear, 'Black', 'left')
        for j, l in enumerate(wrap(str(p.get('text', '')), 40, 760, 'SemiBold')[:2]):
            text(cv, l, x + 62, y + 76 + j * 48, 40, MUTED, appear, 'SemiBold', 'left')
    headline(cv, sc.get('head'), t, 400, 100)
    brand_tag(cv)
    fade_in(cv, t)


def sc_fact(cv, t, D, sc):
    bg(cv, t)
    kicker = str(sc.get('kicker', '') or '').upper()
    if kicker:
        text(cv, kicker, 540, 620, 48, S['accent'], ease(t / 0.4), 'Bold', spacing=8)
    big = str(sc.get('big', '') or sc.get('label', '') or '')
    lines = wrap(big.upper(), 110, 940)
    size = fit_size(lines, 940, 110)
    lines = wrap(big.upper(), size, 940)
    y0 = 1000 - (len(lines) - 1) * size * 0.55
    for j, l in enumerate(lines):
        kk = ease((t - 0.2 - j * 0.12) / 0.45)
        text(cv, l, 540, y0 + j * size * 1.12 + (1 - kk) * 30, size, WHITE if j % 2 == 0 else S['accent'], kk)
    if sc.get('sublabel'):
        text(cv, str(sc['sublabel']), 540, y0 + len(lines) * size * 1.12 + 60, 44, MUTED, ease((t - 0.8) / 0.5), 'SemiBold')
    headline(cv, sc.get('head'), t, 400, 96)
    brand_tag(cv)
    fade_in(cv, t)


def sc_brand(cv, t, D, sc):
    bg(cv, t)
    k = ease(t / 1.0)
    if not S.get('duo'):
        twin_mark(cv, 540, 980, 1.9 + 0.25 * k, k)
    text(cv, 'WELCOME TO', 540, 1400, 54, MUTED, ease((t - 0.5) / 0.5), 'Bold', spacing=8)
    text(cv, 'PARALLEL', 540, 1520, 130, WHITE, ease((t - 0.7) / 0.5), 'Black', spacing=6)
    text(cv, 'EARTH', 540, 1640, 110, S['accent'], ease((t - 0.85) / 0.5), 'Black', spacing=24)
    headline(cv, sc.get('head'), t)
    fade_in(cv, t)


def sc_cta(cv, t, D, sc):
    skia = S['skia']
    bg(cv, t)
    if not S.get('duo'):
        twin_mark(cv, 540, 700, 1.0, ease(t / 0.6))
    head = sc.get('head') or ['WOULD YOU', 'SURVIVE IN THIS', '*TIMELINE?*']
    headline(cv, head, t - 0.2, 1020, 100)
    pulse = 1 + 0.04 * math.sin(t * 6)
    k = ease((t - 0.8) / 0.4)
    btn = str(sc.get('button', 'FOLLOW FOR MORE')).upper()
    if sc.get('a') and sc.get('b'):
        for j, (lab, val, col) in enumerate((('A', sc['a'], BLUE_L), ('B', sc['b'], 0xFFFFB862))):
            kk = ease((t - 0.6 - j * 0.25) / 0.35)
            yy = 1290 + j * 150
            cv.save(); cv.translate(540, yy); cv.scale(0.7 + 0.3 * kk, 0.7 + 0.3 * kk)
            cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-430, -60, 430, 60), 60, 60), paint(col, 0.18 * kk))
            cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-430, -60, 430, 60), 60, 60), paint(col, kk, stroke=5))
            cv.drawCircle(-370, 0, 42, paint(col, kk))
            cv.restore()
            text(cv, lab, 540 - 370 * (0.7 + 0.3 * kk), yy + 20, 54, BG, kk, 'Black')
            vv = str(val).upper()
            vs = fit_size([vv], 640, 50, 'Black', 30)
            text(cv, vv, 590, yy + vs * 0.36, vs, WHITE, kk, 'Black')
        text(cv, 'COMMENT A OR B', 540, 1600, 44, S['accent'], ease((t - 1.2) / 0.4) * (0.75 + 0.25 * math.sin(t * 6)), 'Black', spacing=4)
        fade_in(cv, t)
        return
    bs = fit_size([btn], 480, 54)
    cv.save(); cv.translate(540, 1470); cv.scale(pulse, pulse)
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-280, -70, 280, 70), 70, 70), paint(S['accent'], k))
    cv.restore()
    text(cv, btn, 540, 1470 + bs * 0.36, bs, BG, k, 'Black', scale=pulse)
    fade_in(cv, t)


def sc_split(cv, t, D, sc):
    skia = S['skia']
    cv.clear(BG)
    sides = [(S.get('bgL'), 0, 0xFF3A7BFF, BODIES['earth']), (S.get('bgR'), 540, 0xFFFF8A3D, BODIES['parallel'])]
    for img, x0, tint, colors in sides:
        cv.save(); cv.clipRect(skia.Rect(x0, 0, x0 + 540, H))
        if img is not None:
            cv.drawImage(img, x0, 0)
        else:
            bg(cv, t)
            globe(cv, x0 + 270, 980, 190, colors[0], colors[1], t * 0.2)
        cv.drawRect(skia.Rect(x0, 0, x0 + 540, H), paint(tint, 0.13))
        shade = skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)], [acol(BG, 0.85), acol(BG, 0.2), acol(BG, 0.3), acol(BG, 0.9)], [0, 0.35, 0.7, 1])
        cv.drawRect(skia.Rect(x0, 0, x0 + 540, H), skia.Paint(Shader=shade))
        cv.restore()
    # glowing rift in the middle
    pth = skia.Path(); pth.moveTo(540, 0)
    for yy in range(0, H + 60, 60):
        pth.lineTo(540 + math.sin(yy * 0.05 + t * 8) * 14, yy)
    cv.drawPath(pth, paint(WHITE, 0.25, stroke=26)); cv.drawPath(pth, paint(WHITE, 0.95, stroke=6))
    k = ease((t - 0.2) / 0.4)
    text(cv, str(sc.get('left_name', 'OUR EARTH')).upper(), 270, 640, 44, BLUE_L, k, 'Black', spacing=3)
    text(cv, str(sc.get('right_name', "PARA'S EARTH")).upper(), 810, 640, 44, 0xFFFFB862, k, 'Black', spacing=3)
    for val, xx, col, st in ((sc.get('left_value', ''), 270, WHITE, 0.5), (sc.get('right_value', ''), 810, 0xFFFFB862, 0.9)):
        val = str(val or '').upper()
        if val:
            kk = ease((t - st) / 0.4)
            vs = fit_size(wrap(val, 84, 470) or [val], 470, 84, 'Black', 40)
            for j, ln in enumerate(wrap(val, vs, 470)[:3]):
                text(cv, ln, xx, 1060 + j * vs * 1.08, vs, col, kk, 'Black', scale=0.8 + 0.2 * kk)
    headline(cv, sc.get('head'), t, 380, 96)
    brand_tag(cv)
    fade_in(cv, t)


SCENES = {'split': sc_split, 'title': sc_title, 'globe': sc_globe, 'counter': sc_counter, 'bars': sc_bars, 'words': sc_words,
          'compare': sc_compare, 'distance': sc_distance, 'wave': sc_wave, 'timeline': sc_timeline,
          'fact': sc_fact, 'brand': sc_brand, 'cta': sc_cta}



# ------------------------------------------------------------ duo mascots: Terra (our Earth) & Para (parallel Earth)
CHAR = {
    'terra': {'light': 0xFF7CCBFF, 'dark': 0xFF1C5FC4, 'patch': 0xFF52C98E, 'name': 'TERRA', 'hi': 0xFF7CD8FF},
    'para': {'light': 0xFFFFC98A, 'dark': 0xFFD24E16, 'patch': 0xFFB8350F, 'name': 'PARA', 'hi': 0xFFFFB862},
}
INK = 0xFF0A0F24


def _arc(cv, cx, cy, w, h, start, sweep, color, width):
    skia = S['skia']
    cv.drawArc(skia.Rect(cx - w, cy - h, cx + w, cy + h), start, sweep, False, paint(color, 1, stroke=width))


def mascot(cv, who, cx, cy, r, mood='happy', mouth=0.0, t=0.0, look=0.0, talking=False, a=1.0, wave=0.0,
           action='', point=None, spin=0.0):
    skia = S['skia']
    c = CHAR.get(who, CHAR['terra'])
    if a <= 0 or r <= 2:
        return
    breathe = 1 + 0.018 * math.sin(t * 2.2 + (0 if who == 'terra' else 1.7))
    bob = -abs(math.sin(t * 9)) * 7 * mouth if talking else 0
    dx = dy = rot = 0.0
    sy = 1.0
    if action == 'jump':
        ph = abs(math.sin(t * 5.5)); dy = -ph * r * 0.45; sy = 1 - 0.12 * (1 - ph) ** 6
    elif action == 'tremble':
        dx = math.sin(t * 70) * r * 0.045
    elif action == 'shiver':
        dx = math.sin(t * 55) * r * 0.06; dy = math.cos(t * 47) * r * 0.02
    elif action == 'float':
        dy = math.sin(t * 1.6) * r * 0.28; rot = math.sin(t * 1.1) * 12
    elif action == 'dizzy':
        rot = math.sin(t * 3) * 16
    elif action == 'sweat':
        sy = 0.94 + 0.02 * math.sin(t * 3)
    rot += spin
    cv.save()
    cv.translate(cx + dx, cy + bob + dy); cv.rotate(rot); cv.scale(breathe, sy / breathe); cv.translate(-cx, -cy)
    # little arms (wave on brand/cta)
    arm = paint(c['dark'], a, stroke=r * 0.1)
    for side in (-1, 1):
        sx = cx + side * r * 0.92
        ang = (0.6 + 0.5 * math.sin(t * 10)) if (wave and side == (1 if who == 'terra' else -1)) else 0.25
        ex, ey = sx + side * r * 0.35 * math.cos(ang), cy + r * 0.15 - r * 0.45 * math.sin(ang)
        if point and side == (1 if who == 'terra' else -1):
            vx, vy = point[0] - sx, point[1] - (cy + r * 0.2)
            L = math.hypot(vx, vy) or 1
            ex, ey = sx + vx / L * r * 0.75, cy + r * 0.2 + vy / L * r * 0.75
        cv.drawLine(sx, cy + r * 0.2, ex, ey, arm)
        cv.drawCircle(ex, ey, r * 0.11, paint(c['light'], a))
    # body
    sh = skia.GradientShader.MakeRadial(skia.Point(cx - r * 0.35, cy - r * 0.4), r * 1.6, [c['light'], c['dark']])
    p = skia.Paint(AntiAlias=True, Shader=sh); p.setAlphaf(a)
    cv.drawCircle(cx, cy, r, p)
    cv.save()
    pth = skia.Path(); pth.addCircle(cx, cy, r); cv.clipPath(pth, doAntiAlias=True)
    for (dx, dy, rx, ry) in [(-0.55, 0.45, 0.42, 0.28), (0.62, -0.55, 0.35, 0.22), (0.5, 0.7, 0.3, 0.2), (-0.75, -0.35, 0.22, 0.3)]:
        cv.drawOval(skia.Rect(cx + (dx - rx) * r, cy + (dy - ry) * r, cx + (dx + rx) * r, cy + (dy + ry) * r), paint(c['patch'], 0.55 * a))
    cv.restore()
    cv.drawCircle(cx, cy, r, paint(INK, a, stroke=r * 0.07))
    # cheeks
    if mood in ('happy', 'excited'):
        for side in (-1, 1):
            cv.drawOval(skia.Rect(cx + side * r * 0.55 - r * 0.15, cy + r * 0.12, cx + side * r * 0.55 + r * 0.15, cy + r * 0.26), paint(0xFFFF7FA0, 0.55 * a))
    # eyes
    blink = ((t + (0.0 if who == 'terra' else 1.3)) % 3.4) < 0.12
    ey = cy - r * 0.12
    erx, ery = r * 0.2, r * 0.25
    if mood == 'shocked':
        erx, ery = r * 0.23, r * 0.29
    pr = r * (0.06 if mood in ('shocked', 'scared') else 0.105)
    for side in (-1, 1):
        ex = cx + side * r * 0.33
        if blink:
            cv.drawLine(ex - erx, ey, ex + erx, ey, paint(INK, a, stroke=r * 0.06))
            continue
        cv.drawOval(skia.Rect(ex - erx, ey - ery, ex + erx, ey + ery), paint(WHITE, a))
        cv.drawOval(skia.Rect(ex - erx, ey - ery, ex + erx, ey + ery), paint(INK, a, stroke=r * 0.04))
        lx = look * erx * 0.45
        ly = (-ery * 0.35) if mood == 'thinking' else (ery * 0.1)
        cv.drawCircle(ex + lx, ey + ly, pr, paint(INK, a))
        cv.drawCircle(ex + lx - pr * 0.35, ey + ly - pr * 0.4, pr * 0.35, paint(WHITE, a))
    # brows
    bw = r * 0.06
    for side in (-1, 1):
        bx = cx + side * r * 0.33
        by = ey - r * 0.36
        if mood == 'shocked':
            _arc(cv, bx, by - r * 0.08, r * 0.17, r * 0.1, 200, 140, INK, bw)
        elif mood == 'scared':
            cv.drawLine(bx - side * r * 0.16, by - r * 0.12, bx + side * r * 0.14, by + r * 0.02, paint(INK, a, stroke=bw))
        elif mood == 'sad':
            cv.drawLine(bx - side * r * 0.16, by - r * 0.06, bx + side * r * 0.14, by + r * 0.04, paint(INK, a, stroke=bw))
        elif mood == 'thinking':
            off = -r * 0.1 if side == 1 else 0
            cv.drawLine(bx - r * 0.15, by + off, bx + r * 0.15, by + off - r * 0.03, paint(INK, a, stroke=bw))
        else:
            _arc(cv, bx, by + r * 0.02, r * 0.15, r * 0.08, 210, 120, INK, bw)
    # mouth
    my = cy + r * 0.38
    open_ = clamp(mouth)
    if talking and open_ > 0.05 or mood == 'shocked':
        h = r * (0.07 + 0.26 * open_) if talking else r * 0.16
        w = r * (0.2 + 0.08 * open_) if mood != 'shocked' else r * 0.14
        cv.drawOval(skia.Rect(cx - w, my - h, cx + w, my + h), paint(0xFF3A0D18, a))
        if h > r * 0.12:
            cv.save()
            mp = skia.Path(); mp.addOval(skia.Rect(cx - w, my - h, cx + w, my + h)); cv.clipPath(mp, doAntiAlias=True)
            cv.drawOval(skia.Rect(cx - w * 0.7, my + h * 0.2, cx + w * 0.7, my + h * 1.4), paint(0xFFFF6F8A, a))
            cv.restore()
        cv.drawOval(skia.Rect(cx - w, my - h, cx + w, my + h), paint(INK, a, stroke=r * 0.045))
    elif mood == 'scared':
        pth = skia.Path(); pth.moveTo(cx - r * 0.22, my)
        for i in range(1, 7):
            pth.lineTo(cx - r * 0.22 + i * r * 0.073, my + (r * 0.05 if i % 2 else -r * 0.05))
        cv.drawPath(pth, paint(INK, a, stroke=r * 0.05))
        cv.drawOval(skia.Rect(cx + r * 0.62, cy - r * 0.55, cx + r * 0.76, cy - r * 0.33), paint(0xFF9FE3FF, 0.9 * a))
    elif mood == 'sad':
        _arc(cv, cx, my + r * 0.12, r * 0.2, r * 0.12, 200, 140, INK, r * 0.055)
    elif mood == 'excited':
        grin = skia.Path()
        grin.addArc(skia.Rect(cx - r * 0.28, my - r * 0.22, cx + r * 0.28, my + r * 0.22), 0, 180)
        grin.close()
        cv.drawPath(grin, paint(0xFF3A0D18, a))
        cv.save(); cv.clipPath(grin, doAntiAlias=True)
        cv.drawOval(skia.Rect(cx - r * 0.18, my + r * 0.08, cx + r * 0.18, my + r * 0.34), paint(0xFFFF6F8A, a))
        cv.restore()
        cv.drawPath(grin, paint(INK, a, stroke=r * 0.045))
    elif mood == 'thinking':
        cv.drawLine(cx - r * 0.12, my + r * 0.02, cx + r * 0.16, my - r * 0.04, paint(INK, a, stroke=r * 0.05))
    else:
        _arc(cv, cx, my - r * 0.08, r * 0.24, r * 0.16, 20, 140, INK, r * 0.055)
    if action == 'shiver':
        cv.drawCircle(cx, cy, r, paint(0xFFBFEFFF, 0.28 * a))
        for k_ in range(5):
            ix = cx - r * 0.6 + k_ * r * 0.3
            iy = cy + math.sqrt(max(0, r * r - (ix - cx) ** 2)) - r * 0.05
            pth = skia.Path(); pth.moveTo(ix - r * 0.06, iy); pth.lineTo(ix + r * 0.06, iy); pth.lineTo(ix, iy + r * (0.18 + 0.06 * (k_ % 2))); pth.close()
            cv.drawPath(pth, paint(0xFFE6FAFF, 0.9 * a))
    if action == 'sweat':
        for k_ in range(3):
            ph = (t * 0.9 + k_ * 0.33) % 1.0
            ddx = cx + (k_ - 1) * r * 0.55
            ddy = cy - r * 0.7 + ph * r * 1.4
            drop = skia.Path(); drop.addCircle(ddx, ddy, r * 0.07)
            drop.moveTo(ddx - r * 0.065, ddy - r * 0.02); drop.lineTo(ddx, ddy - r * 0.17); drop.lineTo(ddx + r * 0.065, ddy - r * 0.02)
            cv.drawPath(drop, paint(0xFF9FE3FF, (1 - ph) * a))
    if action == 'dizzy':
        for k_ in range(3):
            an = t * 4 + k_ * 2.09
            star(cv, cx + math.cos(an) * r * 0.75, cy - r * 1.05 + math.sin(an) * r * 0.18, r * 0.12, 0xFFFFE14D, a)
    cv.restore()


def star(cv, x, y, rr, col, a=1.0):
    skia = S['skia']
    p = skia.Path()
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rad = rr if k % 2 == 0 else rr * 0.45
        (p.moveTo if k == 0 else p.lineTo)(x + math.cos(ang) * rad, y + math.sin(ang) * rad)
    p.close()
    cv.drawPath(p, paint(col, a))


def react_icon(cv, kind, x, y, s, a):
    """Small pop-up reaction above a character: ! ? sweat stars snow."""
    skia = S['skia']
    if a <= 0 or s <= 0:
        return
    cv.save(); cv.translate(x, y); cv.scale(s, s)
    if kind in ('!', '?'):
        cv.drawCircle(0, 0, 38, paint(WHITE, a)); cv.drawCircle(0, 0, 38, paint(INK, a, stroke=5))
        text(cv, kind, 0, 19, 54, 0xFFFF4D4D if kind == '!' else 0xFF3A7BFF, a, 'Black')
    elif kind == 'stars':
        star(cv, -26, 0, 22, 0xFFFFE14D, a); star(cv, 22, -18, 16, 0xFFFFE14D, a); star(cv, 18, 22, 12, 0xFFFFFFFF, a)
    elif kind == 'sweat':
        for dx_, dy_ in ((-14, 0), (16, -14)):
            p = skia.Path(); p.addCircle(dx_, dy_, 11); p.moveTo(dx_ - 10, dy_ - 3); p.lineTo(dx_, dy_ - 28); p.lineTo(dx_ + 10, dy_ - 3)
            cv.drawPath(p, paint(0xFF9FE3FF, a))
    elif kind == 'snow':
        sp_ = paint(0xFFE6FAFF, a, stroke=5)
        for k in range(3):
            an = k * math.pi / 3
            cv.drawLine(-math.cos(an) * 26, -math.sin(an) * 26, math.cos(an) * 26, math.sin(an) * 26, sp_)
    cv.restore()


def mascot_layer(cv, t, sc):
    """Draw Terra & Para over every scene, mouths driven by the voice track."""
    if not S.get('duo'):
        return
    kind = str(sc.get('type', '')).lower()
    tl = sc.get('_tl') or []
    env = sc.get('_env') or []
    fi = int(t * FPS)
    amp = env[fi] if 0 <= fi < len(env) else 0.0
    speaker, mood_of = None, {}
    for (st, en, who, mood) in tl:
        if st <= t < en + 0.05:
            speaker = who
        if st <= t + 0.3:
            mood_of[who] = mood
    default = str(sc.get('mood', 'happy'))
    pos = {'terra': (150, 1470, 100), 'para': (930, 1470, 100)}
    wave = 0.0
    if kind == 'brand':
        pos = {'terra': (330, 1060, 150), 'para': (750, 1060, 150)}; wave = 1.0
    elif kind == 'cta':
        pos = {'terra': (330, 720, 115), 'para': (750, 720, 115)}; wave = 1.0
    eff = str(sc.get('effect', '')).lower()
    forced = str(sc.get('action', '') or '').lower()
    line_start = {}
    for (st, en, who_, mood_) in tl:
        if st <= t:
            line_start[who_] = (st, mood_)
    for who in ('terra', 'para'):
        x, y, r = pos[who]
        talking = speaker == who
        if talking:
            r *= 1.08
        mood = mood_of.get(who, default)
        look = 0.0 if talking else (1.0 if who == 'terra' else -1.0) if speaker else (0.6 if who == 'terra' else -0.6)
        act = forced
        if not act:
            if eff == 'freeze' or 'freez' in str(sc.get('bg', '')):
                act = 'shiver'
            elif eff == 'heat':
                act = 'sweat'
            elif eff == 'wobble':
                act = 'dizzy'
            elif mood == 'scared':
                act = 'tremble'
            elif mood == 'excited' and talking:
                act = 'jump'
        point = None
        if talking and kind in ('counter', 'bars', 'compare', 'timeline', 'wave', 'globe', 'distance', 'split') and act in ('', 'tremble'):
            point = (540, 960)
        pop, spin = 1.0, 0.0
        if kind == 'title' and who == 'para':
            pop = ease((t - 0.7) / 0.5); spin = (1 - pop) * 540
            pr_ = clamp((t - 0.45) / 0.35) * (1 - clamp((t - 1.9) / 0.5))
            if pr_ > 0:
                portal_ring(cv, x, y, r * 1.45 * pr_, t, pr_)
        elif kind == 'title':
            pop = ease(t / 0.35)
        if pop <= 0:
            continue
        mascot(cv, who, x, y + (1 - pop) * (0 if who == 'para' and kind == 'title' else 200), r * (pop if (kind == 'title' and who == 'para') else 1.0),
               mood, amp if talking else 0.0, t, look, talking, 1.0, wave, act, point, spin)
        if who in line_start:
            st_, m_ = line_start[who]
            icon = {'shocked': '!', 'thinking': '?', 'scared': 'sweat', 'excited': 'stars'}.get(m_)
            if act == 'shiver':
                icon = 'snow'
            age = t - st_
            if icon and age < 1.1:
                k_ = ease(age / 0.18) * (1 - clamp((age - 0.85) / 0.25))
                react_icon(cv, icon, x + (40 if who == 'terra' else -40), y - r - 55 - 20 * ease(age / 0.5), k_, k_)
        if kind not in ('brand', 'cta'):
            nm = CHAR[who]['name']
            text(cv, nm, x, y + r + 44, 28, CHAR[who]['hi'], 0.9, 'Black', spacing=3)


def portal_ring(cv, x, y, rad, t, a):
    skia = S['skia']
    if rad <= 2:
        return
    glow = skia.GradientShader.MakeRadial(skia.Point(x, y), rad * 1.3, [acol(0xFF8A5CFF, 0.55 * a), acol(0xFF2E1A6B, 0.35 * a), 0x00000000])
    cv.drawCircle(x, y, rad * 1.3, skia.Paint(AntiAlias=True, Shader=glow))
    for k in range(4):
        col = BLUE_L if k % 2 == 0 else 0xFFFFB862
        rr = rad * (1 - k * 0.16)
        cv.drawArc(skia.Rect(x - rr, y - rr, x + rr, y + rr), (t * 320 * (1 if k % 2 else -1) + k * 70) % 360, 230, False, paint(col, 0.9 * a, stroke=max(3, rad * 0.06)))

# ------------------------------------------------------------ captions + camera
NO_CAPTION = ('title', 'words', 'brand', 'cta')


def word_times(say, start, end):
    words = str(say).split()
    if not words:
        return []
    wts = [len(w) + (4 if w[-1] in '.!?' else 2 if w[-1] in ',;:' else 0) + 2 for w in words]
    tot = float(sum(wts))
    out, c = [], start
    for w, wt in zip(words, wts):
        d = (end - start) * wt / tot
        out.append((w, c, c + d)); c += d
    return out


def chunks(wt, maxw=3):
    groups, cur = [], []
    for item in wt:
        cur.append(item)
        if len(cur) >= maxw or item[0][-1] in '.!?,;:':
            groups.append(cur); cur = []
    if cur:
        groups.append(cur)
    return groups


def captions(cv, t, sc):
    skia = S['skia']
    if str(sc.get('type', '')).lower() in NO_CAPTION or not sc.get('_speech'):
        return
    tl = sc.get('_tl') or [(sc['_speech'][0], sc['_speech'][1], 'narrator', '')]
    lines = sc.get('_texts') or [sc.get('say', '')]
    groups, who_of = [], []
    for (st, en, who, _), txt in zip(tl, lines):
        for g in chunks(word_times(txt, st, en)):
            groups.append(g); who_of.append(who)
    cur, cur_who = None, None
    for g, w_ in zip(groups, who_of):
        if g[0][1] <= t < g[-1][2] + 0.05:
            cur, cur_who = g, w_; break
    if cur is None:
        return
    words = [re.sub(r'[^\w%°+\-.\'’$,]', '', w).strip('.,;:') .upper() for w, _, _ in cur]
    words = [w for w in words if w] or ['']
    if S.get('duo') and cur_who in CHAR:
        bubble(cv, t, cur, words, cur_who)
        return
    size = 86
    f = font(size, 'Black')
    sp = f.measureText(' ')
    total = sum(f.measureText(w) for w in words) + sp * (len(words) - 1)
    maxw = 600 if S.get('duo') else 980
    while total > maxw and size > 46:
        size -= 6; f = font(size, 'Black'); sp = f.measureText(' ')
        total = sum(f.measureText(w) for w in words) + sp * (len(words) - 1)
    pop = ease((t - cur[0][1]) / 0.12)
    y = 1585
    cv.save(); cv.translate(540, y); s_ = 0.85 + 0.15 * pop; cv.scale(s_, s_); cv.translate(-540, -y)
    x = 540 - total / 2
    stroke = skia.Paint(AntiAlias=True, Color=0xFF05070F, Style=skia.Paint.kStroke_Style, StrokeWidth=14)
    stroke.setStrokeJoin(skia.Paint.kRound_Join)
    for w, (_, a, b) in zip(words, cur):
        active = a <= t < b + 0.05
        cv.drawString(w, x, y + size * 0.35, f, stroke)
        hi = CHAR[cur_who]['hi'] if cur_who in CHAR else CAPTION_HI
        cv.drawString(w, x, y + size * 0.35, f, paint(hi if active else WHITE))
        x += f.measureText(w) + sp
    cv.restore()


CAPTION_HI = 0xFFFFE14D


def bubble(cv, t, cur, words, who):
    skia = S['skia']
    size = 64
    f = font(size, 'Black'); sp = f.measureText(' ')
    total = sum(f.measureText(w) for w in words) + sp * (len(words) - 1)
    while total > 560 and size > 40:
        size -= 4; f = font(size, 'Black'); sp = f.measureText(' ')
        total = sum(f.measureText(w) for w in words) + sp * (len(words) - 1)
    bw, bh = max(240, total + 64), size + 50
    cx, cy = 540, 1575
    pop = ease((t - cur[0][1]) / 0.14)
    sx_ = 0.6 + 0.4 * pop
    tail_x = 300 if who == 'terra' else 780
    cv.save(); cv.translate(tail_x, cy); cv.scale(sx_, sx_); cv.translate(-tail_x, -cy)
    rect = skia.Rect(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    tail = skia.Path()
    if who == 'terra':
        tail.moveTo(rect.left() + 30, cy - 10); tail.lineTo(250, cy - 70); tail.lineTo(rect.left() + 36, cy + 22)
    else:
        tail.moveTo(rect.right() - 30, cy - 10); tail.lineTo(830, cy - 70); tail.lineTo(rect.right() - 36, cy + 22)
    tail.close()
    shadow = paint(0xFF000000, 0.35)
    cv.drawRRect(skia.RRect.MakeRectXY(rect.makeOffset(0, 8), 34, 34), shadow)
    cv.drawPath(tail, paint(WHITE)); cv.drawPath(tail, paint(INK, 1, stroke=6))
    cv.drawRRect(skia.RRect.MakeRectXY(rect, 34, 34), paint(WHITE))
    cv.drawRRect(skia.RRect.MakeRectXY(rect, 34, 34), paint(INK, 1, stroke=6))
    cv.drawPath(tail, paint(WHITE))
    x = cx - total / 2
    hi = CHAR[who]['dark']
    for w, (_, a, b) in zip(words, cur):
        active = a <= t < b + 0.05
        cv.drawString(w, x, cy + size * 0.36, f, paint(hi if active else INK))
        x += f.measureText(w) + sp
    cv.restore()


def camera(cv, t, D, kind):
    z = 1 + 0.09 * (1 - ease(t / 0.32)) + 0.045 * (t / max(D, 0.1))
    dx = dy = 0.0
    if kind == 'title' and t < 0.45:
        k = 1 - t / 0.45
        dx, dy = math.sin(t * 90) * 14 * k, math.cos(t * 77) * 14 * k
    cv.translate(540 + dx, 960 + dy); cv.scale(z, z); cv.translate(-540, -960)


def open_bg(path, D, w=W, h=H):
    cmd = ['ffmpeg', '-loglevel', 'error', '-stream_loop', '-1', '-i', path, '-t', f'{D + 0.5:.2f}', '-an',
           '-vf', f'scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={FPS},format=rgba', '-f', 'rawvideo', '-']
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def next_bg(proc, last, w=W, h=H):
    skia = S['skia']
    buf = proc.stdout.read(w * h * 4) if proc else b''
    if len(buf) < w * h * 4:
        return last
    return skia.Image.fromarray(np.frombuffer(buf, np.uint8).reshape(h, w, 4).copy(), colorType=skia.kRGBA_8888_ColorType)


def glitch(arr, t, seed):
    """Timeline-jump glitch on the first frames of a scene."""
    k = 1 - t / 0.24
    if k <= 0:
        return arr
    rng = np.random.default_rng(seed)
    out = arr.copy()
    sh = int(34 * k)
    out[:, :, 0] = np.roll(arr[:, :, 0], sh, axis=1)
    out[:, :, 2] = np.roll(arr[:, :, 2], -sh, axis=1)
    for _ in range(7):
        y0 = int(rng.integers(0, H - 90)); hh = int(rng.integers(14, 90))
        out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], int(rng.integers(-90, 90) * k), axis=1)
    return out


def portal_wipe(cv, t):
    """Iris that opens from a swirling portal at the start of a scene in the other timeline."""
    skia = S['skia']
    if t >= 0.75:
        return
    k = ease(t / 0.75)
    R = 30 + 1250 * k
    mask = skia.Path(); mask.setFillType(skia.PathFillType.kEvenOdd)
    mask.addRect(skia.Rect(-10, -10, W + 10, H + 10)); mask.addCircle(540, 960, R)
    cv.drawPath(mask, paint(0xFF05030F, 1.0))
    for j in range(5):
        rr = R + 10 + j * 18
        col = BLUE_L if j % 2 == 0 else 0xFFFFB862
        cv.drawArc(skia.Rect(540 - rr, 960 - rr, 540 + rr, 960 + rr), (t * 700 * (1 if j % 2 else -1) + j * 60) % 360, 250, False,
                   paint(col, 0.9 * (1 - k * 0.6), stroke=10 - j))


# ------------------------------------------------------------ render
def setup_state(cfg):
    import skia
    S['skia'] = skia
    S['tf'] = {w: skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, f'InterDisplay-{w}.otf')) for w in ['Black', 'Bold', 'SemiBold']}
    S['accent'] = ACCENTS.get(str(cfg.get('accent', 'orange')).lower(), ACCENTS['orange'])
    S['style'] = str(cfg.get('style', 'space')).lower()
    S['duo'] = any(isinstance(x, dict) and x.get('lines') for x in cfg.get('scenes', []))
    rnd = random.Random(int(hashlib.md5(str(cfg.get('title', '')).encode()).hexdigest()[:6], 16))
    S['stars'] = [(rnd.uniform(0, W), rnd.uniform(0, H), rnd.uniform(1.2, 3.4), rnd.uniform(0, 6.28), rnd.uniform(0.3, 1.5)) for _ in range(140)]


def render_scene(args):
    cfg, i, D, path = args
    setup_state(cfg)
    skia = S['skia']
    sc = cfg['scenes'][i]
    fn = SCENES.get(str(sc.get('type', 'fact')).lower(), sc_fact)
    surf = skia.Surface(W, H)
    n = int(round(D * FPS))
    tmp = path[:-4] + '.part.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-r', str(FPS), tmp]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    kind = str(sc.get('type', 'fact')).lower()
    bgp = open_bg(sc['_bg'], D) if sc.get('_bg') and os.path.exists(sc['_bg']) else None
    bgl = open_bg(sc['_bgL'], D, 540, H) if sc.get('_bgL') and os.path.exists(sc['_bgL']) else None
    bgr = open_bg(sc['_bgR'], D, 540, H) if sc.get('_bgR') and os.path.exists(sc['_bgR']) else None
    S['bgimg'] = S['bgL'] = S['bgR'] = None
    idx = int(sc.get('_idx', 0))
    portal = bool(sc.get('_portal'))
    for f in range(n):
        t = f / FPS
        if bgp:
            S['bgimg'] = next_bg(bgp, S['bgimg'])
        if bgl:
            S['bgL'] = next_bg(bgl, S['bgL'], 540, H)
        if bgr:
            S['bgR'] = next_bg(bgr, S['bgR'], 540, H)
        with surf as cv:
            cv.clear(BG)
            cv.save(); camera(cv, t, D, kind); fn(cv, t, D, sc); cv.restore()
            mascot_layer(cv, t, sc)
            captions(cv, t, sc)
            if portal:
                portal_wipe(cv, t)
        arr = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)
        if idx > 0 and not portal and t < 0.24:
            arr = glitch(arr, t, idx * 1000 + f)
        pr.stdin.write(arr.tobytes())
    pr.stdin.close()
    for p_ in (bgp, bgl, bgr):
        if p_:
            p_.kill()
    ok = pr.wait() == 0
    if ok:
        os.replace(tmp, path)
    return ok


def still(cfg, i, D, t, path):
    setup_state(cfg)
    skia = S['skia']
    sc = cfg['scenes'][i]
    fn = SCENES.get(str(sc.get('type', 'fact')).lower(), sc_fact)
    surf = skia.Surface(W, H)
    S['bgimg'] = S['bgL'] = S['bgR'] = None
    if sc.get('_bg') and os.path.exists(sc['_bg']):
        p = open_bg(sc['_bg'], D); S['bgimg'] = next_bg(p, None); p.kill()
    for key_, sk in (('_bgL', 'bgL'), ('_bgR', 'bgR')):
        if sc.get(key_) and os.path.exists(sc[key_]):
            p = open_bg(sc[key_], D, 540, H); S[sk] = next_bg(p, None, 540, H); p.kill()
    with surf as cv:
        cv.clear(BG)
        cv.save(); camera(cv, t, D, str(sc.get('type', ''))); fn(cv, t, D, sc); cv.restore()
        mascot_layer(cv, t, sc)
        captions(cv, t, sc)
    surf.makeImageSnapshot().save(path, skia.kPNG)


def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')[:50] or 'video'


def trim_silence(a, thr=0.01):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    return a[max(0, idx[0] - int(SR * 0.03)):min(len(a), idx[-1] + int(SR * 0.08))]


def sfx_track(n, starts, kinds, scenes=None):
    rng = np.random.default_rng(5)
    out = np.zeros(n, dtype=np.float32)

    def place(sig, at):
        i = int(at * SR)
        if i >= n:
            return
        m = min(len(sig), n - i)
        out[i:i + m] += sig[:m]
    # whoosh: band-limited noise swell
    L = int(SR * 0.55)
    tt = np.arange(L) / SR
    noise = rng.standard_normal(L).astype(np.float32)
    spec = np.fft.rfft(noise)
    fr = np.fft.rfftfreq(L, 1 / SR)
    spec[(fr < 400) | (fr > 7000)] = 0
    noise = np.fft.irfft(spec, L).astype(np.float32)
    env = np.minimum(tt / 0.3, 1) * np.exp(-np.maximum(tt - 0.3, 0) * 14)
    whoosh = (noise / (np.abs(noise).max() + 1e-9) * env * 0.16).astype(np.float32)
    # boom: falling sine
    Lb = int(SR * 1.2)
    tb = np.arange(Lb) / SR
    freq = 38 + 50 * np.exp(-tb * 6)
    boom = (np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-tb * 3.2) * 0.55).astype(np.float32)
    # cartoon pop, boing, zap, whirl
    Lp = int(SR * 0.12); tp = np.arange(Lp) / SR
    pop = (np.sin(2 * np.pi * (900 - 2500 * tp) * tp) * np.exp(-tp * 40) * 0.22).astype(np.float32)
    Lg = int(SR * 0.45); tg = np.arange(Lg) / SR
    fg = 220 + 160 * np.sin(2 * np.pi * 7 * tg) * np.exp(-tg * 3) + 200 * tg
    boing = (np.sin(2 * np.pi * np.cumsum(fg) / SR) * np.exp(-tg * 6) * 0.2).astype(np.float32)
    Lz = int(SR * 0.22); tz = np.arange(Lz) / SR
    zap = (np.sign(np.sin(2 * np.pi * (1800 - 6000 * tz) * tz)) * rng.uniform(0.6, 1, Lz) * np.exp(-tz * 18) * 0.07).astype(np.float32)
    Lw = int(SR * 0.9); tw = np.arange(Lw) / SR
    fw = 150 + 900 * (tw / 0.9) ** 1.5
    whirl = (np.sin(2 * np.pi * np.cumsum(fw) / SR) * np.sin(np.pi * tw / 0.9) * 0.12 + whoosh[:Lw] * 0.6 if Lw <= len(whoosh) else
             np.sin(2 * np.pi * np.cumsum(fw) / SR) * np.sin(np.pi * tw / 0.9) * 0.12).astype(np.float32)
    for j, (st, kd) in enumerate(zip(starts, kinds)):
        sc = (scenes or [{}] * len(starts))[j] if scenes else {}
        if sc.get('_portal'):
            place(whirl, max(0, st - 0.1))
        elif j > 0:
            place(zap, st)
            place(whoosh, max(0, st - 0.3))
        else:
            place(whoosh, max(0, st - 0.3))
        if kd in ('title', 'words', 'brand'):
            place(boom, st)
        for (ls, le, who, mood) in (sc.get('_tl') or []):
            if mood in ('shocked', 'thinking', 'scared'):
                place(pop, st + ls)
            elif mood == 'excited':
                place(boing, st + ls)
        if kd == 'title' and scenes:
            place(whirl * 0.8, st + 0.45)  # Para's portal entrance
    return out


def pixabay_pick(key, q, used):
    import requests
    r = requests.get('https://pixabay.com/api/videos/', timeout=30,
                     params={'key': key, 'q': q[:100], 'per_page': 20, 'safesearch': 'true', 'video_type': 'film'})
    hits = [h for h in r.json().get('hits', []) if h['id'] not in used and h.get('duration', 0) >= 4]
    if not hits:
        r = requests.get('https://pixabay.com/api/videos/', timeout=30, params={'key': key, 'q': ' '.join(q.split()[:2]), 'per_page': 20, 'safesearch': 'true'})
        hits = [h for h in r.json().get('hits', []) if h['id'] not in used and h.get('duration', 0) >= 4]
    if not hits:
        return None, None
    h = random.Random(q).choice(hits[:6])
    vids = h.get('videos', {})
    for size in ('medium', 'large', 'small', 'tiny'):
        v = vids.get(size) or {}
        if v.get('url') and v.get('height', 0) >= 540:
            return h['id'], v['url']
    for size in ('medium', 'large', 'small', 'tiny'):
        if (vids.get(size) or {}).get('url'):
            return h['id'], vids[size]['url']
    return None, None


def fetch_backgrounds(scenes, work, log=print):
    key = os.environ.get('PEXELS_API_KEY', '').strip()
    pkey = os.environ.get('PIXABAY_API_KEY', '').strip()
    if not key and not pkey:
        return
    import requests
    used = set()

    def grab(q, path):
        if key:
            return False
        vid, url = pixabay_pick(pkey, q, used)
        if not url:
            return False
        with requests.get(url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            with open(path, 'wb') as fh:
                for chunk in dl.iter_content(1 << 20):
                    fh.write(chunk)
        used.add(vid)
        return True
    for i, sc in enumerate(scenes):
        if str(sc.get('type', '')).lower() == 'split':
            for side, k_ in (('bg_left', '_bgL'), ('bg_right', '_bgR')):
                q = str(sc.get(side, '') or '').strip()
                if q:
                    try:
                        pth = os.path.join(work, f'bg_{i+1:02d}_{side}.mp4')
                        if grab(q, pth):
                            sc[k_] = pth; log(f'  split {side} "{q}": ok')
                    except Exception as e:
                        log(f'  split {side} "{q}" failed: {str(e)[:100]}')
            continue
        q = str(sc.get('bg', '') or '').strip()
        if not q or str(sc.get('type', '')).lower() in ('brand', 'cta'):
            continue
        if not key:
            try:
                vid, url = pixabay_pick(pkey, q, used)
                if not url:
                    log(f'  bg "{q}": nothing found'); continue
                path = os.path.join(work, f'bg_{i+1:02d}.mp4')
                with requests.get(url, stream=True, timeout=120) as dl:
                    dl.raise_for_status()
                    with open(path, 'wb') as fh:
                        for chunk in dl.iter_content(1 << 20):
                            fh.write(chunk)
                used.add(vid)
                sc['_bg'] = path
                log(f'  bg "{q}": pixabay video {vid}')
            except Exception as e:
                log(f'  bg "{q}" failed: {str(e)[:120]}')
            continue
        try:
            r = requests.get('https://api.pexels.com/videos/search', headers={'Authorization': key}, timeout=30,
                             params={'query': q, 'orientation': 'portrait', 'size': 'medium', 'per_page': 12})
            vids = [v for v in r.json().get('videos', []) if v['id'] not in used and v.get('duration', 0) >= 4]
            if not vids:
                r = requests.get('https://api.pexels.com/videos/search', headers={'Authorization': key}, timeout=30,
                                 params={'query': q, 'per_page': 12})
                vids = [v for v in r.json().get('videos', []) if v['id'] not in used and v.get('duration', 0) >= 4]
            if not vids:
                log(f'  bg "{q}": nothing found'); continue
            v = random.Random(q).choice(vids[:5])
            files = [f for f in v['video_files'] if f.get('file_type') == 'video/mp4' and f.get('width') and f.get('height')]
            files.sort(key=lambda f: (abs(min(f['width'], f['height']) - 1080), -f['height']))
            if not files:
                continue
            path = os.path.join(work, f'bg_{i+1:02d}.mp4')
            with requests.get(files[0]['link'], stream=True, timeout=120) as dl:
                dl.raise_for_status()
                with open(path, 'wb') as fh:
                    for chunk in dl.iter_content(1 << 20):
                        fh.write(chunk)
            used.add(v['id'])
            sc['_bg'] = path
            log(f'  bg "{q}": pexels video {v["id"]}')
        except Exception as e:
            log(f'  bg "{q}" failed: {str(e)[:120]}')


def S_duo_portal(scenes, j):
    sc = scenes[j]
    if str(sc.get('transition', '')).lower() == 'portal':
        return True
    if str(sc.get('type', '')).lower() == 'split':
        return True
    duo = any(x.get('lines') for x in scenes)
    return duo and j == 1


def build(cfg, out_dir=None, workers=2, log=print):
    import soundfile as sf
    from kokoro_onnx import Kokoro
    from multiprocessing import Pool
    out_dir = out_dir or OUT_DIR
    name = slug(cfg.get('title', 'parallel_earth'))
    work = os.path.join(out_dir, 'pe_work_' + name)
    os.makedirs(work, exist_ok=True)
    scenes = [s for s in cfg.get('scenes', []) if isinstance(s, dict)]
    cfg['scenes'] = scenes
    log(f'Scenes: {len(scenes)}  ->  making voice-over...')
    k = Kokoro(os.path.join(MODEL_DIR, 'kokoro-v1.0.onnx'), os.path.join(MODEL_DIR, 'voices-v1.0.bin'))
    voice = str(cfg.get('voice', 'am_michael'))
    lang = 'en-gb' if voice.startswith('b') else 'en-us'
    speed = float(cfg.get('speed', 1.0) or 1.0)
    min_s = float(cfg.get('min_seconds', 0) or 0)

    vmap = {'terra': 'am_puck', 'para': 'af_bella'}
    vmap.update(cfg.get('voices') or {})

    def tts(txt, v, spd):
        lg = 'en-gb' if v.startswith('b') else 'en-us'
        a, _ = k.create(txt, voice=v, speed=spd, lang=lg)
        return trim_silence(np.asarray(a, dtype=np.float32))

    def voice_pass(spd):
        sg, du = [], []
        for sc in scenes:
            lines = [l for l in (sc.get('lines') or []) if isinstance(l, dict) and str(l.get('text', '')).strip()]
            if lines:
                parts, tl, texts, c = [np.zeros(int(SR * 0.15), dtype=np.float32)], [], [], 0.15
                for j, l in enumerate(lines):
                    who = 'para' if str(l.get('who', '')).lower().startswith('p') else 'terra'
                    a = tts(str(l['text']), vmap[who], spd)
                    tl.append((c, c + len(a) / SR, who, str(l.get('mood', sc.get('mood', 'happy'))).lower()))
                    texts.append(str(l['text']))
                    parts.append(a); c += len(a) / SR
                    if j < len(lines) - 1:
                        parts.append(np.zeros(int(SR * 0.22), dtype=np.float32)); c += 0.22
                parts.append(np.zeros(int(SR * 0.35), dtype=np.float32))
                a = np.concatenate(parts)
                sc['_tl'], sc['_texts'] = tl, texts
                sc['_speech'] = [tl[0][0], tl[-1][1]]
                sc['say'] = ' '.join(texts)
            else:
                say = str(sc.get('say', '') or '').strip()
                a = tts(say, voice, spd) if say else np.zeros(int(SR * 1.2), dtype=np.float32)
                sc['_speech'] = [0.15, 0.15 + len(a) / SR]
                a = np.concatenate([np.zeros(int(SR * 0.15)), a, np.zeros(int(SR * 0.35))]).astype(np.float32)
            hop = SR // FPS
            n = len(a) // hop + 1
            rms = np.array([np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(a) else 0 for i in range(n)])
            ref = np.percentile(rms[rms > 0], 90) if np.any(rms > 0) else 1
            sc['_env'] = [round(float(min(1.0, v / (ref + 1e-9))), 2) for v in rms]
            sg.append(a.astype(np.float32)); du.append(len(a) / SR)
        return sg, du
    segs, durs = voice_pass(speed)
    if min_s and sum(durs) < min_s - 2:
        log(f'Narration {sum(durs):.1f} s is short, slowing voice slightly')
        segs, durs = voice_pass(max(0.85, speed * 0.92))
    if min_s and sum(durs) < min_s:
        extra = min_s - sum(durs) + 0.3  # hold the last scene a little longer
        segs[-1] = np.concatenate([segs[-1], np.zeros(int(SR * extra), dtype=np.float32)])
        durs[-1] += extra
    frames, cum = [], 0.0
    for d in durs:
        s0 = round(cum * FPS); cum += d; frames.append((round(cum * FPS) - s0) / FPS)
    total = sum(frames)
    vo = np.concatenate(segs)
    vo = np.pad(vo, (0, max(0, int(total * SR) - len(vo))))[:int(total * SR)]
    fetch_backgrounds(scenes, work, log)
    for j, sc in enumerate(scenes):
        sc['_idx'] = j
        sc['_portal'] = S_duo_portal(scenes, j)
    starts = np.cumsum([0] + frames[:-1])
    # ambient pad, root note varies per title
    root = [49.0, 55.0, 58.3, 61.7, 65.4][int(hashlib.md5(name.encode()).hexdigest()[:2], 16) % 5]
    tt = np.arange(len(vo)) / SR
    pad = sum(w * np.sin(2 * np.pi * root * m * tt + np.sin(2 * np.pi * 0.07 * tt) * 2) for m, w in [(1, 1.0), (1.5, 0.6), (2, 0.5), (3, 0.25)])
    pad *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * tt)
    fade = np.minimum(1, np.minimum(tt / 2.0, (total - tt) / 2.5))
    pad = pad / np.abs(pad).max() * 0.06 * fade
    mix = vo / max(1e-6, np.abs(vo).max()) * 0.92 + pad + sfx_track(len(vo), starts, [str(sc.get('type', '')) for sc in scenes], scenes)
    sf.write(os.path.join(work, 'audio.wav'), mix.astype(np.float32), SR)
    log(f'Voice-over: {total:.1f} s  ->  rendering {len(scenes)} scenes...')
    jobs = [(cfg, i, frames[i], os.path.join(work, f'scene_{i+1:02d}.mp4')) for i in range(len(scenes))]
    with Pool(workers) as pool:
        res = pool.map(render_scene, jobs)
    if not all(res):
        raise RuntimeError('scene render failed')
    lst = os.path.join(work, 'list.txt')
    open(lst, 'w').write(''.join(f"file 'scene_{i+1:02d}.mp4'\n" for i in range(len(scenes))))
    out = os.path.join(out_dir, f'{name}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-i', os.path.join(work, 'audio.wav'),
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-ar', '48000',
                    '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], check=True)
    json.dump({'frames': frames, 'titles': [str(sc.get('chapter', '')) for sc in scenes]}, open(os.path.join(work, 'timing.json'), 'w'))
    log(f'Done: {out}')
    return out


if __name__ == '__main__':
    cfg = json.load(open(sys.argv[1]))
    if len(sys.argv) > 2 and sys.argv[2] == 'stills':
        os.makedirs(os.path.join(OUT_DIR, 'stills'), exist_ok=True)
        for i in range(len(cfg['scenes'])):
            still(cfg, i, 6.0, 4.0, os.path.join(OUT_DIR, 'stills', f's{i+1:02d}.png'))
    else:
        build(cfg)
