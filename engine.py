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
    twin_mark(cv, 540, 980, 1.9 + 0.25 * k, k)
    text(cv, 'WELCOME TO', 540, 1400, 54, MUTED, ease((t - 0.5) / 0.5), 'Bold', spacing=8)
    text(cv, 'PARALLEL', 540, 1520, 130, WHITE, ease((t - 0.7) / 0.5), 'Black', spacing=6)
    text(cv, 'EARTH', 540, 1640, 110, S['accent'], ease((t - 0.85) / 0.5), 'Black', spacing=24)
    headline(cv, sc.get('head'), t)
    fade_in(cv, t)


def sc_cta(cv, t, D, sc):
    skia = S['skia']
    bg(cv, t)
    twin_mark(cv, 540, 700, 1.0, ease(t / 0.6))
    head = sc.get('head') or ['WOULD YOU', 'SURVIVE IN THIS', '*TIMELINE?*']
    headline(cv, head, t - 0.2, 1020, 100)
    pulse = 1 + 0.04 * math.sin(t * 6)
    k = ease((t - 0.8) / 0.4)
    btn = str(sc.get('button', 'FOLLOW FOR MORE')).upper()
    bs = fit_size([btn], 480, 54)
    cv.save(); cv.translate(540, 1470); cv.scale(pulse, pulse)
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-280, -70, 280, 70), 70, 70), paint(S['accent'], k))
    cv.restore()
    text(cv, btn, 540, 1470 + bs * 0.36, bs, BG, k, 'Black', scale=pulse)
    fade_in(cv, t)


SCENES = {'title': sc_title, 'globe': sc_globe, 'counter': sc_counter, 'bars': sc_bars, 'words': sc_words,
          'compare': sc_compare, 'distance': sc_distance, 'wave': sc_wave, 'timeline': sc_timeline,
          'fact': sc_fact, 'brand': sc_brand, 'cta': sc_cta}


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
    st, en = sc['_speech']
    groups = chunks(word_times(sc.get('say', ''), st, en))
    cur = None
    for g in groups:
        if g[0][1] <= t < g[-1][2] + 0.05:
            cur = g; break
    if cur is None:
        return
    words = [re.sub(r'[^\w%°+\-.\'’$,]', '', w).strip('.,;:') .upper() for w, _, _ in cur]
    words = [w for w in words if w] or ['']
    size = 86
    f = font(size, 'Black')
    sp = f.measureText(' ')
    total = sum(f.measureText(w) for w in words) + sp * (len(words) - 1)
    while total > 980 and size > 52:
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
        cv.drawString(w, x, y + size * 0.35, f, paint(CAPTION_HI if active else WHITE))
        x += f.measureText(w) + sp
    cv.restore()


CAPTION_HI = 0xFFFFE14D


def camera(cv, t, D, kind):
    z = 1 + 0.09 * (1 - ease(t / 0.32)) + 0.045 * (t / max(D, 0.1))
    dx = dy = 0.0
    if kind == 'title' and t < 0.45:
        k = 1 - t / 0.45
        dx, dy = math.sin(t * 90) * 14 * k, math.cos(t * 77) * 14 * k
    cv.translate(540 + dx, 960 + dy); cv.scale(z, z); cv.translate(-540, -960)


def open_bg(path, D):
    cmd = ['ffmpeg', '-loglevel', 'error', '-stream_loop', '-1', '-i', path, '-t', f'{D + 0.5:.2f}', '-an',
           '-vf', f'scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=rgba', '-f', 'rawvideo', '-']
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def next_bg(proc, last):
    skia = S['skia']
    buf = proc.stdout.read(W * H * 4) if proc else b''
    if len(buf) < W * H * 4:
        return last
    return skia.Image.fromarray(np.frombuffer(buf, np.uint8).reshape(H, W, 4).copy(), colorType=skia.kRGBA_8888_ColorType)


# ------------------------------------------------------------ render
def setup_state(cfg):
    import skia
    S['skia'] = skia
    S['tf'] = {w: skia.Typeface.MakeFromFile(os.path.join(FONT_DIR, f'InterDisplay-{w}.otf')) for w in ['Black', 'Bold', 'SemiBold']}
    S['accent'] = ACCENTS.get(str(cfg.get('accent', 'orange')).lower(), ACCENTS['orange'])
    S['style'] = str(cfg.get('style', 'space')).lower()
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
    S['bgimg'] = None
    for f in range(n):
        t = f / FPS
        if bgp:
            S['bgimg'] = next_bg(bgp, S['bgimg'])
        with surf as cv:
            cv.clear(BG)
            cv.save(); camera(cv, t, D, kind); fn(cv, t, D, sc); cv.restore()
            captions(cv, t, sc)
        pr.stdin.write(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType).tobytes())
    pr.stdin.close()
    if bgp:
        bgp.kill()
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
    S['bgimg'] = None
    if sc.get('_bg') and os.path.exists(sc['_bg']):
        p = open_bg(sc['_bg'], D); S['bgimg'] = next_bg(p, None); p.kill()
    with surf as cv:
        cv.clear(BG)
        cv.save(); camera(cv, t, D, str(sc.get('type', ''))); fn(cv, t, D, sc); cv.restore()
        captions(cv, t, sc)
    surf.makeImageSnapshot().save(path, skia.kPNG)


def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')[:50] or 'video'


def trim_silence(a, thr=0.01):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0:
        return a
    return a[max(0, idx[0] - int(SR * 0.03)):min(len(a), idx[-1] + int(SR * 0.08))]


def sfx_track(n, starts, kinds):
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
    for st, kd in zip(starts, kinds):
        place(whoosh, max(0, st - 0.3))
        if kd in ('title', 'words', 'brand'):
            place(boom, st)
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
    for i, sc in enumerate(scenes):
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

    def voice_pass(spd):
        sg, du = [], []
        for sc in scenes:
            say = str(sc.get('say', '') or '').strip()
            a = np.zeros(int(SR * 1.2), dtype=np.float32)
            if say:
                a, _ = k.create(say, voice=voice, speed=spd, lang=lang)
                a = trim_silence(np.asarray(a, dtype=np.float32))
            sc['_speech'] = [0.15, 0.15 + len(a) / SR]
            a = np.concatenate([np.zeros(int(SR * 0.15)), a, np.zeros(int(SR * 0.35))]).astype(np.float32)
            sg.append(a); du.append(len(a) / SR)
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
    starts = np.cumsum([0] + frames[:-1])
    # ambient pad, root note varies per title
    root = [49.0, 55.0, 58.3, 61.7, 65.4][int(hashlib.md5(name.encode()).hexdigest()[:2], 16) % 5]
    tt = np.arange(len(vo)) / SR
    pad = sum(w * np.sin(2 * np.pi * root * m * tt + np.sin(2 * np.pi * 0.07 * tt) * 2) for m, w in [(1, 1.0), (1.5, 0.6), (2, 0.5), (3, 0.25)])
    pad *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * tt)
    fade = np.minimum(1, np.minimum(tt / 2.0, (total - tt) / 2.5))
    pad = pad / np.abs(pad).max() * 0.06 * fade
    mix = vo / max(1e-6, np.abs(vo).max()) * 0.92 + pad + sfx_track(len(vo), starts, [str(sc.get('type', '')) for sc in scenes])
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
