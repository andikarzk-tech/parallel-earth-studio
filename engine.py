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
    wordmark(cv, t - 0.6, 540, 1530, 1.0, 1.0)
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


# ------------------------------------------------------------ animated props (peraga) + weather overlays
def _rnd(i, k=0):
    """Deterministic pseudo-random 0..1 for particle i, channel k."""
    x = math.sin(i * 12.9898 + k * 78.233) * 43758.5453
    return x - math.floor(x)


FX_WORDS = [('storm', ('storm', 'thunder', 'lightning', 'hurricane')), ('rain', ('rain', 'flood', 'downpour', 'monsoon')),
            ('snow', ('snow', 'ice age', 'frozen', 'freez', 'blizzard', 'winter')), ('embers', ('fire', 'burn', 'heat', 'lava', 'hot', 'melt')),
            ('meteors', ('meteor', 'asteroid', 'comet', 'shooting star')), ('bubbles', ('ocean', 'underwater', 'breathe water', 'sea ')),
            ('wind', ('wind', 'tornado', 'gust', 'storm wind')), ('ash', ('volcano', 'eruption', 'ash'))]
FX_LIST = ['rain', 'storm', 'snow', 'embers', 'meteors', 'bubbles', 'wind', 'ash']
PROP_LIST = ['volcano', 'clock', 'planet_crack', 'tornado', 'rocket', 'tree', 'tsunami', 'ice', 'heart', 'brain', 'city', 'thermometer']


def scene_fx(sc):
    fx = str(sc.get('fx', '') or '').lower().strip()
    if fx in FX_LIST:
        return fx
    if fx == 'none' or str(sc.get('type', '')).lower() in ('brand', 'cta', 'fact'):
        return ''
    eff = str(sc.get('effect', '')).lower()
    if eff == 'freeze':
        return 'snow'
    if eff == 'heat':
        return 'embers'
    txt = ' '.join([str(sc.get('bg', ''))] + [str(l.get('text', '')) for l in (sc.get('lines') or []) if isinstance(l, dict)]
                   + [str(x) for x in (sc.get('head') or [])]).lower()
    for name, words in FX_WORDS:
        if any(re.search(r'\b' + re.escape(w.strip()), txt) for w in words):
            return name
    return ''


def fx_layer(cv, t, sc, front=False):
    """Full-screen weather / particle overlay. back layer behind mascots, front layer (rain streaks) above."""
    fx = sc.get('_fx', '')
    if not fx:
        return
    skia = S['skia']
    k = clamp(t / 0.5)
    if fx in ('rain', 'storm'):
        n = 90 if fx == 'rain' else 140
        p = paint(0xFFBFDFFF, 0.55 * k, stroke=3)
        for i in range(n):
            if (i % 2 == 0) != front:
                continue
            sp = 1500 + 700 * _rnd(i, 1)
            x = (_rnd(i) * (W + 300) - 150 + t * 260) % (W + 300) - 150
            y = (_rnd(i, 2) * H + t * sp) % (H + 200) - 100
            cv.drawLine(x, y, x - 22, y - 70, p)
        if fx == 'storm' and not front:
            fl = t % 2.6
            if fl < 0.16:
                cv.drawRect(skia.Rect(0, 0, W, H), paint(0xFFE8F0FF, 0.35 * (1 - fl / 0.16)))
                bolt = skia.Path(); x0 = 300 + 500 * _rnd(int(t // 2.6), 5); bolt.moveTo(x0, 0)
                yy = 0
                for j in range(9):
                    yy += 90; bolt.lineTo(x0 + (_rnd(j, int(t // 2.6)) - 0.5) * 160, yy)
                cv.drawPath(bolt, paint(0xFFFFFFFF, 0.9 * (1 - fl / 0.16), stroke=8))
                cv.drawPath(bolt, paint(0xFF9FD0FF, 0.5 * (1 - fl / 0.16), stroke=22))
        return
    if front:
        return
    if fx == 'snow':
        for i in range(110):
            x = (_rnd(i) * W + math.sin(t * 1.3 + i) * 40) % W
            y = (_rnd(i, 2) * H + t * (90 + 120 * _rnd(i, 1))) % H
            cv.drawCircle(x, y, 3 + 6 * _rnd(i, 3), paint(WHITE, (0.5 + 0.4 * _rnd(i, 4)) * k))
    elif fx == 'embers':
        cv.drawRect(skia.Rect(0, H * 0.55, W, H), skia.Paint(Shader=skia.GradientShader.MakeLinear(
            [skia.Point(0, H * 0.55), skia.Point(0, H)], [0x00000000, acol(0xFFFF5A1F, 0.28 * k)])))
        for i in range(70):
            life = (t * (0.35 + 0.3 * _rnd(i, 1)) + _rnd(i, 2)) % 1.0
            x = _rnd(i) * W + math.sin(t * 2 + i) * 30
            y = H + 40 - life * H * 1.1
            cv.drawCircle(x, y, 3 + 5 * _rnd(i, 3), paint(0xFFFFB347 if i % 3 else 0xFFFF5A1F, (1 - life) * 0.9 * k))
    elif fx == 'meteors':
        for i in range(7):
            per = 1.6 + _rnd(i, 1) * 1.4
            ph = ((t + _rnd(i, 2) * per) % per) / per
            if ph > 0.45:
                continue
            q = ph / 0.45
            x0, y0 = W * (0.3 + 0.9 * _rnd(i)), -60 + 500 * _rnd(i, 3)
            x, y = x0 - q * 900, y0 + q * 700
            sh = skia.GradientShader.MakeLinear([skia.Point(x, y), skia.Point(x + 260, y - 200)], [acol(0xFFFFE6A8, 0.95), 0x00000000])
            cv.drawLine(x, y, x + 260, y - 200, skia.Paint(AntiAlias=True, Shader=sh, StrokeWidth=10, Style=skia.Paint.kStroke_Style))
            cv.drawCircle(x, y, 11, paint(0xFFFFF4D6, 1.0))
    elif fx == 'bubbles':
        cv.drawRect(skia.Rect(0, 0, W, H), paint(0xFF0A5CA8, 0.16 * k))
        for i in range(45):
            life = (t * (0.18 + 0.2 * _rnd(i, 1)) + _rnd(i, 2)) % 1.0
            x = _rnd(i) * W + math.sin(t * 3 + i) * 20
            y = H + 30 - life * (H + 60)
            rr = 8 + 22 * _rnd(i, 3)
            cv.drawCircle(x, y, rr, paint(0xFFCFF1FF, 0.6 * k, stroke=3))
            cv.drawCircle(x - rr * 0.35, y - rr * 0.35, rr * 0.22, paint(WHITE, 0.6 * k))
    elif fx == 'wind':
        for i in range(26):
            y = _rnd(i) * H
            x = ((t * (900 + 600 * _rnd(i, 1)) + _rnd(i, 2) * W * 2) % (W * 2)) - W * 0.5
            ln = 160 + 260 * _rnd(i, 3)
            pth = skia.Path(); pth.moveTo(x, y)
            pth.cubicTo(x + ln * 0.3, y - 18, x + ln * 0.7, y + 18, x + ln, y)
            cv.drawPath(pth, paint(WHITE, 0.35 * k, stroke=4))
    elif fx == 'ash':
        cv.drawRect(skia.Rect(0, 0, W, H), paint(0xFF3A2A20, 0.22 * k))
        for i in range(90):
            x = (_rnd(i) * W + t * 40 + math.sin(t + i) * 25) % W
            y = (_rnd(i, 2) * H + t * (50 + 60 * _rnd(i, 1))) % H
            cv.drawCircle(x, y, 2 + 4 * _rnd(i, 3), paint(0xFF9A928C, 0.7 * k))


# ---- object props: each draws around (cx, cy) at scale s (s=1 -> ~300 px tall); g = growth 0..1 over the scene
def prop_volcano(cv, cx, cy, s, t, g):
    skia = S['skia']
    base, top = cy + 210 * s, cy - 60 * s
    m = skia.Path(); m.moveTo(cx - 330 * s, base); m.lineTo(cx - 70 * s, top); m.lineTo(cx + 70 * s, top); m.lineTo(cx + 330 * s, base); m.close()
    cv.drawPath(m, skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeLinear(
        [skia.Point(cx, top), skia.Point(cx, base)], [0xFF6B4A3A, 0xFF2E1E18])))
    for i in range(3):  # lava streams
        L = clamp(g * 1.6 - i * 0.2)
        x0 = cx + (i - 1) * 40 * s
        pth = skia.Path(); pth.moveTo(x0, top)
        pth.quadTo(x0 + (i - 1) * 90 * s, top + 140 * s * L, x0 + (i - 1) * 160 * s * L, top + 270 * s * L)
        cv.drawPath(pth, paint(0xFFFF6A1A, 0.95, stroke=18 * s))
        cv.drawPath(pth, paint(0xFFFFD166, 0.8, stroke=6 * s))
    for i in range(10):  # smoke
        life = (t * 0.4 + i / 10) % 1.0
        cv.drawCircle(cx + math.sin(i * 2.1 + t) * 60 * s * life, top - 40 * s - life * 380 * s, (30 + 70 * life) * s, paint(0xFF55504E, 0.55 * (1 - life)))
    for i in range(16):  # lava bombs
        per = 1.2 + _rnd(i) * 0.8
        ph = ((t + _rnd(i, 1) * per) % per) / per
        vx = (_rnd(i, 2) - 0.5) * 520 * s
        x = cx + vx * ph
        y = top - (900 * ph - 1100 * ph * ph) * s
        cv.drawCircle(x, y, (8 + 10 * _rnd(i, 3)) * s, paint(0xFFFF8A2A, 1 - ph * 0.5))
    cv.drawOval(skia.Rect(cx - 72 * s, top - 14 * s, cx + 72 * s, top + 14 * s), paint(0xFFFFB347, 0.9))


def prop_clock(cv, cx, cy, s, t, g):
    skia = S['skia']
    R = 230 * s
    cv.drawCircle(cx, cy, R + 18 * s, paint(S['accent'], 1))
    cv.drawCircle(cx, cy, R, paint(0xFFF7F3E8, 1))
    for i in range(12):
        an = i * math.pi / 6
        cv.drawLine(cx + math.sin(an) * R * 0.82, cy - math.cos(an) * R * 0.82, cx + math.sin(an) * R * 0.94, cy - math.cos(an) * R * 0.94, paint(INK, 1, stroke=8 * s))
    speed = 1 + 18 * g * g
    mins = t * speed * 2.2
    for frac, ln, wd in ((mins / 12.0, 0.55, 16), (mins, 0.82, 10)):
        an = frac * 2 * math.pi
        for trail in range(5 if speed > 6 else 1):
            a_ = an - trail * 0.12 * (speed / 19)
            cv.drawLine(cx, cy, cx + math.sin(a_) * R * ln, cy - math.cos(a_) * R * ln, paint(INK, 1 - trail * 0.2, stroke=wd * s))
    cv.drawCircle(cx, cy, 16 * s, paint(0xFFFF4D4D, 1))


def prop_planet_crack(cv, cx, cy, s, t, g):
    skia = S['skia']
    R = 230 * s
    sh_ = (math.sin(t * 60) * 6 * s * g, math.cos(t * 51) * 6 * s * g)
    cx, cy = cx + sh_[0], cy + sh_[1]
    globe(cv, cx, cy, R, *BODIES['earth'], t * 0.1, 1.0)
    cv.save(); pth = skia.Path(); pth.addCircle(cx, cy, R); cv.clipPath(pth, doAntiAlias=True)
    for c_ in range(6):
        an = c_ * 1.05 + 0.3
        pts = [(cx, cy)]
        L = R * 1.1 * clamp(g * 1.4 - c_ * 0.08)
        steps = 7
        for j in range(1, steps + 1):
            d = L * j / steps
            jit = (_rnd(c_ * 10 + j) - 0.5) * 0.5
            pts.append((cx + math.cos(an + jit) * d, cy + math.sin(an + jit) * d))
        cp = skia.Path(); cp.moveTo(*pts[0])
        for q in pts[1:]:
            cp.lineTo(*q)
        cv.drawPath(cp, paint(0xFFFF6A1A, 0.55, stroke=26 * s))
        cv.drawPath(cp, paint(0xFFFFE08A, 1, stroke=8 * s))
    cv.restore()
    if g > 0.7:
        for i in range(10):
            q = (g - 0.7) / 0.3
            an = i * 0.63
            cv.drawCircle(cx + math.cos(an) * (R + 120 * s * q), cy + math.sin(an) * (R + 120 * s * q), 14 * s, paint(0xFF6B4A3A, 1 - q * 0.5))


def prop_tornado(cv, cx, cy, s, t, g):
    for i in range(14):
        f = i / 13
        y = cy - 230 * s + f * 460 * s
        w = (230 - 190 * f) * s
        off = math.sin(t * 3 + f * 4) * 60 * s * f
        cv.drawOval(S['skia'].Rect(cx + off - w, y - 26 * s, cx + off + w, y + 26 * s), paint(0xFF9AA3B5, 0.5, stroke=14 * s))
    for i in range(14):
        an = t * 5 + i * 0.9
        f = _rnd(i)
        y = cy - 200 * s + f * 380 * s
        w = (230 - 190 * f) * s * 1.1
        cv.drawRect(S['skia'].Rect(cx + math.cos(an) * w - 9 * s, y + math.sin(an) * 20 * s - 9 * s, cx + math.cos(an) * w + 9 * s, y + math.sin(an) * 20 * s + 9 * s), paint(0xFF6B4A3A, 0.9))


def prop_rocket(cv, cx, cy, s, t, g):
    skia = S['skia']
    y = cy + 120 * s - g * 520 * s + math.sin(t * 40) * 3 * s
    for i in range(16):  # smoke trail
        life = (t * 0.8 + i / 16) % 1.0
        cv.drawCircle(cx + math.sin(i * 1.7) * 50 * s * life, y + 200 * s + life * 520 * s, (30 + 60 * life) * s, paint(0xFFD8DCE6, 0.5 * (1 - life)))
    fl = 1 + 0.25 * math.sin(t * 50)
    flame = skia.Path(); flame.moveTo(cx - 40 * s, y + 140 * s); flame.quadTo(cx, y + (260 + 60 * fl) * s, cx + 40 * s, y + 140 * s); flame.close()
    cv.drawPath(flame, paint(0xFFFF8A2A, 1)); cv.drawPath(flame, paint(0xFFFFE08A, 0.8, stroke=8 * s))
    bodyp = skia.Path(); bodyp.moveTo(cx, y - 200 * s); bodyp.quadTo(cx + 80 * s, y - 120 * s, cx + 70 * s, y + 140 * s)
    bodyp.lineTo(cx - 70 * s, y + 140 * s); bodyp.quadTo(cx - 80 * s, y - 120 * s, cx, y - 200 * s); bodyp.close()
    for side in (-1, 1):
        fin = skia.Path(); fin.moveTo(cx + side * 60 * s, y + 40 * s); fin.lineTo(cx + side * 130 * s, y + 160 * s); fin.lineTo(cx + side * 60 * s, y + 140 * s); fin.close()
        cv.drawPath(fin, paint(0xFFFF4D4D, 1)); cv.drawPath(fin, paint(INK, 1, stroke=6 * s))
    cv.drawPath(bodyp, paint(0xFFF4F7FF, 1)); cv.drawPath(bodyp, paint(INK, 1, stroke=7 * s))
    cv.drawCircle(cx, y - 60 * s, 34 * s, paint(0xFF5AB0FF, 1)); cv.drawCircle(cx, y - 60 * s, 34 * s, paint(INK, 1, stroke=7 * s))


def prop_tree(cv, cx, cy, s, t, g):
    base = cy + 230 * s

    def branch(x, y, ang, ln, depth, gk):
        if depth == 0 or gk <= 0:
            return
        L = ln * clamp(gk)
        x2, y2 = x + math.sin(ang) * L, y - math.cos(ang) * L
        cv.drawLine(x, y, x2, y2, paint(0xFF6B4A2A, 1, stroke=max(3, depth * 6 * s)))
        if depth <= 2 and gk > 0.6:
            cv.drawCircle(x2, y2, (34 + 6 * math.sin(t * 2 + depth)) * s * clamp((gk - 0.6) / 0.4), paint(0xFF3FBF6A, 0.9))
        sway = math.sin(t * 1.5 + depth) * 0.06
        branch(x2, y2, ang - 0.45 + sway, ln * 0.72, depth - 1, gk * 1.3 - 0.25)
        branch(x2, y2, ang + 0.4 + sway, ln * 0.72, depth - 1, gk * 1.3 - 0.3)
    cv.drawOval(S['skia'].Rect(cx - 220 * s, base - 20 * s, cx + 220 * s, base + 20 * s), paint(0xFF2E6B3A, 0.8))
    branch(cx, base, 0.0, 170 * s, 6, 0.25 + g * 1.2)


def prop_tsunami(cv, cx, cy, s, t, g):
    skia = S['skia']
    for i in range(7):  # city
        bx = cx - 300 * s + i * 95 * s
        bh = (80 + 120 * _rnd(i)) * s
        cv.drawRect(skia.Rect(bx, cy + 230 * s - bh, bx + 70 * s, cy + 230 * s), paint(0xFF2A3350, 1))
    H_ = (120 + 300 * g) * s
    x_front = cx + 420 * s - g * 520 * s
    wv = skia.Path(); wv.moveTo(cx - 600 * s, cy + 240 * s)
    for k in range(0, 41):
        x = cx - 600 * s + k * 30 * s
        if x > x_front:
            break
        wv.lineTo(x, cy + 240 * s - H_ + math.sin(t * 4 + k * 0.5) * 12 * s)
    wv.quadTo(x_front + 120 * s, cy + 240 * s - H_ * 1.1, x_front + 60 * s, cy + 240 * s - H_ * 0.55)
    wv.lineTo(x_front + 20 * s, cy + 240 * s); wv.close()
    cv.drawPath(wv, skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeLinear(
        [skia.Point(0, cy + 240 * s - H_), skia.Point(0, cy + 240 * s)], [0xFF5FD3F5, 0xFF0A4E8C])))
    for i in range(12):
        an = t * 3 + i
        cv.drawCircle(x_front + 60 * s + math.cos(an) * 40 * s, cy + 240 * s - H_ * 1.0 + math.sin(an) * 30 * s, 14 * s, paint(WHITE, 0.85))


def prop_ice(cv, cx, cy, s, t, g):
    skia = S['skia']
    R = 260 * s * clamp(0.2 + g)
    for arm in range(6):
        an = arm * math.pi / 3 + t * 0.15
        ex, ey = cx + math.cos(an) * R, cy + math.sin(an) * R
        cv.drawLine(cx, cy, ex, ey, paint(0xFFDDF6FF, 1, stroke=14 * s))
        for j in (0.4, 0.65, 0.85):
            bx, by = cx + math.cos(an) * R * j, cy + math.sin(an) * R * j
            for sd in (-1, 1):
                a2 = an + sd * 0.7
                cv.drawLine(bx, by, bx + math.cos(a2) * R * 0.22, by + math.sin(a2) * R * 0.22, paint(0xFFBFEFFF, 1, stroke=9 * s))
    cv.drawCircle(cx, cy, 34 * s, paint(WHITE, 1))
    cv.drawCircle(cx, cy, R * 1.15, paint(0xFFBFEFFF, 0.25 * (0.5 + 0.5 * math.sin(t * 3)), stroke=6 * s))


def prop_heart(cv, cx, cy, s, t, g):
    skia = S['skia']
    bpm = 1.2 + 1.6 * g
    ph = (t * bpm) % 1.0
    beat = 1 + 0.12 * math.exp(-ph * 9) + 0.07 * math.exp(-max(0, ph - 0.18) * 9) * (ph > 0.18)
    cv.save(); cv.translate(cx, cy - 40 * s); cv.scale(beat * s, beat * s)
    hp = skia.Path(); hp.moveTo(0, 150)
    hp.cubicTo(-260, -20, -150, -210, 0, -90); hp.cubicTo(150, -210, 260, -20, 0, 150); hp.close()
    cv.drawPath(hp, skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeRadial(skia.Point(-60, -80), 260, [0xFFFF7A8A, 0xFFC0182E])))
    cv.drawPath(hp, paint(INK, 1, stroke=8))
    cv.restore()
    ecg = skia.Path(); base = cy + 210 * s
    for k in range(0, 61):
        x = cx - 330 * s + k * 11 * s
        q = ((k / 60.0) * 2 + t * bpm) % 1.0
        y = base - (120 * s if 0.45 < q < 0.5 else -50 * s if 0.5 <= q < 0.54 else 0)
        (ecg.moveTo if k == 0 else ecg.lineTo)(x, y)
    cv.drawPath(ecg, paint(0xFF5EE6A8, 1, stroke=7 * s))


def prop_brain(cv, cx, cy, s, t, g):
    skia = S['skia']
    for side in (-1, 1):
        cv.drawOval(skia.Rect(cx + (side * 115 - 140) * s, cy - 170 * s, cx + (side * 115 + 140) * s, cy + 150 * s), paint(0xFFFF9EBB, 1))
        cv.drawOval(skia.Rect(cx + (side * 115 - 140) * s, cy - 170 * s, cx + (side * 115 + 140) * s, cy + 150 * s), paint(INK, 1, stroke=8 * s))
        for j in range(4):
            _arc(cv, cx + side * (80 + 25 * (j % 2)) * s, cy - 110 * s + j * 70 * s, 70 * s, 28 * s, 200 if j % 2 else 20, 140, 0xFFD66A8A, 7 * s)
    for i in range(int(6 + 14 * g)):
        x1, y1 = cx + (_rnd(i) - 0.5) * 420 * s, cy + (_rnd(i, 1) - 0.5) * 280 * s
        x2, y2 = cx + (_rnd(i, 2) - 0.5) * 420 * s, cy + (_rnd(i, 3) - 0.5) * 280 * s
        cv.drawLine(x1, y1, x2, y2, paint(0xFFFFE14D, 0.35, stroke=3 * s))
        q = (t * 1.5 + _rnd(i, 4)) % 1.0
        cv.drawCircle(x1 + (x2 - x1) * q, y1 + (y2 - y1) * q, 9 * s, paint(0xFFFFF4A8, 1))


def prop_city(cv, cx, cy, s, t, g):
    skia = S['skia']
    for i in range(9):
        bx = cx - 380 * s + i * 86 * s
        bh = (150 + 230 * _rnd(i)) * s
        top = cy + 240 * s - bh
        cv.drawRect(skia.Rect(bx, top, bx + 76 * s, cy + 240 * s), paint(0xFF1E2744 if i % 2 else 0xFF263157, 1))
        for wy in range(int(bh // (34 * s))):
            for wx in range(2):
                on = _rnd(i * 50 + wy * 2 + wx, int(t * 2)) < 0.35 + 0.5 * g
                cv.drawRect(skia.Rect(bx + (12 + wx * 32) * s, top + (14 + wy * 34) * s, bx + (32 + wx * 32) * s, top + (34 + wy * 34) * s),
                            paint(0xFFFFD84D if on else 0xFF39456B, 1))
    for i in range(6):
        q = (t * 0.4 + i / 6) % 1.0
        cv.drawCircle(cx - 420 * s + q * 840 * s, cy + 255 * s, 8 * s, paint(0xFFFF5A4E if i % 2 else WHITE, 0.9))


def prop_thermometer(cv, cx, cy, s, t, g):
    skia = S['skia']
    lvl = clamp(0.25 + 0.7 * ease(g))
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(cx - 40 * s, cy - 250 * s, cx + 40 * s, cy + 170 * s), 40 * s, 40 * s), paint(WHITE, 1))
    cv.drawCircle(cx, cy + 200 * s, 75 * s, paint(WHITE, 1))
    col = lerp_col(0xFF5AB0FF, 0xFFFF4D2E, lvl)
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(cx - 20 * s, cy + 170 * s - lvl * 400 * s, cx + 20 * s, cy + 190 * s), 20 * s, 20 * s), paint(col, 1))
    cv.drawCircle(cx, cy + 200 * s, 55 * s, paint(col, 1))
    for j in range(8):
        yy = cy + 150 * s - j * 50 * s
        cv.drawLine(cx + 50 * s, yy, cx + 80 * s, yy, paint(WHITE, 0.8, stroke=5 * s))


PROPS = {'volcano': prop_volcano, 'clock': prop_clock, 'planet_crack': prop_planet_crack, 'tornado': prop_tornado,
         'rocket': prop_rocket, 'tree': prop_tree, 'tsunami': prop_tsunami, 'ice': prop_ice, 'heart': prop_heart,
         'brain': prop_brain, 'city': prop_city, 'thermometer': prop_thermometer}


def sc_prop(cv, t, D, sc):
    """Big animated illustration of the scenario (peraga)."""
    bg(cv, t)
    name = str(sc.get('prop', 'planet_crack')).lower().replace(' ', '_')
    fn = PROPS.get(name, prop_planet_crack)
    k = ease(t / 0.45)
    g = clamp(t / max(D * 0.85, 0.5))
    cv.save(); cv.translate(540, 960); cv.scale(0.6 + 0.4 * k, 0.6 + 0.4 * k); cv.translate(-540, -960)
    fn(cv, 540, 960, 1.0, t, g)
    cv.restore()
    labels(cv, t, sc, 1330, 0.8)
    headline(cv, sc.get('head'), t, 400, 100)
    brand_tag(cv)
    fade_in(cv, t)


# ------------------------------------------------------------ motion graphics: kinetic type, pictograms, transitions, logo sting
def ease_back(x, s=1.7):
    x = clamp(x) - 1
    return 1 + x * x * ((s + 1) * x + s)


def headline(cv, head, t, y0=430, maxsize=108):
    """Kinetic headline: words pop in one by one with overshoot; accent lines get a marker swipe behind them."""
    skia = S['skia']
    lines = parse_head(head)
    if not lines:
        return
    size = fit_size([l for l, _ in lines], 960, maxsize)
    gap = size * 1.14
    f = font(size, 'Black')
    sp = f.measureText(' ')
    wi = 0
    for i, (s, c) in enumerate(lines):
        y = y0 + i * gap
        words = s.split()
        widths = [f.measureText(w.strip('*')) for w in words]
        total = sum(widths) + sp * (len(words) - 1)
        if c != WHITE:  # marker swipe behind the accent line
            k = ease((t - 0.15 - i * 0.1) / 0.35)
            if k > 0:
                pad = size * 0.18
                r = skia.Rect(W / 2 - total / 2 - pad, y - size * 0.82, W / 2 - total / 2 - pad + (total + 2 * pad) * k, y + size * 0.2)
                cv.drawRRect(skia.RRect.MakeRectXY(r, size * 0.16, size * 0.16), paint(c, 0.22))
                cv.drawRect(skia.Rect(r.left(), y + size * 0.12, r.right(), y + size * 0.2), paint(c, 0.9))
        x = W / 2 - total / 2
        for w, wd in zip(words, widths):
            wc = c
            if w.startswith('*') or w.endswith('*'):
                wc = S['accent']
            w = w.strip('*')
            k = clamp((t - wi * 0.055) / 0.32)
            if k > 0:
                sc_ = ease_back(k)
                cv.save(); cv.translate(x + wd / 2, y - size * 0.35); cv.rotate((1 - ease(k)) * -8); cv.scale(sc_, sc_)
                cv.drawString(w, -wd / 2 + 4, size * 0.35 + 5, f, paint(0xFF000000, 0.35 * ease(k)))
                cv.drawString(w, -wd / 2, size * 0.35, f, paint(wc, ease(k)))
                cv.restore()
            x += wd + sp
            wi += 1


# ---- pictogram icons (drawn in a 100x100 box centred on 0,0)
def _ic_person(cv, p):
    cv.drawCircle(0, -28, 17, p)
    cv.drawRRect(S['skia'].RRect.MakeRectXY(S['skia'].Rect(-24, -8, 24, 44), 16, 16), p)


def _ic_house(cv, p):
    skia = S['skia']
    roof = skia.Path(); roof.moveTo(-42, -4); roof.lineTo(0, -40); roof.lineTo(42, -4); roof.close(); cv.drawPath(roof, p)
    cv.drawRect(skia.Rect(-30, -6, 30, 40), p)


def _ic_tree(cv, p):
    cv.drawCircle(0, -14, 30, p); cv.drawRect(S['skia'].Rect(-6, 10, 6, 42), p)


def _ic_car(cv, p):
    skia = S['skia']
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-44, -6, 44, 20), 10, 10), p)
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-26, -28, 22, 0), 10, 10), p)
    cv.drawCircle(-24, 24, 11, p); cv.drawCircle(24, 24, 11, p)


def _ic_drop(cv, p):
    skia = S['skia']
    d = skia.Path(); d.moveTo(0, -42); d.cubicTo(22, -10, 32, 8, 28, 20); d.cubicTo(22, 44, -22, 44, -28, 20); d.cubicTo(-32, 8, -22, -10, 0, -42); d.close()
    cv.drawPath(d, p)


def _ic_sun(cv, p):
    cv.drawCircle(0, 0, 22, p)
    for k in range(8):
        a = k * math.pi / 4
        cv.drawLine(math.cos(a) * 30, math.sin(a) * 30, math.cos(a) * 42, math.sin(a) * 42, S['skia'].Paint(AntiAlias=True, Color=p.getColor(), StrokeWidth=7, Style=S['skia'].Paint.kStroke_Style, StrokeCap=S['skia'].Paint.kRound_Cap))


def _ic_heart(cv, p):
    skia = S['skia']
    h = skia.Path(); h.moveTo(0, 38); h.cubicTo(-56, 0, -30, -46, 0, -18); h.cubicTo(30, -46, 56, 0, 0, 38); h.close(); cv.drawPath(h, p)


def _ic_bolt(cv, p):
    skia = S['skia']
    b = skia.Path(); b.moveTo(8, -44); b.lineTo(-24, 6); b.lineTo(0, 6); b.lineTo(-8, 44); b.lineTo(26, -8); b.lineTo(2, -8); b.close(); cv.drawPath(b, p)


def _ic_fish(cv, p):
    skia = S['skia']
    cv.drawOval(skia.Rect(-34, -18, 22, 18), p)
    t_ = skia.Path(); t_.moveTo(18, 0); t_.lineTo(42, -20); t_.lineTo(42, 20); t_.close(); cv.drawPath(t_, p)


def _ic_bird(cv, p):
    skia = S['skia']
    b = skia.Path(); b.moveTo(-40, -6); b.quadTo(-20, -30, 0, -4); b.quadTo(20, -30, 40, -6); b.quadTo(20, -14, 0, 12); b.quadTo(-20, -14, -40, -6); b.close()
    cv.drawPath(b, p)


def _ic_coin(cv, p):
    cv.drawCircle(0, 0, 36, p)
    cv.drawCircle(0, 0, 24, S['skia'].Paint(AntiAlias=True, Color=0x55000000, StrokeWidth=6, Style=S['skia'].Paint.kStroke_Style))


def _ic_globe(cv, p):
    skia = S['skia']
    st = skia.Paint(AntiAlias=True, Color=p.getColor(), StrokeWidth=6, Style=skia.Paint.kStroke_Style)
    cv.drawCircle(0, 0, 36, st); cv.drawOval(skia.Rect(-16, -36, 16, 36), st); cv.drawLine(-36, 0, 36, 0, st)


def _ic_plane(cv, p):
    skia = S['skia']
    b = skia.Path(); b.moveTo(-44, 0); b.lineTo(44, -6); b.lineTo(44, 6); b.close(); cv.drawPath(b, p)
    w = skia.Path(); w.moveTo(-4, -2); w.lineTo(-20, -36); w.lineTo(10, -2); w.lineTo(-20, 36); w.lineTo(-4, 2); w.close(); cv.drawPath(w, p)


def _ic_phone(cv, p):
    skia = S['skia']
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-22, -40, 22, 40), 9, 9), p)
    cv.drawRect(skia.Rect(-16, -30, 16, 26), skia.Paint(AntiAlias=True, Color=0x66000000))


def _ic_battery(cv, p):
    skia = S['skia']
    cv.drawRRect(skia.RRect.MakeRectXY(skia.Rect(-40, -20, 34, 20), 6, 6), p); cv.drawRect(skia.Rect(34, -8, 42, 8), p)


def _ic_bulb(cv, p):
    skia = S['skia']
    cv.drawCircle(0, -10, 26, p); cv.drawRect(skia.Rect(-12, 14, 12, 34), p)


ICONS = {'person': _ic_person, 'house': _ic_house, 'tree': _ic_tree, 'car': _ic_car, 'drop': _ic_drop, 'sun': _ic_sun, 'heart': _ic_heart,
         'bolt': _ic_bolt, 'fish': _ic_fish, 'bird': _ic_bird, 'coin': _ic_coin, 'globe': _ic_globe, 'plane': _ic_plane,
         'phone': _ic_phone, 'battery': _ic_battery, 'bulb': _ic_bulb}


def sc_icons(cv, t, D, sc):
    """Pictogram infographic: a grid of icons pops in, then `highlight` of them light up in the accent colour."""
    bg(cv, t)
    name = str(sc.get('icon', 'person')).lower()
    fn = ICONS.get(name, _ic_person)
    total = int(clamp(float(sc.get('total', 10) or 10), 1, 40))
    hi = int(clamp(float(sc.get('highlight', 3) or 0), 0, total))
    cols = 5 if total <= 25 else 8
    rows = (total + cols - 1) // cols
    cell = min(170, 900 / cols, 420 / rows)
    s = cell / 120
    x0 = W / 2 - (cols - 1) * cell / 2
    y0 = 690 + cell / 2
    lit_t = 0.4 + total * 0.05
    for i in range(total):
        r_, c_ = divmod(i, cols)
        k = clamp((t - 0.25 - i * 0.05) / 0.3)
        if k <= 0:
            continue
        lit = i < hi and t > lit_t + i * 0.08
        col = S['accent'] if lit else 0xFF6B7396
        pop = ease_back(k) * (1 + (0.12 * math.sin(clamp((t - lit_t - i * 0.08) / 0.3) * math.pi) if lit else 0))
        cv.save(); cv.translate(x0 + c_ * cell, y0 + r_ * cell); cv.scale(s * pop, s * pop)
        fn(cv, paint(col, ease(k)))
        cv.restore()
    big = str(sc.get('big', '') or (f'{hi} OUT OF {total}' if hi else '')).upper()
    if big:
        text(cv, big, 540, 690 + rows * cell + 95, 92, S['accent'], ease((t - lit_t) / 0.4), 'Black')
    labels(cv, t, sc, 690 + rows * cell + 180, lit_t + 0.3)
    headline(cv, sc.get('head'), t, 400, 100)
    brand_tag(cv)
    fade_in(cv, t)


# ---- scene-entry transitions (drawn over the first ~0.35 s of a scene)
TRANSITIONS = ['slice', 'iris', 'glitch', 'bars', 'shutter']


def transition(cv, kind, t):
    skia = S['skia']
    d = 0.38
    if t >= d:
        return
    q = ease_io(t / d)
    acc = S['accent']
    if kind == 'iris':
        pth = skia.Path(); pth.addRect(skia.Rect(-10, -10, W + 10, H + 10))
        hole = skia.Path(); hole.addCircle(540, 960, 1150 * q)
        cut = skia.Op(pth, hole, skia.PathOp.kDifference_PathOp)
        cv.drawPath(cut, paint(BG, 1)); cv.drawCircle(540, 960, 1150 * q, paint(acc, 1 - q, stroke=40))
    elif kind == 'slice':
        for j in range(3):
            qq = ease_io(clamp(t / d * 1.2 - j * 0.12))
            off = qq * (W + 900)
            pth = skia.Path()
            y0 = j * H / 3
            pth.moveTo(-200 + off, y0); pth.lineTo(W + 700 + off, y0); pth.lineTo(W + 500 + off, y0 + H / 3 + 2); pth.lineTo(-400 + off, y0 + H / 3 + 2); pth.close()
            cv.drawPath(pth, paint([acc, BG, acc][j], 1))
    elif kind == 'bars':
        for j in range(8):
            qq = ease_io(clamp(t / d * 1.4 - j * 0.05))
            h_ = H / 8
            cv.drawRect(skia.Rect(0, j * h_ + h_ * qq, W, (j + 1) * h_ + 1), paint(acc if j % 2 else BG, 1))
    elif kind == 'shutter':
        cv.drawRect(skia.Rect(0, 0, W, H / 2 * (1 - q)), paint(BG, 1))
        cv.drawRect(skia.Rect(0, H - H / 2 * (1 - q), W, H), paint(BG, 1))
        cv.drawLine(0, H / 2 * (1 - q), W, H / 2 * (1 - q), paint(acc, 1 - q * 0.5, stroke=10))
        cv.drawLine(0, H - H / 2 * (1 - q), W, H - H / 2 * (1 - q), paint(acc, 1 - q * 0.5, stroke=10))


# ---- logo sting: portal opens, Terra & Para fly in, kinetic wordmark, flash
def wordmark(cv, t, cx, cy, sc_=1.0, a=1.0):
    skia = S['skia']
    f1 = font(int(124 * sc_), 'Black'); f2 = font(int(104 * sc_), 'Black')
    word = 'PARALLEL'
    widths = [f1.measureText(ch) for ch in word]
    sp1 = 6 * sc_
    total = sum(widths) + sp1 * (len(word) - 1)
    x = cx - total / 2
    for i, (ch, w) in enumerate(zip(word, widths)):
        k = clamp((t - i * 0.045) / 0.3)
        if k > 0:
            yy = cy - (1 - ease_back(k)) * 140 * sc_
            cv.drawString(ch, x, yy, f1, paint(WHITE, ease(k) * a))
        x += w + sp1
    k2 = ease((t - 0.45) / 0.5)
    spacing = (60 - 36 * k2) * sc_
    word2 = 'EARTH'
    w2 = [f2.measureText(ch) for ch in word2]
    total2 = sum(w2) + spacing * 4
    x = cx - total2 / 2
    for ch, w in zip(word2, w2):
        cv.drawString(ch, x, cy + 120 * sc_, f2, paint(S['accent'], k2 * a))
        x += w + spacing


def logo_sting(cv, t, w, h, with_hosts=True):
    """2.6 s channel sting. Works for 1080x1920 and 1920x1080."""
    skia = S['skia']
    cx, cy = w / 2, h * (0.42 if h > w else 0.4)
    unit = min(w, h) / 1080
    cv.clear(BG)
    rng = random.Random(7)
    for i in range(140):
        x, y = rng.random() * w, rng.random() * h
        zoom = 1 + t * 0.25
        cv.drawCircle(cx + (x - cx) * zoom, cy + (y - cy) * zoom, rng.random() * 2.4 + 0.6, paint(WHITE, 0.25 + 0.6 * rng.random()))
    ring = ease_back(clamp(t / 0.6)) * 300 * unit
    portal_ring(cv, cx, cy, ring, t, clamp(t / 0.3))
    if with_hosts:
        for who, side in (('terra', -1), ('para', 1)):
            k = ease_back(clamp((t - 0.3) / 0.6), 1.2)
            x = cx + side * (w * 0.75 - (w * 0.75 - 330 * unit) * k)
            y = cy + 40 * unit + math.sin(t * 3 + side) * 8 * unit
            mascot(cv, who, x, y, 110 * unit, 'excited' if t > 1.2 else 'happy', 0.0, t, -side * 0.6, False, 1.0, 0.0, '', None, side * (1 - k) * 360,
                   gesture='cheer' if t > 1.3 else '', inner=-side)
    wy = cy + (360 if h > w else 420) * unit
    wordmark(cv, t - 0.75, cx, wy, unit, 1.0)
    fl = clamp((t - 1.55) / 0.5)
    if 0 < fl < 1:
        cv.drawCircle(cx, cy, 300 * unit + fl * 900 * unit, paint(S['accent'], (1 - fl) * 0.8, stroke=24 * unit * (1 - fl) + 2))
        cv.drawRect(skia.Rect(0, 0, w, h), paint(WHITE, (1 - fl) ** 3 * 0.35))
    tg = ease((t - 1.8) / 0.5)
    text(cv, 'WHAT IF THE WORLD WAS DIFFERENT?', cx, wy + 230 * unit, int(40 * unit), MUTED, tg, 'Bold', spacing=4 * unit)


def sting_audio(dur, sr=24000):
    n = int(dur * sr); tt = np.arange(n) / sr
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(n)
    sweep = np.clip(tt / 0.9, 0, 1)
    whoosh = np.convolve(noise, np.ones(30) / 30, mode='same') * np.sin(np.pi * np.clip(tt / 1.0, 0, 1)) * 0.25 * sweep
    boom = np.where(tt > 1.55, np.sin(2 * np.pi * (55 + 40 * np.exp(-(tt - 1.55) * 8)) * (tt - 1.55)) * np.exp(-(tt - 1.55) * 3.5), 0) * 0.6
    chime = sum(np.where(tt > 1.6, np.sin(2 * np.pi * f * (tt - 1.6)) * np.exp(-(tt - 1.6) * 2.5), 0) for f in (880, 1320, 1760)) * 0.08
    out = whoosh + boom + chime
    return (out / max(1e-6, np.abs(out).max()) * 0.8).astype(np.float32)


def render_sting(path, w=1080, h=1920, dur=2.6, cfg=None):
    import soundfile as sf
    setup_state(cfg or {'title': 'sting', 'accent': 'orange', 'scenes': [{'lines': [1]}]})
    skia = S['skia']
    surf = skia.Surface(w, h)
    wav = path[:-4] + '.wav'
    sf.write(wav, sting_audio(dur), 24000)
    pr = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                           '-i', wav, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                           '-ar', '48000', '-ac', '2', '-shortest', path], stdin=subprocess.PIPE)
    for f in range(int(dur * FPS)):
        with surf as cv:
            logo_sting(cv, f / FPS, w, h)
        pr.stdin.write(surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType).tobytes())
    pr.stdin.close(); pr.wait()
    os.remove(wav)
    return path


SCENES = {'split': sc_split, 'title': sc_title, 'globe': sc_globe, 'counter': sc_counter, 'bars': sc_bars, 'words': sc_words,
          'compare': sc_compare, 'distance': sc_distance, 'wave': sc_wave, 'timeline': sc_timeline,
          'fact': sc_fact, 'brand': sc_brand, 'cta': sc_cta, 'prop': sc_prop, 'icons': sc_icons}



# ------------------------------------------------------------ duo mascots: Terra (our Earth) & Para (parallel Earth)
CHAR = {
    'terra': {'light': 0xFF7CCBFF, 'dark': 0xFF1C5FC4, 'patch': 0xFF52C98E, 'name': 'TERRA', 'hi': 0xFF7CD8FF},
    'para': {'light': 0xFFFFC98A, 'dark': 0xFFD24E16, 'patch': 0xFFB8350F, 'name': 'PARA', 'hi': 0xFFFFB862},
}
INK = 0xFF0A0F24


def _arc(cv, cx, cy, w, h, start, sweep, color, width):
    skia = S['skia']
    cv.drawArc(skia.Rect(cx - w, cy - h, cx + w, cy + h), start, sweep, False, paint(color, 1, stroke=width))


def _limb(cv, x0, y0, x1, y1, ln1, ln2, bend, col, width, a):
    """Two-segment limb from (x0,y0) to (x1,y1); elbow pushed to side `bend` (+1/-1)."""
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy) or 1.0
    reach = ln1 + ln2
    if d > reach:
        x1, y1 = x0 + dx / d * reach, y0 + dy / d * reach
        d = reach
    a1 = (ln1 * ln1 + d * d - ln2 * ln2) / (2 * d)
    h = math.sqrt(max(0.0, ln1 * ln1 - a1 * a1))
    mx, my = x0 + dx / d * a1, y0 + dy / d * a1
    ex, ey = mx - dy / d * h * bend, my + dx / d * h * bend
    p = paint(col, a, stroke=width)
    p.setStrokeCap(S['skia'].Paint.kRound_Cap)
    cv.drawLine(x0, y0, ex, ey, p)
    cv.drawLine(ex, ey, x1, y1, p)
    return x1, y1


def _hand_targets(gesture, side, r, t, amp, inner, point_vec):
    """Hand position relative to body centre for one side (-1 left, +1 right)."""
    sw = math.sin(t * 2.4 + side) * 0.05 * r
    idle = (side * 1.12 * r, 0.78 * r + sw)
    g = gesture
    if g == 'talk':
        ph = math.sin(t * 7 + (0 if side < 0 else 2.2))
        return (side * (1.22 + 0.12 * ph) * r, (0.05 - 0.35 * amp - 0.15 * ph) * r)
    if g == 'wave' and side == inner:
        ang = math.sin(t * 11) * 0.35
        return (side * (1.05 + 0.3 * math.sin(ang)) * r, (-1.05 + 0.15 * math.cos(ang)) * r)
    if g == 'point' and point_vec and side == (1 if point_vec[0] >= 0 else -1):
        return (side * 0.9 * r + point_vec[0] * 0.8 * r, 0.15 * r + point_vec[1] * 0.8 * r)
    if g == 'cheeks':
        return (side * 0.62 * r, 0.22 * r)
    if g == 'cover':
        return (side * 0.2 * r, 0.42 * r)
    if g == 'chin' and side == inner:
        return (side * 0.12 * r, 0.7 * r)
    if g == 'scratch' and side == inner:
        return (side * (0.32 + 0.06 * math.sin(t * 16)) * r, -0.98 * r)
    if g == 'cheer':
        return (side * 1.0 * r, (-1.25 - 0.08 * math.sin(t * 9)) * r)
    if g == 'droop':
        return (side * 0.92 * r, 1.05 * r)
    if g == 'hug':
        return (side * 0.55 * r, 0.55 * r)
    return idle


def mascot(cv, who, cx, cy, r, mood='happy', mouth=0.0, t=0.0, look=0.0, talking=False, a=1.0, wave=0.0,
           action='', point=None, spin=0.0, gesture='', walk=None, umbrella=False, hop=0.0, inner=None):
    """Full-body cartoon planet: legs with shoes, two-segment arms with mitten hands, face with moods."""
    skia = S['skia']
    c = CHAR.get(who, CHAR['terra'])
    if a <= 0 or r <= 2:
        return
    inner = inner if inner is not None else (1 if who == 'terra' else -1)
    breathe = 1 + 0.018 * math.sin(t * 2.2 + (0 if who == 'terra' else 1.7))
    bob = -abs(math.sin(t * 9)) * 7 * mouth if talking else 0
    dx = dy = rot = 0.0
    sy = 1.0
    tuck = 0.0
    if action == 'jump':
        ph = abs(math.sin(t * 5.5)); dy = -ph * r * 0.45; sy = 1 - 0.12 * (1 - ph) ** 6; tuck = ph
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
    elif action == 'lean':
        rot = -10 + math.sin(t * 6) * 3
    if hop:
        dy -= hop * r * 0.55; tuck = max(tuck, hop)
    if walk is not None:
        dy -= abs(math.sin(walk)) * r * 0.08
        rot += math.sin(walk) * 4
    rot += spin
    if (S.get('spr') or {}).get(who):
        sh = skia.Paint(AntiAlias=True, Color=0xFF000000); sh.setAlphaf(0.32 * a * max(0.35, 1 - abs(dy) / (r * 0.8)))
        sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, r * 0.08))
        cv.drawOval(skia.Rect(cx - r * 0.78, cy + r * 1.55, cx + r * 0.78, cy + r * 1.78), sh)
    cv.save()
    cv.translate(cx + dx, cy + bob + dy); cv.rotate(rot); cv.scale(breathe, sy / breathe); cv.translate(-cx, -cy)
    dark, light = c['dark'], c['light']
    spr = (S.get('spr') or {}).get(who)
    if spr:
        _sprite_body(cv, spr, who, cx, cy, r, mood, mouth, t, talking, gesture, wave, point, walk, tuck, umbrella, inner, a)
    else:
        # legs + shoes (behind the body)
        for side in (-1, 1):
            hx, hy = cx + side * r * 0.34, cy + r * 0.8
            if walk is not None:
                ph = walk + (0 if side < 0 else math.pi)
                fx, fy = hx + math.sin(ph) * r * 0.32, cy + r * 1.42 - max(0.0, math.cos(ph)) * r * 0.16
            else:
                fx, fy = hx + side * r * 0.06, cy + r * (1.42 - 0.3 * tuck)
            _limb(cv, hx, hy, fx, fy, r * 0.34, r * 0.34, -side, dark, r * 0.21, a)
            sw_ = r * 0.25
            shoe = skia.Rect(fx - sw_ + side * r * 0.09, fy - r * 0.09, fx + sw_ + side * r * 0.09, fy + r * 0.13)
            cv.drawRRect(skia.RRect.MakeRectXY(shoe, r * 0.11, r * 0.11), paint(WHITE, a))
            cv.drawRect(skia.Rect(shoe.left(), fy + r * 0.06, shoe.right(), fy + r * 0.13), paint(c['hi'], a))
            cv.drawRRect(skia.RRect.MakeRectXY(shoe, r * 0.11, r * 0.11), paint(INK, a, stroke=r * 0.045))
        # body
        sh = skia.GradientShader.MakeRadial(skia.Point(cx - r * 0.35, cy - r * 0.4), r * 1.6, [light, dark])
        p = skia.Paint(AntiAlias=True, Shader=sh); p.setAlphaf(a)
        cv.drawCircle(cx, cy, r, p)
        cv.save()
        pth = skia.Path(); pth.addCircle(cx, cy, r); cv.clipPath(pth, doAntiAlias=True)
        for (ddx, ddy, rx, ry) in [(-0.55, 0.45, 0.42, 0.28), (0.62, -0.55, 0.35, 0.22), (0.5, 0.7, 0.3, 0.2), (-0.75, -0.35, 0.22, 0.3)]:
            cv.drawOval(skia.Rect(cx + (ddx - rx) * r, cy + (ddy - ry) * r, cx + (ddx + rx) * r, cy + (ddy + ry) * r), paint(c['patch'], 0.55 * a))
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
        # arms + mitten hands (in front, so hands can touch the face)
        g = gesture or ('wave' if wave else '')
        pv = None
        if point is not None:
            vx, vy = point[0] - cx, point[1] - cy
            L = math.hypot(vx, vy) or 1.0
            pv = (vx / L, vy / L)
            if not g or g == 'talk':
                g = 'point'
        for side in (-1, 1):
            sx_, sy_ = cx + side * r * 0.86, cy + r * 0.22
            if umbrella and side == -inner:
                hx, hy = cx + side * r * 1.02, cy - r * 0.1
            else:
                tx, ty = _hand_targets(g, side, r, t, clamp(mouth), inner, pv)
                hx, hy = cx + tx, cy + ty
            hx, hy = _limb(cv, sx_, sy_, hx, hy, r * 0.5, r * 0.5, side if hy > sy_ else -side, dark, r * 0.13, a)
            cv.drawCircle(hx, hy, r * 0.15, paint(light, a))
            cv.drawCircle(hx, hy, r * 0.15, paint(INK, a, stroke=r * 0.04))
            if umbrella and side == -inner:
                _umbrella(cv, hx, hy, cx + side * r * 0.3, cy - r * 1.55, r, c['hi'] if who == 'para' else 0xFF5AB0FF, a, t)
    # extra effects
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


def _sprite_cell(cv, spr, name, cx, cy, s, a):
    img, meta = spr
    xy = meta['index'].get(name)
    if not xy:
        return
    cw, ch = meta['cell']; ax, ay = meta['anchor']
    p = S['skia'].Paint(AntiAlias=True); p.setAlphaf(a)
    cv.drawImageRect(img, S['skia'].Rect(xy[0], xy[1], xy[0] + cw, xy[1] + ch),
                     S['skia'].Rect(cx - ax * s, cy - ay * s, cx + (cw - ax) * s, cy + (ch - ay) * s),
                     S['skia'].SamplingOptions(S['skia'].FilterMode.kLinear, S['skia'].MipmapMode.kLinear), p)


def _sprite_body(cv, spr, who, cx, cy, r, mood, mouth, t, talking, gesture, wave, point, walk, tuck, umbrella, inner, a):
    """3D-rendered sprite layers: legs, body, face, arms (atlas made with Blender)."""
    img, meta = spr
    s = r / meta['px_per_r']
    # legs
    if walk is not None:
        legs = 'legs_walk%d' % (int((walk % (2 * math.pi)) / (2 * math.pi) * 8) % 8)
    elif tuck > 0.5:
        legs = 'legs_tuck'
    else:
        legs = 'legs_stand'
    _sprite_cell(cv, spr, legs, cx, cy, s, a)
    _sprite_cell(cv, spr, 'body', cx, cy, s, a)
    # face
    m = mood if mood in meta['moods'] else 'happy'
    blink = ((t + (0.0 if who == 'terra' else 1.3)) % 3.4) < 0.12
    amp = clamp(mouth)
    if blink:
        ms = 'blink'
    elif talking and amp > 0.08:
        ms = 'o1' if amp < 0.3 else 'o2' if amp < 0.6 else 'o3'
    else:
        ms = 'closed'
    _sprite_cell(cv, spr, f'face_{m}_{ms}', cx, cy, s, a)
    # arms
    g = gesture or ('wave' if wave else '')
    if point is not None and (not g or g == 'talk'):
        g = 'point'
    if umbrella:
        arm = 'umbrella0'
    elif walk is not None:
        arm = 'walk%d' % (int((walk % (2 * math.pi)) / (2 * math.pi) * 8) % 8)
    elif g == 'talk':
        arm = 'talk%d' % (int(t * 7) % 6)
    elif g == 'wave':
        arm = 'wave%d' % [0, 1, 2, 1][int(t * 9) % 4]
    elif g == 'cheer':
        arm = 'cheer%d' % (int(t * 6) % 2)
    elif g == 'scratch':
        arm = 'scratch%d' % (int(t * 9) % 2)
    elif g in ('cheeks', 'cover', 'chin', 'droop', 'point'):
        arm = g + '0'
    else:
        arm = 'idle0'
    _sprite_cell(cv, spr, 'arms_' + arm, cx, cy, s, a)
    if umbrella:
        hx, hy = meta['hands']['umbrella0'][str(-inner)]
        ax, ay = meta['anchor']
        X, Y = cx + (hx - ax) * s, cy + (hy - ay) * s
        _umbrella(cv, X, Y, cx - inner * r * 0.3, cy - r * 1.55, r, CHAR[who]['hi'] if who == 'para' else 0xFF5AB0FF, a, t)


def _umbrella(cv, hx, hy, tx, ty, r, col, a, t):
    """Umbrella held at (hx, hy); canopy centred at (tx, ty) above the head."""
    skia = S['skia']
    cv.save(); cv.translate(hx, hy); cv.rotate(math.sin(t * 2.3) * 3); cv.translate(-hx, -hy)
    cv.drawLine(hx, hy + r * 0.12, tx, ty - r * 0.5, paint(0xFFE3E7F0, a, stroke=r * 0.07))
    w = r * 1.3
    can = skia.Path()
    can.moveTo(tx - w, ty + r * 0.12)
    can.cubicTo(tx - w * 0.9, ty - r * 0.8, tx + w * 0.9, ty - r * 0.8, tx + w, ty + r * 0.12)
    for k in range(3, -1, -1):
        x0 = tx - w + k * w * 0.5
        can.quadTo(x0 + w * 0.25, ty - r * 0.08, x0, ty + r * 0.12)
    can.close()
    cv.drawPath(can, paint(col, a))
    for k in (-0.5, 0.0, 0.5):
        cv.drawLine(tx, ty - r * 0.58, tx + k * w, ty + r * 0.1, paint(INK, 0.35 * a, stroke=r * 0.03))
    cv.drawPath(can, paint(INK, a, stroke=r * 0.05))
    cv.drawCircle(tx, ty - r * 0.62, r * 0.07, paint(INK, a))
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


GESTURE_OF = {'shocked': 'cheeks', 'scared': 'cover', 'excited': 'cheer', 'sad': 'droop', 'thinking': 'scratch'}


def mascot_layer(cv, t, sc):
    """Draw Terra & Para (full body) over every scene: walk-ins, gestures, reactions, mouths driven by the voice."""
    if not S.get('duo'):
        return
    kind = str(sc.get('type', '')).lower()
    idx = int(sc.get('_idx', 0))
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
    home = {'terra': (175, 1440, 100), 'para': (890, 1440, 100)}
    pos = dict(home)
    wave = 0.0
    walk_in = {}
    if kind == 'brand':
        pos = {'terra': (330, 1080, 135), 'para': (750, 1080, 135)}; wave = 1.0
    elif kind == 'cta':
        pos = {'terra': (330, 740, 105), 'para': (750, 740, 105)}; wave = 1.0
    if kind in ('brand', 'cta'):
        for who in ('terra', 'para'):     # walk from the corners to centre stage
            q = ease(t / 0.9)
            (x0, y0, r0), (x1, y1, r1) = home[who], pos[who]
            pos[who] = (x0 + (x1 - x0) * q, y0 + (y1 - y0) * q, r0 + (r1 - r0) * q)
            if t < 0.9:
                walk_in[who] = t * 14
    elif idx == 0 and kind == 'title' and t < 1.1:
        q = ease(t / 1.1)
        x, y, r = home['terra']
        pos['terra'] = (-160 + (x + 160) * q, y, r)
        walk_in['terra'] = t * 14
    eff = str(sc.get('effect', '')).lower()
    forced = str(sc.get('action', '') or '').lower()
    fx = sc.get('_fx', '')
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
            if eff == 'freeze' or fx == 'snow' or 'freez' in str(sc.get('bg', '')):
                act = 'shiver'
            elif eff == 'heat' or fx == 'embers':
                act = 'sweat'
            elif eff == 'wobble':
                act = 'dizzy'
            elif fx == 'wind':
                act = 'lean'
            elif mood == 'scared':
                act = 'tremble'
            elif mood == 'excited' and talking:
                act = 'jump'
        # gesture: what the hands do
        if wave:
            gest = 'wave'
        elif talking:
            gest = GESTURE_OF.get(mood, 'talk') if mood in ('shocked', 'scared', 'excited') else 'talk'
        else:
            gest = GESTURE_OF.get(mood, '')
            if gest == 'scratch' and who == 'para':
                gest = 'chin'
        point = None
        if talking and kind in ('counter', 'bars', 'compare', 'timeline', 'wave', 'globe', 'distance', 'split', 'prop') and act in ('', 'tremble') \
                and mood not in ('shocked', 'scared', 'excited'):
            point = (540, 960)
        # quick hop when a line starts with a big emotion
        hop = 0.0
        if who in line_start:
            st_, m_ = line_start[who]
            age_ = t - st_
            if m_ in ('shocked', 'excited') and age_ < 0.4:
                hop = math.sin(math.pi * age_ / 0.4)
        pop, spin = 1.0, 0.0
        if kind == 'title' and who == 'para' and idx == 0:
            pop = ease((t - 0.7) / 0.5); spin = (1 - pop) * 540
            pr_ = clamp((t - 0.45) / 0.35) * (1 - clamp((t - 1.9) / 0.5))
            if pr_ > 0:
                portal_ring(cv, x, y, r * 1.45 * pr_, t, pr_)
        if pop <= 0:
            continue
        mascot(cv, who, x, y, r * (pop if (kind == 'title' and who == 'para' and idx == 0) else 1.0),
               mood, amp if talking else 0.0, t, look, talking, 1.0, wave, act, point, spin,
               gesture=gest, walk=walk_in.get(who), umbrella=fx in ('rain', 'storm') and kind not in ('brand', 'cta'), hop=hop)
        if who in line_start:
            st_, m_ = line_start[who]
            icon = {'shocked': '!', 'thinking': '?', 'scared': 'sweat', 'excited': 'stars'}.get(m_)
            if act == 'shiver':
                icon = 'snow'
            age = t - st_
            if icon and age < 1.1:
                k_ = ease(age / 0.18) * (1 - clamp((age - 0.85) / 0.25))
                react_icon(cv, icon, x + (40 if who == 'terra' else -40), y - r * 1.3 - 55 - 20 * ease(age / 0.5), k_, k_)
        if kind not in ('brand', 'cta') and idx <= 1:
            text(cv, CHAR[who]['name'], x, y + r * 1.5 + 40, 28, CHAR[who]['hi'], 0.9, 'Black', spacing=3)


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
    if path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        dd = D + 0.5
        vf = (f"scale={int(w * 1.18)}:{int(h * 1.18)}:force_original_aspect_ratio=increase,"
              f"crop={w}:{h}:x='(iw-{w})*(0.15+0.7*t/{dd:.2f})':y='(ih-{h})*(0.5+0.25*sin(t/1.7))',fps={FPS},format=rgba")
        cmd = ['ffmpeg', '-loglevel', 'error', '-loop', '1', '-framerate', str(FPS), '-i', path, '-t', f'{dd:.2f}',
               '-vf', vf, '-f', 'rawvideo', '-']
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
    S['spr'] = {}
    sdir = os.environ.get('PE_SPRITE_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sprites')
    if str(cfg.get('hosts', '3d')).lower() != '2d':
        for who_ in ('terra', 'para'):
            pi, pj = os.path.join(sdir, who_ + '.png'), os.path.join(sdir, who_ + '.json')
            if os.path.exists(pi) and os.path.exists(pj):
                S['spr'][who_] = (skia.Image.open(pi), json.load(open(pj)))
    rnd = random.Random(int(hashlib.md5(str(cfg.get('title', '')).encode()).hexdigest()[:6], 16))
    S['stars'] = [(rnd.uniform(0, W), rnd.uniform(0, H), rnd.uniform(1.2, 3.4), rnd.uniform(0, 6.28), rnd.uniform(0.3, 1.5)) for _ in range(140)]


def render_scene(args):
    cfg, i, D, path = args
    setup_state(cfg)
    skia = S['skia']
    sc = cfg['scenes'][i]
    sc['_fx'] = scene_fx(sc)
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
    trans = str(sc.get('transition', '') or '').lower()
    if trans not in TRANSITIONS:
        trans = TRANSITIONS[idx % len(TRANSITIONS)]
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
            fx_layer(cv, t, sc)
            mascot_layer(cv, t, sc)
            fx_layer(cv, t, sc, front=True)
            captions(cv, t, sc)
            if portal:
                portal_wipe(cv, t)
            if idx > 0 and not portal and trans != 'glitch':
                transition(cv, trans, t)
        arr = surf.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)
        if idx > 0 and not portal and trans == 'glitch' and t < 0.24:
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
    sc['_fx'] = scene_fx(sc)
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
        fx_layer(cv, t, sc)
        mascot_layer(cv, t, sc)
        fx_layer(cv, t, sc, front=True)
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


def pixabay_image(key, q, used):
    """Photo fallback when no video matches: returns (id, url)."""
    import requests
    for params in ({'orientation': 'vertical'}, {}):
        for qq in (q, ' '.join(q.split()[:2])):
            r = requests.get('https://pixabay.com/api/', timeout=30,
                             params={'key': key, 'q': qq[:100], 'image_type': 'photo', 'per_page': 20, 'safesearch': 'true', **params})
            hits = [h for h in r.json().get('hits', []) if h['id'] not in used and h.get('largeImageURL')]
            if hits:
                h = random.Random(q).choice(hits[:6])
                return h['id'], h['largeImageURL']
    return None, None


def scene_query(sc, cfg_title=''):
    """What the background should show: the scene's own 'bg', else words from its headline / text."""
    q = str(sc.get('bg', '') or '').strip()
    if q:
        return q
    parts = [str(x) for x in (sc.get('head') or [])] + [str(x) for x in (sc.get('words') or [])]
    for k in ('big', 'label', 'body', 'prop'):
        if sc.get(k):
            parts.append(str(sc[k]))
    words = re.findall(r"[A-Za-z]{3,}", ' '.join(parts).replace('*', ' '))
    stop = {'what', 'the', 'and', 'next', 'time', 'day', 'one', 'how', 'long', 'could', 'last', 'with', 'into', 'for', 'your', 'our', 'para', 'earth'}
    words = [w.lower() for w in words if w.lower() not in stop][:3]
    if not words:
        words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", cfg_title) if w.lower() not in stop][:3]
    return ' '.join(words)


def fetch_backgrounds(scenes, work, log=print, title=''):
    """Every scene (except the CTA) gets real footage of what is being explained; a photo with slow pan if no video exists."""
    pkey = os.environ.get('PIXABAY_API_KEY', '').strip()
    if not pkey:
        return
    import requests
    used = set()

    def grab(q, base):
        vid, url = pixabay_pick(pkey, q, used)
        ext = '.mp4'
        if not url:
            vid, url = pixabay_image(pkey, q, used)
            ext = '.jpg'
        if not url:
            return None, None
        path = base + ext
        with requests.get(url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            with open(path, 'wb') as fh:
                for chunk in dl.iter_content(1 << 20):
                    fh.write(chunk)
        used.add(vid)
        return path, ('video' if ext == '.mp4' else 'photo') + f' {vid}'
    for i, sc in enumerate(scenes):
        kind = str(sc.get('type', '')).lower()
        if kind == 'split':
            for side, k_ in (('bg_left', '_bgL'), ('bg_right', '_bgR')):
                q = str(sc.get(side, '') or '').strip() or scene_query(sc, title)
                try:
                    pth, info = grab(q, os.path.join(work, f'bg_{i+1:02d}_{side}'))
                    if pth:
                        sc[k_] = pth; log(f'  split {side} "{q}": {info}')
                except Exception as e:
                    log(f'  split {side} "{q}" failed: {str(e)[:100]}')
            continue
        if kind == 'cta' or (kind == 'brand' and not sc.get('bg')):
            continue
        q = scene_query(sc, title)
        if not q:
            continue
        try:
            pth, info = grab(q, os.path.join(work, f'bg_{i+1:02d}'))
            if pth:
                sc['_bg'] = pth; log(f'  bg "{q}": {info}')
            else:
                log(f'  bg "{q}": nothing found')
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
    fetch_backgrounds(scenes, work, log, str(cfg.get('title', '')))
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
