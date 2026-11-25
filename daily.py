"""Parallel Earth daily run: Gemini writes N configs -> engine renders N videos -> sent to Telegram.
Env: GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional), VIDEOS (default 3), MOCK=1 for a dry run.
"""
import os, json, re, time, random, traceback, glob, subprocess, datetime
import requests
import engine

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'out')
os.makedirs(OUT, exist_ok=True)
DONE_FILE = os.path.join(ROOT, 'topics_done.txt')
CONF_DIR = os.path.join(ROOT, 'configs')
COMPILED_FILE = os.path.join(ROOT, 'compiled.txt')
MODE = (os.environ.get('MODE', 'auto') or 'auto').lower()   # auto | daily | compile | both
COMPILE_MIN = int(os.environ.get('COMPILE_MIN', '5') or 5)
MIN_SECONDS = 66
WIB = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(hours=7)
CHAT_FILE = os.path.join(ROOT, 'telegram_chat_id.txt')
N = int(os.environ.get('VIDEOS', '3') or 3)
KEY = os.environ.get('GEMINI_API_KEY', '').strip()
TG = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
MOCK = os.environ.get('MOCK') == '1'

PROMPT = open(os.path.join(ROOT, 'prompt.txt'), encoding='utf-8').read()
ACCENTS = ['orange', 'red', 'green', 'purple', 'cyan', 'yellow']
STYLES = ['space', 'grid', 'dust']
VOICES = ['am_michael', 'am_fenrir', 'bm_george', 'bm_fable', 'af_heart', 'am_puck']


def log(*a):
    print(*a, flush=True)


def done_topics():
    if not os.path.exists(DONE_FILE):
        return []
    return [l.strip() for l in open(DONE_FILE, encoding='utf-8') if l.strip()]


def pick_model():
    r = requests.get('https://generativelanguage.googleapis.com/v1beta/models', params={'key': KEY, 'pageSize': 200}, timeout=30)
    r.raise_for_status()
    names = [m['name'] for m in r.json().get('models', []) if 'generateContent' in m.get('supportedGenerationMethods', [])]
    bad = ('image', 'tts', 'audio', 'live', 'embedding', 'vision', 'learnlm', 'gemma', 'robotics', 'computer')
    good = [n for n in names if 'flash' in n and not any(b in n for b in bad)]

    def score(n):
        v = re.findall(r'(\d+(?:\.\d+)?)', n)
        ver = float(v[0]) if v else 0
        return (ver, 'lite' not in n, 'preview' not in n and 'exp' not in n)
    good.sort(key=score, reverse=True)
    cands = good + [n for n in names if n not in good and not any(b in n for b in bad)]
    log('Gemini models to try:', cands[:4])
    return cands[:7]


def ask_gemini(models, avoid, used_styles):
    user = (PROMPT + '\n\nTOPIC: invent ONE fresh, surprising "What if" topic yourself (science, space, nature, history, human body, technology). '
            'It must be different from all of these already-made topics:\n- ' + '\n- '.join(avoid[-150:] or ['(none yet)']) +
            f'\n\nUse accent "{used_styles[0]}", style "{used_styles[1]}", voice "{used_styles[2]}".'
            '\nAlso add these top-level keys: "yt_title" (max 70 chars, catchy, ends with #shorts), '
            '"description" (2 short sentences), "hashtags" (5-8 hashtags in one string).')
    last = None
    for rnd in range(3):
        if rnd:
            log(f'All Gemini models busy, waiting {90 * rnd} s before round {rnd + 1}')
            time.sleep(90 * rnd)
        cfg = try_models(models, user)
        if cfg:
            return cfg
    raise RuntimeError('Gemini is overloaded right now (503). Try again later.')


def try_models(models, user):
    last = None
    for model in models:
        for attempt in range(2):
            try:
                r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/{model}:generateContent', params={'key': KEY}, timeout=150,
                                  json={'contents': [{'role': 'user', 'parts': [{'text': user}]}],
                                        'generationConfig': {'temperature': 1.0, 'responseMimeType': 'application/json'}})
                if r.status_code == 429:
                    log('rate limited, waiting 40 s'); time.sleep(40); continue
                r.raise_for_status()
                txt = r.json()['candidates'][0]['content']['parts'][0]['text']
                cfg = json.loads(txt[txt.find('{'):txt.rfind('}') + 1])
                validate(cfg)
                return cfg
            except requests.exceptions.Timeout as e:
                last = e
                log(f'{model} timed out, trying next model')
                break
            except Exception as e:
                last = e
                log(f'{model} attempt {attempt+1} failed: {str(e)[:200]}')
                time.sleep(10)
    return None


def validate(cfg):
    sc = cfg.get('scenes')
    assert isinstance(sc, list) and 8 <= len(sc) <= 14, 'need 8-14 scenes'
    def scene_text(s):
        ls = s.get('lines') or []
        if ls:
            return ' '.join(str(l.get('text', '')) for l in ls if isinstance(l, dict))
        return str(s.get('say', ''))
    for s in sc:
        assert isinstance(s, dict) and s.get('type') and scene_text(s).strip(), 'scene missing type/lines'
    words = sum(len(scene_text(s).split()) for s in sc)
    assert 150 <= words <= 250, f'narration length {words} words'
    assert cfg.get('title'), 'missing title'


def mock_cfg(i):
    return json.load(open(os.path.join(ROOT, 'sample_config.json'), encoding='utf-8'))


def chat_id():
    cid = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
    if cid:
        return cid
    if os.path.exists(CHAT_FILE):
        cid = open(CHAT_FILE).read().strip()
        if cid:
            return cid
    r = requests.get(f'https://api.telegram.org/bot{TG}/getUpdates', timeout=30).json()
    for u in reversed(r.get('result', [])):
        m = u.get('message') or u.get('channel_post') or {}
        if m.get('chat', {}).get('id'):
            cid = str(m['chat']['id'])
            open(CHAT_FILE, 'w').write(cid)
            return cid
    return ''


def tg_send(cid, path, cfg, idx):
    cap = (f"🎬 Video {idx}/{N}\n\n<b>{cfg.get('yt_title') or cfg.get('title')}</b>\n\n{cfg.get('description', '')}\n\n{cfg.get('hashtags', '')}\n\n"
           "Stock footage: Pixabay\n\n✅ Cek fakta dulu sebelum upload. Centang label konten AI.")
    with open(path, 'rb') as f:
        r = requests.post(f'https://api.telegram.org/bot{TG}/sendVideo', timeout=300,
                          data={'chat_id': cid, 'caption': cap[:1024], 'parse_mode': 'HTML', 'supports_streaming': 'true', 'width': 1080, 'height': 1920},
                          files={'video': (os.path.basename(path), f, 'video/mp4')})
    log('telegram', r.status_code, r.text[:200])
    def fmt(s):
        if s.get('lines'):
            return ' / '.join(f"{str(l.get('who','')).upper()}: {l.get('text','')}" for l in s['lines'] if isinstance(l, dict))
        return s.get('say', '')
    script = '\n'.join(f"{i+1}. {fmt(s)}" for i, s in enumerate(cfg['scenes']))
    requests.post(f'https://api.telegram.org/bot{TG}/sendMessage', timeout=60,
                  data={'chat_id': cid, 'text': ('📝 Naskah (untuk cek fakta):\n' + script)[:4000]})


def tg_text(cid, text):
    if TG and cid:
        requests.post(f'https://api.telegram.org/bot{TG}/sendMessage', timeout=60, data={'chat_id': cid, 'text': text[:4000]})


def daily_run(cid):
    avoid = done_topics()
    models = [] if MOCK else pick_model()
    made = []
    for i in range(1, N + 1):
        try:
            style = (ACCENTS[(len(avoid) + i) % len(ACCENTS)], STYLES[(len(avoid) + i) % len(STYLES)], random.choice(VOICES))
            cfg = mock_cfg(i) if MOCK else ask_gemini(models, avoid + [m['title'] for m in made], style)
            log(f'[{i}/{N}] {cfg["title"]}')
            cfg['min_seconds'] = MIN_SECONDS
            cfg['speed'] = max(1.05, float(cfg.get('speed', 1.0) or 1.0))
            os.makedirs(CONF_DIR, exist_ok=True)
            if not MOCK:
                json.dump(cfg, open(os.path.join(CONF_DIR, f'{WIB:%Y-%m-%d}_{i:02d}.json'), 'w'), ensure_ascii=False, indent=1)
            path = engine.build(cfg, OUT, workers=2, log=log)
            made.append(cfg)
            if TG and cid:
                tg_send(cid, path, cfg, i)
            if not MOCK:
                time.sleep(8)
        except Exception as e:
            traceback.print_exc()
            tg_text(cid, f'⚠️ Video {i} gagal dibuat: {str(e)[:300]}')
    if made and not MOCK:
        with open(DONE_FILE, 'a', encoding='utf-8') as f:
            for c in made:
                f.write(c['title'].strip() + '\n')
    if TG and cid:
        tg_text(cid, f'✅ Selesai: {len(made)}/{N} video hari ini.')
    return len(made)


def fmt_ts(sec):
    sec = int(sec)
    return f'{sec // 60}:{sec % 60:02d}' if sec < 3600 else f'{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


def compile_run(cid):
    done = set(l.strip() for l in open(COMPILED_FILE)) if os.path.exists(COMPILED_FILE) else set()
    files = [f for f in sorted(glob.glob(os.path.join(CONF_DIR, '*.json'))) if os.path.basename(f) not in done][-10:]
    if len(files) < COMPILE_MIN:
        log(f'compilation skipped: only {len(files)} new configs'); return 0
    cfgs = [json.load(open(f, encoding='utf-8')) for f in files]
    n = len(cfgs)
    scenes = [{'type': 'title', 'visual': 'twin', 'effect': 'pulse',
               'lines': [{'who': 'terra', 'mood': 'excited', 'text': f'Para, show me {n} worlds where everything went differently!'},
                         {'who': 'para', 'mood': 'happy', 'text': 'Buckle up, Terra. Let us begin.'}],
               'head': [f'{n} WHAT IFS', 'THAT WOULD', '*CHANGE EVERYTHING*']}]
    for i, c in enumerate(cfgs, 1):
        t = str(c.get('title', 'What if')).strip().rstrip('?')
        if i == 1:
            scenes[0]['chapter'] = f'{i}. {t}?'  # first chapter must start at 0:00 and be 10 s+
        scenes.append({'type': 'fact', 'chapter': f'{i}. {t}?' if i > 1 else '', 'kicker': f'WHAT IF #{i}', 'big': t + '?',
                       'lines': [{'who': 'para', 'mood': 'excited', 'text': f'Number {i}. {t}?'}]})
        scenes += [x for x in c.get('scenes', []) if x.get('type') not in ('brand', 'cta')]
    scenes += [{'type': 'brand', 'head': ['WHICH ONE', '*SCARED YOU?*'],
                'lines': [{'who': 'terra', 'mood': 'scared', 'text': 'Which timeline scared you the most?'},
                          {'who': 'para', 'mood': 'excited', 'text': 'Welcome to Parallel Earth!'}]},
               {'type': 'cta', 'head': ['NEW WORLDS', '*EVERY WEEK*'], 'button': 'SUBSCRIBE',
                'lines': [{'who': 'terra', 'mood': 'happy', 'text': 'Subscribe for a new parallel world every week!'}]}]
    ccfg = {'title': f'Compilation {WIB:%Y-%m-%d}', 'voice': 'am_michael', 'accent': random.choice(ACCENTS), 'style': 'space', 'scenes': scenes}
    log(f'Compilation: {n} videos, {len(scenes)} scenes')
    vert = engine.build(ccfg, OUT, workers=2, log=log)
    timing = json.load(open(os.path.join(os.path.dirname(vert), 'pe_work_' + engine.slug(ccfg['title']), 'timing.json')))
    wide = vert[:-4] + '_16x9.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', vert, '-filter_complex',
                    '[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=40:2,eq=brightness=-0.3[bg];'
                    '[0:v]scale=-2:1080[fg];[bg][fg]overlay=(W-w)/2:0,format=yuv420p',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'copy', '-movflags', '+faststart', wide], check=True)
    # chapters
    t, chapters = 0.0, []
    for sc, fr in zip(scenes, timing['frames']):
        if sc.get('chapter'):
            chapters.append(f"{fmt_ts(t)} {sc['chapter']}")
        t += fr
    title = f'{n} What If Scenarios That Would Change Earth Forever | Parallel Earth'
    desc = 'What would happen if the rules of our planet changed? ' + '\n\n' + '\n'.join(chapters) + '\n\n#whatif #science #space #parallelearth'
    open(os.path.join(OUT, 'compilation_description.txt'), 'w', encoding='utf-8').write(title + '\n\n' + desc)
    log(f'Compilation done: {wide} ({t/60:.1f} min)')
    if TG and cid:
        send = wide
        if os.path.getsize(wide) > 48 * 1024 * 1024:
            send = wide[:-4] + '_tg.mp4'
            kbps = max(300, int(44 * 8 * 1024 / t) - 128)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wide, '-vf', 'scale=1280:720', '-c:v', 'libx264', '-preset', 'veryfast',
                            '-b:v', f'{kbps}k', '-maxrate', f'{kbps}k', '-bufsize', f'{2*kbps}k', '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', send], check=True)
        cap = f'📺 <b>Video panjang mingguan</b> ({t/60:.1f} menit, 16:9)\n\n<b>{title}</b>\n\nUpload sebagai video biasa (bukan Shorts). Versi kualitas penuh ada di GitHub → Actions → run terbaru → artifact.'
        with open(send, 'rb') as f:
            r = requests.post(f'https://api.telegram.org/bot{TG}/sendVideo', timeout=600,
                              data={'chat_id': cid, 'caption': cap[:1024], 'parse_mode': 'HTML', 'supports_streaming': 'true'},
                              files={'video': (os.path.basename(send), f, 'video/mp4')})
        log('telegram', r.status_code, r.text[:200])
        tg_text(cid, '📝 Deskripsi + chapter (salin ke YouTube):\n\n' + title + '\n\n' + desc)
    if not MOCK:
        with open(COMPILED_FILE, 'a') as f:
            for p in files:
                f.write(os.path.basename(p) + '\n')
    return 1


def main():
    cid = chat_id() if TG else ''
    log('telegram chat:', 'found' if cid else 'not set', '| mode:', MODE, '| WIB', f'{WIB:%A %Y-%m-%d}')
    made = 0
    if MODE in ('auto', 'daily', 'both'):
        made += daily_run(cid)
    if MODE in ('compile', 'both') or (MODE == 'auto' and WIB.weekday() == 6):
        try:
            made += compile_run(cid)
        except Exception as e:
            traceback.print_exc()
            tg_text(cid, f'⚠️ Video kompilasi gagal: {str(e)[:300]}')
    if not made:
        raise SystemExit('no video made')


if __name__ == '__main__':
    main()
