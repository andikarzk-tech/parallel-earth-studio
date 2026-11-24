"""Parallel Earth daily run: Gemini writes N configs -> engine renders N videos -> sent to Telegram.
Env: GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional), VIDEOS (default 3), MOCK=1 for a dry run.
"""
import os, json, re, time, random, traceback
import requests
import engine

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'out')
os.makedirs(OUT, exist_ok=True)
DONE_FILE = os.path.join(ROOT, 'topics_done.txt')
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
    return cands[:4]


def ask_gemini(models, avoid, used_styles):
    user = (PROMPT + '\n\nTOPIC: invent ONE fresh, surprising "What if" topic yourself (science, space, nature, history, human body, technology). '
            'It must be different from all of these already-made topics:\n- ' + '\n- '.join(avoid[-150:] or ['(none yet)']) +
            f'\n\nUse accent "{used_styles[0]}", style "{used_styles[1]}", voice "{used_styles[2]}".'
            '\nAlso add these top-level keys: "yt_title" (max 70 chars, catchy, ends with #shorts), '
            '"description" (2 short sentences), "hashtags" (5-8 hashtags in one string).')
    last = None
    for model in models:
        for attempt in range(2):
            try:
                r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/{model}:generateContent', params={'key': KEY}, timeout=120,
                                  json={'contents': [{'role': 'user', 'parts': [{'text': user}]}],
                                        'generationConfig': {'temperature': 1.0, 'responseMimeType': 'application/json'}})
                if r.status_code == 429:
                    log('rate limited, waiting 40 s'); time.sleep(40); continue
                r.raise_for_status()
                txt = r.json()['candidates'][0]['content']['parts'][0]['text']
                cfg = json.loads(txt[txt.find('{'):txt.rfind('}') + 1])
                validate(cfg)
                return cfg
            except Exception as e:
                last = e
                log(f'{model} attempt {attempt+1} failed: {str(e)[:200]}')
                time.sleep(5)
    raise RuntimeError(f'Gemini failed: {last}')


def validate(cfg):
    sc = cfg.get('scenes')
    assert isinstance(sc, list) and 5 <= len(sc) <= 12, 'need 5-12 scenes'
    for s in sc:
        assert isinstance(s, dict) and s.get('type') and str(s.get('say', '')).strip(), 'scene missing type/say'
    words = sum(len(str(s['say']).split()) for s in sc)
    assert 70 <= words <= 220, f'narration length {words} words'
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
           "✅ Cek fakta dulu sebelum upload. Centang label konten AI.")
    with open(path, 'rb') as f:
        r = requests.post(f'https://api.telegram.org/bot{TG}/sendVideo', timeout=300,
                          data={'chat_id': cid, 'caption': cap[:1024], 'parse_mode': 'HTML', 'supports_streaming': 'true', 'width': 1080, 'height': 1920},
                          files={'video': (os.path.basename(path), f, 'video/mp4')})
    log('telegram', r.status_code, r.text[:200])
    script = '\n'.join(f"{i+1}. {s.get('say','')}" for i, s in enumerate(cfg['scenes']))
    requests.post(f'https://api.telegram.org/bot{TG}/sendMessage', timeout=60,
                  data={'chat_id': cid, 'text': ('📝 Naskah (untuk cek fakta):\n' + script)[:4000]})


def tg_text(cid, text):
    if TG and cid:
        requests.post(f'https://api.telegram.org/bot{TG}/sendMessage', timeout=60, data={'chat_id': cid, 'text': text[:4000]})


def main():
    avoid = done_topics()
    models = [] if MOCK else pick_model()
    cid = chat_id() if TG else ''
    log('telegram chat:', 'found' if cid else 'not set')
    made = []
    for i in range(1, N + 1):
        try:
            style = (ACCENTS[(len(avoid) + i) % len(ACCENTS)], STYLES[(len(avoid) + i) % len(STYLES)], random.choice(VOICES))
            cfg = mock_cfg(i) if MOCK else ask_gemini(models, avoid + [m['title'] for m in made], style)
            log(f'[{i}/{N}] {cfg["title"]}')
            json.dump(cfg, open(os.path.join(OUT, f'config_{i:02d}.json'), 'w'), ensure_ascii=False, indent=1)
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
    if not made:
        raise SystemExit('no video made')


if __name__ == '__main__':
    main()
