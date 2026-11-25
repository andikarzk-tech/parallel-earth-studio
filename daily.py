"""Parallel Earth daily run: Gemini writes N configs -> engine renders N videos -> sent to Telegram.
Env: GEMINI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional), VIDEOS (default 3), MOCK=1 for a dry run.
YouTube auto-upload (optional): YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN; YT_SLOTS (WIB publish times, default 19:00,07:00,23:00,15:00,11:00),
YT_COMPILE_SLOT (default 20:00). Videos are uploaded private + scheduled (publishAt), so they can be deleted before they go live.
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
NEXT_FILE = os.path.join(ROOT, 'next_topic.txt')
IDEA_POOL = ['the Sun turned blue', 'humans could breathe underwater', 'Earth had rings like Saturn', 'the Moon was twice as close',
             'oxygen levels doubled', 'Earth stopped spinning', 'there was no night', 'insects grew as big as dogs',
             'the oceans were made of fresh water', 'Earth had two suns', 'gravity was half as strong', 'it rained every day everywhere',
             'humans hibernated in winter', 'Earth was flat for one day', 'volcanoes never existed', 'trees could walk']
N = int(os.environ.get('VIDEOS', '3') or 3)
KEY = os.environ.get('GEMINI_API_KEY', '').strip()
TG = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
MOCK = os.environ.get('MOCK') == '1'
YT_ID = os.environ.get('YT_CLIENT_ID', '').strip()
YT_SECRET = os.environ.get('YT_CLIENT_SECRET', '').strip()
YT_REFRESH = os.environ.get('YT_REFRESH_TOKEN', '').strip()
YT_ON = bool(YT_ID and YT_SECRET and YT_REFRESH) and not MOCK
YT_SLOTS = [x.strip() for x in (os.environ.get('YT_SLOTS') or '19:00,07:00,23:00,15:00,11:00').split(',') if x.strip()]
YT_COMPILE_SLOT = (os.environ.get('YT_COMPILE_SLOT') or '20:00').strip()

PROMPT = open(os.path.join(ROOT, 'prompt.txt'), encoding='utf-8').read()
ACCENTS = ['orange', 'red', 'green', 'purple', 'cyan', 'yellow']
STYLES = ['space', 'grid', 'dust']
VOICES = ['am_michael', 'am_fenrir', 'bm_george', 'bm_fable', 'af_heart', 'am_puck']


# ---------------- Self-improvement loop (also in learn.py) ----------------
import math
import xml.etree.ElementTree as ET
CHANNEL_ID = os.environ.get('YT_CHANNEL_ID', 'UCsBqaoVeFY0owFLLZkiITMA')

GENES = {
    'accent': ['orange', 'red', 'green', 'purple', 'cyan', 'yellow'],
    'style': ['space', 'grid', 'dust'],
    'speed': ['1.05', '1.1', '1.15'],
    'hook': ['shock', 'question', 'countdown', 'imagine', 'myth'],
    'category': ['space', 'planet earth', 'human body', 'animals', 'history', 'technology', 'everyday life', 'oceans'],
}
HOOKS = {
    'shock': 'Terra\'s FIRST line states one shocking, concrete consequence with a number (e.g. "In 3 seconds every ocean would leave its bed!").',
    'question': 'Terra\'s FIRST line is a direct question to the viewer that creates instant curiosity (e.g. "Would you survive one day here?").',
    'countdown': 'Tell the story as a timeline: Terra\'s FIRST line starts a countdown ("Second 1...", "Day 1...") and later scenes jump forward in time.',
    'imagine': 'Terra\'s FIRST line puts the viewer inside it: "Imagine you wake up and ...".',
    'myth': 'Terra\'s FIRST line breaks a common belief: "Everyone thinks ... They are wrong."',
}
EXPLORE = 0.3     # share of choices spent on trying new things
MIN_AGE_H = 12    # a video needs this many hours online before it is judged
NS = {'a': 'http://www.w3.org/2005/Atom', 'yt': 'http://www.youtube.com/xml/schemas/2015', 'media': 'http://search.yahoo.com/mrss/'}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def fetch_rss():
    """{video_id: (views, published_utc)} for the newest ~15 public videos."""
    out = {}
    try:
        r = requests.get('https://www.youtube.com/feeds/videos.xml', params={'channel_id': CHANNEL_ID}, timeout=30)
        if r.status_code != 200:
            return out
        root = ET.fromstring(r.content)
        for e in root.findall('a:entry', NS):
            vid = e.findtext('yt:videoId', default='', namespaces=NS)
            pub = e.findtext('a:published', default='', namespaces=NS)
            st = e.find('media:group/media:community/media:statistics', NS)
            views = int(st.get('views', '0')) if st is not None else 0
            try:
                p = datetime.datetime.fromisoformat(pub.replace('Z', '+00:00')).astimezone(datetime.timezone.utc).replace(tzinfo=None)
            except Exception:
                p = None
            if vid:
                out[vid] = (views, p)
    except Exception as e:
        print('rss failed:', e)
    return out


def update_stats(conf_dir):
    """Write the latest view counts into every config that was uploaded. Returns list of (path, cfg)."""
    rss = fetch_rss()
    items = []
    for p in sorted(glob.glob(os.path.join(conf_dir, '*.json'))):
        try:
            c = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        vid = c.get('yt_id')
        if vid and vid in rss:
            views, pub = rss[vid]
            c['stats'] = {'views': views, 'published': pub.isoformat() if pub else c.get('stats', {}).get('published'),
                          'checked': _now().isoformat(timespec='minutes')}
            json.dump(c, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        items.append((p, c))
    return items


def _score(c):
    st = c.get('stats') or {}
    if not st.get('published') or 'views' not in st:
        return None
    try:
        pub = datetime.datetime.fromisoformat(st['published'])
        chk = datetime.datetime.fromisoformat(st.get('checked') or _now().isoformat())
    except Exception:
        return None
    age = (chk - pub).total_seconds() / 3600
    if age < MIN_AGE_H:
        return None
    return st['views'] / math.log2(2 + min(age, 24 * 14))


def _first_line(c):
    for s in c.get('scenes', []):
        for l in s.get('lines') or []:
            if isinstance(l, dict) and l.get('text'):
                return l['text']
    return ''


def learn_from_data(conf_dir):
    """Returns (state, brief_for_gemini, report_for_telegram)."""
    items = update_stats(conf_dir)
    scored = [(s, c) for _, c in items if (s := _score(c)) is not None]
    state = {g: {} for g in GENES}
    if len(scored) < 3:
        brief = ('CHANNEL DATA: not enough viewer data yet. Experiment boldly: unusual angles, strong numbers, a twist at the end.')
        report = f'📈 Belajar: baru {len(scored)} video punya data penonton (perlu ≥3, umur ≥{MIN_AGE_H} jam). Mesin masih mode eksplorasi.'
        return state, brief, report
    med = sorted(s for s, _ in scored)[len(scored) // 2] or 1.0
    for s, c in scored:
        rel = s / med
        for g, v in (c.get('genes') or {}).items():
            if g in state and v is not None:
                n, tot = state[g].get(v, (0, 0.0))
                state[g][v] = (n + 1, tot + rel)
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    top, bottom = ranked[:3], ranked[-3:][::-1]

    def fmt(lst):
        return '\n'.join(f'- "{c.get("title")}" ({s / med:.1f}x average) opening line: "{_first_line(c)[:120]}" '
                         f'[hook={c.get("genes", {}).get("hook")}, category={c.get("genes", {}).get("category")}]' for s, c in lst)
    winners = []
    for g in ('hook', 'category', 'style', 'accent', 'speed'):
        vals = {v: tot / n for v, (n, tot) in state[g].items() if n >= 2}
        if vals:
            best = max(vals, key=vals.get)
            winners.append(f'{g}={best} ({vals[best]:.1f}x)')
    brief = ('CHANNEL DATA (real viewers, use it):\nOur BEST videos so far:\n' + fmt(top) +
             '\nOur WEAKEST videos:\n' + fmt(bottom) +
             '\nCopy what made the best ones work (their opening energy, kind of topic, tension). Avoid what the weakest did.'
             + ('\nWinning patterns: ' + ', '.join(winners) if winners else ''))
    report = (f'📈 Laporan belajar ({len(scored)} video dinilai)\n🏆 Terbaik: {top[0][1].get("title")} ({top[0][0] / med:.1f}x rata-rata)\n'
              f'🐢 Terlemah: {bottom[0][1].get("title")} ({bottom[0][0] / med:.1f}x)\n'
              + ('🧬 Gen unggul: ' + ', '.join(winners) if winners else '🧬 Gen unggul: belum cukup data per gen'))
    return state, brief, report


def pick_genes(state, rnd=random):
    """Choose one value per gene: exploit winners, keep exploring."""
    out = {}
    for g, values in GENES.items():
        seen = state.get(g, {})
        untested = [v for v in values if v not in seen]
        if rnd.random() < EXPLORE or not seen:
            out[g] = rnd.choice(untested or values)
            continue
        weights = []
        for v in values:
            n, tot = seen.get(v, (0, 0.0))
            mean = (tot + 1.0) / (n + 1)          # prior of 1.0 = channel average
            weights.append(max(0.05, mean) ** 2)
        out[g] = rnd.choices(values, weights=weights)[0]
    return out


def log(*a):
    print(*a, flush=True)


def done_topics():
    if not os.path.exists(DONE_FILE):
        return []
    return [l.strip() for l in open(DONE_FILE, encoding='utf-8') if l.strip() and not l.startswith('#')]


def get_next():
    """Queued topic teased by the previous video: returns (topic, category)."""
    if os.path.exists(DONE_FILE):
        for l in open(DONE_FILE, encoding='utf-8'):
            if l.startswith('#next '):
                t, _, cat = l[6:].strip().partition('|')
                return t.strip(), (cat.strip() or None)
    return '', None


def set_next(topic, category=None):
    lines = [l for l in open(DONE_FILE, encoding='utf-8')] if os.path.exists(DONE_FILE) else []
    lines = [l for l in lines if not l.startswith('#next ')]
    lines.append(f'#next {topic}' + (f' | {category}' if category else '') + '\n')
    open(DONE_FILE, 'w', encoding='utf-8').write(''.join(l if l.endswith('\n') else l + '\n' for l in lines))


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


def ask_gemini(models, avoid, genes, brief, topic=None, next_cat='space'):
    topic_txt = (f'TOPIC (use exactly this): What if {topic}?' if topic else
                 f'TOPIC: invent ONE fresh, surprising "What if" topic yourself in the category "{genes["category"]}".')
    user = (PROMPT + f'\n\n{topic_txt}\nNEXT TOPIC (you choose it): a fresh, surprising "What if" in the category "{next_cat}". '
            'Tease it in the brand scene and put it in "next_topic".\n'
            'Both must be different from all of these already-made topics:\n- ' + '\n- '.join(avoid[-150:] or ['(none yet)']) +
            f'\n\nHOOK STYLE: {HOOKS[genes["hook"]]}'
            f'\nUse accent "{genes["accent"]}", style "{genes["style"]}".'
            f'\n\n{brief}'
            '\nAlso add these top-level keys: "yt_title" (max 70 chars, catchy, ends with #shorts), '
            '"description" (2 short sentences), "hashtags" (5-8 hashtags in one string).')
    for rnd in range(3):
        if rnd:
            log(f'All Gemini models busy, waiting {90 * rnd} s before round {rnd + 1}')
            time.sleep(90 * rnd)
        cfg = try_models(models, user)
        if cfg:
            return cfg
    raise RuntimeError('Gemini is overloaded right now (503). Try again later.')


EDITOR = ('''You are a ruthless YouTube Shorts retention editor and a careful science fact-checker.
Improve this Parallel Earth script (JSON below) so more viewers watch to the end:
- the first line must hook in under 2 seconds; every line must make the viewer want the next one
- cut filler words, use vivid concrete numbers and images, keep Terra and Para's personalities and the running gags
- FIX any fact that is wrong or exaggerated (real-science parts must be accurate)
- keep the exact same JSON schema, keys, scene types, scene order, visual fields and "next_topic"; keep 8-14 scenes and 150-250 spoken words in total
Return ONLY the full improved JSON.''')


def polish(models, cfg):
    """Second Gemini pass: retention + fact check. Falls back to the original on any problem."""
    keep = {k: cfg.get(k) for k in ('next_topic', 'accent', 'style')}
    try:
        better = try_models(models[:3], EDITOR + '\n\n' + json.dumps(cfg, ensure_ascii=False))
        if better and len(better.get('scenes', [])) == len(cfg['scenes']):
            better.update({k: v for k, v in keep.items() if v})
            log('polish: improved script accepted')
            return better
    except Exception as e:
        log('polish failed:', str(e)[:200])
    return cfg


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


# ---------------- YouTube auto-upload ----------------
def yt_token():
    r = requests.post('https://oauth2.googleapis.com/token', timeout=30,
                      data={'client_id': YT_ID, 'client_secret': YT_SECRET, 'refresh_token': YT_REFRESH, 'grant_type': 'refresh_token'})
    if r.status_code != 200:
        raise RuntimeError(f'YouTube token error {r.status_code}: {r.text[:200]}')
    return r.json()['access_token']


def publish_time(slot):
    """Next WIB time for slot 'HH:MM' that is at least 45 min away. Returns (utc datetime, wib datetime)."""
    h, m = (int(x) for x in slot.split(':'))
    now_wib = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(hours=7)
    t = now_wib.replace(hour=h, minute=m, second=0, microsecond=0)
    while t < now_wib + datetime.timedelta(minutes=45):
        t += datetime.timedelta(days=1)
    return t - datetime.timedelta(hours=7), t


def tags_from(hashtags):
    tags = [w.lstrip('#') for w in re.split(r'[\s,]+', hashtags or '') if w.strip('#')]
    return list(dict.fromkeys(tags + ['what if', 'science', 'Parallel Earth']))[:15]


def yt_upload(path, title, desc, tags, slot):
    when_utc, when_wib = publish_time(slot)
    body = {'snippet': {'title': title[:100], 'description': desc[:4900], 'tags': tags, 'categoryId': '28',
                        'defaultLanguage': 'en', 'defaultAudioLanguage': 'en'},
            'status': {'privacyStatus': 'private', 'publishAt': when_utc.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                       'selfDeclaredMadeForKids': False, 'embeddable': True}}
    size = os.path.getsize(path)
    last = None
    for attempt in range(3):
        try:
            tok = yt_token()
            r = requests.post('https://www.googleapis.com/upload/youtube/v3/videos', timeout=60,
                              params={'uploadType': 'resumable', 'part': 'snippet,status', 'notifySubscribers': 'true'},
                              headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/json; charset=UTF-8',
                                       'X-Upload-Content-Type': 'video/mp4', 'X-Upload-Content-Length': str(size)},
                              data=json.dumps(body))
            if r.status_code != 200:
                raise RuntimeError(f'init {r.status_code}: {r.text[:300]}')
            with open(path, 'rb') as f:
                r = requests.put(r.headers['Location'], data=f, timeout=1800,
                                 headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'video/mp4', 'Content-Length': str(size)})
            if r.status_code not in (200, 201):
                raise RuntimeError(f'upload {r.status_code}: {r.text[:300]}')
            vid = r.json()['id']
            log('youtube uploaded', vid, 'publish', f'{when_wib:%Y-%m-%d %H:%M} WIB')
            return vid, when_wib
        except Exception as e:
            last = e
            log('youtube upload attempt', attempt + 1, 'failed:', e)
            if 'invalid_grant' in str(e) or 'quotaExceeded' in str(e) or 'uploadLimitExceeded' in str(e):
                break
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(str(last)[:400])


def yt_short(path, cfg, idx):
    title = str(cfg.get('yt_title') or cfg.get('title') or 'What If?').strip()
    if '#shorts' not in title.lower() and len(title) < 90:
        title += ' #shorts'
    desc = (f"{cfg.get('description', '')}\n\nWhich world would you choose? Tell Terra and Para in the comments!\n"
            f"Subscribe for a new parallel world every day.\n\nStock footage: Pixabay\n\n{cfg.get('hashtags', '')}")
    slot = YT_SLOTS[(idx - 1) % len(YT_SLOTS)]
    return yt_upload(path, title, desc, tags_from(cfg.get('hashtags', '')), slot)


def tg_send(cid, path, cfg, idx, yt=None):
    cap = (f"🎬 Video {idx}/{N}\n\n<b>{cfg.get('yt_title') or cfg.get('title')}</b>\n\n{cfg.get('description', '')}\n\n{cfg.get('hashtags', '')}\n\n"
           "Stock footage: Pixabay\n\n")
    if yt and yt[0]:
        cap += (f"📅 SUDAH DIUNGGAH ke YouTube, tayang otomatis {yt[1]:%d %b %H:%M} WIB.\n"
                f"Cek fakta di naskah. Kalau ada yang salah, hapus sebelum jam tayang:\nhttps://studio.youtube.com/video/{yt[0]}/edit")
    elif yt:
        cap += f"⚠️ Upload YouTube gagal: {yt[1]}\nUpload manual dari video ini."
    else:
        cap += "✅ Cek fakta dulu sebelum upload."
    if os.path.getsize(path) > 48 * 1024 * 1024:  # Telegram bots accept up to 50 MB
        small = path[:-4] + '_tg.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '27',
                        '-c:a', 'copy', '-movflags', '+faststart', small], check=True)
        cap += '\n(Versi Telegram dikompres; kualitas penuh ada di GitHub → Actions → artifact.)'
        path = small
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
    state, brief, report = ({g: {} for g in GENES}, '', '') if MOCK else learn_from_data(CONF_DIR)
    log(report)
    tg_text(cid, report)
    plan = [pick_genes(state) for _ in range(N + 1)]   # genes for today's videos (+1 = tomorrow's first topic category)
    made = []
    for i in range(1, N + 1):
        try:
            genes = plan[i - 1]
            queued, queued_cat = get_next()
            if queued and queued_cat:
                genes['category'] = queued_cat       # topic was teased yesterday / by the previous video
            cfg = mock_cfg(i) if MOCK else ask_gemini(models, avoid + [m['title'] for m in made], genes, brief,
                                                      queued or None, plan[i]['category'])
            if not MOCK:
                cfg = polish(models, cfg)
                nt = str(cfg.get('next_topic') or '').strip()
                nt = re.sub(r'(?i)^what if\s+', '', nt).rstrip('?').strip()
                if nt:
                    set_next(nt, plan[i]['category'])
            log(f'[{i}/{N}] {cfg["title"]} | genes {genes}')
            cfg['genes'] = genes
            cfg['accent'], cfg['style'] = genes['accent'], genes['style']
            cfg['min_seconds'] = MIN_SECONDS
            cfg['speed'] = float(genes['speed'])
            os.makedirs(CONF_DIR, exist_ok=True)
            conf_path = os.path.join(CONF_DIR, f'{WIB:%Y-%m-%d}_{i:02d}.json')
            if not MOCK:
                json.dump(cfg, open(conf_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            path = engine.build(cfg, OUT, workers=2, log=log)
            made.append(cfg)
            yt = None
            if YT_ON:
                try:
                    yt = yt_short(path, cfg, i)
                    cfg['yt_id'], cfg['publish_at'] = yt[0], yt[1].isoformat(timespec='minutes')
                    json.dump(cfg, open(conf_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                except Exception as e:
                    traceback.print_exc()
                    yt = ('', str(e)[:300])
            if TG and cid:
                tg_send(cid, path, cfg, i, yt)
            if not MOCK:
                time.sleep(8)
        except Exception as e:
            traceback.print_exc()
            tg_text(cid, f'⚠️ Video {i} gagal dibuat: {str(e)[:300]}')
    if made and not MOCK:
        with open(DONE_FILE, 'a', encoding='utf-8') as f:
            for c in made:
                f.write(c['title'].strip() + '\n')
        with open(os.path.join(CONF_DIR, 'learning_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f'\n=== {WIB:%Y-%m-%d} ===\n{report}\nToday genes: ' + json.dumps([c.get('genes') for c in made]) + '\n')
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
    yt = None
    if YT_ON:
        try:
            yt = yt_upload(wide, title, desc, ['what if', 'science', 'space', 'Parallel Earth', 'compilation'], YT_COMPILE_SLOT)
        except Exception as e:
            traceback.print_exc()
            yt = ('', str(e)[:300])
    if TG and cid:
        send = wide
        if os.path.getsize(wide) > 48 * 1024 * 1024:
            send = wide[:-4] + '_tg.mp4'
            kbps = max(300, int(44 * 8 * 1024 / t) - 128)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wide, '-vf', 'scale=1280:720', '-c:v', 'libx264', '-preset', 'veryfast',
                            '-b:v', f'{kbps}k', '-maxrate', f'{kbps}k', '-bufsize', f'{2*kbps}k', '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', send], check=True)
        cap = f'📺 <b>Video panjang mingguan</b> ({t/60:.1f} menit, 16:9)\n\n<b>{title}</b>\n\n'
        if yt and yt[0]:
            cap += f'📅 SUDAH DIUNGGAH, tayang {yt[1]:%d %b %H:%M} WIB.\nhttps://studio.youtube.com/video/{yt[0]}/edit'
        else:
            cap += 'Upload sebagai video biasa (bukan Shorts). Versi kualitas penuh ada di GitHub → Actions → run terbaru → artifact.'
            if yt:
                cap += f'\n⚠️ Upload otomatis gagal: {yt[1]}'
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
    log('telegram chat:', 'found' if cid else 'not set', '| mode:', MODE, '| WIB', f'{WIB:%A %Y-%m-%d}', '| youtube:', 'on' if YT_ON else 'off')
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
