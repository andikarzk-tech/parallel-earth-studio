# Parallel Earth Studio — otomatis harian

Setiap hari pukul 04:00 WIB, GitHub Actions:
1. meminta Gemini (gratis) menulis naskah + config video "What if" dengan topik baru,
2. membuat voice-over (Kokoro, open-source), visual motion graphic, dan musik latar,
3. mengirim video MP4 + judul, deskripsi, hashtag, dan naskah ke Telegram.

## Pengaturan
- Secrets (Settings → Secrets and variables → Actions):
  - `GEMINI_API_KEY` — dari aistudio.google.com
  - `TELEGRAM_BOT_TOKEN` — dari @BotFather (kirim /start ke bot Anda sekali)
- Variable opsional `VIDEOS` — jumlah video per hari (default 3).
- Jalankan manual: tab Actions → Parallel Earth Daily → Run workflow.
- Video juga tersimpan 7 hari sebagai artifact di halaman run.
