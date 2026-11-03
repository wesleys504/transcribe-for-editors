# Transcribe

A Mac app that transcribes your clips on your own computer. For each clip you get:

- **`Clip.srt`**: captions, ready to import into Premiere
- **`Clip - transcript.docx`** and **`.txt`**: a timecoded transcript with speaker names, using the clip's source timecode when it has one
- **A summary (optional):** `Clip - summary.docx` with a short summary, the topics, and the best soundbites with timecodes
- **Translations (optional):** `Clip - ES.srt` with the same timings, plus a translated transcript. Add more languages to any clip later.

Transcription runs locally with Whisper, so it's free and works offline. With a Claude API key, Claude also fixes names and terms, guesses who's speaking, and does the translations.

## Install (once)

1. Unzip the download. You'll get a **Transcribe** folder.
2. Open **Terminal** (Cmd+Space, type "Terminal").
3. Type `bash ` (with a space), drag **install.command** from the Transcribe folder into the window, and press Return. Press Return again to start.

The first install downloads 2–3 GB and takes 5–15 minutes. Updates take about a minute. When it's done, **Transcribe** opens. It lives in your Applications folder, with a shortcut on your Desktop. Drag it to your Dock to keep it handy.

## Using it

- **Drop clips or folders** onto the window, or use **Add clips…** / **Add folder…**. Each clip shows its progress.
- **Who's talking?** When a clip with several speakers finishes, a window shows a few of each speaker's lines as subtitles. Press ▶ to hear a line, then type a name. Names you've used before are suggested. Typing the same name for two labels merges them into one person.
- **Summary** opens the summary and soundbites. Press ▶ to hear a bite, or copy them all.
- **More → Translate…** translates a clip you've already transcribed, into one language or several. **More → Write summary** adds a summary to an older clip.
- **Recent** lists everything you've transcribed, so you can open transcripts, rename speakers, translate or summarize later.
- **Claude this month** (top right) shows an estimate of what Claude has cost. Each clip shows its own cost too. Your exact bill is on console.anthropic.com under Usage and Billing.
- **Watch folder** (top right): while it's on, anything put in `Transcribe/Watch Folder` in your home folder is transcribed automatically, even with the app closed.
- **Settings:**
  - your Claude API key (stored in your Mac's Keychain)
  - name fixes on or off
  - which language to translate to
  - how many people are talking
  - the spoken language
  - your list of **names and terms**
  - caption line length (42 characters for broadcast, 32 for vertical social)

For names that only apply to one project, put a `names.txt` file in that project's footage folder.

## If something goes wrong

The **Activity** tab shows what happened. The full log is in `Transcribe/transcribe.log` in your home folder.

## Uninstall

Turn off the watch folder, then delete `Transcribe` from your home folder and from Applications.

---

© 2026 TBD Media Post. All rights reserved.
