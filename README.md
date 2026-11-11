# Transcribe

A Mac app for editors that transcribes your clips on your own computer. Drop clips in, choose what you want, and choose where it goes:

- **Captions** (`Clip.srt`), ready to import into Premiere
- **A transcript** as Word and/or text: timecoded, with speaker names, using the clip's source timecode when it has one
- **A summary** with the topics and the best soundbites with timecodes
- **Translations** in as many languages as you need, with captions on the same timings
- **Clean audio** (`Clip - clean.wav`): a cleaned-up dialogue track exactly as long as the clip, so it drops in and stays in sync

Every feature has a free option. Transcription always runs on your Mac with Whisper. Name fixes, summaries and translation use either **Claude** (best quality, about a cent per transcript) or a **free local model** through Ollama. Audio cleanup runs **on your Mac for free**, or through **ElevenLabs**, **Auphonic** or **ai-coustics** if you have an account.

## Install (once)

1. Unzip the download. You'll get a **Transcribe** folder.
2. Open **Terminal** (Cmd+Space, type "Terminal").
3. Type `bash ` (with a space), drag **install.command** from the Transcribe folder into the window, and press Return. Press Return again to start.

The first install downloads 2–3 GB and takes 5–15 minutes. Updates take about a minute. When it's done, **Transcribe** opens. It lives in your Applications folder, with a shortcut on your Desktop.

## Using it

- **Drop clips or folders** onto the window, or use **Add clips…** / **Add folder…**.
- **Create files** opens with each batch. Tick what you want (captions, Word or text transcript, summary, clean audio, translations) and where to save it:
  - next to each clip
  - in a **Transcripts** folder next to each clip
  - in **one folder** you choose

  Tick **Save these as my choices** to make them your defaults. Tick **Don't ask again** to skip the window; you can turn it back on in Settings.
- **Who's talking?** When a clip with several speakers finishes, a window shows a few of each speaker's lines as subtitles. Press ▶ to hear a line, then type a name. Giving two labels the same name merges them.
- **Summary** opens the summary and soundbites. Press ▶ to hear a bite, or copy them all.
- **More → Translate…** or **More → Write summary** adds those to a clip you've already done. They're saved in the same place as the clip's other files.
- **Recent** lists everything you've transcribed, so you can open, rename, translate or summarize later.
- **Watch folder** (top right): anything put in `Transcribe/Watch Folder` in your home folder is processed automatically with your saved choices, even with the app closed.

If two clips have the same name (like `A001.mov` from two camera cards) and you save to one folder, the second gets its folder name added, so nothing is overwritten.

## The free options

**Name fixes, summaries and translation on your Mac:**
1. Install **Ollama** (free) from [ollama.com](https://ollama.com) and open it.
2. In Transcribe, go to **Settings → AI** and choose **On this Mac (free)**.
3. Pick a model and click **Download model**:
   - **Qwen 2.5 7B** if your Mac has 16 GB of memory
   - **Qwen 2.5 3B** if it has 8 GB

It's slower than Claude and a little less polished, especially for translation, but nothing leaves your Mac.

**Audio cleanup on your Mac:** choose **On this Mac (free)** under Audio cleanup. It removes hum, room noise and wind, and needs an Apple Silicon Mac. It runs at several times real speed.

## Paid services (optional)

Add your keys in **Settings**. They're stored in your Mac's Keychain, never in the app's files.

| Service | Used for | Cost |
| --- | --- | --- |
| Claude | name fixes, summaries, translation | about 1¢ per transcript ([console.anthropic.com](https://console.anthropic.com)) |
| ElevenLabs Voice Isolator | pulling voices out of noise or music | your ElevenLabs credits |
| Auphonic | noise and reverb reduction, leveling, broadcast loudness | 2 free hours a month, then paid |
| ai-coustics | studio-style speech enhancement | paid, with a free trial |

Auphonic and ai-coustics also level the audio to broadcast loudness (-23 LUFS). Adobe Podcast's Enhance Speech isn't included because Adobe doesn't offer a public API for it.

**Settings → Spending** shows estimated Claude costs and how many minutes you've sent to each audio service.

## If something goes wrong

The **Activity** tab shows what happened. The full log is in `Transcribe/transcribe.log` in your home folder.

## Uninstall

Turn off the watch folder, then delete `Transcribe` from your home folder and from Applications.

## License

© 2026 TBD Media Post. Free for personal and other noncommercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE.md).

Using Transcribe on paid or commercial work, or building it into a product, needs permission. Get in touch through GitHub.
