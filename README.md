# Transcribe

A Mac app for editors that transcribes your clips on your own computer. Drop clips in, choose what you want, and choose where it goes:

- **Captions** (`Clip.srt`), ready to import into Premiere
- **A transcript** as Word and/or text: timecoded, with speaker names, using the clip's source timecode when it has one
- **A summary** with the topics and the best soundbites with timecodes
- **Translations** in as many languages as you need, with captions on the same timings
- **Clean audio** (`Clip - clean.wav`): a cleaned-up dialogue track exactly as long as the clip, so it drops in and stays in sync
- **Dialogue, music and effects stems** (`Clip - dialogue.wav`, `Clip - music.wav`, `Clip - effects.wav`): the clip's mix split three ways, each exactly as long as the clip

It also has **Projects**: group clips and write paper edits, VO scripts and promos next to them, with an assistant that has read the transcripts. Every soundbite in a script is checked against what was actually said. For pickup lines, you can clone a speaker's voice with ElevenLabs, with their permission recorded, and every generated line is clearly marked.

Every feature has a free option. Transcription always runs on your Mac with Whisper. Name fixes, summaries and translation use either **Claude** (best quality, about a cent per transcript) or a **free local model** through Ollama. Audio cleanup runs **on your Mac for free**, or through **ElevenLabs**, **Auphonic** or **ai-coustics** if you have an account.

## Install (once)

1. Unzip the download. You'll get a **Transcribe** folder.
2. Open **Terminal** (Cmd+Space, type "Terminal").
3. Type `bash ` (with a space), drag **install.command** from the Transcribe folder into the window, and press Return. Press Return again to start.

The first install downloads about 3 GB and takes 5–15 minutes. Updates take about a minute. When it's done, **Transcribe** opens. It lives in your Applications folder, with a shortcut on your Desktop.

## Using it

- **Drop clips or folders** onto the window, or use **Add clips…** / **Add folder…**.
- **Create files** opens with each batch. Tick what you want (captions, Word or text transcript, summary, clean audio, stems, translations) and where to save it:
  - next to each clip
  - in a **Transcripts** folder next to each clip
  - in **one folder** you choose

  Tick **Save these as my choices** to make them your defaults. Tick **Don't ask again** to skip the window; you can turn it back on in Settings.
- **Who's talking?** When a clip with several speakers finishes, a window shows a few of each speaker's lines as subtitles. Press ▶ to hear a line, then type a name. Giving two labels the same name merges them.
- **Summary** opens the summary and soundbites. Press ▶ to hear a bite, or copy them all.
- **More → Translate…** or **More → Write summary** adds those to a clip you've already done. They're saved in the same place as the clip's other files.
- **Recent** lists everything you've transcribed, so you can open, rename, translate or summarize later.
- **Watch folder** (top right): anything put in `Transcribe/Watch Folder` in your home folder is processed automatically with your saved choices, even with the app closed.

## Stems

Tick **Dialogue, music & effects stems** in Create files. You get three 48 kHz, 24-bit stereo WAVs next to the clip. Line them up at the clip's start and they stay in sync, because each one is exactly as long as the clip's audio. When you also make captions or a transcript, Transcribe reads the dialogue stem, so music under an interview doesn't confuse it.

Stems run on your Mac for free and need an Apple Silicon Mac. They take a while: plan on several minutes for a 10-minute clip. The Queue shows an estimate. The separation model is BandIt Plus (Watcharasupat et al.), downloaded once by the installer.

## Projects and scripts

1. Open the **Projects** tab and make a project, like *Chef Doc* or *Spring Promo*.
2. Under **Clips**, choose **+ Add**. Pick clips you've already transcribed (**From Recent**) or new ones (**Transcribe new clips…**).
3. Under **Scripts**, choose **+ New** and pick a kind: **Paper edit**, **VO / narration**, **Promo / social cut** or **Free writing**.
4. Write in the middle, or ask the **Assistant** on the right: "Build a 3-minute string-out", "Find every bite about the menu", "Cut a 30-second promo with a strong hook". It writes straight into the script. **Undo** puts it back.
5. Select part of the script before you ask, and the assistant rewrites just that part.
6. **Notes for the assistant** is read with every request. Put the audience, tone, length and names that must be spelled right there.

Soundbites are written like this, one per line:

```
A001 [01:02:03:04 - 01:02:09:12] AMY: "Exact words from the transcript."
```

**Check bites** compares every soundbite with the transcript. ✓ means the words are exact. ! means the words are right but the timecodes are off. ≈ means the words are close but not exact. ✕ means the words aren't in the transcript. **Fix** swaps in the transcript's exact words and timecodes. Press ▶ to hear a bite, or click a row to jump to it. Checks also run after every assistant edit.

The status line shows soundbite runtime and VO read time (about 150 words a minute). **Export** saves the script as a Word document. Scripts save as you type, in `Transcribe/projects` in your home folder.

The assistant uses whichever AI you chose in Settings. With Claude, it reads about 30,000 words of transcript per project, roughly 3 to 4 hours of interviews. On Sonnet, the first request on a full project costs up to about 15¢. Follow-ups within a few minutes cost much less, because Claude keeps the transcripts cached. Each reply shows its cost. With the free local model, it reads about 4,000 words (around 25 minutes of interview), so it suits short projects. When a project is bigger than the assistant can read, it tells you how much it saw.

## Cloned voices for pickup lines

For the line that isn't there: a word that got stepped on, a sentence cut off, a line the person approved. Cloned voices use ElevenLabs and need a plan that includes voice cloning (Starter and up).

1. Name the speakers in your clips (**Name speakers**). Only named people can be cloned.
2. In a project, under **Voices**, choose **Clone…** next to the person. Transcribe picks 30 seconds to 2½ minutes of their cleanest solo speech from the project's clips. It uses dialogue stems or clean audio when you've made them.
3. Tick that they've agreed, and type who gave permission and how (for example, *appearance release signed Nov 3*). Cloning won't start without both.
4. Choose **+ Line**, or type a line in the script:

   ```
   SYNTH AMY: "and that's when we knew."
   ```

5. **Check bites** lists it as **SYNTHETIC**. Choose **Make it** to generate it from the text, or **From my read…** to pick a recording of you saying the line. That one keeps your timing and emphasis and usually sounds more natural.

The WAV (48 kHz, 24-bit) is saved in `Transcribe/Generated audio/<project>`, and its name always contains `SYNTHETIC`. You'll still need to match room tone and mic color in your mix.

**What keeps it honest:**
- Generated lines are never counted as soundbites.
- Word exports mark them in red as *SYNTHETIC VOICE, NOT RECORDED*.
- **Voices → ••• → Generated-voice log** lists every voice cloned and every line made, with who gave permission, ready to send to a producer or network.
- The assistant writes a SYNTH line only when you ask for one, and it says so when it does.
- Removing a voice (×) deletes it from your ElevenLabs account too. Files you've made stay.

Many networks and clients require disclosure or sign-off for synthetic dialogue. Check their standards before you deliver.

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
| ElevenLabs voice cloning | pickup lines in a cloned voice, with permission | your ElevenLabs credits (Starter plan or higher) |
| Claude, in Projects | the script assistant | up to about 15¢ a request on a big project, less on follow-ups |

Auphonic and ai-coustics also level the audio to broadcast loudness (-23 LUFS). Adobe Podcast's Enhance Speech isn't included because Adobe doesn't offer a public API for it.

**Settings → Spending** shows estimated Claude costs and how many minutes of audio went to each service.

## If something goes wrong

The **Activity** tab shows what happened. The full log is in `Transcribe/transcribe.log` in your home folder.

## Uninstall

Turn off the watch folder, then delete `Transcribe` from your home folder and from Applications.

## Credits

Dialogue, music and effects separation uses the BandIt Plus model by Karn Watcharasupat and colleagues, through Roman Solovyev's Music-Source-Separation-Training code (MIT License, see `dnr/LICENSE-MSST.txt`). The model weights are downloaded from that project's release page.

## License

© 2026 TBD Media Post. Free for personal and other noncommercial use under the [PolyForm Noncommercial License 1.0.0](LICENSE.md).

Using Transcribe on paid or commercial work, or building it into a product, needs permission. Get in touch through GitHub.
