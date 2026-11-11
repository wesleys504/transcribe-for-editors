#!/usr/bin/env python3
# Copyright (c) 2026 TBD Media Post. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE.md).
"""
Transcribe — local transcription for editors.

For each media file: transcribes on this Mac with Whisper, labels speakers,
optionally has Claude fix names/terms, and writes next to the file:
  <name>.srt                 captions for Premiere
  <name> - transcript.txt    timecoded transcript
  <name> - transcript.docx   the same, as a Word doc for producers

Usage:
  transcribe.py FILE_OR_FOLDER [...]     transcribe these
  transcribe.py --watch                  process new files in the watch folder (used by the watch-folder agent)
"""
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path(os.environ.get("TRANSCRIBE_HOME", Path.home() / "Transcribe"))
MODELS = HOME / "models"
SETTINGS_FILE = HOME / "settings.json"
GLOBAL_NAMES = HOME / "names.txt"
WATCH_DIR = HOME / "Watch Folder"
LOG_FILE = HOME / "transcribe.log"

MEDIA_EXTS = {
    ".mov", ".mp4", ".m4v", ".mxf", ".avi", ".mts", ".m2ts", ".mkv", ".webm", ".mpg", ".mpeg", ".3gp",
    ".wav", ".aif", ".aiff", ".mp3", ".m4a", ".aac", ".flac", ".bwf", ".caf",
}
SR = 16000

DEFAULTS = {
    "model": "large-v3-turbo",      # Whisper model
    "language": "auto",             # or "en", "es", ...
    "speakers": "auto",             # or a number, e.g. 2
    "speaker_labels": True,
    "claude_cleanup": True,         # only runs if an API key is saved
    "claude_model": "claude-sonnet-5-5",
    "ask_speaker_names": True,      # ask in Terminal who each speaker is
    "translate_to": "",             # e.g. "English" — also writes translated captions + transcript
    "summary": True,                # (older setting, folded into "outputs")
    # --- what to make, and where (the app asks per batch unless ask_outputs is off)
    "outputs": ["srt", "docx", "summary"],   # any of: srt, docx, txt, summary, clean
    "dest": "beside",               # beside | subfolder | folder
    "dest_dir": "",                 # used when dest == "folder"
    "ask_outputs": True,
    # --- audio cleanup
    "audio_cleanup": "local",       # local | elevenlabs | auphonic | aicoustics
    "transcribe_cleaned": True,     # transcribe from the cleaned audio when cleaning
    # --- which AI does name fixes, summaries and translation
    "ai_provider": "claude",        # claude | local
    "local_model": "qwen2.5:7b",
    "ollama_url": "http://127.0.0.1:11434",
    "max_chars_per_line": 42,
    "max_lines": 2,
    "max_caption_seconds": 6.0,
}

SEG_MODEL = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS / "nemo_en_titanet_small.onnx"
KEYCHAIN_SERVICE = "Transcribe-Claude-API-Key"


# ---------------------------------------------------------------- small helpers
def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d ") + line + "\n")
    except OSError:
        pass


PROGRESS_HOOK = None  # the app sets this to show progress bars


def progress(stage, frac):
    if PROGRESS_HOOK:
        try:
            PROGRESS_HOOK(stage, frac)
        except Exception:
            pass


def load_settings():
    s = dict(DEFAULTS)
    try:
        s.update(json.loads(SETTINGS_FILE.read_text()))
    except (OSError, ValueError):
        pass
    return s


def notify(title, text):
    if platform.system() != "Darwin":
        return
    t = text.replace('"', "'")
    subprocess.run(["osascript", "-e", f'display notification "{t}" with title "{title}"'], capture_output=True)


SECRET_SERVICES = {
    "claude": ("Transcribe-Claude-API-Key", "ANTHROPIC_API_KEY"),
    "elevenlabs": ("Transcribe-ElevenLabs-API-Key", "ELEVENLABS_API_KEY"),
    "auphonic": ("Transcribe-Auphonic-API-Key", "AUPHONIC_API_KEY"),
    "aicoustics": ("Transcribe-aicoustics-API-Key", "AICOUSTICS_API_KEY"),
}


def get_secret(name):
    service, env = SECRET_SERVICES[name]
    if os.environ.get(env):
        return os.environ[env].strip()
    if platform.system() != "Darwin":
        return ""
    r = subprocess.run(["security", "find-generic-password", "-s", service, "-w"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def set_secret(name, value):
    service, env = SECRET_SERVICES[name]
    if platform.system() != "Darwin":
        os.environ[env] = value
        return
    r = subprocess.run(["security", "add-generic-password", "-U", "-s", service, "-a", "Transcribe", "-w", value],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "Keychain refused the key")


def delete_secret(name):
    service, env = SECRET_SERVICES[name]
    os.environ.pop(env, None)
    if platform.system() == "Darwin":
        subprocess.run(["security", "delete-generic-password", "-s", service], capture_output=True)


def get_api_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"].strip()
    if platform.system() != "Darwin":
        return ""
    r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def read_names(media_dir):
    """Names & terms: ~/Transcribe/names.txt plus a names.txt next to the clips."""
    names = []
    for p in (GLOBAL_NAMES, Path(media_dir) / "names.txt"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and line not in names:
                    names.append(line)
        except OSError:
            pass
    return names


# ---------------------------------------------------------------- audio + timecode
def load_audio(path):
    """Decode any audio/video file to 16 kHz mono float32 using PyAV (no ffmpeg install needed)."""
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise NoAudio("no audio track")
        stream = c.streams.audio[0]
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for frame in c.decode(stream):
            for out in res.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in res.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise NoAudio("audio track is empty")
    return np.concatenate(chunks).astype(np.float32)


def load_audio_span(path, start, end):
    """Decode just [start, end] seconds (for play buttons) without reading the whole file."""
    import av
    import numpy as np
    start = max(0.0, float(start)); end = max(start + 0.1, float(end))
    chunks = []
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise NoAudio("no audio track")
        stream = c.streams.audio[0]
        try:
            c.seek(int(max(0, start - 1) * 1_000_000))  # microseconds, container time base
        except Exception:
            pass
        res = av.AudioResampler(format="s16", layout="mono", rate=SR)
        t_first = None
        for frame in c.decode(stream):
            ft = float(frame.time) if frame.time is not None else 0.0
            if ft > end:
                break
            for out in res.resample(frame):
                if t_first is None:
                    t_first = ft
                chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    a = np.concatenate(chunks)
    i0 = max(0, int((start - (t_first or 0)) * SR))
    return a[i0:i0 + int((end - start) * SR)]


def read_start_timecode(path):
    """Source timecode (e.g. 14:02:10:05) and fps if the file carries them, else (None, None)."""
    try:
        import av
        with av.open(str(path)) as c:
            fps = None
            for s in c.streams.video:
                if s.average_rate:
                    fps = float(s.average_rate)
                    break
            tc = c.metadata.get("timecode")
            for s in c.streams:
                tc = tc or s.metadata.get("timecode")
            if tc and fps and re.match(r"^\d{2}:\d{2}:\d{2}[:;.]\d{2}$", tc):
                return tc, fps
    except Exception:
        pass
    return None, None


def tc_to_seconds(tc, fps):
    h, m, s, f = [int(x) for x in re.split(r"[:;.]", tc)]
    return h * 3600 + m * 60 + s + f / round(fps)


def fmt_tc(sec, fps=None):
    """HH:MM:SS:FF when fps is known, else HH:MM:SS."""
    sec = max(0.0, sec)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    if fps:
        fr = int(round((sec - int(sec)) * round(fps)))
        fr = min(fr, round(fps) - 1)
        return f"{h:02d}:{m:02d}:{s:02d}:{fr:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt(sec):
    sec = max(0.0, sec)
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------- transcription engines
# Every engine returns a list of words: {"start", "end", "text", "prob"}; text includes punctuation.

def glossary_prompt(names):
    if not names:
        return None
    return "Names and terms: " + ", ".join(names[:60]) + "."


def engine_name():
    forced = os.environ.get("TRANSCRIBE_ENGINE")
    if forced:
        return forced
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "faster-whisper"


def transcribe_words(audio, settings, names, turns=None):
    eng = engine_name()
    lang = None if settings["language"] == "auto" else settings["language"]
    prompt = glossary_prompt(names)

    if eng == "mlx":
        import mlx_whisper
        repo = {
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "small": "mlx-community/whisper-small-mlx",
        }.get(settings["model"], settings["model"])
        r = mlx_whisper.transcribe(
            audio, path_or_hf_repo=repo, word_timestamps=True, language=lang,
            initial_prompt=prompt, condition_on_previous_text=False,
            hallucination_silence_threshold=2.0, verbose=None,
        )
        return [
            {"start": float(w["start"]), "end": float(w["end"]), "text": w["word"].strip(), "prob": float(w.get("probability", 1))}
            for seg in r.get("segments", []) for w in seg.get("words", []) if w["word"].strip()
        ], r.get("language")

    if eng == "faster-whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel(settings["model"], device="auto", compute_type="int8")
        segments, info = model.transcribe(
            audio, word_timestamps=True, language=lang, initial_prompt=prompt,
            vad_filter=True, condition_on_previous_text=False, hotwords=prompt,
        )
        words, total = [], max(len(audio) / SR, 1)
        for seg in segments:
            progress("Transcribing", 0.25 + 0.5 * min(seg.end / total, 1))
            for w in seg.words or []:
                if w.word.strip():
                    words.append({"start": float(w.start), "end": float(w.end), "text": w.word.strip(), "prob": float(w.probability)})
        return words, info.language

    if eng == "sherpa":  # small CPU engine, used for testing
        import sherpa_onnx
        d = Path(os.environ["TRANSCRIBE_SHERPA_WHISPER"])
        enc = next(d.glob("*encoder.int8.onnx")); dec = next(d.glob("*decoder.int8.onnx")); tok = next(d.glob("*tokens.txt"))
        rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=str(enc), decoder=str(dec), tokens=str(tok), language=lang or "en", num_threads=4)
        spans = [(t["start"], t["end"]) for t in (turns or [])] or [(0.0, len(audio) / SR)]
        words = []
        for a, b in spans:
            st = rec.create_stream()
            st.accept_waveform(SR, audio[int(a * SR):int(b * SR)])
            rec.decode_stream(st)
            toks = st.result.text.split()
            total = sum(len(t) for t in toks) or 1
            cur = a
            for t in toks:
                dur = (b - a) * len(t) / total
                words.append({"start": cur, "end": cur + dur, "text": t, "prob": 1.0})
                cur += dur
        return words, lang or "en"

    raise ValueError(f"unknown engine {eng}")


# ---------------------------------------------------------------- speaker labels
def diarize(audio, settings):
    import sherpa_onnx
    if not (SEG_MODEL.exists() and EMB_MODEL.exists()):
        log("Speaker models missing — run the installer again. Continuing without speaker labels.")
        return []
    n = settings.get("speakers", "auto")
    n = int(n) if str(n).isdigit() and int(n) > 0 else -1
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL)),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=n, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    res = sd.process(audio).sort_by_start_time()
    # Renumber speakers in order of first appearance: Speaker 1, Speaker 2, ...
    order, turns = {}, []
    for r in res:
        if r.speaker not in order:
            order[r.speaker] = len(order) + 1
        turns.append({"start": float(r.start), "end": float(r.end), "speaker": order[r.speaker]})
    return turns


def assign_speakers(words, turns):
    if not turns:
        for w in words:
            w["speaker"] = 0
        return words
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(w["end"], t["end"]) - max(w["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        if best is None:  # word in a gap: nearest turn
            mid = (w["start"] + w["end"]) / 2
            best = min(turns, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))["speaker"]
        w["speaker"] = best
    # Smooth single stray words that flip speaker mid-sentence.
    for i in range(1, len(words) - 1):
        if words[i - 1]["speaker"] == words[i + 1]["speaker"] != words[i]["speaker"]:
            words[i]["speaker"] = words[i - 1]["speaker"]
    return words


# ---------------------------------------------------------------- paragraphs
def ends_sentence(text):
    return bool(re.search(r"[.!?…][\"')\]]*$", text))


def build_paragraphs(words, max_gap=1.5):
    """Group words into speaker paragraphs: break on speaker change or a long pause."""
    paras = []
    for i, w in enumerate(words):
        new = not paras or w["speaker"] != paras[-1]["speaker"] or (w["start"] - words[i - 1]["end"] > max_gap and ends_sentence(words[i - 1]["text"]))
        if new:
            paras.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "w": []})
        p = paras[-1]
        p["w"].append(i)
        p["end"] = w["end"]
    return paras


def para_text(words, p):
    return " ".join(words[i]["text"] for i in p["w"] if words[i]["text"])


# ---------------------------------------------------------------- Claude cleanup
FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "integer"},
                    "original": {"type": "string"},
                    "replacement": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["item", "original", "replacement", "reason"],
                "additionalProperties": False,
            },
        },
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "name": {"type": "string"}},
                "required": ["label", "name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["fixes", "speakers"],
    "additionalProperties": False,
}


def claude_system(names):
    g = ("PROJECT NAMES & TERMS (authoritative spellings — fix any variant or mishearing of these):\n" + "\n".join(names)) if names else \
        "No names list was given; use context to spot names and keep each spelled consistently."
    return "\n\n".join([
        "You proofread automatic speech-to-text transcripts for a TV/documentary editor.",
        "Each line is `number | Speaker N | words`. Fix only TRANSCRIPTION errors: misheard words (use the topic and nearby lines), "
        "names and terms, capitalization of proper nouns, and clearly wrong punctuation.",
        "Do NOT fix the speaker's grammar, slang, filler or sentence structure — the transcript must match what was said. "
        "Only suggest a fix when you are confident. Leave correct lines out.",
        "`original` must be copied exactly from that line, and kept short (just the words that change).",
        "`speakers`: only if dialogue makes it clear who a label is (they introduce themselves or are addressed by name), give their name. Otherwise empty.",
        g,
    ])


def _ssl_context():
    """Python from uv doesn't see the Mac's certificate store; use the bundled one."""
    import ssl
    try:
        import truststore  # uses the macOS Keychain's trusted certificates
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()



# ---------------------------------------------------------------- small HTTP helpers
def _multipart(fields, files):
    """fields: {name: str}; files: {name: (filename, bytes, content_type)} -> (body, content_type)"""
    import uuid as _uuid
    boundary = "----transcribe" + _uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fname, data, ctype) in files.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def http(method, url, headers=None, data=None, timeout=600, tls=True):
    """Returns (status, body bytes, content-type). Raises RuntimeError with the service's message on HTTP errors."""
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    ctx = _ssl_context() if tls and url.startswith("https") else None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        raise ServiceError(e.code, body)


class ServiceError(RuntimeError):
    def __init__(self, code, body):
        super().__init__(f"HTTP {code}: {body}")
        self.code, self.body = code, body


# ---------------------------------------------------------------- free local AI (Ollama)
LOCAL_MODELS = {
    "qwen2.5:7b": "Qwen 2.5 7B (better; needs 16 GB of memory, 4.7 GB download)",
    "qwen2.5:3b": "Qwen 2.5 3B (faster; fine with 8 GB of memory, 1.9 GB download)",
}


def ollama_url(settings):
    return (settings.get("ollama_url") or "http://127.0.0.1:11434").rstrip("/")


def ollama_status(settings):
    try:
        _, body, _ = http("GET", ollama_url(settings) + "/api/tags", timeout=4)
        names = [m.get("name", "") for m in json.loads(body).get("models", [])]
        return {"running": True, "models": names}
    except Exception:
        return {"running": False, "models": []}


def ollama_has_model(settings):
    st = ollama_status(settings)
    want = settings.get("local_model") or "qwen2.5:7b"
    return st["running"] and any(n == want or n.split(":")[0] == want for n in st["models"])


def ollama_pull(settings, model, on_progress=None):
    """Download a model through Ollama, reporting 0..1 progress."""
    req = urllib.request.Request(ollama_url(settings) + "/api/pull", method="POST",
                                 data=json.dumps({"model": model, "stream": True}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("error"):
                raise RuntimeError(ev["error"])
            if on_progress and ev.get("total"):
                on_progress(ev.get("status", ""), (ev.get("completed") or 0) / ev["total"])
            if ev.get("status") == "success":
                return True
    return True


def ollama_json(settings, system, text, schema, lead, ctx=16384):
    body = {
        "model": settings.get("local_model") or "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": lead + "\n\n" + text + "\n\nReply only with JSON that matches this schema:\n" + json.dumps(schema)},
        ],
        "format": schema, "stream": False,
        "options": {"temperature": 0, "num_ctx": ctx},
    }
    try:
        _, raw, _ = http("POST", ollama_url(settings) + "/api/chat", {"content-type": "application/json"},
                         json.dumps(body).encode(), timeout=3600)
    except ServiceError as e:
        if e.code == 404:
            raise RuntimeError(f"The local model {body['model']} isn't downloaded yet. Download it in Settings.")
        raise
    except urllib.error.URLError:
        raise RuntimeError("Ollama isn't running. Open the Ollama app, then try again.")
    content = json.loads(raw).get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except ValueError:
        m = re.search(r"\{.*\}", content, re.S)  # small models sometimes wrap the JSON
        if m:
            return json.loads(m.group(0))
        raise RuntimeError("The local model returned something that wasn't valid JSON.")


def ai_provider(settings):
    return "local" if settings.get("ai_provider") == "local" else "claude"


def ai_ready(settings):
    return ollama_has_model(settings) if ai_provider(settings) == "local" else bool(get_api_key())


def ai_not_ready_reason(settings):
    if ai_provider(settings) == "local":
        st = ollama_status(settings)
        if not st["running"]:
            return "Ollama isn't running (or isn't installed)"
        return f"the local model {settings.get('local_model')} isn't downloaded"
    return "no Claude API key is saved"


def ai_chunk(settings, claude_chars):
    return claude_chars if ai_provider(settings) == "claude" else min(claude_chars, 5000)


def ai_json(settings, system, text, schema, lead):
    """Ask the chosen AI (Claude or the free local model) for JSON matching `schema`."""
    if ai_provider(settings) == "local":
        log(f"  local model {settings.get('local_model')} (free, on this Mac)…")
        return ollama_json(settings, system, text, schema, lead)
    key = get_api_key()
    if not key:
        raise RuntimeError("no Claude API key is saved")
    out, _ = claude_call(key, settings["claude_model"], system, text, schema=schema, lead=lead)
    return out


# ---------------------------------------------------------------- audio cleanup
CLEAN_SR = 48000
CLEANUP_SERVICES = {
    "local": "On this Mac (free)",
    "elevenlabs": "ElevenLabs Voice Isolator",
    "auphonic": "Auphonic",
    "aicoustics": "ai-coustics",
}
SERVICE_URLS = {
    "elevenlabs": os.environ.get("TRANSCRIBE_ELEVENLABS_URL", "https://api.elevenlabs.io"),
    "auphonic": os.environ.get("TRANSCRIBE_AUPHONIC_URL", "https://auphonic.com"),
    "aicoustics": os.environ.get("TRANSCRIBE_AICOUSTICS_URL", "https://api.ai-coustics.io"),
}
CHUNK_MINUTES = {"elevenlabs": 30, "auphonic": 60, "aicoustics": 60}


def local_cleanup_available():
    try:
        import deepfilternet_rs  # noqa: F401
        return True
    except Exception:
        return False


def load_audio_at(path, rate):
    import av
    import numpy as np
    chunks = []
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise NoAudio("no audio track")
        res = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in c.decode(c.streams.audio[0]):
            for out in res.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in res.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise NoAudio("audio track is empty")
    return np.concatenate(chunks).astype(np.float32)


def _decode_bytes(data, rate=CLEAN_SR):
    import av
    import numpy as np
    chunks = []
    with av.open(io.BytesIO(data)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in c.decode(c.streams.audio[0]):
            for out in res.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in res.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def _encode_flac(samples, rate=CLEAN_SR):
    import av
    import numpy as np
    buf = io.BytesIO()
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with av.open(buf, "w", format="flac") as c:
        st = c.add_stream("flac", rate=rate, layout="mono")
        step = rate  # one second per frame
        for i in range(0, len(pcm), step):
            fr = av.AudioFrame.from_ndarray(pcm[i:i + step].reshape(1, -1), format="s16", layout="mono")
            fr.sample_rate = rate
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode(None):
            c.mux(pkt)
    return buf.getvalue()


def _fit(x, n):
    import numpy as np
    return x[:n] if len(x) >= n else np.concatenate([x, np.zeros(n - len(x), np.float32)])


def write_wav(path, samples, rate=CLEAN_SR):
    import numpy as np
    import wave as _wave
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _clean_local(x):
    import numpy as np
    import deepfilternet_rs
    p = deepfilternet_rs.DeepFilterNetRealtime(model_path=None)
    fl = p.frame_length
    n = len(x)
    padded = np.concatenate([x, np.zeros((-n) % fl, np.float32)])
    outs, step = [], fl * 100
    for i in range(0, len(padded), step):
        outs.append(np.asarray(p.process_chunk(padded[i:i + step]), np.float32))
        if i % (step * 60) == 0:
            progress("Cleaning audio on this Mac", 0.02 + 0.2 * i / max(len(padded), 1))
    outs.append(np.asarray(p.finalize(), np.float32))
    try:
        p.close()
    except Exception:
        pass
    return _fit(np.concatenate(outs), n)


def _clean_elevenlabs(flac, key):
    body, ctype = _multipart({}, {"audio": ("audio.flac", flac, "audio/flac")})
    try:
        _, out, _ = http("POST", SERVICE_URLS["elevenlabs"] + "/v1/audio-isolation",
                         {"xi-api-key": key, "Content-Type": ctype}, body, timeout=3600)
    except ServiceError as e:
        if e.code == 401:
            raise RuntimeError("ElevenLabs rejected the API key. Check it in Settings.")
        if "quota" in e.body.lower() or e.code == 402:
            raise RuntimeError("Your ElevenLabs account is out of credits.")
        raise RuntimeError(f"ElevenLabs error {e.code}: {e.body[:200]}")
    return out


def _clean_auphonic(flac, key, title):
    base = SERVICE_URLS["auphonic"]
    auth = {"Authorization": f"bearer {key}"}
    spec = {
        "metadata": {"title": title}, "output_basename": "transcribe-clean",
        "output_files": [{"format": "wav"}],
        "algorithms": {"filtering": True, "leveler": True, "normloudness": True, "loudnesstarget": -23,
                       "denoise": True, "denoisemethod": "dynamic", "denoiseamount": 0},
    }
    try:
        _, raw, _ = http("POST", base + "/api/productions.json", dict(auth, **{"Content-Type": "application/json"}), json.dumps(spec).encode())
        uuid = json.loads(raw)["data"]["uuid"]
        body, ctype = _multipart({}, {"input_file": ("audio.flac", flac, "audio/flac")})
        http("POST", f"{base}/api/production/{uuid}/upload.json", dict(auth, **{"Content-Type": ctype}), body, timeout=3600)
        http("POST", f"{base}/api/production/{uuid}/start.json", auth, b"")
    except ServiceError as e:
        if e.code in (401, 403):
            raise RuntimeError("Auphonic rejected the API key. Check it in Settings.")
        raise RuntimeError(f"Auphonic error {e.code}: {e.body[:200]}")
    t0 = time.time()
    while time.time() - t0 < 3 * 3600:
        _, raw, _ = http("GET", f"{base}/api/production/{uuid}.json", auth)
        d = json.loads(raw)["data"]
        status, label = d.get("status"), str(d.get("status_string", ""))
        if status == 3 or label.lower() == "done":
            url = d["output_files"][0]["download_url"]
            sep = "&" if "?" in url else "?"
            _, out, _ = http("GET", f"{url}{sep}bearer_token={key}", timeout=3600)
            return out
        if status == 2 or "error" in label.lower() or d.get("error_message"):
            raise RuntimeError(f"Auphonic couldn't process it: {d.get('error_message') or label}")
        time.sleep(5)
    raise RuntimeError("Auphonic took too long; check your Auphonic dashboard.")


def _clean_aicoustics(flac, key):
    base = SERVICE_URLS["aicoustics"]
    auth = {"X-API-Key": key}
    settings_json = json.dumps({"loudness_target": -23, "true_peak": -1, "enhancement_level": 100,
                                "enhancement_model": "LARK_V2", "transcode": "WAV"})
    body, ctype = _multipart({"media_enhancement": settings_json}, {"file": ("audio.flac", flac, "audio/flac")})
    try:
        _, raw, _ = http("POST", base + "/v2/medias", dict(auth, **{"Content-Type": ctype}), body, timeout=3600)
    except ServiceError as e:
        if e.code in (401, 403):
            raise RuntimeError("ai-coustics rejected the API key. Check it in Settings.")
        raise RuntimeError(f"ai-coustics error {e.code}: {e.body[:200]}")
    uid = json.loads(raw)["uid"]
    t0 = time.time()
    while time.time() - t0 < 3 * 3600:
        _, raw, _ = http("GET", f"{base}/v2/medias/{uid}/metadata", auth)
        status = str(json.loads(raw).get("enhancement_status", "")).upper()
        if status == "COMPLETED":
            _, out, _ = http("GET", f"{base}/v2/medias/{uid}/file", auth, timeout=3600)
            return out
        if status in ("FAILED", "ERROR", "CANCELLED"):
            raise RuntimeError("ai-coustics couldn't process it.")
        time.sleep(4)
    raise RuntimeError("ai-coustics took too long; check your ai-coustics dashboard.")



# ---------------------------------------------------------------- dialogue / music / effects stems
STEMS_DIR = MODELS / "stems"
STEM_NAMES = {"speech": "dialogue", "music": "music", "effects": "effects"}
STEMS_SR = 44100
STEMS_OUT_SR = 48000
STEMS_URL = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/"
STEMS_FILES = {"cfg.yaml": "config_dnr_bandit_bsrnn_multi_mus64.yaml", "bandit.chpt": "model_bandit_plus_dnr_sdr_11.47.chpt"}
_STEMS_MODEL = {}


def stems_available():
    if not all((STEMS_DIR / f).exists() for f in STEMS_FILES):
        return False
    import importlib.util
    return importlib.util.find_spec("torch") is not None   # don't import it: that takes seconds


def load_audio_stereo(path, rate):
    import av
    import numpy as np
    chunks = []
    with av.open(str(path)) as c:
        if not c.streams.audio:
            raise NoAudio("no audio track")
        res = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        for frame in c.decode(c.streams.audio[0]):
            for out in res.resample(frame):
                chunks.append(out.to_ndarray())
        for out in res.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        raise NoAudio("audio track is empty")
    return np.concatenate(chunks, axis=1).astype(np.float32)


def _resample(x, src, dst):
    """Resample (channels, n) float audio with PyAV."""
    import av
    import numpy as np
    if src == dst:
        return x
    res = av.AudioResampler(format="fltp", layout="stereo", rate=dst)
    out = []
    step = src * 10
    for i in range(0, x.shape[1], step):
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(x[:, i:i + step]), format="fltp", layout="stereo")
        fr.sample_rate = src
        out += [o.to_ndarray() for o in res.resample(fr)]
    out += [o.to_ndarray() for o in res.resample(None)]
    return np.concatenate(out, axis=1)


def write_wav24(path, x, rate):
    """(channels, n) float audio → 24-bit WAV."""
    import numpy as np
    import wave as _wave
    v = np.ascontiguousarray((np.clip(x.T, -1, 1) * 8388607).astype("<i4"))
    b = v.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[0]); w.setsampwidth(3); w.setframerate(rate)
        w.writeframes(b)


def _stems_model():
    if "m" in _STEMS_MODEL:
        return _STEMS_MODEL["m"], _STEMS_MODEL["dev"], _STEMS_MODEL["cfg"]
    import torch
    import yaml
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from dnr import MultiMaskMultiSourceBandSplitRNNSimple
    cfg = yaml.safe_load((STEMS_DIR / "cfg.yaml").read_text())
    model = MultiMaskMultiSourceBandSplitRNNSimple(**cfg["model"])
    sd = torch.load(STEMS_DIR / "bandit.chpt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("state_dict", sd) if isinstance(sd, dict) else sd)
    model.eval()
    dev = "cpu"
    if os.environ.get("TRANSCRIBE_STEMS_DEVICE"):
        dev = os.environ["TRANSCRIBE_STEMS_DEVICE"]
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    model.to(dev)
    _STEMS_MODEL.update(m=model, dev=dev, cfg=cfg)
    return model, dev, cfg


def _run_chunks(model, dev, x, chunk, step, on_progress):
    import numpy as np
    import torch
    n = x.shape[1]
    fade = chunk // 10
    win = np.ones(chunk, np.float32)
    win[:fade] = np.linspace(0, 1, fade)
    win[-fade:] = np.linspace(1, 0, fade)
    pad = chunk - step
    xp = np.pad(x, ((0, 0), (pad, pad + chunk)))
    out = np.zeros((3, 2, xp.shape[1]), np.float32)
    wsum = np.zeros(xp.shape[1], np.float32)
    starts = list(range(0, xp.shape[1] - chunk + 1, step))
    with torch.inference_mode():
        for k, s in enumerate(starts):
            y = model(torch.from_numpy(np.ascontiguousarray(xp[None, :, s:s + chunk])).to(dev))[0].float().cpu().numpy()
            out[:, :, s:s + chunk] += y * win
            wsum[s:s + chunk] += win
            on_progress((k + 1) / len(starts))
    return (out / np.maximum(wsum, 1e-8))[:, :, pad:pad + n]


def separate_stems(media, p0=0.02, p1=0.3):
    """Split a clip's audio into dialogue, music and effects. Returns {name: (2, n) array at 48 kHz},
    each exactly as long as the clip's audio, so the stems line up with the picture."""
    import numpy as np
    if not stems_available():
        raise RuntimeError("stems aren't installed yet (run the installer again on an Apple Silicon Mac)")
    progress("Reading audio for stems", p0)
    x = load_audio_stereo(media, STEMS_SR)
    secs = x.shape[1] / STEMS_SR
    n_out = int(round(secs * STEMS_OUT_SR))
    model, dev, cfg = _stems_model()
    chunk = int(cfg["audio"]["chunk_size"])
    step = chunk // 2              # 2x overlap: same quality as 4x in testing, twice as fast
    log(f"  splitting dialogue, music and effects ({secs / 60:.1f} min, {'GPU' if dev == 'mps' else 'CPU'})…")
    t0 = time.time()

    def on_progress(f):
        el = time.time() - t0
        left = el / f * (1 - f) if f > 0.02 else None
        msg = "Splitting dialogue, music & effects" + (f" (about {max(1, int(left // 60))} min left)" if left and left > 90 else "")
        progress(msg, p0 + (p1 - p0) * f)

    try:
        y = _run_chunks(model, dev, x, chunk, step, on_progress)
    except Exception as e:
        if dev == "cpu":
            raise
        log(f"  GPU stems failed ({e}); trying the CPU")
        model.to("cpu"); _STEMS_MODEL["dev"] = "cpu"
        y = _run_chunks(model, "cpu", x, chunk, step, on_progress)
    log(f"  stems took {int(time.time() - t0)}s")
    names = [STEM_NAMES.get(s, s) for s in cfg["training"]["instruments"]]
    res = {}
    for k, name in enumerate(names):
        z = _resample(y[k], STEMS_SR, STEMS_OUT_SR)
        res[name] = np.stack([_fit(z[0].astype(np.float32), n_out), _fit(z[1].astype(np.float32), n_out)])
    record_service_usage("stems", secs / 60)
    return res


def clean_audio(media, service):
    """Returns cleaned 48 kHz mono audio, exactly as long as the original (keeps sync with the video)."""
    import numpy as np
    if service not in CLEANUP_SERVICES:
        raise RuntimeError(f"unknown cleanup service {service}")
    progress("Reading audio for cleanup", 0.02)
    x = load_audio_at(media, CLEAN_SR)
    minutes = len(x) / CLEAN_SR / 60
    if service == "local":
        if not local_cleanup_available():
            raise RuntimeError("free on-Mac cleanup needs an Apple Silicon Mac")
        log(f"  cleaning audio on this Mac ({minutes:.1f} min)…")
        y = _clean_local(x)
    else:
        key = get_secret(service)
        if not key:
            raise RuntimeError(f"no {CLEANUP_SERVICES[service]} API key is saved (add it in Settings)")
        step = CHUNK_MINUTES[service] * 60 * CLEAN_SR
        parts = []
        nparts = max(1, -(-len(x) // step))
        for k, i in enumerate(range(0, len(x), step)):
            chunk = x[i:i + step]
            progress(f"Cleaning audio with {CLEANUP_SERVICES[service]}" + (f" (part {k + 1} of {nparts})" if nparts > 1 else ""),
                     0.02 + 0.2 * k / nparts)
            log(f"  sending audio to {CLEANUP_SERVICES[service]}" + (f", part {k + 1} of {nparts}" if nparts > 1 else "") + "…")
            flac = _encode_flac(chunk)
            if service == "elevenlabs":
                out = _clean_elevenlabs(flac, key)
            elif service == "auphonic":
                out = _clean_auphonic(flac, key, f"{Path(media).stem} part {k + 1}")
            else:
                out = _clean_aicoustics(flac, key)
            parts.append(_fit(_decode_bytes(out), len(chunk)))
        y = np.concatenate(parts)
    record_service_usage(service, minutes)
    return y


def record_service_usage(service, minutes):
    try:
        data = json.loads(USAGE_FILE.read_text())
    except (OSError, ValueError):
        data = {}
    month = time.strftime("%Y-%m")
    m = data.setdefault(month, {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0})
    sv = m.setdefault("services", {})
    sv[service] = round(sv.get(service, 0) + minutes, 2)
    try:
        USAGE_FILE.write_text(json.dumps(data, indent=2))
    except OSError:
        pass


# Prices per million tokens (USD), from platform.claude.com/docs/en/about-claude/pricing (Nov 2026).
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-opus-5-5": (4.00, 20.00),
}
USAGE_FILE = HOME / "usage.json"
CLIP_COST = {"usd": 0.0}  # running cost for the clip being worked on


def cost_of(model, usage):
    pin, pout = PRICES.get(model, (2.00, 10.00))
    tin = (usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0) * 1.25 + (usage.get("cache_read_input_tokens") or 0) * 0.1
    if model == "claude-haiku-5-5" and (usage.get("input_tokens") or 0) > 100_000:
        pin, pout = 0.50, 2.50
    return tin / 1e6 * pin + (usage.get("output_tokens") or 0) / 1e6 * pout


def record_usage(model, usage):
    usd = cost_of(model, usage)
    CLIP_COST["usd"] += usd
    try:
        data = json.loads(USAGE_FILE.read_text())
    except (OSError, ValueError):
        data = {}
    month = time.strftime("%Y-%m")
    m = data.setdefault(month, {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0})
    m["usd"] += usd
    m["input_tokens"] += usage.get("input_tokens") or 0
    m["output_tokens"] += usage.get("output_tokens") or 0
    m["requests"] += 1
    try:
        USAGE_FILE.write_text(json.dumps(data, indent=2))
    except OSError:
        pass
    return usd


def usage_summary():
    try:
        data = json.loads(USAGE_FILE.read_text())
    except (OSError, ValueError):
        data = {}
    month = time.strftime("%Y-%m")
    return {"month": month, "this_month": dict({"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "requests": 0, "services": {}}, **data.get(month, {})),
            "all_time_usd": sum(v.get("usd", 0) for v in data.values())}


def claude_call(api_key, model, system, text, schema=None, lead="Proofread these lines:"):
    body = json.dumps({
        "model": model, "max_tokens": 16000, "system": system,
        "messages": [{"role": "user", "content": lead + "\n\n" + text}],
        "output_config": {"format": {"type": "json_schema", "schema": schema or FIX_SCHEMA}},
    }).encode()
    for attempt in range(3):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST", headers={
            "content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01",
        })
        try:
            with urllib.request.urlopen(req, timeout=300, context=_ssl_context()) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            if e.code == 401:
                raise RuntimeError("Claude rejected the API key. Re-enter it in Transcribe's Settings.")
            raise RuntimeError(f"Claude error {e.code}: {msg}")
    usd = record_usage(model, data.get("usage", {}))
    log(f"  Claude: {data.get('usage', {}).get('input_tokens', 0):,} in / {data.get('usage', {}).get('output_tokens', 0):,} out tokens ≈ ${usd:.3f}")
    if data.get("stop_reason") == "max_tokens":
        raise RuntimeError("Claude's answer was cut off.")
    block = next((b for b in data.get("content", []) if b.get("type") == "text"), None)
    if not block:
        raise RuntimeError("Claude returned no result.")
    return json.loads(block["text"]), data.get("usage", {})


def norm(s):
    return re.sub(r"[^\w']+", "", s.lower().replace("’", "'"))


def apply_word_fix(words, idxs, original, replacement):
    """Replace the run of words matching `original` inside paragraph word indexes `idxs`. Returns new idxs or None."""
    target = [norm(t) for t in original.split() if norm(t)]
    if not target:
        return None
    for s in range(len(idxs) - len(target) + 1):
        if all(norm(words[idxs[s + k]]["text"]) == target[k] for k in range(len(target))):
            first, last = words[idxs[s]], words[idxs[s + len(target) - 1]]
            parts = replacement.split()
            span = idxs[s:s + len(target)]
            if not parts:
                for i in span:
                    words[i]["text"] = ""
                return idxs
            start, end = first["start"], max(last["end"], first["start"])
            total = sum(len(p) for p in parts) or 1
            new, cur = [], start
            for j, ptxt in enumerate(parts):
                dur = (end - start) * len(ptxt) / total
                new.append({"start": cur, "end": cur + dur if j < len(parts) - 1 else end, "text": ptxt, "prob": 1.0, "speaker": first["speaker"]})
                cur += dur
            # Reuse slots: overwrite the span, blank leftovers, insert extras at the end of the span.
            for k, i in enumerate(span):
                words[i] = new[k] if k < len(new) else dict(words[i], text="")
            extra = new[len(span):]
            if extra:
                pos = span[-1] + 1
                words[pos:pos] = extra
                # shift indexes after pos
                return [i if i < pos else i + len(extra) for i in idxs[:s + len(target)]] + list(range(pos, pos + len(extra))) + \
                       [i + len(extra) for i in idxs[s + len(target):]]
            return idxs
    return None


def claude_cleanup(words, paras, names, settings, labels):
    if not ai_ready(settings):
        log(f"  Skipping name fixes: {ai_not_ready_reason(settings)}.")
        return words, paras, labels, 0
    limit = ai_chunk(settings, 12000)
    lines, chunks, size = [], [], 0
    for i, p in enumerate(paras):
        ln = f"{i} | {labels.get(p['speaker'], 'Speaker')} | {para_text(words, p)}"
        if size + len(ln) > limit and lines:
            chunks.append("\n".join(lines)); lines, size = [], 0
        lines.append(ln); size += len(ln) + 1
    if lines:
        chunks.append("\n".join(lines))

    fixes, speaker_names = [], {}
    for n, ch in enumerate(chunks):
        if len(chunks) > 1:
            log(f"  Claude cleanup, part {n + 1} of {len(chunks)}…")
        out = ai_json(settings, claude_system(names), ch, FIX_SCHEMA, "Proofread these lines:")
        fixes += out.get("fixes", [])
        for s in out.get("speakers", []):
            speaker_names[s["label"]] = s["name"].strip()

    # Apply from the last paragraph backwards so inserted words don't shift earlier indexes.
    applied = 0
    for f in sorted(fixes, key=lambda f: -int(f.get("item", -1))):
        i = f.get("item")
        if not isinstance(i, int) or not (0 <= i < len(paras)) or f["original"] == f["replacement"]:
            continue
        new_idxs = apply_word_fix(words, paras[i]["w"], f["original"], f["replacement"])
        if new_idxs is None:
            continue
        shift = len(new_idxs) - len(paras[i]["w"])
        paras[i]["w"] = new_idxs
        if shift:
            for q in paras[i + 1:]:
                q["w"] = [x + shift for x in q["w"]]
        applied += 1
        log(f"  fixed: {f['original']} → {f['replacement']}  ({f.get('reason', '')})")

    for num, label in list(labels.items()):
        if label in speaker_names and speaker_names[label]:
            labels[num] = speaker_names[label]
            log(f"  {label} is {speaker_names[label]}")
    return words, paras, labels, applied



# ---------------------------------------------------------------- translation
LANG_CODES = {
    "english": "EN", "spanish": "ES", "catalan": "CA", "french": "FR", "italian": "IT", "german": "DE",
    "portuguese": "PT", "japanese": "JA", "chinese": "ZH", "korean": "KO", "dutch": "NL", "arabic": "AR",
    "hebrew": "HE", "russian": "RU", "hindi": "HI", "polish": "PL", "swedish": "SV", "turkish": "TR",
}
WHISPER_LANG = {"en": "english", "es": "spanish", "ca": "catalan", "fr": "french", "it": "italian", "de": "german",
                "pt": "portuguese", "ja": "japanese", "zh": "chinese", "ko": "korean", "nl": "dutch", "ar": "arabic",
                "he": "hebrew", "ru": "russian", "hi": "hindi", "pl": "polish", "sv": "swedish", "tr": "turkish"}

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"item": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["item", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def lang_code(name):
    return LANG_CODES.get(name.strip().lower(), name.strip()[:2].upper())


def translate_items(texts, target, source, names, settings, kind):
    """Translate a list of strings with Claude, keeping one output per input."""
    if not ai_ready(settings):
        raise RuntimeError(f"translation needs AI: {ai_not_ready_reason(settings)}")
    limit = ai_chunk(settings, 8000)
    rules = [
        f"Translate {kind} from {source or 'the source language'} into {target}.",
        "Return exactly one translation per numbered line, with the same `item` number. Never merge or split lines.",
        "Keep people's names, restaurant and brand names, and place names as they are.",
        "Translate meaning naturally, the way a professional subtitler would — not word for word. Keep the speaker's tone.",
    ]
    if kind == "subtitles":
        rules.append("Subtitles must be quick to read: keep each one about as short as the original, and drop filler words like 'um'.")
    if names:
        rules.append("Correct spellings of names and terms in this project:\n" + "\n".join(names))
    system = "\n\n".join(rules)
    out = [None] * len(texts)
    lines, chunks, idx, size = [], [], [], 0
    for i, t in enumerate(texts):
        ln = f"{i} | {t}"
        if size + len(ln) > limit and lines:
            chunks.append("\n".join(lines)); lines, size = [], 0
        lines.append(ln); size += len(ln) + 1
    if lines:
        chunks.append("\n".join(lines))
    for ch in chunks:
        res = ai_json(settings, system, ch, TRANSLATE_SCHEMA, "Translate these lines:")
        for t in res.get("translations", []):
            if isinstance(t.get("item"), int) and 0 <= t["item"] < len(out):
                out[t["item"]] = t["text"].strip()
    # anything missing: fall back to the original rather than leave a hole
    return [o if o else texts[i] for i, o in enumerate(out)]


def split_text_lines(text, maxc):
    """Two balanced lines for a translated caption (no word timings to go on)."""
    text = " ".join(text.split())
    if len(text) <= maxc:
        return [text]
    ws = text.split(" ")
    best, best_score = [text], 1e9
    for k in range(1, len(ws)):
        a, b = " ".join(ws[:k]), " ".join(ws[k:])
        score = max(0, len(a) - maxc) * 10 + max(0, len(b) - maxc) * 10 + abs(len(a) - len(b)) - (10 if re.search(r"[,.;:!?]$", ws[k - 1]) else 0)
        if score < best_score:
            best, best_score = [a, b], score
    return best



# ---------------------------------------------------------------- summary + soundbites
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "soundbites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"item": {"type": "integer"}, "quote": {"type": "string"}, "why": {"type": "string"}},
                "required": ["item", "quote", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "topics", "soundbites"],
    "additionalProperties": False,
}


def _quote_span(words, para, quote):
    """Find the quote inside the paragraph's words; return (start, end) seconds."""
    idx = [i for i in para["w"] if words[i]["text"]]
    q = [norm(t) for t in quote.split() if norm(t)]
    toks = [norm(words[i]["text"]) for i in idx]
    if q:
        n = len(q)
        for s0 in range(len(toks) - n + 1):
            if toks[s0:s0 + n] == q:
                return words[idx[s0]]["start"], words[idx[s0 + n - 1]]["end"]
        # looser: match the first and last three words
        head, tail = q[:3], q[-3:]
        starts = [k for k in range(len(toks)) if toks[k:k + len(head)] == head]
        ends = [k + len(tail) - 1 for k in range(len(toks)) if toks[k:k + len(tail)] == tail]
        for a in starts:
            for b in ends:
                if b >= a:
                    return words[idx[a]]["start"], words[idx[b]]["end"]
    return para["start"], para["end"]


def summarize_state(st, settings):
    if not ai_ready(settings):
        raise RuntimeError(f"summaries need AI: {ai_not_ready_reason(settings)}")
    tc_offset = tc_to_seconds(st["tc"], st["fps"]) if st["tc"] else 0.0
    lines = []
    for i, p in enumerate(st["paras"]):
        t = para_text(st["words"], p).strip()
        if t:
            lines.append(f"{i} | [{fmt_tc(p['start'] + tc_offset)}] {st['labels'].get(str(p['speaker']), 'Speaker')} | {t}")
    text = "\n".join(lines)[:400_000]
    system = "\n\n".join([
        "You are an associate producer logging an interview or conversation for a TV/documentary editor.",
        "Each line is `number | [timecode] speaker | what they said`.",
        "summary: 3–6 plain sentences on what this conversation is about, who is in it, and its key points or story beats. Refer to speakers exactly by the labels given.",
        "topics: 3–8 short phrases (2–5 words each), in the order they come up.",
        "soundbites: the 5–10 strongest bites an editor would pull: clear, self-contained, emotional, funny or story-driving. "
        "`quote` must be copied word-for-word from one line (1–3 sentences, no ellipses) and `item` is that line's number. "
        "`why` is a few words on what it's good for, e.g. 'opens the story', 'emotional peak', 'explains the menu'. Order them by time.",
        "If the conversation is very short or has no usable bites, return fewer.",
    ])
    log("  writing the summary…")
    limit = ai_chunk(settings, 400_000)
    if len(text) <= limit:
        res = ai_json(settings, system, text, SUMMARY_SCHEMA, "Log this conversation:")
    else:
        # small local models: summarize in parts, then combine
        parts, cur, size = [], [], 0
        for ln in lines:
            if size + len(ln) > limit and cur:
                parts.append("\n".join(cur)); cur, size = [], 0
            cur.append(ln); size += len(ln) + 1
        if cur:
            parts.append("\n".join(cur))
        sums, bites = [], []
        for k, part in enumerate(parts):
            log(f"  summary, part {k + 1} of {len(parts)}…")
            r = ai_json(settings, system, part, SUMMARY_SCHEMA, "Log this part of the conversation:")
            sums.append(r.get("summary", "")); bites += r.get("soundbites", [])[:3]
        combine = {"type": "object", "properties": {"summary": {"type": "string"}, "topics": {"type": "array", "items": {"type": "string"}}},
                   "required": ["summary", "topics"], "additionalProperties": False}
        r = ai_json(settings, "Combine these partial summaries of one conversation into a single 3–6 sentence summary and 3–8 short topics, in order. Refer to speakers exactly by the labels used.",
                    "\n\n".join(f"Part {i + 1}: {t}" for i, t in enumerate(sums)), combine, "Combine:")
        res = {"summary": r.get("summary", ""), "topics": r.get("topics", []), "soundbites": bites[:10]}
    bites = []
    for b in res.get("soundbites", []):
        i = b.get("item")
        if not isinstance(i, int) or not (0 <= i < len(st["paras"])):
            continue
        p = st["paras"][i]
        a, z = _quote_span(st["words"], p, b["quote"])
        bites.append({"item": i, "speaker": p["speaker"], "quote": b["quote"].strip(), "why": b["why"].strip(), "start": a, "end": z})
    st["summary"] = {"text": res.get("summary", "").strip(), "topics": [t.strip() for t in res.get("topics", []) if t.strip()],
                     "soundbites": sorted(bites, key=lambda b: b["start"])}
    return st["summary"]


def _named(text, labels):
    """Swap 'Speaker 2' for the person's name once speakers are named."""
    for k, v in sorted(labels.items(), key=lambda kv: -int(kv[0])):
        if not re.fullmatch(r"Speaker \d+", v):
            text = re.sub(rf"\bSpeaker {int(k)}\b", v, text)
    return text


def summary_view(st):
    sm = st.get("summary")
    if not sm:
        return None
    tc_offset = tc_to_seconds(st["tc"], st["fps"]) if st["tc"] else 0.0
    return {
        "summary": _named(sm["text"], st["labels"]),
        "topics": sm["topics"],
        "soundbites": [{**b, "who": st["labels"].get(str(b["speaker"]), ""), "tc": fmt_tc(b["start"] + tc_offset, st["fps"]),
                        "tc_out": fmt_tc(b["end"] + tc_offset, st["fps"])} for b in sm["soundbites"]],
    }


def write_summary(st, out, txt=True):
    v = summary_view(st)
    if not v:
        return
    media = Path(st["media"])
    lab = shown_labels({int(k): x for k, x in st["labels"].items()})
    if txt:
      with open(out["sum_txt"], "w", encoding="utf-8") as f:
        f.write(f"{media.name}\nSummary\n\n{v['summary']}\n\nTopics\n")
        f.write("".join(f"- {t}\n" for t in v["topics"]))
        f.write("\nSoundbites\n")
        for b in v["soundbites"]:
            who = lab.get(b["speaker"])
            f.write(f"\n[{b['tc']} – {b['tc_out']}]  {who.upper() + ':  ' if who else ''}\"{b['quote']}\"\n    {b['why']}\n")
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
    except ImportError:
        return
    doc = Document()
    doc.styles["Normal"].font.name = "Helvetica"
    doc.styles["Normal"].font.size = Pt(11)
    doc.add_heading(media.stem, level=1)
    doc.add_paragraph(v["summary"])
    doc.add_heading("Topics", level=2)
    for t in v["topics"]:
        doc.add_paragraph(t, style="List Bullet")
    doc.add_heading("Soundbites", level=2)
    for b in v["soundbites"]:
        para = doc.add_paragraph()
        r = para.add_run(f"{b['tc']} – {b['tc_out']}   ")
        r.font.color.rgb = RGBColor(0x88, 0x88, 0x88); r.font.size = Pt(9)
        who = lab.get(b["speaker"])
        if who:
            para.add_run(who.upper() + "   ").bold = True
        para.add_run(f"\u201c{b['quote']}\u201d")
        w = doc.add_paragraph(b["why"])
        w.runs[0].font.color.rgb = RGBColor(0x77, 0x77, 0x77); w.runs[0].font.size = Pt(9); w.runs[0].italic = True
    doc.save(out["sum_docx"])


# ---------------------------------------------------------------- captions
def _text(ws):
    return " ".join(w["text"] for w in ws)


def _split_lines(ws, maxc):
    """Break one caption into at most two balanced lines, preferring a break after punctuation."""
    txt = _text(ws)
    if len(txt) <= maxc:
        return [txt]
    best, best_score = None, 1e9
    for k in range(1, len(ws)):
        a, b = _text(ws[:k]), _text(ws[k:])
        over = max(0, len(a) - maxc) + max(0, len(b) - maxc)
        score = over * 10 + abs(len(a) - len(b)) - (10 if re.search(r"[,.;:!?]$", ws[k - 1]["text"]) else 0)
        if score < best_score:
            best, best_score = [a, b], score
    return best


def _balanced_parts(ws, limit, maxs):
    """Split a sentence that's too long into k roughly equal captions, snapping to commas when close."""
    total = len(_text(ws))
    dur = ws[-1]["end"] - ws[0]["start"]
    k = max(1, -(-total // limit), int(-(-dur // maxs)))
    if k == 1:
        return [ws]
    parts, start = [], 0
    for n in range(1, k):
        target = total * n / k
        pos, best, best_score = 0, None, 1e9
        for i in range(start + 1, len(ws)):
            pos = len(_text(ws[:i]))
            score = abs(pos - target) - (limit * 0.25 if re.search(r"[,;:]$", ws[i - 1]["text"]) else 0)
            if len(_text(ws[start:i])) > limit:
                score += 1000
            if score < best_score:
                best, best_score = i, score
        if best is None or best >= len(ws):
            break
        parts.append(ws[start:best]); start = best
    parts.append(ws[start:])
    return [p for p in parts if p]


def build_captions(words, settings):
    maxc, maxl, maxs = settings["max_chars_per_line"], settings["max_lines"], settings["max_caption_seconds"]
    limit = maxc * maxl

    # 1. sentences: break on sentence end, speaker change, or a pause
    sents, cur = [], []
    for w in words:
        if cur and (w["speaker"] != cur[-1]["speaker"] or w["start"] - cur[-1]["end"] > 1.0):
            sents.append(cur); cur = []
        cur.append(w)
        if ends_sentence(w["text"]):
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)

    # 2. long sentences -> balanced parts; short neighbours -> merged when they fit
    pieces = []
    for snt in sents:
        pieces += _balanced_parts(snt, limit, maxs)
    caps = []
    for pc in pieces:
        if caps:
            prev = caps[-1]
            same = prev[-1]["speaker"] == pc[0]["speaker"] and pc[0]["start"] - prev[-1]["end"] < 0.6
            fits = len(_text(prev + pc)) <= limit and pc[-1]["end"] - prev[0]["start"] <= maxs
            short = len(_text(pc)) < 20 or len(_text(prev)) < 20
            if same and fits and short:
                caps[-1] = prev + pc
                continue
        caps.append(pc)

    out = []
    for ws in caps:
        lines = _split_lines(ws, maxc) if maxl >= 2 else [_text(ws)]
        start, end = ws[0]["start"], ws[-1]["end"]
        out.append({"start": start, "end": max(end, start + 0.8), "lines": lines, "speaker": ws[0]["speaker"]})
    for a, b in zip(out, out[1:]):
        if b["start"] > a["start"]:
            a["end"] = min(a["end"], b["start"] - 0.04)
    return out


def write_srt(path, caps):
    with open(path, "w", encoding="utf-8") as f:
        for n, c in enumerate(caps, 1):
            f.write(f"{n}\n{fmt_srt(c['start'])} --> {fmt_srt(c['end'])}\n" + "\n".join(c["lines"]) + "\n\n")


# ---------------------------------------------------------------- transcripts
def shown_labels(labels):
    """One unnamed speaker: no label needed. Otherwise show every label."""
    if len(labels) == 1 and re.fullmatch(r"Speaker \d+", next(iter(labels.values()))):
        return {}
    return labels


def _para_rows(words, paras, texts):
    """(paragraph, text) pairs; `texts` overrides the words (used for translations)."""
    for i, p in enumerate(paras):
        t = (texts[i] if texts else para_text(words, p)).strip()
        if t:
            yield p, t


def write_txt(path, media, words, paras, labels, tc_offset, fps, meta, texts=None):
    labels = shown_labels(labels)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{media.name}\n")
        f.write(" · ".join(meta) + "\n\n")
        for p, t in _para_rows(words, paras, texts):
            who = labels.get(p["speaker"])
            f.write(f"[{fmt_tc(p['start'] + tc_offset, fps)}]  {who.upper() + ':  ' if who else ''}{t}\n\n")


def write_docx(path, media, words, paras, labels, tc_offset, fps, meta, texts=None):
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
    except ImportError:
        return False
    labels = shown_labels(labels)
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Helvetica"
    st.font.size = Pt(11)
    doc.add_heading(media.stem, level=1)
    m = doc.add_paragraph(" · ".join(meta))
    m.runs[0].font.color.rgb = RGBColor(0x77, 0x77, 0x77)
    m.runs[0].font.size = Pt(9)
    for p, t in _para_rows(words, paras, texts):
        para = doc.add_paragraph()
        r = para.add_run(fmt_tc(p["start"] + tc_offset, fps) + "   ")
        r.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        r.font.size = Pt(9)
        who = labels.get(p["speaker"])
        if who:
            rb = para.add_run(who.upper() + "   ")
            rb.bold = True
        para.add_run(t)
    doc.save(path)
    return True


# ---------------------------------------------------------------- one file
TRANSCRIPT_FILES = {"srt", "docx", "txt", "summary"}


def default_opts(settings):
    """What to make and where, from Settings (used by the watch folder and as the window's defaults)."""
    files = list(settings.get("outputs") or ["srt", "docx", "summary"])
    if settings.get("summary") is False and "summary" in files:  # honor the older on/off setting
        files.remove("summary")
    tr = [t.strip() for t in (settings.get("translate_to") or "").split(",") if t.strip()]
    return {"files": files, "translate": tr, "dest": settings.get("dest", "beside"),
            "dir": settings.get("dest_dir", ""), "cleanup": settings.get("audio_cleanup", "local")}


def out_dir(media, out):
    media = Path(media)
    dest = (out or {}).get("dest", "beside")
    if dest == "folder" and (out or {}).get("dir"):
        return Path(out["dir"]).expanduser()
    if dest == "subfolder":
        return media.parent / "Transcripts"
    return media.parent


def _base_taken(base):
    return any(Path(f"{base}{sfx}").exists() for sfx in (".srt", " - transcript.docx", " - transcript.txt", " - summary.docx", " - clean.wav", " - dialogue.wav"))


def resolve_out(media, opts):
    """Fix the output location for a clip once, so renames and later translations land in the same place."""
    media = Path(media)
    out = {"files": list(opts.get("files", [])), "dest": opts.get("dest", "beside"), "dir": opts.get("dir", "")}
    d = out_dir(media, out)
    base = d / media.stem
    prev = load_state_for(media)
    if prev and prev.get("out", {}).get("base") and Path(prev["out"]["base"]).parent == d:
        base = Path(prev["out"]["base"])       # same clip, same place: overwrite its own files
    elif prev and not prev.get("out") and d == media.parent:
        pass                                   # transcribed with an older version: its files are the plain names
    elif _base_taken(base):
        # another clip with the same name already wrote here: keep both
        alt = d / f"{media.stem} ({media.suffix.lstrip('.').lower()})"
        if out["dest"] == "folder":
            alt = d / f"{media.parent.name} - {media.stem}"
        base = alt
    out["base"] = str(base)
    return out


def outputs_for(media, tr_codes=None, out=None):
    """Every file Transcribe could write for a clip. Callers check .exists() for what was actually made."""
    base = Path(out["base"]) if out and out.get("base") else out_dir(media, out) / Path(media).stem
    res = {
        "srt": Path(f"{base}.srt"),
        "txt": Path(f"{base} - transcript.txt"),
        "docx": Path(f"{base} - transcript.docx"),
        "sum_docx": Path(f"{base} - summary.docx"),
        "sum_txt": Path(f"{base} - summary.txt"),
        "clean": Path(f"{base} - clean.wav"),
        "stem_dialogue": Path(f"{base} - dialogue.wav"),
        "stem_music": Path(f"{base} - music.wav"),
        "stem_effects": Path(f"{base} - effects.wav"),
    }
    if isinstance(tr_codes, str):
        tr_codes = [tr_codes]
    for code in tr_codes or []:
        res[f"srt_{code}"] = Path(f"{base} - {code}.srt")
        res[f"txt_{code}"] = Path(f"{base} - transcript {code}.txt")
        res[f"docx_{code}"] = Path(f"{base} - transcript {code}.docx")
    return res


def translations(st):
    """All translations saved for a clip: {code: {...}} (older saves kept one under 'tr')."""
    trs = dict(st.get("trs") or {})
    if st.get("tr"):
        trs.setdefault(st["tr"]["code"], st["tr"])
    return trs


def outputs_for_state(st):
    return outputs_for(Path(st["media"]), list(translations(st)), st.get("out"))


def translate_state(st, target, settings, names=None):
    """Translate an already-transcribed clip (captions + transcript) and remember it."""
    target = target.strip()
    src = WHISPER_LANG.get((st.get("lang") or "").lower(), st.get("lang") or "")
    if src and src.lower() == target.lower():
        log(f"  already in {target} — no translation needed")
        return False
    if names is None:
        names = read_names(Path(st["media"]).parent)
    if not st.get("caps"):
        caps = build_captions([w for w in st["words"] if w["text"]], settings)
        st["caps"] = [{"start": c["start"], "end": c["end"], "lines": c["lines"]} for c in caps]
    log(f"  translating to {target}…")
    progress(f"Translating captions to {target}", 0.3)
    tr_caps = translate_items([" ".join(c["lines"]) for c in st["caps"]], target, src, names, settings, "subtitles")
    progress(f"Translating transcript to {target}", 0.65)
    tr_paras = translate_items([para_text(st["words"], p) for p in st["paras"]], target, src, names, settings, "an interview transcript")
    code = lang_code(target)
    st.setdefault("trs", {})[code] = {"lang": target, "code": code, "caps": tr_caps, "paras": tr_paras}
    st.pop("tr", None) if st.get("tr", {}).get("code") == code else None
    return True


def process(media, settings, opts=None):
    media = Path(media)
    t0 = time.time()
    CLIP_COST["usd"] = 0.0
    opts = opts or default_opts(settings)
    files = set(opts.get("files") or [])
    want_transcript = bool(files & TRANSCRIPT_FILES) or bool(opts.get("translate"))
    out = resolve_out(media, opts)
    outs = outputs_for(media, None, out)
    Path(out["base"]).parent.mkdir(parents=True, exist_ok=True)
    log(f"▶ {media.name}")
    names = read_names(media.parent)
    if names:
        log(f"  using {len(names)} names & terms")

    cleaned = None
    if "clean" in files:
        service = opts.get("cleanup") or settings.get("audio_cleanup") or "local"
        try:
            cleaned = clean_audio(media, service)
            write_wav(outs["clean"], cleaned)
            log(f"  clean audio → {outs['clean'].name}")
        except NoAudio:
            raise
        except Exception as e:
            log(f"  audio cleanup failed: {e}")
            if not want_transcript:
                raise RuntimeError(f"audio cleanup failed: {e}")

    dialogue = None
    if "stems" in files:
        try:
            stems = separate_stems(media, 0.22 if cleaned is not None else 0.02, 0.45 if cleaned is not None else 0.3)
            for name, audio_ in stems.items():
                write_wav24(outs[f"stem_{name}"], audio_, STEMS_OUT_SR)
            dialogue = stems.get("dialogue")
            log(f"  stems → {outs['stem_dialogue'].name}, {outs['stem_music'].name}, {outs['stem_effects'].name}")
        except NoAudio:
            raise
        except Exception as e:
            log(f"  stems failed: {e}")
            if not want_transcript:
                raise RuntimeError(f"stems failed: {e}")

    if not want_transcript:
        secs = len(cleaned) / CLEAN_SR if cleaned is not None else (dialogue.shape[1] / STEMS_OUT_SR if dialogue is not None else 0)
        state = {"media": str(media), "out": out, "audio_only": True, "secs": secs,
                 "labels": {}, "words": [], "paras": [], "tc": None, "fps": None, "lang": None, "fixes": 0}
        save_state(state)
        progress("Done", 1.0)
        log(f"  done in {int(time.time() - t0)}s")
        return state

    pre = cleaned is not None or dialogue is not None
    progress("Reading audio", 0.45 if dialogue is not None else 0.25 if pre else 0.02)
    if cleaned is None and dialogue is not None and settings.get("transcribe_cleaned", True):
        import numpy as np
        mono = dialogue.mean(0)
        audio = np.interp(np.arange(0, len(mono), STEMS_OUT_SR / SR), np.arange(len(mono)), mono).astype(np.float32)
        log("  transcribing from the dialogue stem")
    elif cleaned is not None and settings.get("transcribe_cleaned", True):
        import numpy as np
        audio = np.interp(np.arange(0, len(cleaned), CLEAN_SR / SR), np.arange(len(cleaned)), cleaned).astype(np.float32)
        log("  transcribing from the cleaned audio")
    else:
        audio = load_audio(media)
    secs = len(audio) / SR
    log(f"  {fmt_tc(secs)} of audio")

    turns = []
    if settings["speaker_labels"]:
        progress("Finding speakers", 0.5 if dialogue is not None else 0.3 if pre else 0.08)
        log("  finding speakers…")
        try:
            turns = diarize(audio, settings)
        except Exception as e:
            log(f"  speaker labels failed ({e}); continuing without them")

    progress("Transcribing", 0.55 if dialogue is not None else 0.35 if pre else 0.25)
    log(f"  transcribing ({engine_name()}, {settings['model']})…")
    words, lang = transcribe_words(audio, settings, names, turns)
    if not words:
        log("  no speech found.")
        return False
    words = assign_speakers(words, turns)
    speakers = sorted({w["speaker"] for w in words if w["speaker"]})
    labels = {s: f"Speaker {s}" for s in speakers}
    paras = build_paragraphs(words)

    fixes = 0
    if settings["claude_cleanup"]:
        progress("Checking names & terms", 0.75)
        try:
            words, paras, labels, fixes = claude_cleanup(words, paras, names, settings, labels)
        except Exception as e:
            log(f"  name fixes skipped: {e}")

    caps = build_captions([w for w in words if w["text"]], settings)
    tc, fps = read_start_timecode(media)
    state = {
        "media": str(media), "words": words, "paras": paras, "out": out,
        "labels": {str(k): v for k, v in labels.items()},
        "tc": tc, "fps": fps, "secs": secs, "lang": lang, "fixes": fixes,
        "caps": [{"start": c["start"], "end": c["end"], "lines": c["lines"]} for c in caps],
    }

    if "summary" in files:
        if ai_ready(settings):
            progress("Writing the summary", 0.8)
            try:
                summarize_state(state, settings)
            except Exception as e:
                log(f"  summary skipped: {e}")
        else:
            log(f"  summary skipped: {ai_not_ready_reason(settings)}")

    for target in opts.get("translate") or []:
        progress(f"Translating to {target}", 0.85)
        try:
            translate_state(state, target, settings, names)
        except Exception as e:
            log(f"  translation to {target} skipped: {e}")

    progress("Writing files", 0.97)
    state["cost_usd"] = round(state.get("cost_usd", 0) + CLIP_COST["usd"], 4)
    if CLIP_COST["usd"]:
        log(f"  Claude cost for this clip ≈ ${CLIP_COST['usd']:.3f}")
    write_outputs(state, settings)
    save_state(state)
    made = [p.name for p in outputs_for_state(state).values() if p.exists()]
    log(f"  done in {int(time.time() - t0)}s → " + ", ".join(made))
    progress("Done", 1.0)
    return state


def state_meta(st, translated=None):
    meta = [f"Duration {fmt_tc(st['secs'])}"]
    if translated:
        meta.append(f"Translated from {WHISPER_LANG.get((st['lang'] or '').lower(), st['lang'] or '?').title()} to {translated['lang'].title()} by Claude")
    else:
        meta.append(f"Language {st['lang'] or '?'}")
    if st["tc"]:
        meta.append(f"Source TC {st['tc']} @ {st['fps']:g} fps")
    if len(st["labels"]) > 1:
        meta.append(f"{len(set(st['labels'].values()))} speakers")
    if st["fixes"] and not translated:
        meta.append(f"{st['fixes']} names/terms fixed by Claude")
    return meta


def write_outputs(st, settings, captions=True):
    if st.get("audio_only"):
        return
    media = Path(st["media"])
    trs = translations(st)
    out = outputs_for(media, list(trs), st.get("out"))
    files = set((st.get("out") or {}).get("files") or ["srt", "docx", "txt", "summary"])  # older clips made everything
    labels = {int(k): v for k, v in st["labels"].items()}
    tc_offset = tc_to_seconds(st["tc"], st["fps"]) if st["tc"] else 0.0
    caps = st.get("caps") or build_captions([w for w in st["words"] if w["text"]], settings)
    if captions and "srt" in files:
        write_srt(out["srt"], caps)
    if "txt" in files:
        write_txt(out["txt"], media, st["words"], st["paras"], labels, tc_offset, st["fps"], state_meta(st))
    if "docx" in files:
        write_docx(out["docx"], media, st["words"], st["paras"], labels, tc_offset, st["fps"], state_meta(st))
    if st.get("summary"):
        write_summary(st, out, txt="txt" in files)
    # translations follow the same choices; if no caption or transcript file was picked, make captions + Word
    tr_files = files & {"srt", "docx", "txt"} or {"srt", "docx"}
    maxc = settings["max_chars_per_line"]
    for code, tr in trs.items():
        if "srt" in tr_files and (captions or not out[f"srt_{code}"].exists()):
            write_srt(out[f"srt_{code}"], [dict(c, lines=split_text_lines(t, maxc)) for c, t in zip(caps, tr["caps"])])
        meta = state_meta(st, tr)
        if "txt" in tr_files:
            write_txt(out[f"txt_{code}"], media, st["words"], st["paras"], labels, tc_offset, st["fps"], meta, texts=tr["paras"])
        if "docx" in tr_files:
            write_docx(out[f"docx_{code}"], media, st["words"], st["paras"], labels, tc_offset, st["fps"], meta, texts=tr["paras"])


# ---------------------------------------------------------------- saved state (for naming speakers later)
SESSIONS = HOME / "sessions"


def _state_path(media):
    import hashlib
    return SESSIONS / (hashlib.sha1(str(Path(media).resolve()).encode()).hexdigest()[:16] + ".json")


def save_state(st):
    try:
        SESSIONS.mkdir(parents=True, exist_ok=True)
        _state_path(st["media"]).write_text(json.dumps(st))
    except OSError as e:
        log(f"  (couldn't save session for later renaming: {e})")


def load_state_for(path):
    """Accept the clip itself, or any of its .srt / transcript files."""
    p = Path(path)
    name = p.name
    for suffix in (" - transcript.txt", " - transcript.docx", ".srt"):
        if name.endswith(suffix):
            stem = p.with_name(name[: -len(suffix)])
            for f in p.parent.iterdir():
                if f.suffix.lower() in MEDIA_EXTS and f.with_suffix("") == stem:
                    p = f
                    break
            break
    sp = _state_path(p)
    if sp.exists():
        return json.loads(sp.read_text())
    # outputs saved somewhere else: search saved clips for one that wrote this file
    target = str(Path(path).resolve())
    if SESSIONS.exists():
        for f in SESSIONS.glob("*.json"):
            try:
                st = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if target == str(Path(st.get("media", "")).resolve()) or any(str(o.resolve()) == target for o in outputs_for_state(st).values()):
                return st
    return None


# ---------------------------------------------------------------- naming speakers in Terminal
def _samples(st, num, n=2, width=110):
    """The longest things this speaker said, with timecode, to jog your memory."""
    tc_offset = tc_to_seconds(st["tc"], st["fps"]) if st["tc"] else 0.0
    ps = [p for p in st["paras"] if p["speaker"] == num]
    ps = sorted(ps, key=lambda p: -len(para_text(st["words"], p)))[:n]
    out = []
    for p in sorted(ps, key=lambda p: p["start"]):
        t = para_text(st["words"], p)
        t = t if len(t) <= width else t[: width - 1].rsplit(" ", 1)[0] + "…"
        out.append(f"      [{fmt_tc(p['start'] + tc_offset, st['fps'])}] \"{t}\"")
    return out


def ask_speaker_names(states, settings):
    states = [st for st in states if st and st.get("labels")]
    if not states:
        return
    try:
        if not sys.stdin.isatty():
            return
    except Exception:
        return
    print("\n" + "─" * 60)
    print("Name the speakers  (Return = keep what's in [brackets],  s = skip the rest)")
    print("Type the same name twice to merge two labels into one person.")
    used = []
    for st in states:
        nums = sorted(int(k) for k in st["labels"])
        print(f"\n▶ {Path(st['media']).name}")
        changed = False
        for num in nums:
            cur = st["labels"][str(num)]
            words_said = sum(len(para_text(st["words"], p).split()) for p in st["paras"] if p["speaker"] == num)
            print(f"\n   {cur}  ({words_said} words)")
            print("\n".join(_samples(st, num)))
            if used:
                print("      " + "   ".join(f"{i}) {n}" for i, n in enumerate(used, 1)))
            try:
                ans = input(f"   Who is this? [{cur}] ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return
            if ans.lower() == "s":
                if changed:
                    write_outputs(st, settings, captions=False); save_state(st)
                return
            if ans.isdigit() and 1 <= int(ans) <= len(used):
                ans = used[int(ans) - 1]
            name = ans or cur
            if name != cur:
                st["labels"][str(num)] = name
                changed = True
            if not re.fullmatch(r"Speaker \d+", name) and name not in used:
                used.append(name)
        if changed:
            write_outputs(st, settings, captions=False)
            save_state(st)
            print("   ✓ transcript updated")


# ---------------------------------------------------------------- batch + watch folder
SKIP_FILE = HOME / ".no-audio.json"


def _skip_key(media):
    try:
        st = media.stat()
        return f"{media}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return str(media)


def _skipped():
    try:
        return set(json.loads(SKIP_FILE.read_text()))
    except (OSError, ValueError):
        return set()


def remember_no_audio(media):
    s = _skipped()
    s.add(_skip_key(media))
    try:
        SKIP_FILE.write_text(json.dumps(sorted(s)))
    except OSError:
        pass


def is_done(media):
    return _state_path(media).exists() or outputs_for(media)["srt"].exists() or _skip_key(media) in _skipped()


def expand(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in MEDIA_EXTS and not f.name.startswith("."))
        elif p.suffix.lower() in MEDIA_EXTS:
            files.append(p)
    return files


def stable(path, wait=3.0):
    try:
        a = path.stat().st_size
        time.sleep(wait)
        b = path.stat().st_size
        return a == b and b > 0
    except OSError:
        return False


class NoAudio(Exception):
    pass


def run_batch(files, settings, skip_done=False, states=None):
    ok = fail = 0
    for f in files:
        if skip_done and is_done(f):
            continue
        try:
            st = process(f, settings)
            if st:
                ok += 1
                if states is not None:
                    states.append(st)
        except NoAudio as e:
            remember_no_audio(f)
            log(f"  – skipped {f.name}: {e}")
        except Exception as e:
            fail += 1
            log(f"  ✗ {f.name}: {e}")
    return ok, fail


def watch_once(settings):
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    lock = HOME / ".watch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        # another run is going; if it's stale (>3h), take over
        if time.time() - lock.stat().st_mtime < 3 * 3600:
            return
        lock.unlink(missing_ok=True)
        return watch_once(settings)
    try:
        while True:
            todo = [f for f in expand([WATCH_DIR]) if not is_done(f)]
            todo = [f for f in todo if stable(f)]
            if not todo:
                break
            ok, fail = run_batch(todo, settings)
            if ok or fail:
                notify("Transcribe", f"Watch folder: {ok} done" + (f", {fail} failed" if fail else ""))
            if not ok:
                break
    finally:
        lock.unlink(missing_ok=True)


def main(argv):
    settings = load_settings()
    HOME.mkdir(parents=True, exist_ok=True)
    if argv and argv[0] == "--watch":
        watch_once(settings)
        return 0
    if argv and argv[0] == "--set" and len(argv) == 3:
        key, val = argv[1], argv[2]
        if key not in DEFAULTS:
            print(f"unknown setting {key}"); return 1
        cur = {}
        try:
            cur = json.loads(SETTINGS_FILE.read_text())
        except (OSError, ValueError):
            pass
        if isinstance(DEFAULTS[key], list):
            val = [v.strip() for v in val.split(",") if v.strip()]
        elif isinstance(DEFAULTS[key], bool):
            val = val.lower() in ("1", "true", "yes", "on")
        elif isinstance(DEFAULTS[key], (int, float)) and not isinstance(DEFAULTS[key], bool):
            val = type(DEFAULTS[key])(val)
        cur[key] = val
        SETTINGS_FILE.write_text(json.dumps(cur, indent=2))
        return 0
    if argv and argv[0] == "--get" and len(argv) == 2:
        print(settings.get(argv[1], ""))
        return 0
    if argv and argv[0] == "--translate" and len(argv) >= 3:
        target = argv[1]
        for a in argv[2:]:
            st = load_state_for(a)
            if not st:
                print(f"No saved transcript data for {Path(a).name} — transcribe it first."); continue
            if translate_state(st, target, settings):
                write_outputs(st, settings)
                save_state(st)
                log(f"  ✓ {Path(st['media']).name} translated to {target}")
        return 0
    if argv and argv[0] == "--name-speakers":
        states, missing = [], []
        for a in argv[1:]:
            st = load_state_for(a)
            (states.append(st) if st else missing.append(Path(a).name))
        for m in missing:
            print(f"No saved transcript data for {m} — transcribe it again with this version first.")
        ask_speaker_names(states, settings)
        return 0
    files = expand(argv)
    if not files:
        print("No audio or video files found in what you dropped.")
        return 1
    log(f"{len(files)} file(s) to transcribe")
    states = []
    ok, fail = run_batch(files, settings, states=states)
    msg = f"{ok} transcribed" + (f", {fail} failed" if fail else "")
    log(msg)
    notify("Transcribe", msg)
    if settings.get("ask_speaker_names", True):
        ask_speaker_names(states, settings)
    return 0 if not fail else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
