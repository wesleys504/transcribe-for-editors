# Copyright (c) 2026 TBD Media Post. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE.md).
"""Cloned voices for pickup lines, through ElevenLabs.

Guardrails, on purpose: a voice is only made for a named person, only after the editor records who gave
permission, and every generated line is flagged as synthetic in scripts, checks, exports, file names and a log."""
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

T = None
P = None
TTS_MODEL = "eleven_multilingual_v2"
STS_MODEL = "eleven_multilingual_sts_v2"
SAMPLE_TARGET = 150      # seconds of the person's speech to send for cloning
SAMPLE_MIN = int(os.environ.get("TRANSCRIBE_VOICE_MIN_SECS", "30"))   # override only for testing
SYNTH = re.compile(r'^\s*SYNTH\s+(?P<who>[^:\n"“]{1,40}):\s*["“](?P<text>[^"”\n]+)["”]\s*$')


def bind(transcribe_module, projects_module):
    global T, P
    T, P = transcribe_module, projects_module


def _url():
    return T.SERVICE_URLS["elevenlabs"]


def _key():
    k = T.get_secret("elevenlabs")
    if not k:
        raise RuntimeError("Cloned voices use ElevenLabs. Add your ElevenLabs key in Settings first.")
    return k


def _multipart(fields, files):
    """fields: {name: str}; files: [(field, filename, bytes, content_type)]."""
    boundary = "----transcribe" + uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, fname, data, ctype in files:
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
        out.write(data)
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def _call(method, path, key, body=None, ctype=None, timeout=600):
    headers = {"xi-api-key": key}
    if ctype:
        headers["Content-Type"] = ctype
    try:
        return T.http(method, _url() + path, headers, body, timeout=timeout)[1]
    except T.ServiceError as e:
        low = e.body.lower()
        if e.code == 401:
            raise RuntimeError("ElevenLabs rejected the API key. Check it in Settings.")
        if e.code == 402 or "quota" in low or "credits" in low:
            raise RuntimeError("Your ElevenLabs account is out of credits.")
        if "can_not_use_instant_voice_cloning" in low or "subscription" in low or "plan" in low:
            raise RuntimeError("Your ElevenLabs plan doesn't include voice cloning. It's on the Starter plan and up.")
        if e.code == 404 and "voice" in low:
            raise RuntimeError("ElevenLabs doesn't have that voice any more. Remove it here and clone it again.")
        raise RuntimeError(f"ElevenLabs error {e.code}: {e.body[:200]}")


# ---------------------------------------------------------------- whose voice, and which audio

def speakers(p):
    """Named people in the project's clips (unnamed 'Speaker 2' can't be cloned: name them first)."""
    out = {}
    for c in p["clips"]:
        st = T.load_state_for(c)
        if not st or not st.get("words"):
            continue
        for k, name in st["labels"].items():
            if re.fullmatch(r"Speaker \d+", name):
                continue
            secs = sum(pa["end"] - pa["start"] for pa in st["paras"] if str(pa["speaker"]) == str(k))
            out[name] = out.get(name, 0) + secs
    return out


def _best_audio(media, st):
    """The cleanest version of a clip's audio we have: dialogue stem, then clean audio, then the clip."""
    outs = T.outputs_for(Path(media), None, st.get("out"))
    for k in ("stem_dialogue", "clean"):
        if outs.get(k) and outs[k].exists():
            return outs[k], k
    return Path(media), "clip"


def gather(p, name, target=SAMPLE_TARGET):
    """Pick the person's longest uninterrupted stretches of speech, across the project's clips."""
    cands = []
    for c in p["clips"]:
        st = T.load_state_for(c)
        if not st or not st.get("words") or not Path(c).exists():
            continue
        nums = {str(k) for k, v in st["labels"].items() if v == name}
        paras = st["paras"]
        for i, pa in enumerate(paras):
            if str(pa["speaker"]) not in nums or pa["end"] - pa["start"] < 3:
                continue
            # stay away from the edges, where the other person may overlap
            a = pa["start"] + (0.3 if i else 0)
            b = pa["end"] - (0.3 if i + 1 < len(paras) else 0)
            if b - a >= 2.5:
                cands.append((b - a, c, st, a, min(b, a + 25)))
    cands.sort(key=lambda x: -x[0])
    picked, total = [], 0.0
    for _, c, st, a, b in cands:
        if total >= target:
            break
        picked.append((c, st, a, b))
        total += b - a
    return picked, total


def sample_plan(pid, name):
    p = P.load(pid)
    picked, total = gather(p, name)
    srcs = sorted({_best_audio(c, st)[1] for c, st, _, _ in picked})
    return {"name": name, "secs": round(total, 1), "segments": len(picked), "clips": len({c for c, *_ in picked}),
            "sources": srcs, "enough": total >= SAMPLE_MIN, "voice": p.get("voices", {}).get(name)}


# ---------------------------------------------------------------- clone

def clone(pid, name, consent_by, consent_note="", remove_noise=False):
    consent_by = (consent_by or "").strip()
    if not consent_by:
        raise RuntimeError("Say who gave permission to clone this voice.")
    key = _key()
    p = P.load(pid)
    picked, total = gather(p, name)
    if total < SAMPLE_MIN:
        raise RuntimeError(f"Only {int(total)} seconds of {name} talking alone. Cloning needs at least {SAMPLE_MIN}; add more of their clips.")
    T.log(f"  cloning {name}'s voice from {total:.0f}s of speech ({len(picked)} pieces)…")
    files = []
    cache = {}
    import numpy as np
    picked.sort(key=lambda x: (x[0], x[2]))      # one clip at a time, in order
    for n, (c, st, a, b) in enumerate(picked):
        if c not in cache:
            cache = {c: T.load_audio_at(_best_audio(c, st)[0], 48000)}
        seg = cache[c][int(a * 48000):int(b * 48000)]
        pk = float(np.abs(seg).max() or 1)
        files.append(("files", f"{n + 1:02d}.flac", T._encode_flac((seg / pk * 0.9).astype(np.float32), 48000), "audio/flac"))
    fields = {"name": f"{name} ({p['name']}, Transcribe)",
              "description": f"Cloned in Transcribe for the project {p['name']}. Permission: {consent_by}. {consent_note}".strip(),
              "remove_background_noise": "true" if remove_noise else "false"}
    body, ctype = _multipart(fields, files)
    out = json.loads(_call("POST", "/v1/voices/add", key, body, ctype))
    vid = out.get("voice_id")
    if not vid:
        raise RuntimeError("ElevenLabs didn't return a voice.")
    with P.LOCK:
        p = P.load(pid)
        p.setdefault("voices", {})[name] = {"voice_id": vid, "created": time.time(), "secs": round(total, 1),
                                           "consent_by": consent_by, "consent_note": consent_note[:500],
                                           "needs_verification": bool(out.get("requires_verification"))}
        p.setdefault("synth_log", []).append({"t": time.time(), "event": "voice cloned", "who": name, "consent_by": consent_by})
        P.save(p)
    T.log(f"  {name}'s voice is ready")
    return p["voices"][name]


def forget(pid, name, delete_remote=True):
    with P.LOCK:
        p = P.load(pid)
        v = p.get("voices", {}).pop(name, None)
        if v:
            p.setdefault("synth_log", []).append({"t": time.time(), "event": "voice removed", "who": name})
        P.save(p)
    if v and delete_remote:
        try:
            _call("DELETE", f"/v1/voices/{v['voice_id']}", _key())
        except Exception as e:
            T.log(f"  couldn't delete {name}'s voice at ElevenLabs ({e}); delete it on elevenlabs.io")


# ---------------------------------------------------------------- generate lines

def _out_dir(p):
    d = Path(p.get("audio_dir") or (T.HOME / "Generated audio" / re.sub(r'[\\/:*?"<>|]+', "-", p["name"])))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out_path(p, name, text, how):
    words = re.sub(r"[^\w' ]+", "", text).split()[:7]
    stem = f"{name} - SYNTHETIC - {' '.join(words)}" + (" (from recording)" if how == "recording" else "")
    path = _out_dir(p) / f"{stem}.wav"
    k = 2
    while path.exists():
        path = _out_dir(p) / f"{stem} ({k}).wav"
        k += 1
    return path


def _save(p, pid, name, text, how, audio_bytes, recording=""):
    import numpy as np
    y = T._decode_bytes(audio_bytes, 48000)
    if not len(y):
        raise RuntimeError("ElevenLabs sent back empty audio.")
    path = _out_path(p, name, text, how)
    T.write_wav24(path, np.stack([y, y]), 48000)
    secs = len(y) / 48000
    T.record_service_usage("elevenlabs_voice", secs / 60)
    with P.LOCK:
        p = P.load(pid)
        p.setdefault("synth_log", []).append({"t": time.time(), "event": "line generated", "who": name, "text": text,
                                              "how": how, "file": str(path), "secs": round(secs, 2),
                                              "recording": recording, "consent_by": p["voices"][name]["consent_by"]})
        P.save(p)
    T.log(f"  synthetic line for {name} → {path.name}")
    return {"file": str(path), "secs": round(secs, 2)}


def _voice(p, name):
    v = p.get("voices", {}).get(name)
    if not v:
        raise RuntimeError(f"{name}'s voice isn't cloned yet. Clone it under Voices first.")
    return v


def say(pid, name, text, stability=0.5, similarity=0.8):
    """Text to speech in the cloned voice."""
    text = (text or "").strip()
    if not text:
        raise RuntimeError("The line is empty.")
    p = P.load(pid)
    v = _voice(p, name)
    body = json.dumps({"text": text, "model_id": TTS_MODEL,
                       "voice_settings": {"stability": stability, "similarity_boost": similarity, "style": 0.0, "use_speaker_boost": True}}).encode()
    audio = _call("POST", f"/v1/text-to-speech/{v['voice_id']}?output_format=mp3_44100_128", _key(), body, "application/json")
    return _save(p, pid, name, text, "text", audio)


def say_like(pid, name, text, recording):
    """Speech to speech: the editor's own read of the line, turned into the person's voice (keeps the timing)."""
    p = P.load(pid)
    v = _voice(p, name)
    rec = T.load_audio_at(recording, 48000)
    if len(rec) > 48000 * 60 * 5:
        raise RuntimeError("That recording is over 5 minutes. Trim it to the line you need.")
    body, ctype = _multipart({"model_id": STS_MODEL, "remove_background_noise": "true"},
                             [("audio", "line.flac", T._encode_flac(rec, 48000), "audio/flac")])
    audio = _call("POST", f"/v1/speech-to-speech/{v['voice_id']}?output_format=mp3_44100_128", _key(), body, ctype)
    return _save(p, pid, name, text or Path(recording).stem, "recording", audio, str(recording))


def lines_in(body):
    return [(n, m["who"].strip(), m["text"].strip()) for n, line in enumerate(body.split("\n")) for m in [SYNTH.match(line)] if m]


def latest_file(p, name, text):
    for e in reversed(p.get("synth_log", [])):
        if e.get("event") == "line generated" and e.get("who") == name and e.get("text") == text and Path(e["file"]).exists():
            return e
    return None


def canonical(p, who):
    """SYNTH AMY → the voice saved as 'Amy'."""
    for name in list(p.get("voices", {})) + list(speakers(p)):
        if name.lower() == who.strip().lower():
            return name
    return who.strip()


def check_lines(p, body):
    """Results for SYNTH lines, shown with the soundbite check. Never counted as a match."""
    res = []
    for n, who, text in lines_in(body):
        who = canonical(p, who)
        voice = p.get("voices", {}).get(who)
        f = latest_file(p, who, text)
        r = {"line": n, "clip": "SYNTH", "who": who, "quote": text, "status": "synth", "voice": bool(voice),
             "file": f["file"] if f else "", "secs": f["secs"] if f else 0}
        if not voice:
            r["note"] = f"Generated voice line. {who}'s voice isn't cloned in this project yet."
        elif f:
            r["note"] = f"Generated voice, not something {who} said. Permission: {voice['consent_by']}."
        else:
            r["note"] = f"Generated voice line, not made yet. Permission: {voice['consent_by']}."
        res.append(r)
    return res


def log_text(p):
    rows = []
    for e in p.get("synth_log", []):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e["t"]))
        if e["event"] == "line generated":
            rows.append(f"{when}  {e['who']}: \"{e['text']}\"  ({'from a recording' if e['how'] == 'recording' else 'from text'}; "
                        f"permission: {e.get('consent_by', '')})  → {Path(e['file']).name}")
        else:
            rows.append(f"{when}  {e['event']}: {e['who']}" + (f" (permission: {e['consent_by']})" if e.get("consent_by") else ""))
    return "\n".join(rows)
