# Copyright (c) 2026 TBD Media Post. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE.md).
"""Projects and scripts: group clips, write paper edits / VO / promos, with an assistant that
knows the project's transcripts, and checks every quoted soundbite against them."""
import difflib
import json
import os
import functools
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

T = None  # the transcribe module; set by bind()
LOCK = threading.RLock()   # the app serves requests on several threads; one project change at a time


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with LOCK:
            return fn(*a, **k)
    return wrapper


def bind(transcribe_module):
    global T
    T = transcribe_module


def _dir():
    d = T.HOME / "projects"
    d.mkdir(parents=True, exist_ok=True)
    return d


SCRIPT_TYPES = {
    "paper": "Paper edit",
    "vo": "VO / narration",
    "promo": "Promo / social cut",
    "free": "Free writing",
}

# ---------------------------------------------------------------- storage


def _path(pid):
    if not re.fullmatch(r"[a-f0-9]{10}", pid or ""):
        raise RuntimeError("Unknown project.")
    return _dir() / f"{pid}.json"


def load(pid):
    try:
        return json.loads(_path(pid).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError("That project doesn't exist any more.")


def save(p):
    p["updated"] = time.time()
    path = _path(p["id"])
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(p, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return p


def list_projects():
    out = []
    for f in _dir().glob("*.json"):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"id": p["id"], "name": p["name"], "clips": len(p["clips"]), "scripts": len(p["scripts"]),
                    "updated": p.get("updated", 0)})
    return sorted(out, key=lambda x: -x["updated"])


@_locked
def create(name):
    name = (name or "").strip()
    if not name:
        raise RuntimeError("Give the project a name.")
    p = {"id": uuid.uuid4().hex[:10], "name": name[:120], "created": time.time(), "notes": "", "clips": [], "scripts": []}
    return save(p)


@_locked
def update(pid, name=None, notes=None):
    p = load(pid)
    if name is not None and name.strip():
        p["name"] = name.strip()[:120]
    if notes is not None:
        p["notes"] = notes[:4000]
    return save(p)


@_locked
def remove(pid):
    """Moves the project into projects/Deleted, so nothing is lost by accident."""
    src = _path(pid)
    trash = _dir() / "Deleted"
    trash.mkdir(exist_ok=True)
    if src.exists():
        os.replace(src, trash / src.name)


@_locked
def add_clips(pid, paths):
    p = load(pid)
    have = set(p["clips"])
    for x in paths:
        x = str(x)
        if x not in have:
            p["clips"].append(x)
            have.add(x)
    return save(p)


@_locked
def remove_clip(pid, path):
    p = load(pid)
    p["clips"] = [c for c in p["clips"] if c != path]
    return save(p)


def _script(p, sid):
    for s in p["scripts"]:
        if s["id"] == sid:
            return s
    raise RuntimeError("That script doesn't exist any more.")


@_locked
def new_script(pid, kind, title=""):
    p = load(pid)
    kind = kind if kind in SCRIPT_TYPES else "free"
    n = sum(1 for s in p["scripts"] if s["type"] == kind) + 1
    s = {"id": uuid.uuid4().hex[:8], "type": kind, "title": (title or "").strip()[:120] or f"{SCRIPT_TYPES[kind]} {n}",
         "body": "", "chat": [], "created": time.time(), "updated": time.time()}
    p["scripts"].append(s)
    save(p)
    return s


@_locked
def save_script(pid, sid, title=None, body=None, kind=None):
    p = load(pid)
    s = _script(p, sid)
    if title is not None and title.strip():
        s["title"] = title.strip()[:120]
    if body is not None:
        s["body"] = body
    if kind in SCRIPT_TYPES:
        s["type"] = kind
    s["updated"] = time.time()
    save(p)
    return s


@_locked
def delete_script(pid, sid):
    p = load(pid)
    s = _script(p, sid)
    p.setdefault("deleted_scripts", []).append(s)   # kept in the file, just hidden
    p["scripts"] = [x for x in p["scripts"] if x["id"] != sid]
    save(p)


@_locked
def clear_chat(pid, sid):
    p = load(pid)
    _script(p, sid)["chat"] = []
    save(p)


# ---------------------------------------------------------------- clips as the assistant sees them


def clip_labels(paths):
    """Short names for clips ('A001'), made unique with the folder when two clips share a name."""
    stems = {}
    for x in paths:
        stems.setdefault(Path(x).stem.lower(), []).append(x)
    out = {}
    for x in paths:
        pp = Path(x)
        out[x] = f"{pp.parent.name}/{pp.stem}" if len(stems[pp.stem.lower()]) > 1 else pp.stem
    return out


def _tc_offset(st):
    return T.tc_to_seconds(st["tc"], st["fps"]) if st.get("tc") else 0.0


def clip_lines(st):
    """Transcript as sentence-sized lines with in/out timecodes and speaker names."""
    words, labels = st["words"], T.shown_labels({int(k): v for k, v in st["labels"].items()})
    off, fps = _tc_offset(st), st.get("fps")
    lines = []
    for p in st["paras"]:
        idx = [i for i in p["w"] if words[i]["text"]]
        cur = []
        for i in idx:
            cur.append(i)
            if T.ends_sentence(words[i]["text"]) or i == idx[-1]:
                text = " ".join(words[j]["text"].strip() for j in cur).strip()
                if text:
                    lines.append({"in": words[cur[0]]["start"], "out": words[cur[-1]]["end"], "who": labels.get(p["speaker"], ""),
                                  "text": text, "tin": T.fmt_tc(words[cur[0]]["start"] + off, fps),
                                  "tout": T.fmt_tc(words[cur[-1]]["end"] + off, fps)})
                cur = []
    return lines


def clip_info(path):
    st = T.load_state_for(path)
    info = {"path": path, "name": Path(path).name, "exists": Path(path).exists(), "ready": False}
    if st and st.get("words"):
        labels = sorted(set(v for v in st["labels"].values() if not re.fullmatch(r"Speaker \d+", v)))
        info.update(ready=True, secs=st["words"][-1]["end"] if st["words"] else 0, speakers=labels,
                    nspeakers=len(set(st["labels"].values())), has_summary=bool(st.get("summary")))
    return info


def project_view(pid):
    p = load(pid)
    labels = clip_labels(p["clips"])
    clips = [dict(clip_info(c), label=labels[c]) for c in p["clips"]]
    scripts = [{k: s[k] for k in ("id", "type", "title", "updated")} for s in p["scripts"]]
    return {"id": p["id"], "name": p["name"], "notes": p.get("notes", ""), "clips": clips, "scripts": scripts,
            "types": SCRIPT_TYPES}


def project_context(p, budget):
    """Text the assistant reads: each clip's summary, then its transcript lines, within `budget` characters."""
    labels = clip_labels(p["clips"])
    heads, bodies = [], []
    for c in p["clips"]:
        st = T.load_state_for(c)
        if not st or not st.get("words"):
            continue
        lab = labels[c]
        lines = clip_lines(st)
        names = sorted(set(l["who"] for l in lines if l["who"]))
        secs = st["words"][-1]["end"]
        h = f"## CLIP {lab}  (file {Path(c).name}, {T.fmt_tc(secs)} long" + (f", speakers: {', '.join(names)}" if names else "") + ")\n"
        sv = T.summary_view(st)
        if sv:
            h += f"Summary: {sv['summary']}\nTopics: {', '.join(sv['topics'])}\n"
        heads.append(h)
        bodies.append([f"[{l['tin']} - {l['tout']}] {l['who'].upper() + ': ' if l['who'] else ''}{l['text']}" for l in lines])
    if not heads:
        return "", 0, 0
    left = budget - sum(len(h) for h in heads)
    shown = total = 0
    per = max(0, left // len(heads))
    parts = []
    # give unused room from short clips to long ones
    sizes = [sum(len(x) + 1 for x in b) for b in bodies]
    spare = sum(max(0, per - s) for s in sizes)
    long_n = sum(1 for s in sizes if s > per) or 1
    for h, b, s in zip(heads, bodies, sizes):
        room = s if s <= per else per + spare // long_n
        keep, used = [], 0
        for x in b:
            if used + len(x) + 1 > room:
                break
            keep.append(x)
            used += len(x) + 1
        total += len(b)
        shown += len(keep)
        tail = "" if len(keep) == len(b) else f"\n[… {len(b) - len(keep)} more lines of this transcript aren't shown, to stay within the AI's reading limit]"
        parts.append(h + "Transcript:\n" + "\n".join(keep) + tail)
    return "\n\n".join(parts), shown, total


# ---------------------------------------------------------------- the assistant

FORMAT_RULES = """How to write soundbites so the app can check them and editors can find them:
  CLIP [IN - OUT] SPEAKER: "exact words"
for example:
  A001 [01:02:03:04 - 01:02:09:12] AMY: "We opened the doors at six and the line was already around the block."
Rules for soundbites:
- Copy the words exactly from a transcript line. You may use part of a line, or join neighbouring lines from the same clip, but never reword, tidy or invent.
- Use the clip name and timecodes shown in the transcript. When you use part of a line, keep the line's timecodes; the app corrects them.
- Put every soundbite on its own line.
Other lines:
- Section headings start with "## ".
- Narration or VO lines start with "VO: ".
- Editor notes go in square brackets, like [B-ROLL: crowd outside] or [MUSIC: builds] or [SUPER: Opening night, 2019].
Generated (cloned-voice) lines:
- Only when the editor explicitly asks for a pickup or generated line, write it as SYNTH NAME: "text", for example SYNTH AMY: "and that's when we knew."
- Never present a generated line as something the person said, never write one on your own initiative, and keep it as close to
  their real words and meaning as possible. In your reply, say it's a generated line that needs the person's OK.
"""

TYPE_GUIDES = {
    "paper": "This is a PAPER EDIT (a string-out): a sequence of soundbites from the transcripts, in story order, with short "
             "editor notes and section headings. Build a clear beginning, middle and end. Prefer complete thoughts with clean "
             "in and out points. Mostly soundbites; keep notes short.",
    "vo": "This is a VO / NARRATION script. Write for the ear: short sentences, active verbs, one idea per line, easy to read aloud. "
          "Narration lines start with 'VO: '. You can weave in soundbites from the transcripts where a real voice says it better. "
          "About 150 spoken words is one minute.",
    "promo": "This is a PROMO or SOCIAL CUT. Lead with the strongest hook in the first seconds. Keep it tight, usually 15, 30 or 60 "
             "seconds; pick bites that work out of context. Include [SUPER: …] and a closing call to action when it fits.",
    "free": "This is free writing. Help with whatever the editor asks: outlines, treatments, interview questions, show notes, "
            "titles, descriptions. Quote the transcripts in the soundbite format when you quote them.",
}

CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string", "description": "What you say back in the chat. Short and plain."},
        "edit": {"type": "string", "enum": ["none", "replace_all", "append", "replace_selection"],
                 "description": "How to change the script: none (just answer), replace_all (new full script), append (add to the end), "
                                "replace_selection (replace only the text the editor selected)."},
        "text": {"type": "string", "description": "The script text for the edit. Empty when edit is none."},
    },
    "required": ["reply", "edit", "text"],
    "additionalProperties": False,
}


def _system(p, s, context, shown, total):
    sys_ = ("You are the writing assistant inside Transcribe, a Mac app for video editors. You help an editor write scripts "
            "for their project, using the project's interview transcripts. Work like a sharp story producer: honest about what "
            "the footage supports, specific, no filler. Never invent a quote or put words in someone's mouth. If the transcripts "
            "don't contain what the editor wants, say so plainly.\n\n"
            "When you change the script, return the script text in `text` and say in `reply` what you did in a sentence or two. "
            "When the editor only asks a question, answer in `reply` and set edit to none. Write the script in plain text.\n\n"
            + FORMAT_RULES + "\n" + TYPE_GUIDES[s["type"]])
    if p.get("notes"):
        sys_ += "\n\nThe editor's notes for this project (audience, tone, length, style):\n" + p["notes"]
    if context:
        note = "" if shown == total else f" (only {shown} of {total} transcript lines fit; say so if it matters)"
        ctx = f"PROJECT: {p['name']}\nTRANSCRIPTS{note}:\n\n{context}"
    else:
        ctx = f"PROJECT: {p['name']}\nThis project has no transcribed clips yet, so there is nothing to quote. You can still help with structure and writing."
    return sys_, ctx


def _user_turn(s, message, selection):
    t = f"CURRENT SCRIPT ({SCRIPT_TYPES[s['type']]}, titled \"{s['title']}\"):\n"
    t += (s["body"] if s["body"].strip() else "(empty)") + "\n\n"
    if selection:
        t += f"THE EDITOR HAS SELECTED THIS PART OF THE SCRIPT:\n{selection}\n\n"
    t += "EDITOR: " + message
    return t


def _claude_chat(settings, system, context, history, user):
    key = T.get_api_key()
    if not key:
        raise RuntimeError("The assistant needs your Claude key, or switch to the free local model in Settings → AI.")
    model = settings.get("claude_model") or "claude-sonnet-5-5"
    msgs = []
    for m in history:
        msgs.append({"role": "user" if m["role"] == "user" else "assistant",
                     "content": m["text"] if m["role"] == "user" else json.dumps({"reply": m["text"], "edit": "none", "text": ""})})
    msgs.append({"role": "user", "content": user})
    body = json.dumps({
        "model": model, "max_tokens": 16000,
        # the transcripts are cached, so follow-up messages cost about a tenth as much to read them again
        "system": [{"type": "text", "text": system}, {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}}],
        "messages": msgs,
        "output_config": {"format": {"type": "json_schema", "schema": CHAT_SCHEMA}},
    }).encode()
    url = os.environ.get("TRANSCRIBE_CLAUDE_URL", "https://api.anthropic.com") + "/v1/messages"
    for attempt in range(3):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=300, context=T._ssl_context() if url.startswith("https") else None) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            if e.code == 401:
                raise RuntimeError("Claude rejected the API key. Re-enter it in Settings.")
            raise RuntimeError(f"Claude error {e.code}: {msg}")
        except (urllib.error.URLError, OSError, ConnectionError) as e:
            if attempt < 2:
                time.sleep(2)
                continue
            raise RuntimeError(f"Couldn't reach Claude ({e.__class__.__name__}). Check your internet connection.")
    usd = T.record_usage(model, data.get("usage", {}))
    if data.get("stop_reason") == "max_tokens":
        raise RuntimeError("Claude's answer was too long and got cut off. Ask for a shorter piece.")
    block = next((b for b in data.get("content", []) if b.get("type") == "text"), None)
    if not block:
        raise RuntimeError("Claude returned nothing.")
    return json.loads(block["text"]), usd


def _local_chat(settings, system, context, history, user):
    convo = "\n".join(("EDITOR: " if m["role"] == "user" else "YOU: ") + m["text"] for m in history[-6:])
    text = context + ("\n\nEARLIER IN THIS CHAT:\n" + convo if convo else "") + "\n\n" + user
    out = T.ollama_json(settings, system, text, CHAT_SCHEMA, "Here is the project and the editor's request.", ctx=32768)
    return out, 0.0


def chat(pid, sid, message, selection=""):
    message = (message or "").strip()
    if not message:
        raise RuntimeError("Type a message first.")
    settings = T.load_settings()
    p = load(pid)
    s = _script(p, sid)
    local = T.ai_provider(settings) == "local"
    if local and not T.ollama_has_model(settings):
        raise RuntimeError("The free local model isn't ready. Set it up in Settings → AI, or switch to Claude.")
    budget = 40_000 if local else 300_000
    context, shown, total = project_context(p, budget)
    system, ctx = _system(p, s, context, shown, total)
    history = [m for m in s["chat"] if m["role"] in ("user", "assistant")][-12:]
    user = _user_turn(s, message, selection)
    T.log(f"  script assistant: {p['name']} / {s['title']}")
    out, usd = (_local_chat if local else _claude_chat)(settings, system, ctx, history, user)
    edit = out.get("edit") if out.get("edit") in ("none", "replace_all", "append", "replace_selection") else "none"
    text = out.get("text") or ""
    if edit == "replace_selection" and not selection:
        edit = "append"
    if edit != "none" and not text.strip():
        edit = "none"
    now = time.time()
    with LOCK:
        p = load(pid)
        s = _script(p, sid)
        _append_chat(s, message, selection, out, usd, now, edit)
        save(p)
    return {"reply": out.get("reply") or "Done.", "edit": edit, "text": text, "cost": round(usd, 4),
            "context": {"shown": shown, "total": total}}


def _append_chat(s, message, selection, out, usd, now, edit):
    s["chat"].append({"role": "user", "text": message, "t": now, "selection": bool(selection)})
    s["chat"].append({"role": "assistant", "text": out.get("reply") or "Done.", "t": time.time(), "edit": edit, "cost": round(usd, 4)})
    del s["chat"][:-60]


# ---------------------------------------------------------------- checking soundbites

TCRE = r"\d{1,2}:\d{2}:\d{2}(?:[:;.]\d{2})?"
BITE = re.compile(
    rf"^(?P<lead>\s*(?:[-•*]\s*)?)(?P<clip>[^\[\]\n\"“]{{1,80}}?)\s*\[\s*(?P<tin>{TCRE})\s*[-–—]\s*(?P<tout>{TCRE})\s*\]\s*"
    r"(?:(?P<who>[^:\"“\n]{1,40}):\s*)?[\"“](?P<q>[^\"”\n]+)[\"”]\s*$")


def _tc_secs(tc, fps):
    parts = [int(x) for x in re.split(r"[:;.]", tc)]
    if len(parts) == 3:
        parts.append(0)
    h, m, s, f = parts
    return h * 3600 + m * 60 + s + (f / round(fps) if fps else 0)


def _clip_index(p):
    labels = clip_labels(p["clips"])
    idx = {}
    for c in p["clips"]:
        st = T.load_state_for(c)
        if not st or not st.get("words"):
            continue
        ws = [w for w in st["words"] if w["text"] and T.norm(w["text"])]
        entry = {"path": c, "st": st, "ws": ws, "toks": [T.norm(w["text"]) for w in ws], "label": labels[c]}
        for key in {labels[c].lower(), Path(c).stem.lower(), Path(c).name.lower()}:
            idx.setdefault(key, entry)
    return idx


def _find(entry, q):
    """Exact token run → (a, b, 1.0); else best fuzzy window → (a, b, ratio)."""
    toks, n = entry["toks"], len(q)
    if not n or not toks:
        return None
    first = q[0]
    for i in range(len(toks) - n + 1):
        if toks[i] == first and toks[i:i + n] == q:
            return i, i + n - 1, 1.0
    best = None
    anchors = set(q[:3]) | set(q[-3:])
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(q)
    for i, t in enumerate(toks):
        if t not in anchors:
            continue
        for start in {max(0, i - q.index(t)), max(0, i - (n - 1 - q[::-1].index(t)))}:
            for L in (n - 2, n, n + 2):
                if L < 1:
                    continue
                w = toks[start:start + L]
                sm.set_seq1(w)
                r = sm.ratio()
                if not best or r > best[2]:
                    best = (start, start + len(w) - 1, r)
    return best


def check(pid, body):
    """Check each soundbite line against the transcripts. Returns one result per soundbite."""
    p = load(pid)
    idx = _clip_index(p)
    res = []
    lines = body.split("\n")
    for n, line in enumerate(lines):
        m = BITE.match(line)
        if not m:
            continue
        r = {"line": n, "clip": m["clip"].strip(), "quote": m["q"].strip(), "tin": m["tin"], "tout": m["tout"]}
        entry = idx.get(r["clip"].lower())
        q = [T.norm(x) for x in r["quote"].split() if T.norm(x)]
        if not entry:
            # maybe the clip name is off: look for the words in every clip
            hits = [(e, _find(e, q)) for e in {id(v): v for v in idx.values()}.values()]
            hits = [(e, h) for e, h in hits if h and h[2] == 1.0]
            if len(hits) == 1:
                entry = hits[0][0]
                r["clip_fix"] = entry["label"]
            else:
                r.update(status="clip", note=f"No clip called {r['clip']} in this project.")
                res.append(r)
                continue
        found = _find(entry, q)
        st, fps = entry["st"], entry["st"].get("fps")
        off = _tc_offset(st)
        if not found or found[2] < 0.75:
            r.update(status="missing", note="These words aren't in the transcript.")
            res.append(r)
            continue
        a, b, ratio = found
        ws = entry["ws"]
        tin, tout = T.fmt_tc(ws[a]["start"] + off, fps), T.fmt_tc(ws[b]["end"] + off, fps)
        real = " ".join(w["text"].strip() for w in ws[a:b + 1])
        r.update(fix_tin=tin, fix_tout=tout, secs=round(ws[b]["end"] - ws[a]["start"], 2),
                 path=entry["path"], start=ws[a]["start"], end=ws[b]["end"])
        if ratio < 1.0:
            r.update(status="close", note="Close, but not the exact words.", fix_quote=real)
        else:
            same = (r["tin"], r["tout"]) == (tin, tout)
            if not same:
                try:   # within half a second counts as right
                    same = abs(_tc_secs(r["tin"], fps) - (ws[a]["start"] + off)) <= 0.5 and abs(_tc_secs(r["tout"], fps) - (ws[b]["end"] + off)) <= 0.5
                except (ValueError, ZeroDivisionError):
                    same = False
            if r.get("clip_fix"):
                r.update(status="tc", note=f"Exact words, from clip {r['clip_fix']}.")
            elif same:
                r.update(status="ok", note="Exact words.")
            else:
                r.update(status="tc", note="Exact words, but the timecodes are off. Fix moves them to where it's said.")
        res.append(r)
    return res


def apply_fixes(body, results):
    """Rewrite soundbite lines with the transcript's exact words and timecodes."""
    lines = body.split("\n")
    for r in results:
        if r.get("status") not in ("tc", "close", "ok") or r["line"] >= len(lines):
            continue
        m = BITE.match(lines[r["line"]])
        if not m:
            continue
        who = f"{m['who'].strip()}: " if m["who"] else ""
        clip = r.get("clip_fix") or m["clip"].strip()
        quote = r.get("fix_quote") or m["q"].strip()
        lines[r["line"]] = f'{m["lead"]}{clip} [{r["fix_tin"]} - {r["fix_tout"]}] {who}"{quote}"'
    return "\n".join(lines)


def stats(body, results=None):
    """Runtime of soundbites plus read time of VO, for the status line under the editor."""
    bite_secs = sum(r.get("secs", 0) for r in (results or []) if r.get("status") not in ("missing", "synth", "clip"))
    vo_words = 0
    for line in body.split("\n"):
        if BITE.match(line):
            continue
        s = line.strip()
        if s.upper().startswith("VO:"):
            vo_words += len(s[3:].split())
    words = len(re.findall(r"\b\w+\b", body))
    synth = sum(1 for line in body.split("\n") if re.match(r'\s*SYNTH\s+[^:]+:', line))
    return {"bites": sum(1 for line in body.split("\n") if BITE.match(line)), "bite_secs": bite_secs, "synth": synth,
            "vo_words": vo_words, "vo_secs": vo_words / 150 * 60, "words": words}


# ---------------------------------------------------------------- Word export


def export_docx(pid, sid, folder):
    from docx import Document
    from docx.shared import Pt, RGBColor
    p = load(pid)
    s = _script(p, sid)
    folder = Path(folder).expanduser()
    safe = re.sub(r'[\\/:*?"<>|]+', "-", f"{p['name']} - {s['title']}").strip() or "Script"
    path = folder / f"{safe}.docx"
    k = 2
    while path.exists():
        path = folder / f"{safe} ({k}).docx"
        k += 1
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Helvetica"
    st.font.size = Pt(11)
    doc.add_heading(s["title"], level=1)
    meta = doc.add_paragraph(f"{p['name']} · {SCRIPT_TYPES[s['type']]} · {time.strftime('%B %-d, %Y')}")
    meta.runs[0].font.color.rgb = RGBColor(0x77, 0x77, 0x77)
    meta.runs[0].font.size = Pt(9)
    grey = RGBColor(0x88, 0x88, 0x88)
    for line in s["body"].split("\n"):
        m = BITE.match(line)
        if m:
            para = doc.add_paragraph()
            r = para.add_run(f"{m['clip'].strip()}  {m['tin']} – {m['tout']}   ")
            r.font.size = Pt(9)
            r.font.color.rgb = grey
            if m["who"]:
                para.add_run(m["who"].strip().upper() + "   ").bold = True
            para.add_run(f"“{m['q'].strip()}”")
        elif re.match(r'\s*SYNTH\s+[^:]+:', line):
            who, said = line.strip()[6:].split(":", 1)
            para = doc.add_paragraph()
            r = para.add_run(f"SYNTHETIC VOICE ({who.strip().upper()}), NOT RECORDED   ")
            r.bold = True
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0xC8, 0x45, 0x3A)
            para.add_run(said.strip()).italic = True
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.strip().upper().startswith("VO:"):
            para = doc.add_paragraph()
            para.add_run("VO   ").bold = True
            para.add_run(line.strip()[3:].strip())
        elif re.fullmatch(r"\s*\[.*\]\s*", line):
            r = doc.add_paragraph().add_run(line.strip())
            r.italic = True
            r.font.color.rgb = grey
        elif line.strip():
            doc.add_paragraph(line.strip())
    doc.save(path)
    return str(path)
