#!/usr/bin/env python3
# Copyright (c) 2026 TBD Media Post. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE.md).
"""Transcribe — the app window. A small local server + a native Mac window (pywebview)."""
import importlib.util
import io
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("transcribe", APP_DIR / "transcribe.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

IS_MAC = platform.system() == "Darwin"
AGENT = "com.transcribe.watchfolder"
UI_FILE = APP_DIR / "ui" / "index.html"

# ---------------------------------------------------------------- jobs
JOBS = {}            # id -> job dict
ORDER = []           # ids in display order
LOCK = threading.Lock()
WORK = queue.Queue()
LOG_LINES = []
CURRENT = {"id": None}
WINDOW = {"w": None}
TEST_KEY = {"k": ""}  # used only when not on a Mac (testing)

_orig_log = T.log


def _log(msg):
    _orig_log(msg)
    with LOCK:
        LOG_LINES.append(time.strftime("%H:%M:%S ") + msg)
        del LOG_LINES[:-300]
        jid = CURRENT["id"]
        if jid in JOBS:
            JOBS[jid]["log"].append(msg.strip())


T.log = _log


def _progress(stage, frac):
    with LOCK:
        jid = CURRENT["id"]
        if jid in JOBS:
            JOBS[jid]["stage"], JOBS[jid]["frac"] = stage, frac


T.PROGRESS_HOOK = _progress


def enqueue(paths):
    files = T.expand([p for p in paths if p])
    added = 0
    with LOCK:
        active = {j["path"] for j in JOBS.values() if j["status"] in ("queued", "running")}
        for f in files:
            if str(f) in active:
                continue
            jid = uuid.uuid4().hex[:8]
            JOBS[jid] = {"id": jid, "path": str(f), "name": f.name, "folder": str(f.parent).replace(str(Path.home()), "~", 1), "status": "queued",
                         "stage": "Waiting", "frac": 0.0, "error": "", "outputs": [], "log": [], "added": time.time(),
                         "speakers": 0, "named": False}
            ORDER.append(jid)
            WORK.put(jid)
            added += 1
    return added, len(files)


def enqueue_translation(path, target, kind="translate"):
    st = T.load_state_for(path)
    if not st:
        raise RuntimeError("No saved transcript for that clip — transcribe it first.")
    media = Path(st["media"])
    label = f"{media.name} → {target}" if kind == "translate" else f"{media.name}: summary"
    with LOCK:
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"id": jid, "kind": kind, "target": target, "path": str(media), "name": label,
                     "folder": str(media.parent).replace(str(Path.home()), "~", 1), "status": "queued", "stage": "Waiting",
                     "frac": 0.0, "error": "", "outputs": [], "log": [], "added": time.time(), "speakers": 0, "named": True}
        ORDER.append(jid)
        WORK.put(jid)
    return jid


def run_summary(job, settings):
    st = T.load_state_for(job["path"])
    if not st:
        raise RuntimeError("No saved transcript for that clip.")
    T.summarize_state(st, settings)
    T.progress("Writing files", 0.95)
    st["cost_usd"] = round(st.get("cost_usd", 0) + T.CLIP_COST["usd"], 4)
    T.write_outputs(st, settings, captions=False)
    T.save_state(st)
    outs = T.outputs_for(Path(st["media"]))
    with LOCK:
        job.update(status="done", stage="Summary ready", frac=1.0, has_summary=True,
                   outputs=[str(outs[k]) for k in ("sum_docx", "docx") if outs[k].exists()])


def run_translation(job, settings):
    st = T.load_state_for(job["path"])
    if not st:
        raise RuntimeError("No saved transcript for that clip.")
    if not T.translate_state(st, job["target"], settings):
        with LOCK:
            job.update(status="skipped", stage=f"Already in {job['target']}", frac=1.0)
        return
    T.progress("Writing files", 0.95)
    st["cost_usd"] = round(st.get("cost_usd", 0) + T.CLIP_COST["usd"], 4)
    T.write_outputs(st, settings, captions=False)
    T.save_state(st)
    code = T.lang_code(job["target"])
    outs = T.outputs_for(Path(st["media"]), [code])
    with LOCK:
        job.update(status="done", stage=f"Translated to {job['target']}", frac=1.0,
                   outputs=[str(outs[k]) for k in (f"docx_{code}", f"srt_{code}", f"txt_{code}") if outs[k].exists()])


def worker():
    while True:
        jid = WORK.get()
        with LOCK:
            job = JOBS.get(jid)
            if not job or job["status"] != "queued":
                continue
            job["status"], job["stage"], job["started"] = "running", "Starting", time.time()
            CURRENT["id"] = jid
        settings = T.load_settings()
        T.CLIP_COST["usd"] = 0.0
        if job.get("kind") in ("translate", "summarize"):
            try:
                (run_translation if job["kind"] == "translate" else run_summary)(job, settings)
            except Exception as e:
                with LOCK:
                    job.update(status="failed", stage="Failed", error=str(e), frac=1.0)
                _log(f"  ✗ {job['name']}: {e}")
            finally:
                with LOCK:
                    job["finished"] = time.time()
                    job["cost"] = round(T.CLIP_COST["usd"], 4)
                    CURRENT["id"] = None
            continue
        try:
            st = T.process(Path(job["path"]), settings)
            with LOCK:
                if st:
                    job["has_summary"] = bool(st.get("summary"))
                    outs = T.outputs_for_state(st)
                    job.update(status="done", stage="Done", frac=1.0,
                               outputs=[str(p) for p in outs.values() if p.exists()],
                               speakers=len(st["labels"]))
                else:
                    job.update(status="skipped", stage="No speech found", frac=1.0)
        except T.NoAudio as e:
            T.remember_no_audio(Path(job["path"]))
            with LOCK:
                job.update(status="skipped", stage=f"Skipped: {e}", frac=1.0)
        except Exception as e:
            with LOCK:
                job.update(status="failed", stage="Failed", error=str(e), frac=1.0)
            _log(f"  ✗ {job['name']}: {e}")
        finally:
            with LOCK:
                job["finished"] = time.time()
                job["cost"] = round(T.CLIP_COST["usd"], 4)
                CURRENT["id"] = None
            if WORK.empty() and IS_MAC:
                done = sum(1 for j in JOBS.values() if j["status"] == "done")
                T.notify("Transcribe", f"Finished — {done} transcribed")


# ---------------------------------------------------------------- helpers
def watch_on():
    if not IS_MAC:
        return False
    r = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{AGENT}"], capture_output=True)
    return r.returncode == 0


def set_watch(on):
    if not IS_MAC:
        return
    if on:
        subprocess.run([str(T.HOME / "watch-on.sh")], check=False)
    else:
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{AGENT}"], capture_output=True)
        try:
            (Path.home() / "Library/LaunchAgents" / f"{AGENT}.plist").unlink()
        except OSError:
            pass


def has_key():
    return bool(T.get_api_key() or TEST_KEY["k"])


def save_key(k):
    if IS_MAC:
        r = subprocess.run(["security", "add-generic-password", "-U", "-s", T.KEYCHAIN_SERVICE, "-a", "Transcribe", "-w", k],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or "Keychain refused the key")
    else:
        TEST_KEY["k"] = k
        os.environ["ANTHROPIC_API_KEY"] = k


def delete_key():
    if IS_MAC:
        subprocess.run(["security", "delete-generic-password", "-s", T.KEYCHAIN_SERVICE], capture_output=True)
    TEST_KEY["k"] = ""
    os.environ.pop("ANTHROPIC_API_KEY", None)


def write_settings(changes):
    cur = {}
    try:
        cur = json.loads(T.SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        pass
    for k, v in changes.items():
        if k in T.DEFAULTS:
            d = T.DEFAULTS[k]
            if isinstance(d, bool):
                v = bool(v)
            elif isinstance(d, (int, float)):
                v = type(d)(v)
            else:
                v = str(v)
            cur[k] = v
    T.HOME.mkdir(parents=True, exist_ok=True)
    T.SETTINGS_FILE.write_text(json.dumps(cur, indent=2))


def speakers_view(path):
    st = T.load_state_for(path)
    if not st:
        return None
    tc_offset = T.tc_to_seconds(st["tc"], st["fps"]) if st["tc"] else 0.0
    out = []
    for k in sorted(st["labels"], key=int):
        num = int(k)
        ps = [p for p in st["paras"] if p["speaker"] == num]
        words = sum(len(T.para_text(st["words"], p).split()) for p in ps)
        best = sorted(ps, key=lambda p: -len(T.para_text(st["words"], p)))[:3]
        samples = []
        for p in sorted(best, key=lambda p: p["start"]):
            t = T.para_text(st["words"], p)
            samples.append({"tc": T.fmt_tc(p["start"] + tc_offset, st["fps"]), "start": p["start"],
                            "end": min(p["end"], p["start"] + 12), "text": t if len(t) < 220 else t[:217].rsplit(" ", 1)[0] + "…"})
        out.append({"num": num, "label": st["labels"][k], "words": words, "samples": samples})
    used = sorted({v for s in _recent_states() for v in s["labels"].values()} - {f"Speaker {i}" for i in range(1, 30)})
    return {"media": st["media"], "name": Path(st["media"]).name, "speakers": out, "used": used}


def _recent_states(limit=40):
    out = []
    try:
        files = sorted(T.SESSIONS.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:limit]
    except OSError:
        return out
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    return out


def recent():
    items = []
    for f in sorted(T.SESSIONS.glob("*.json"), key=lambda p: -p.stat().st_mtime)[:30] if T.SESSIONS.exists() else []:
        try:
            st = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        outs = T.outputs_for_state(st)
        items.append({"path": st["media"], "name": Path(st["media"]).name, "when": f.stat().st_mtime,
                      "speakers": len(st["labels"]), "labels": sorted(set(st["labels"].values())),
                      "outputs": [str(p) for p in outs.values() if p.exists()], "exists": Path(st["media"]).exists(),
                      "translations": sorted(T.translations(st)), "has_summary": bool(st.get("summary")),
                      "cost": st.get("cost_usd", 0)})
    return items


def apply_names(path, labels):
    st = T.load_state_for(path)
    if not st:
        raise RuntimeError("No saved data for that clip — transcribe it again.")
    for k, v in labels.items():
        if str(k) in st["labels"] and str(v).strip():
            st["labels"][str(k)] = str(v).strip()
    T.write_outputs(st, T.load_settings(), captions=False)
    T.save_state(st)
    with LOCK:
        for j in JOBS.values():
            if j["path"] == st["media"]:
                j["named"] = True


def wav_bytes(samples):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(T.SR)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def pick(kind):
    w = WINDOW["w"]
    if not w:
        raise RuntimeError("File picker is only available in the app window.")
    import webview
    if kind == "folder":
        r = w.create_file_dialog(webview.FileDialog.FOLDER)
    else:
        r = w.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True)
    return list(r or [])


def open_path(path, reveal=False):
    if IS_MAC:
        subprocess.run(["open", "-R", path] if reveal else ["open", path])


# ---------------------------------------------------------------- HTTP API
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                with LOCK:
                    jobs = [dict(JOBS[i], log=JOBS[i]["log"][-6:]) for i in ORDER]
                names = ""
                try:
                    names = T.GLOBAL_NAMES.read_text(encoding="utf-8")
                except OSError:
                    pass
                return self._send(200, {
                    "jobs": jobs, "settings": T.load_settings(), "watch": watch_on(), "hasKey": has_key(),
                    "names": names, "engine": T.engine_name(), "isMac": IS_MAC,
                    "watchDir": str(T.WATCH_DIR), "log": LOG_LINES[-80:], "usage": T.usage_summary(),
                })
            if u.path == "/api/recent":
                return self._send(200, {"items": recent()})
            if u.path == "/api/speakers":
                v = speakers_view(q.get("path", ""))
                return self._send(200 if v else 404, v or {"error": "No saved data for that clip."})
            if u.path == "/api/summary":
                st = T.load_state_for(q.get("path", ""))
                v = T.summary_view(st) if st else None
                if not v:
                    return self._send(404, {"error": "No summary for that clip yet."})
                outs = T.outputs_for(Path(st["media"]))
                return self._send(200, dict(v, name=Path(st["media"]).name, media=st["media"],
                                            doc=str(outs["sum_docx"]) if outs["sum_docx"].exists() else ""))
            if u.path == "/api/snippet":
                a = T.load_audio_span(q["path"], float(q["start"]), float(q["end"]))
                return self._send(200, wav_bytes(a), "audio/wav")
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self._body()
            if u.path == "/api/add":
                added, found = enqueue(b.get("paths", []))
                return self._send(200, {"added": added, "found": found})
            if u.path == "/api/pick":
                added, found = enqueue(pick(b.get("kind", "files")))
                return self._send(200, {"added": added, "found": found})
            if u.path == "/api/translate":
                langs = [l for l in b.get("langs", []) if str(l).strip()]
                if not langs:
                    return self._send(400, {"error": "Pick a language."})
                if not has_key():
                    return self._send(400, {"error": "Translation needs your Claude API key. Add it in Settings."})
                for path in b.get("paths", []):
                    for lang in langs:
                        enqueue_translation(path, str(lang))
                return self._send(200, {"ok": True})
            if u.path == "/api/summarize":
                if not has_key():
                    return self._send(400, {"error": "Summaries need your Claude API key. Add it in Settings."})
                for path in b.get("paths", []):
                    enqueue_translation(path, "", kind="summarize")
                return self._send(200, {"ok": True})
            if u.path == "/api/settings":
                write_settings(b)
                return self._send(200, {"ok": True})
            if u.path == "/api/names":
                T.HOME.mkdir(parents=True, exist_ok=True)
                T.GLOBAL_NAMES.write_text(b.get("text", ""), encoding="utf-8")
                return self._send(200, {"ok": True})
            if u.path == "/api/key":
                k = str(b.get("key", "")).strip()
                if not k.startswith("sk-ant-"):
                    return self._send(400, {"error": "That doesn't look like a Claude API key (it starts with sk-ant-)."})
                save_key(k)
                return self._send(200, {"ok": True})
            if u.path == "/api/key/delete":
                delete_key()
                return self._send(200, {"ok": True})
            if u.path == "/api/watch":
                set_watch(bool(b.get("on")))
                return self._send(200, {"watch": watch_on()})
            if u.path == "/api/open":
                open_path(b["path"], bool(b.get("reveal")))
                return self._send(200, {"ok": True})
            if u.path == "/api/speakers":
                apply_names(b["path"], b.get("labels", {}))
                return self._send(200, {"ok": True})
            if u.path == "/api/retry":
                with LOCK:
                    j = JOBS.get(b.get("id"))
                if j:
                    enqueue([j["path"]])
                return self._send(200, {"ok": True})
            if u.path == "/api/clear":
                with LOCK:
                    for i in list(ORDER):
                        if JOBS[i]["status"] in ("done", "failed", "skipped"):
                            ORDER.remove(i); JOBS.pop(i)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": str(e)})


def start_server(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/"


def main(argv):
    T.HOME.mkdir(parents=True, exist_ok=True)
    if "--headless" in argv:  # for testing in a browser
        port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8765
        url = start_server(port)
        print(url, flush=True)
        threading.Event().wait()
        return
    url = start_server()
    try:
        import webview
        from webview.dom import DOMEventHandler
    except Exception as e:  # no native window available: use the browser instead
        _log(f"Native window unavailable ({e}); opening in your browser.")
        if IS_MAC:
            subprocess.run(["open", url])
        threading.Event().wait()
        return

    if IS_MAC:
        try:  # menu bar name
            from Foundation import NSBundle
            info = NSBundle.mainBundle().infoDictionary()
            info["CFBundleName"] = "Transcribe"
        except Exception:
            pass

    window = webview.create_window("Transcribe", url, width=1040, height=740, min_size=(820, 580),
                                   background_color="#1c1c1e", text_select=True)
    WINDOW["w"] = window

    def on_drop(e):
        files = (e.get("dataTransfer") or {}).get("files") or []
        enqueue([f.get("pywebviewFullPath") for f in files])

    def bind(w):
        w.dom.document.events.dragover += DOMEventHandler(lambda e: None, True, True, debounce=500)
        w.dom.document.events.drop += DOMEventHandler(on_drop, True, True)
        # files passed on the command line (e.g. dropped on the Dock icon)
        if len(argv) > 0:
            enqueue(argv)

    icon = APP_DIR / "icon.png"
    webview.start(bind, window, icon=str(icon) if icon.exists() else None)


if __name__ == "__main__":
    main(sys.argv[1:])
