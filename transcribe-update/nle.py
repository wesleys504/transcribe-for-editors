# Copyright (c) 2026 TBD Media Post. Licensed under PolyForm Noncommercial 1.0.0 (see LICENSE.md).
"""
Sequences and markers for Premiere Pro, DaVinci Resolve and FCP 7, as FCP 7 XML (xmeml v4).

Everything is laid out in the sequence's timebase. Times in are real seconds from the start of each
source file (the same clock as the transcript's word times), so nothing depends on Premiere's own transcript.
"""
import functools
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import escape

DEFAULT_FPS = 24000 / 1001
T = None  # transcribe module, set by bind()


def bind(transcribe_module):
    global T
    T = transcribe_module

COMMON_RATES = (24000 / 1001, 24, 25, 30000 / 1001, 30, 50, 60000 / 1001, 60)


def snap_rate(fps):
    """Round a measured rate (e.g. 29.97002997) to the standard rate it means."""
    if not fps:
        return None
    best = min(COMMON_RATES, key=lambda r: abs(r - fps))
    return best if abs(best - fps) < 0.05 else float(fps)


@functools.lru_cache(maxsize=256)
def media_info(path):
    """fps (None for audio-only), duration (s), width, height, audio channels, sample rate."""
    import av
    info = {"fps": None, "secs": 0.0, "width": 1920, "height": 1080, "channels": 0, "rate": 48000}
    with av.open(str(path)) as c:
        if c.duration:
            info["secs"] = c.duration / 1_000_000
        for s in c.streams.video:
            if s.average_rate:
                info["fps"] = snap_rate(float(s.average_rate))
                info["width"], info["height"] = s.codec_context.width or 1920, s.codec_context.height or 1080
                if s.duration and s.time_base:
                    info["secs"] = max(info["secs"], float(s.duration * s.time_base))
                break
        for s in c.streams.audio:
            info["channels"] += s.codec_context.channels or 1
            info["rate"] = s.codec_context.sample_rate or 48000
    return info


def _rate(fps):
    ntsc = abs(fps - round(fps)) > 0.01
    return f"<rate><timebase>{round(fps)}</timebase><ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc></rate>"


def _tc(start_frames, fps, df):
    return (f"<timecode>{_rate(fps)}<string>{T.frames_to_tc(start_frames, fps, df)}</string>"
            f"<frame>{start_frames}</frame><displayformat>{'DF' if df else 'NDF'}</displayformat></timecode>")


def _pathurl(path):
    return "file://localhost" + urllib.parse.quote(str(Path(path).resolve()))


def _marker(m, fps):
    a = round(m["at"] * fps)
    b = round(m["end"] * fps) if m.get("end") is not None else -1
    return (f"<marker><name>{escape(m.get('name', ''))[:250]}</name><comment>{escape(m.get('comment', ''))[:2000]}</comment>"
            f"<in>{a}</in><out>{b if b > a else -1}</out></marker>")


def sequence_xml(name, items, fps=None, start_tc=None, markers=(), handles=0):
    """
    items: [{"path", "in", "out", "name"?}] in seconds, laid end to end; {"gap": secs} leaves a hole.
    start_tc: (frames, df) for the sequence's starting timecode. Default 01:00:00:00.
    markers: [{"at", "end"?, "name", "comment"}] in seconds from the sequence start.
    handles: extra frames on each side of every clip, kept inside the source.
    """
    infos = {it["path"]: media_info(it["path"]) for it in items if "path" in it}
    fps = fps or next((i["fps"] for i in infos.values() if i["fps"]), None) or DEFAULT_FPS
    df = bool(start_tc and start_tc[1])
    seq_start = start_tc[0] if start_tc else T.tc_to_frames("01:00:00;00" if df else "01:00:00:00", fps)
    first_video = next((i for i in infos.values() if i["fps"]), None)
    w, h = (first_video["width"], first_video["height"]) if first_video else (1920, 1080)
    n_audio = min(max([i["channels"] for i in infos.values()] + [1]), 8)

    files_done, file_ids = set(), {}
    vtrack, atracks = [], [[] for _ in range(n_audio)]
    t, cid = 0, 0
    for it in items:
        if "gap" in it:
            t += round(it["gap"] * fps)
            continue
        inf = infos[it["path"]]
        total = max(1, round(inf["secs"] * fps))
        a = max(0, round(it["in"] * fps) - handles)
        b = min(total, round(it["out"] * fps) + handles)
        if b <= a:
            continue
        fid = file_ids.setdefault(it["path"], f"file-{len(file_ids) + 1}")
        if it["path"] in files_done:
            file_el = f'<file id="{fid}"/>'
        else:
            files_done.add(it["path"])
            media = ""
            if inf["fps"]:
                media += (f"<video><samplecharacteristics>{_rate(fps)}<width>{inf['width']}</width>"
                          f"<height>{inf['height']}</height></samplecharacteristics></video>")
            if inf["channels"]:
                media += (f"<audio><samplecharacteristics><depth>16</depth><samplerate>{inf['rate']}</samplerate>"
                          f"</samplecharacteristics><channelcount>{inf['channels']}</channelcount></audio>")
            src_tc = it.get("src_tc")
            file_el = (f'<file id="{fid}"><name>{escape(Path(it["path"]).name)}</name><pathurl>{escape(_pathurl(it["path"]))}</pathurl>'
                       f"{_rate(fps)}<duration>{total}</duration>"
                       f"{_tc(round(src_tc[0] / src_tc[2] * fps), fps, src_tc[1]) if src_tc else ''}"
                       f"<media>{media}</media></file>")
        label = escape(it.get("name") or Path(it["path"]).stem)
        common = (f"<name>{label}</name><enabled>TRUE</enabled><duration>{total}</duration>{_rate(fps)}"
                  f"<start>{t}</start><end>{t + b - a}</end><in>{a}</in><out>{b}</out>")
        ids = []
        if inf["fps"]:
            cid += 1
            ids.append(("video", 1, len(vtrack) + 1, f"clipitem-{cid}"))
        chans = min(inf["channels"], n_audio)
        for ch in range(chans):
            cid += 1
            ids.append(("audio", ch + 1, len(atracks[ch]) + 1, f"clipitem-{cid}"))
        links = "".join(f"<link><linkclipref>{i}</linkclipref><mediatype>{mt}</mediatype><trackindex>{tr}</trackindex>"
                        f"<clipindex>{ci}</clipindex></link>" for mt, tr, ci, i in ids) if len(ids) > 1 else ""
        for k, (mt, tr, ci, i) in enumerate(ids):
            src = "" if mt == "video" else f"<sourcetrack><mediatype>audio</mediatype><trackindex>{tr}</trackindex></sourcetrack>"
            el = f'<clipitem id="{i}">{common}{file_el if k == 0 else f"<file id={chr(34)}{fid}{chr(34)}/>"}{src}{links}</clipitem>'
            (vtrack if mt == "video" else atracks[tr - 1]).append(el)
        t += b - a

    video = (f"<video><format><samplecharacteristics>{_rate(fps)}<width>{w}</width><height>{h}</height>"
             f"<pixelaspectratio>square</pixelaspectratio></samplecharacteristics></format>"
             f"<track>{''.join(vtrack)}</track></video>")
    audio = (f"<audio><numOutputChannels>2</numOutputChannels><format><samplecharacteristics><depth>16</depth>"
             f"<samplerate>48000</samplerate></samplecharacteristics></format>"
             + "".join(f"<track>{''.join(tr)}</track>" for tr in atracks) + "</audio>")
    marks = "".join(_marker(m, fps) for m in markers)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n<xmeml version="4">'
            f'<sequence id="sequence-1"><name>{escape(name)}</name><duration>{max(t, 1)}</duration>{_rate(fps)}'
            f"{_tc(seq_start, fps, df)}<media>{video}{audio}</media>{marks}</sequence></xmeml>\n")


def safe_path(folder, stem, ext):
    import re
    folder = Path(folder).expanduser()
    stem = re.sub(r'[\\/:*?"<>|]+', "-", stem).strip() or "Sequence"
    path, k = folder / f"{stem}{ext}", 2
    while path.exists():
        path, k = folder / f"{stem} ({k}){ext}", k + 1
    return path


def source_tc(st):
    """(start frames, df, fps) of a transcript's source timecode, or None."""
    if st.get("tc") and st.get("fps"):
        return T.tc_to_frames(st["tc"], st["fps"]), T.tc_df(st["tc"], st["fps"]), st["fps"]
    return None


def markers_xml(st, kind="soundbites", fps=None):
    """
    A sequence holding the whole clip, starting at the clip's source timecode, with markers:
    kind="soundbites": summary + the summary's soundbites (needs a summary); kind="paragraphs": every transcript paragraph.
    """
    path = st["media"]
    inf = media_info(path)
    fps = fps or inf["fps"] or snap_rate(st.get("fps")) or DEFAULT_FPS
    labels = T.shown_labels({int(k): v for k, v in st["labels"].items()})
    marks = []
    if kind == "soundbites":
        sm = st.get("summary") or {}
        if not sm.get("soundbites"):
            raise RuntimeError("This clip has no summary yet. Write the summary first.")
        if sm.get("text"):
            marks.append({"at": 0, "name": "SUMMARY", "comment": sm["text"] + ("\n\nTopics: " + ", ".join(sm.get("topics") or []) if sm.get("topics") else "")})
        for b in sm["soundbites"]:
            who = b.get("who") or labels.get(b.get("speaker"), "")
            marks.append({"at": b["start"], "end": b["end"], "name": f"BITE{': ' + b['why'] if b.get('why') else ''}",
                          "comment": f"{who.upper() + ': ' if who else ''}“{b.get('quote', '')}”"})
    else:
        for p in st["paras"]:
            text = T.para_text(st["words"], p).strip()
            if text:
                marks.append({"at": p["start"], "end": p["end"], "name": labels.get(p["speaker"], "") or "Transcript", "comment": text})
    stc = source_tc(st)
    start = (round(stc[0] / stc[2] * fps), stc[1]) if stc else (0, False)
    item = {"path": path, "in": 0, "out": inf["secs"] or st.get("secs", 0), "src_tc": stc}
    name = f"{Path(path).stem} - {'soundbites' if kind == 'soundbites' else 'transcript'} markers"
    return name, sequence_xml(name, [item], fps=fps, start_tc=start, markers=marks), len(marks)
