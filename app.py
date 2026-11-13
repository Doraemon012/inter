import os, re, threading, queue, time, difflib
from pathlib import Path
import numpy as np
import requests
import sounddevice as sd
import gradio as gr
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")
SR = 16000
CHUNK_SEC = 4
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER = os.getenv("WHISPER_MODEL", "tiny")

COHERE_MODEL = os.getenv("COHERE_MODEL", "command-a-03-2025")
PROVIDERS = ["Gemini", "Cohere"]
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "Gemini").strip().capitalize()
if DEFAULT_PROVIDER not in PROVIDERS: DEFAULT_PROVIDER = "Gemini"
_clients = {}

SYSTEM = """You are a real-time copilot for a candidate in a live TECHNICAL interview (software / AI engineering). The candidate reads your answer aloud, so write exactly what should be spoken: natural, confident, interview-ready, first person where relevant. The question comes from speech transcription and may contain errors; infer the intended technical term.
Scope: DSA and coding, complexity, OOP, DBMS, OS, computer networks, backend/frontend/full-stack, system design and architecture, programming languages, APIs, databases, AI/ML, LLMs, RAG, agentic AI, and questions about the candidate's projects, internships and engineering decisions.
Format (Markdown, no preamble, no filler):
**Answer:** one or two spoken sentences with the direct answer.
- 3-5 short bullets explaining the key points.
- **Example:** only if useful; keep code minimal.
- **Complexity:** Time O(...), Space O(...) for DSA/coding questions.
- **Trade-offs:** 1-2 bullets only when relevant.
Rules: be technically accurate. Simple questions get 2-4 lines, not essays. For coding problems: approach first, then concise code, then complexity. For questions about the candidate's own projects/experience, never invent specifics; give a strong answer structure with placeholders like [your project]."""
MAX_TOKENS = 1200
INIT_PROMPT = "Technical software interview: algorithms, data structures, Big O time complexity, SQL, REST API, system design, Python, Java, React, Kubernetes, LLM, RAG, embeddings."

def stream_gemini(prompt):
    key = os.getenv("GEMINI_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY missing in .env")
    if "g" not in _clients: _clients["g"] = genai.Client(api_key=key)
    cfgs = [
        types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=MAX_TOKENS, temperature=0.3,
                                    thinking_config=types.ThinkingConfig(thinking_budget=0)),
        types.GenerateContentConfig(system_instruction=SYSTEM, max_output_tokens=MAX_TOKENS * 2, temperature=0.3),
    ]
    got = False
    for n, cfg in enumerate(cfgs):
        try:
            for ch in _clients["g"].models.generate_content_stream(model=GEMINI_MODEL, contents=prompt, config=cfg):
                t = ch.text
                if t: got = True; yield t
            return
        except Exception:
            if got or n == len(cfgs) - 1: raise

def stream_cohere(prompt):
    key = os.getenv("COHERE_API_KEY")
    if not key: raise RuntimeError("COHERE_API_KEY missing in .env")
    import cohere
    if "c" not in _clients: _clients["c"] = cohere.ClientV2(api_key=key)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    for ev in _clients["c"].chat_stream(model=COHERE_MODEL, messages=msgs, max_tokens=MAX_TOKENS, temperature=0.3):
        if getattr(ev, "type", None) == "content-delta":
            t = ev.delta.message.content.text
            if t: yield t

whisper = WhisperModel(WHISPER, device="cpu", compute_type="int8")

lock = threading.Lock()
S = {
    "listening": False, "always": False, "manual": False,
    "provider": DEFAULT_PROVIDER, "warn": "", "question_seq": 0,
    "answers": [], "ans_ver": 0, "inflight": 0,   # answers (question + streamed text)
    "selected": -1, "sel_ver": 0,                 # answer shown in the left panel
    "transcript": [],      # list of chunk texts
    "pending": "",         # text accumulating toward a question
    "pending_ts": 0.0,
    "question": "", "answer": "", "status": "Idle", "error": "",
    "answered": [],        # normalized answered questions
    "qa": [],              # (q, a) history
}
audio_q = queue.Queue()
stop_evt = threading.Event()

Q_START = re.compile(r"^(what|why|how|when|where|who|whom|whose|which|can|could|would|will|should|do|does|did|is|are|was|were|have|has|tell me|explain|describe|walk me|give me|write|implement|design|compare|define|difference)\b", re.I)

def norm(t): return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

SMALLTALK = re.compile(r"\b(can you hear me|how are you|are you there|how('s| is) it going|can you see my screen|is (this|that) (ok|okay|fine)|does that make sense|are you ready|good morning|good evening|nice to meet)\b", re.I)

def looks_like_question(t):
    t = t.strip()
    if len(t.split()) < 3 or SMALLTALK.search(t): return False
    if "?" in t: return True
    return bool(Q_START.search(t)) or bool(re.search(r"\b(tell me about|walk me through|can you explain|could you explain|what is|what are|how do|how does|why is|why do)\b", t, re.I))

def is_duplicate(q):
    n = norm(q)
    return any(difflib.SequenceMatcher(None, n, a).ratio() > 0.8 for a in S["answered"])

def build_prompt(question, ctx, prev):
    parts = []
    if ctx: parts.append("Recent conversation:\n" + ctx)
    if prev: parts.append("Earlier Q&A:\n" + prev)
    parts.append("Interviewer question (may be a follow-up):\n" + question)
    return "\n\n".join(parts)

def run_answer(entry, q, ctx, prev, provider):
    gen = stream_cohere if provider == "Cohere" else stream_gemini
    try:
        for piece in gen(build_prompt(q, ctx, prev)):
            with lock:
                entry["text"] += piece; S["ans_ver"] += 1
        with lock:
            if S["error"].startswith("LLM error"): S["error"] = ""
    except Exception as e:
        with lock:
            entry["text"] += f"\n\n⚠️ {provider} error: {e}"; S["error"] = f"LLM error: {e}"
    finally:
        with lock:
            entry["done"] = True; S["ans_ver"] += 1; S["inflight"] = max(0, S["inflight"] - 1)
            if entry in S["answers"]: S["qa"].append((q, entry["text"]))

def start_answer(q, detected=False, provider=None):
    with lock:
        if provider in PROVIDERS: S["provider"] = provider
        provider = S["provider"]
        ctx = " ".join(S["transcript"][-5:]).replace(q, "")[-600:].strip()
        prev = "\n".join(f"Q: {pq}\nA: {pa[:200]}" for pq, pa in S["qa"][-2:])
        entry = {"q": q, "text": "", "done": False}
        S["answers"].append(entry); S["ans_ver"] += 1; S["inflight"] += 1
        S["selected"] = len(S["answers"]) - 1; S["sel_ver"] += 1
        S["question"] = q
        if detected: S["question_seq"] += 1
        S["answered"].append(norm(q))
    threading.Thread(target=run_answer, args=(entry, q, ctx, prev, provider), daemon=True).start()

def render_selected(sel):
    ans = S["answers"]
    if sel < 0 or sel >= len(ans): return "*Suggested answers will appear here.*"
    a = ans[sel]
    body = a["text"] or "*Thinking...*"
    return f"### Q{sel + 1} of {len(ans)}: {a['q'][:300]}\n\n{body}" + ("" if a["done"] else " ▌")

def nav_labels():
    last = len(S["answers"]) - 1
    return [f"{'🟢 ' if n == last else ''}Q{n + 1} · {a['q'][:80]}" for n, a in enumerate(S["answers"])]

def select_question(idx):
    with lock:
        if idx is not None and 0 <= idx < len(S["answers"]):
            S["selected"] = idx; S["sel_ver"] += 1

def jump_latest():
    with lock:
        if S["answers"]:
            S["selected"] = len(S["answers"]) - 1; S["sel_ver"] += 1

# ---------------- HackMD sync (API v1: https://api.hackmd.io/v1, Bearer token) ----------------
HM_API = "https://api.hackmd.io/v1"
HM_INTERVAL = 3.0   # min seconds between note updates (debounce while answers stream)

def parse_note_id(v):
    v = (v or "").strip()
    m = re.search(r"hackmd\.io/([^/?#]+)", v)
    if m: return "" if m.group(1).startswith("@") else m.group(1)
    return "" if "/" in v else v

HM = {
    "token": os.getenv("HACKMD_API_TOKEN", "").strip(), "note_id": parse_note_id(os.getenv("HACKMD_NOTE_ID", "")),
    "url": "", "status": "starting...", "base": "", "loaded": False, "offset": 0,
    "synced_ver": -1, "last_push": 0.0, "retry_at": 0.0, "fails": 0, "started": time.strftime("%Y-%m-%d %H:%M"),
}
if HM["note_id"]: HM["url"] = f"https://hackmd.io/{HM['note_id']}"

def hm_req(method, path, **kw):
    r = requests.request(method, HM_API + path, headers={"Authorization": f"Bearer {HM['token']}"}, timeout=15, **kw)
    if r.status_code in (401, 403): raise RuntimeError(f"HackMD auth/permission error ({r.status_code}) - check HACKMD_API_TOKEN and note access")
    if r.status_code == 404: raise RuntimeError("HackMD note not found (404) - check the note ID")
    if r.status_code == 429: raise RuntimeError("HackMD rate limit (429)")
    if r.status_code >= 400: raise RuntimeError(f"HackMD HTTP {r.status_code}: {r.text[:150]}")
    return r

def hackmd_content(final=False):  # call with lock held
    answers = S["answers"]
    if not answers: return HM["base"]
    secs = []
    for n, a in enumerate(answers, HM["offset"] + 1):
        q = " ".join(a["q"].split())
        body = a["text"].strip() or "_generating..._"
        secs.append(f"## Q{n}: {q}\n\n{body}" + ("" if (a["done"] or final) else " ▌"))
    session = f"# Interview session — {HM['started']}\n\n" + "\n\n---\n\n".join(secs) + "\n"
    return (HM["base"].rstrip() + "\n\n" + session) if HM["base"].strip() else session

def hackmd_archive():  # lock held; keeps previous answers in the note after Clear
    if S["answers"]:
        HM["base"] = hackmd_content(final=True); HM["offset"] += len(S["answers"])
    HM["started"] = time.strftime("%Y-%m-%d %H:%M")

def hackmd_status_text():  # lock held
    link = f" — [open note]({HM['url']})" if HM["url"] else ""
    return f"**HackMD:** {HM['status']}{link}"

def hackmd_set(text):
    nid = parse_note_id(text)
    if (text or "").strip() and not nid:
        gr.Warning("Use the note ID or a hackmd.io/<id> URL (published @user/... links are not note IDs)."); return gr.update()
    with lock:
        if nid != HM["note_id"]:
            HM.update(note_id=nid, loaded=False, base="", offset=0, synced_ver=-1, fails=0, retry_at=0.0,
                      url=f"https://hackmd.io/{nid}" if nid else "", status="connecting..." if nid else "⚪ no note configured")
    return nid

def hackmd_create():
    if not HM["token"]:
        gr.Warning("Set HACKMD_API_TOKEN in .env and restart."); return gr.update()
    title = "Live Interview Answers"
    try:
        perm = os.getenv("HACKMD_READ_PERMISSION", "owner").strip() or "owner"
        nid = hm_req("POST", "/notes", json={"title": title, "content": f"# {title}\n", "readPermission": perm, "writePermission": "owner"}).json()["id"]
    except Exception as e:
        gr.Warning(f"HackMD create failed: {e}"); return gr.update()
    with lock:
        old = HM["base"].strip()
        HM.update(note_id=nid, loaded=True, base=f"# {title}\n" + (("\n" + old + "\n") if old else ""), offset=HM["offset"] if old else 0,
                  synced_ver=-1, fails=0, retry_at=0.0, url=f"https://hackmd.io/{nid}", status="✅ note created")
    gr.Info(f"Created HackMD note {nid}. Add HACKMD_NOTE_ID={nid} to .env to reuse it next time.")
    return nid

def hackmd_worker():
    while True:
        time.sleep(1.0)
        try:
            with lock:
                nid = HM["note_id"]
                if not HM["token"]:
                    HM["status"] = "⚪ disabled — set HACKMD_API_TOKEN in .env and restart"; continue
                if not nid:
                    HM["status"] = "⚪ no note — enter a note ID or create one"; continue
                now = time.time()
                if now < HM["retry_at"]: continue
                loaded, ver = HM["loaded"], S["ans_ver"]
                if ver != HM["synced_ver"] and not (S["answers"] or HM["offset"]): HM["synced_ver"] = ver
                need = ver != HM["synced_ver"] and now - HM["last_push"] >= HM_INTERVAL
            if not loaded:   # keep whatever is already in the note
                existing = hm_req("GET", f"/notes/{nid}").json().get("content") or ""
                with lock:
                    if HM["note_id"] != nid: continue
                    HM["base"] = (existing.rstrip() + "\n\n" + HM["base"]).strip() + "\n" if (existing.strip() or HM["base"].strip()) else ""
                    HM["loaded"] = True; HM["fails"] = 0
                    HM["status"] = "✅ connected, waiting for answers"
                continue
            if not need: continue
            with lock:
                ver = S["ans_ver"]; content = hackmd_content()
            hm_req("PATCH", f"/notes/{nid}", json={"content": content})
            with lock:
                if HM["note_id"] == nid:
                    HM.update(synced_ver=ver, last_push=time.time(), fails=0, status=f"✅ synced {time.strftime('%H:%M:%S')}")
        except Exception as e:
            with lock:
                HM["fails"] += 1; HM["retry_at"] = time.time() + min(60, 5 * 2 ** (HM["fails"] - 1))
                HM["status"] = f"⚠️ {e} — retrying"

threading.Thread(target=hackmd_worker, daemon=True).start()

def handle_text(text):
    now = time.time()
    with lock:
        S["transcript"].append(text)
        if S["pending"] and now - S["pending_ts"] > 12: S["pending"] = ""
        cand = (S["pending"] + " " + text).strip() if S["pending"] else text
    if looks_like_question(cand):
        # question continuing across chunks: wait one more chunk if no terminal '?' and chunk ended mid-sentence
        ends_complete = cand.rstrip().endswith(("?", ".", "!"))
        with lock:
            if not ends_complete and not S.get("_waited"):
                S["pending"], S["pending_ts"], S["_waited"] = cand, now, True
                return
            S["_waited"] = False; S["pending"] = ""
        if is_duplicate(cand): return
        start_answer(cand, detected=True)
    else:
        with lock:
            S["pending"] = ""; S["_waited"] = False

def worker():
    buf = []
    while not stop_evt.is_set():
        try: data = audio_q.get(timeout=0.5)
        except queue.Empty: continue
        if data is None:
            buf.clear(); continue
        with lock: active = S["listening"]
        if not active:
            buf.clear(); continue
        buf.append(data)
        if sum(len(b) for b in buf) >= CUR_SR * CHUNK_SEC:
            audio = np.concatenate(buf).astype(np.float32).flatten(); buf.clear()
            if CUR_SR != SR:
                audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * SR / CUR_SR)), np.arange(len(audio)), audio).astype(np.float32)
            if np.abs(audio).mean() < 0.002: 
                with lock: S["status"] = "Listening..."
                continue
            try:
                segs, _ = whisper.transcribe(audio, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False, without_timestamps=True, temperature=0.0, initial_prompt=INIT_PROMPT)
                text = " ".join(s.text.strip() for s in segs).strip()
                if text: handle_text(text)
            except Exception as e:
                with lock: S["error"] = f"Transcription error: {e}"
            with lock:
                S["status"] = "Listening..." if S["listening"] else "Idle"

def cb(indata, frames, t, status):
    audio_q.put(indata[:, 0].copy())

STREAM = None
CUR_SR = SR
MIC_HELP = "Check that a microphone is connected/enabled and allowed in OS settings (Windows: Settings > Privacy & security > Microphone; Linux: install libportaudio2 and check your sound settings)."

def ensure_stream():
    global STREAM, CUR_SR
    if STREAM is not None and STREAM.active: return True
    try:
        dev = sd.query_devices(kind="input")
    except Exception as e:
        S["error"] = f"No microphone found ({e}). {MIC_HELP}"; return False
    rates = [SR]
    try:
        d = int(dev["default_samplerate"])
        if d != SR: rates.append(d)
    except Exception: pass
    last = None
    for r in rates:
        try:
            STREAM = sd.InputStream(samplerate=r, channels=1, dtype="float32", callback=cb, blocksize=r // 2)
            STREAM.start(); CUR_SR = r; S["error"] = ""
            return True
        except Exception as e:
            last = e; close_stream()
    S["error"] = f"Could not open microphone: {last}. {MIC_HELP}"
    return False

def close_stream():
    global STREAM
    s, STREAM = STREAM, None
    if s is not None:
        try: s.stop(); s.close()
        except Exception: pass
try: list(whisper.transcribe(np.zeros(SR, dtype=np.float32), language="en", beam_size=1)[0])  # warm-up
except Exception: pass
threading.Thread(target=worker, daemon=True).start()

def set_listening():
    want = S["always"] or S["manual"]
    if want:
        fresh = not (STREAM is not None and STREAM.active)
        if ensure_stream():
            if fresh: audio_q.put(None)
        else:
            S["manual"] = S["always"] = False; want = False
    else:
        close_stream()
    S["listening"] = want
    S["status"] = "Listening..." if want else "Idle"

def toggle_mic(perm=""):
    with lock:
        S["manual"] = not S["manual"]
        S["warn"] = perm if (S["manual"] and perm and perm != "granted") else ""
        set_listening()
        err = S["error"] if not S["listening"] else ""
    if err: gr.Warning(err)
    return mic_label()

def toggle_always(v, perm=""):
    with lock:
        S["always"] = bool(v)
        if S["always"] and perm and perm != "granted": S["warn"] = perm
        elif not S["manual"]: S["warn"] = ""
        set_listening()
        err = S["error"] if not S["listening"] else ""
        val = S["always"]
    if err: gr.Warning(err)
    return gr.update(value=val), mic_label()

def submit_question(q, provider):
    q = (q or "").strip()
    if not q:
        gr.Warning("Type a question first."); return gr.update()
    start_answer(q, detected=False, provider=provider)
    return q

def set_provider(provider):
    with lock:
        if provider in PROVIDERS: S["provider"] = provider

def mic_label():
    return gr.update(value="⏹ STOP" if S["manual"] else "🎤 START", variant="stop" if S["manual"] else "primary")

def clear_all():
    with lock:
        hackmd_archive()
        S.update(transcript=[], pending="", question="", answer="", answered=[], qa=[], error="", answers=[], ans_ver=S["ans_ver"] + 1, selected=-1, sel_ver=S["sel_ver"] + 1)

def refresh(seen, seen_v):
    with lock:
        on = S["listening"]
        dot = "🟢" if on else "⚪"
        status = f"{dot} {S['status']}"
        if S["inflight"]: status += "  ·  ✍️ Answering..."
        if S["warn"]: status += f"  —  ⚠️ {S['warn']}"
        if S["error"]: status += f"  —  ⚠️ {S['error']}"
        seq = S["question_seq"]
        q_upd = gr.update(value=S["question"]) if seq != seen else gr.update()
        ans, av, sv = S["answers"], S["ans_ver"], S["sel_ver"]
        sel = S["selected"] = min(S["selected"], len(ans) - 1)
        if (av, sv) != tuple(seen_v):
            a_upd = render_selected(sel); l_upd = ans[sel]["text"] if sel >= 0 else ""
        else:
            a_upd = l_upd = gr.update()
        if sv != seen_v[1]:
            labels = nav_labels(); nav_upd = gr.update(choices=labels, value=labels[sel] if sel >= 0 else None)
        else:
            nav_upd = gr.update()
        return status, "\n".join(S["transcript"][-30:]), S["question"], a_upd, q_upd, seq, mic_label(), (av, sv), l_upd, nav_upd, hackmd_status_text()

CSS = """
body, .gradio-container {background:#0b0f17 !important; color:#e6e9ef !important}
#mic {height:110px; font-size:28px; border-radius:20px}
#status {font-size:18px; font-weight:600}
textarea {background:#141a26 !important; color:#e6e9ef !important}
#answer-doc {height:calc(100vh - 210px); min-height:520px; overflow-y:auto !important; background:#101624; border:1px solid #232c42; border-radius:14px; padding:18px 28px}
#answer-doc, #answer-doc p, #answer-doc li {font-size:clamp(20px, 1.6vw, 30px) !important; line-height:1.75 !important}
#answer-doc li {margin:6px 0}
#answer-doc h3 {font-size:clamp(22px, 1.8vw, 32px) !important; line-height:1.4 !important; color:#8fa8ff !important; margin:0 0 .6em !important}
#answer-doc strong {color:#fff}
#answer-doc code, #answer-doc pre {font-size:clamp(16px, 1.2vw, 20px) !important; line-height:1.5 !important}
#qnav {max-height:240px; overflow-y:auto}
#qnav .wrap {flex-direction:column !important; flex-wrap:nowrap !important; gap:6px !important}
#qnav label {width:100%; border-radius:10px; padding:8px 12px; background:#141a26; border:1px solid #232c42; cursor:pointer}
#qnav label.selected {background:#3b4fd8 !important; border-color:#8fa8ff !important; color:#fff !important; font-weight:600}
#qnav input[type=radio] {display:none}
@media (max-width: 900px) {#answer-doc {height:60vh; min-height:340px; padding:14px 16px} #mic {height:80px; font-size:22px}}
"""
PROBE_JS = """async () => {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      return "Browser mic API unavailable on this page (open http://localhost:7860 or use HTTPS). The app records from the system mic itself.";
    const s = await navigator.mediaDevices.getUserMedia({audio: true});
    s.getTracks().forEach(t => t.stop());
    return "granted";
  } catch (e) {
    return "Browser microphone permission not granted (" + (e.name || e) + "). Allow it in the browser site settings; if capture fails, also check OS mic privacy settings.";
  }
}"""
SCROLL_JS = """() => { const r = document.getElementById('answer-doc'); if (!r) return;
  const h = r.querySelector('h3'); const key = h ? h.innerText.split(':')[0] : ''; const n = r.innerText.length;
  if (window.__akey !== key) { window.__akey = key; window.__alen = n; r.scrollTop = 0; return; }
  if (window.__alen !== n) { const near = r.scrollHeight - r.scrollTop - r.clientHeight < 500; window.__alen = n; if (near) r.scrollTop = r.scrollHeight; } }"""
COPY_JS = "(a) => { navigator.clipboard.writeText(a); return a; }"

with gr.Blocks(css=CSS, theme=gr.themes.Base(primary_hue="indigo").set(body_background_fill="#0b0f17"), title="Interview Assistant", fill_width=True) as demo:
    gr.Markdown("# 🎯 Live Interview Assistant")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=360):
            with gr.Row():
                gr.Markdown("## Suggested Answer")
                jump = gr.Button("⏩ Jump to latest", scale=0, min_width=170)
                copy = gr.Button("📋 Copy Answer", scale=0, min_width=150)
            answer = gr.Markdown("*Suggested answers will appear here.*", elem_id="answer-doc")
        with gr.Column(scale=1, min_width=360):
            status = gr.Markdown("⚪ Idle", elem_id="status")
            mic = gr.Button("🎤 START", variant="primary", elem_id="mic")
            with gr.Row():
                always = gr.Checkbox(label="Always Listening", value=False, scale=1)
                provider = gr.Dropdown(PROVIDERS, value=DEFAULT_PROVIDER, label="LLM Provider", scale=1)
                clear = gr.Button("Clear", scale=0, min_width=90)
            transcript = gr.Textbox(label="Live Transcript", lines=6, interactive=False)
            qnav = gr.Radio(choices=[], label="Detected Questions (click to view)", type="index", elem_id="qnav")
            question = gr.Textbox(label="Detected Question", lines=2, interactive=False)
            manual_q = gr.Textbox(label="Type or Edit Question", lines=2, interactive=True, placeholder="Type a question (works without a microphone)...")
            submit = gr.Button("Submit", variant="primary")
            hm_status = gr.Markdown("**HackMD:** starting...")
            with gr.Row():
                hm_note = gr.Textbox(label="HackMD Note ID", value=HM["note_id"], scale=3)
                hm_set_btn = gr.Button("Connect", scale=1, min_width=90)
                hm_new_btn = gr.Button("Create new note", scale=1, min_width=140)
    perm = gr.Textbox(visible=False)
    seen = gr.State(0)
    seen_v = gr.State((-1, -1))
    latest = gr.Textbox(visible=False)
    mic.click(toggle_mic, perm, mic, js="async (p) => await (" + PROBE_JS + ")()")
    always.change(toggle_always, [always, perm], [always, mic], js="async (a, p) => [a, a ? await (" + PROBE_JS + ")() : '']")
    provider.change(set_provider, provider, None)
    submit.click(submit_question, [manual_q, provider], [question])
    clear.click(clear_all)
    qnav.input(select_question, qnav, None)
    jump.click(jump_latest, None, None)
    copy.click(None, latest, None, js=COPY_JS)
    hm_set_btn.click(hackmd_set, hm_note, hm_note)
    hm_note.submit(hackmd_set, hm_note, hm_note)
    hm_new_btn.click(hackmd_create, None, hm_note)
    gr.Timer(0.5).tick(refresh, [seen, seen_v], [status, transcript, question, answer, manual_q, seen, mic, seen_v, latest, qnav, hm_status], show_progress="hidden").then(None, None, None, js=SCROLL_JS)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
