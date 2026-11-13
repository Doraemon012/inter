# Live Interview Assistant

Listens via the mic, transcribes (faster-whisper), detects questions, answers via Gemini or Cohere.
You can also type/edit a question and click Submit (no mic needed).

## Linux
```bash
sudo apt install -y libportaudio2 python3-venv
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set GEMINI_API_KEY and/or COHERE_API_KEY
python app.py
```

## Windows (PowerShell)
```powershell
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # set GEMINI_API_KEY and/or COHERE_API_KEY
python app.py
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`. Enable Settings > Privacy & security > Microphone (incl. desktop apps). PortAudio is bundled with `sounddevice` on Windows.

Open http://localhost:7860 (use `localhost`, so the browser can prompt for mic permission) and pick the LLM provider in the UI.

## HackMD live sync (optional)
1. Create an API token at https://hackmd.io/settings#api and set `HACKMD_API_TOKEN` in `.env`.
2. Set `HACKMD_NOTE_ID` to an existing note's ID (the `<id>` in `https://hackmd.io/<id>`), or click **Create new note** in the UI and copy the ID it creates into `.env`.
3. Open the note on your phone; each question and answer is appended and updated while it streams (every ~3 s).

Notes: existing note content is kept and a new "Interview session" section is added below it. The app overwrites the note with its own copy, so don't edit that note while syncing. Clear keeps earlier answers in the note. Only personal notes are supported. HackMD API rate limits apply; failures are retried automatically and never block audio or answers.
# inter
