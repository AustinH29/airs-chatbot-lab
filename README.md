# AIRS Chatbot Lab

**An interactive demo of Palo Alto Networks AIRS (AI Runtime Security)** — watch a live security layer intercept prompt injections, jailbreaks, data leakage, and malicious code requests in real time.

---

## Two ways to use this lab

| | Path A — Browser only | Path B — Run locally |
|---|---|---|
| **Install anything?** | No | Python 3.10+, Ollama |
| **AI credentials?** | No | Optional (AIRS API key) |
| **LLM responses** | Simulated | Real (Ollama) |
| **AIRS scanning** | Simulated | Simulated or live |
| **Best for** | Learning, demos | Full experience |

---

## Path A — Use the hosted app (zero install)

**Visit:** [https://airs-chatbot-lab.onrender.com](https://airs-chatbot-lab.onrender.com) *(link active once deployed)*

The app opens in **Full Demo mode** by default. No account, no download, no terminal.

### What you can do immediately

1. Type any message in the chat bar and press **Enter** — TARS replies.
2. Click **? Help** in the top-right corner for an orientation guide.
3. Open the **Threat Library** panel (left sidebar arrow) to see 28 categorized attack prompts.
4. Click any demo prompt to auto-fill it, then send it — watch AIRS intercept it.
5. Click **{ } API JSON** under any response to see the raw AIRS scan payload.
6. Use the **Pre-Call Scan** and **Post-Call Scan** toggles in the header to turn scanning on and off and compare the difference.

### Understanding the scan badges

Every response shows two colored badges:

| Badge | Meaning |
|---|---|
| **Pre: ALLOW** (green) | Your prompt was scanned and passed through |
| **Pre: BLOCK** (red) | Your prompt was blocked — the LLM never saw it |
| **Post: ALLOW** (green) | The LLM response was scanned and passed through |
| **Post: BLOCK** (red) | The LLM response was blocked before reaching you |
| **Pre/Post: SKIP** (gray) | That scan was disabled via the toggle |

When a block occurs, a gold **Threat Intel** panel appears explaining the detected threat category.

### Suggested exercises

1. Send a normal question ("What is the capital of France?") with both scans on — it passes through.
2. Open the Threat Library → **Jailbreaks** → click a demo prompt → send it — watch the pre-call block.
3. Toggle **Pre-Call Scan off**, resend a jailbreak prompt — it reaches the LLM but the response may still be caught by the post-call scan.
4. Turn off both scans and resend — compare unfiltered output to filtered output.
5. Expand **Multi-Turn Attacks** and click **Run Full Sequence** — watch a 3-turn attack unfold turn by turn.
6. Click **{ } API JSON** on any blocked message and read the `pre_scan.action` and `category` fields.

---

## Path B — Run locally with real AI

Clone the repo and run the app with a real local LLM (Ollama) and, optionally, live AIRS credentials.

### Prerequisites

Install these before starting:

- **Python 3.10 or newer** — [python.org/downloads](https://www.python.org/downloads/)
- **Ollama** — [ollama.com/download](https://ollama.com/download)
- **Git** — [git-scm.com](https://git-scm.com) (or download the ZIP from GitHub)

After installing Ollama, open a terminal and pull a model:

```
ollama pull qwen2.5:7b
```

### Step 1 — Clone the repository

```
git clone https://github.com/AustinH29/airs-chatbot-lab
cd airs-chatbot-lab
```

### Step 2 — Create the Python environment

```
python -m venv .venv
```

**Windows:**
```
.venv\Scripts\activate
```

**macOS / Linux:**
```
source .venv/bin/activate
```

Then install dependencies:

```
pip install -r requirements.txt
```

### Step 3 — Configure environment variables

Copy the example config:

**Windows:**
```
copy .env.example .env
```

**macOS / Linux:**
```
cp .env.example .env
```

Open `.env` in any text editor. The defaults work for Ollama out of the box — you only need to change things if you pulled a different model or want live AIRS scanning:

```
# Which LLM to use (matches the model you pulled with "ollama pull")
LLM_MODEL=ollama/qwen2.5:7b

# Ollama server address (default — usually no change needed)
LLM_API_BASE=http://localhost:11434

# AIRS credentials — optional, only needed for "Live" mode
PANW_PRISMA_AIRS_API_KEY=your-airs-api-key-here
PANW_PRISMA_AIRS_PROFILE_NAME=your-profile-name
```

**AIRS credentials are optional.** Without them the app works in simulated AIRS mode — scanning patterns are detected locally by regex rules without calling the AIRS API. The demo is fully functional either way.

To get real AIRS credentials: log in to [strata.paloaltonetworks.com](https://strata.paloaltonetworks.com) → **AI Security** → **API Applications** (key) and **Security Profiles** (profile name).

### Step 4 — Run the app

```
python app.py
```

Open **http://localhost:5000** in your browser.

To restart after closing the terminal:

```
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux
python app.py
```

### Switching between modes in the app

The header has three mode buttons:

| Mode | What it uses |
|---|---|
| **Full Demo** | Simulated LLM + simulated AIRS — no Ollama needed |
| **Local LLM** | Real Ollama LLM + simulated AIRS |
| **Live** | Real Ollama LLM + real AIRS API (credentials required) |

Switch freely at any time — no restart needed.

### Switching LLM models

Edit `LLM_MODEL` in your `.env` file (or select a different model from the dropdown in the UI):

| `LLM_MODEL` value | Provider |
|---|---|
| `ollama/qwen2.5:7b` | Local Ollama — Qwen (default) |
| `ollama/llama3.2` | Local Ollama — Llama 3.2 |
| `ollama/mistral` | Local Ollama — Mistral |
| `anthropic/claude-sonnet-4-20250514` | Anthropic Claude (needs `ANTHROPIC_API_KEY`) |
| `gpt-4o` | OpenAI (needs `OPENAI_API_KEY`) |

For Ollama models, pull the model first: `ollama pull <model-name>`

---

## Architecture overview

```
User browser
    │
    ▼
Flask app (app.py)
    │
    ├─── [1] Pre-call AIRS scan ──► AIRS API (or local regex in demo mode)
    │         BLOCK? ──► return block notice + Threat Intel explanation
    │         ALLOW? ──► continue
    │
    ├─── [2] LLM call via LiteLLM ──► Ollama (localhost:11434)
    │         (or static response in Full Demo mode)
    │
    └─── [3] Post-call AIRS scan ──► AIRS API (or local regex in demo mode)
              BLOCK? ──► return block notice + Threat Intel explanation
              ALLOW? ──► stream response to browser
```

**Components:**

- **Flask** — single-file web server (`app.py`). All HTML/CSS/JS is inlined — no build step.
- **LiteLLM** — Python library that provides a unified interface to 100+ LLM providers. Change the model string in `.env` to switch providers without changing any code.
- **Ollama** — runs open-source LLM models entirely on your machine. No data leaves your laptop. No API keys required.
- **AIRS** — Palo Alto Networks AI Runtime Security. Scans prompts and responses for six threat categories: prompt injection, jailbreaks, data leakage (DLP), malicious code, toxic content, and malicious URLs.

---

## The Threat Library

The left sidebar contains 28 categorized demo prompts covering every major AI attack vector:

| Category | What it tests |
|---|---|
| **Jailbreaks** | DAN attacks, persona tricks, "no restrictions" prompts |
| **Prompt Injections** | Fake system messages, RAG poisoning, direct overrides |
| **Data Leakage (DLP)** | SSNs, credit cards, API keys, credentials |
| **Malicious Code** | Cookie stealers, reverse shells, ransomware, phishing pages |
| **Toxic Content** | Weapons manufacturing, drug synthesis, violence instructions |
| **Multi-Turn Attacks** | Sequences that build false context across multiple messages |
| **Evasion Attempts** | Foreign language, leetspeak, metaphor, split requests (intentional misses — shows limits) |

### Multi-Turn attacks

Multi-turn sequences demonstrate how an attacker can spread a single attack across several innocent-looking messages. Click **Run Full Sequence (3 turns)** to auto-play all three turns with a 600ms delay between them. The chat will clear first and each turn is labeled so you can follow the progression.

---

## Troubleshooting

**"Connection refused" when in Local LLM or Live mode**
- Make sure Ollama is running: open the Ollama app or run `ollama serve`
- Verify your model is pulled: `ollama list`

**LLM responses are very slow or time out**
- Qwen 7B needs ~8 GB RAM. Try a smaller model: `ollama pull phi3` then set `LLM_MODEL=ollama/phi3` in `.env`

**AIRS badges show "SKIP" instead of ALLOW/BLOCK**
- You are in Full Demo or Local LLM mode — AIRS is simulated, not calling the live API. Switch to **Live** mode and provide credentials to use the real AIRS API.

**AIRS badges show "SKIPPED" in Live mode**
- Fill in `PANW_PRISMA_AIRS_API_KEY` and `PANW_PRISMA_AIRS_PROFILE_NAME` in your `.env` file.

**SSL warnings in the terminal (corporate network)**
- The LiteLLM warning about fetching model costs is harmless — it falls back to a local copy automatically. The Unverified HTTPS warning from the AIRS calls is expected in corporate environments with SSL inspection.

---

## Deploying your own hosted instance

This app is designed for one-click deployment to [Render](https://render.com) (free tier).

1. Fork the repo to your GitHub account.
2. Create a new **Web Service** on Render and connect it to your fork.
3. Render auto-detects the `Procfile` and uses `gunicorn app:app`.
4. Set these environment variables in the Render dashboard:
   - `FLASK_DEBUG=false`
   - `LLM_MODEL=ollama/qwen2.5:7b` *(Full Demo mode works without Ollama — the model selector is ignored in full_demo)*
   - Optionally: `PANW_PRISMA_AIRS_API_KEY` and `PANW_PRISMA_AIRS_PROFILE_NAME` if you want live AIRS scanning
5. Deploy. The hosted app defaults to Full Demo mode (no Ollama needed on the server).

> **Note:** Local LLM and Live modes require Ollama, which cannot run on Render's free tier. The hosted deployment supports Full Demo only. Clone and run locally for the full experience.

---

## Security note

`Network_channel_lab_info.txt` (if present locally) contains live PANW network credentials and is listed in `.gitignore`. It must never be committed or pushed.

---

## License

MIT — see [LICENSE](LICENSE) if present, otherwise all rights reserved by the repository owner.
