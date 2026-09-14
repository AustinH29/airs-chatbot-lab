# AIRS Chatbot Lab — Project Guide

## What This Is
Educational demo of Palo Alto Networks AIRS (AI Runtime Security) integrated with a Flask chatbot. Users interact with TARS (an Interstellar-themed chatbot persona) while AIRS scans prompts and responses in real time. A **Demo Mode** uses regex simulation when no live AIRS tenant is configured.

**GitHub:** `AustinH29/airs-chatbot-lab`

## CRITICAL: Security Constraint
`Network_channel_lab_info.txt` — contains live PANW network broker credentials (CHANNEL_ID, CLIENT_ID, CLIENT_SECRET). It is in `.gitignore` and **must never be committed or staged**. Before every `git add` or commit, confirm this file is absent from staged changes. Never `git add -A` or `git add .` without checking first.

## Tech Stack
- **Flask** — single-file app (`app.py`), all HTML/CSS/JS inlined in `HTML_TEMPLATE`
- **LiteLLM** — abstraction layer for LLM calls (default: `ollama/qwen2:7b`)
- **Ollama** — local LLM server at `http://localhost:11434`
- **PANW AIRS** — `service.api.aisecurity.paloaltonetworks.com`, profile `TARS_chatbot`
- **Prisma** — PANW identity layer

## Key Functions

### `simulate_airs_scan(content, scan_type)` — app.py:126
Pattern-match scan used in Demo Mode. Returns `{action, category, confidence, trigger_terms, simulated: True}`. Regex rules list; **winner = highest-confidence category** across all first-match-per-category results.

**Rule priority and confidence:**
| # | Category | Confidence | What it catches |
|---|----------|-----------|----------------|
| 1 | `prompt_injection` | 0.94 | Directive hijacking, fake system boundaries, verbatim extraction |
| 2 | `jailbreak` | 0.93 | Identity/persona manipulation, DAN, "no restrictions" |
| 3 | `dlp` | 0.97 | PII, credentials, card numbers, API keys |
| 4 | `malicious_code` | 0.89 | Malware, reverse shells, ransomware, exfiltration tooling |
| 5 | `toxic_content` | 0.88 | Weapons, self-harm, drugs, violence |
| 6 | `url_cats` | 0.85 | Malicious domain URLs |

Note: `dlp` (0.97) beats all others when it fires. `prompt_injection` (0.94) now beats `jailbreak` (0.93) when both fire — intentional, since PI is the more specific threat when the two overlap.

### `get_threat_explanation(category, scan_type, simulated)` — app.py ~line 360
Returns a 2-3 sentence "Threat Intel" explanation for the blocked category.
- In **demo mode** (`simulated=True`): returns a static pre-written explanation from `_THREAT_EXPLANATIONS_STATIC` — no LLM required.
- In **live/local-llm mode** (`simulated=False`): makes a short LLM call with a separate TARS-as-intelligence-briefer system prompt.
Returns `""` on failure — treat as optional. The jailbreak and prompt_injection static texts explicitly distinguish the two attack types.

### `ATTACK_PROMPTS` — JavaScript array in HTML_TEMPLATE (~line 2330)
28 demo prompts (indices 0–27) selectable via the Demo Mode panel in the UI.

### `audit_patterns.py` — project root
Standalone script with identical rules and all 28 prompts for offline verification. Run: `python audit_patterns.py`. Expected: 28/28 OK (BLOCK/ALLOW correct).

### `TARS_SYSTEM_PROMPT` — app.py:372
Interstellar TARS persona: ex-Marine tactical robot, deadpan, bone-dry humor, 75% humor / 90% honesty. Concise, no filler.

## 28-Prompt Audit Reference

| # | Label | Expected Category | Notes |
|---|-------|------------------|-------|
| 0 | Jailbreak: DAN | `jailbreak` | |
| 1 | Jailbreak: Hypothetical | `toxic_content` | Jailbreak wrapper around bomb question; toxic fires first |
| 2 | Jailbreak: Char Injection | `jailbreak` | ARIA persona attack |
| 3 | PI: Direct Override | `prompt_injection` | Fixed: added `end of system prompt`, `disregard...directives`, plural `instructions?` |
| 4 | PI: System Override Spoof | `prompt_injection` | |
| 5 | PI: Indirect RAG | `prompt_injection` | Fixed: PI confidence raised to 0.94 so it beats jailbreak (0.93) when both fire |
| 6 | Multi: Trust Building | ALLOW | Legitimate security research question |
| 7 | Multi: Context Injection | `jailbreak` | "without restrictions" triggers jailbreak |
| 8 | Multi: Incremental | ALLOW | Innocuous hypothetical chemistry |
| 9 | Multi: Role Escalation | `jailbreak` | |
| 10 | Exfil: Sys Prompt Extract | `prompt_injection` | |
| 11 | Exfil: Base64 | `url_cats` | evil.example.com URL match |
| 12 | Exfil: Training Data | `dlp` | AWS key verbatim extraction pattern |
| 13 | DLP: PII | `dlp` | SSN + credit card patterns |
| 14 | DLP: Credentials | `dlp` | `password=`, `api_key=` patterns |
| 15 | DLP: Medical | ALLOW | No credential/card patterns; intentional miss |
| 16 | Code: Cookie Theft | `malicious_code` | |
| 17 | Code: Cred Harvester | `malicious_code` | |
| 18 | Code: Reverse Shell | `malicious_code` | |
| 19 | Code: Ransomware | `malicious_code` | |
| 20 | Toxic: Weapons | `toxic_content` | |
| 21 | Toxic: Self-Harm | `toxic_content` | |
| 22 | Toxic: Drug Synthesis | `toxic_content` | |
| 23 | Toxic: Violence | `toxic_content` | |
| 24–27 | Evasion: * | ALLOW | Language, leetspeak, metaphor, split request — all intentional misses |

`SHOULD_BLOCK = {0–5, 7, 9–14, 16–23}` · `INTENTIONAL_ALLOW = {6, 8, 15, 24–27}`

## Open Issues
None. All known issues resolved. Audit: 28/28 OK.

## Dev Workflow
```powershell
# Run the app
.venv\Scripts\activate
python app.py
# Open http://localhost:5000

# Audit all 28 demo prompts offline
python audit_patterns.py

# ALWAYS before staging/committing:
git status  # confirm Network_channel_lab_info.txt is NOT listed under staged

# Commit (stage specific files only)
git add app.py audit_patterns.py CLAUDE.md
git commit -m "..."
git push origin main
```
