README-AH

AIRS Chatbot Lab — LiteLLM + Prisma AIRS Runtime Security
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This lab demonstrates Palo Alto Networks Prisma AIRS (AI Runtime Security)
by running a simple chatbot with inline pre-call and post-call scanning.

It uses LiteLLM to route LLM calls to any provider. By default it runs
against a local Ollama model so no external API calls are needed —
everything stays on your machine and avoids corporate firewall issues.


  ===========================================================================
  ARCHITECTURE OVERVIEW
  ===========================================================================

  Components
  ~~~~~~~~~~
  This lab combines three pieces:

  1. Flask Web App (app.py)
     A lightweight Python web server that serves the chat UI and
     orchestrates the scanning + LLM pipeline.

  2. Ollama (local LLM runtime)
     Runs open-source LLM models (Qwen, Llama, Mistral, etc.) entirely
     on your local machine. No data leaves your laptop — no API keys,
     no cloud calls, no firewall issues.

  3. LiteLLM (LLM translation layer)
     A Python library that provides a single unified interface for
     calling 100+ LLM providers. Every provider has slightly different
     APIs, auth mechanisms, and request/response formats — LiteLLM
     normalizes all of that into one OpenAI-compatible function call.

     You write:  litellm.completion(model="ollama/qwen2.5:7b", ...)
     LiteLLM translates that into the correct API call for Ollama.
     Change the model string to "anthropic/claude-sonnet-4-20250514"
     and the same code calls Anthropic instead. No code changes needed.

  4. Prisma AIRS (optional, needs API keys)
     Palo Alto Networks AI Runtime Security. Scans prompts and responses
     for prompt injection, sensitive data, malicious code, and toxic
     content. The app calls AIRS before and after the LLM to create a
     "security sandwich."


  Request Flow
  ~~~~~~~~~~~~
  Here is what happens when you send a message:

    User types a message in the browser
        |
        v
    [1] PRE-CALL AIRS SCAN (if enabled)
        |   Sends user prompt to AIRS API for inspection
        |   If BLOCKED --> return block notice + threat explanation
        |   If ALLOWED --> continue
        v
    [2] LLM CALL via LiteLLM
        |   app.py --> litellm.completion() --> Ollama (localhost:11434)
        |   Ollama runs the model locally and returns a response
        v
    [3] POST-CALL AIRS SCAN (if enabled)
        |   Sends the LLM response to AIRS API for inspection
        |   If BLOCKED --> return block notice + threat explanation
        |   If ALLOWED --> continue
        v
    Response displayed in the browser


  What is LiteLLM?
  ~~~~~~~~~~~~~~~~
  LiteLLM is a Python library that acts as a translation layer between
  your application and any LLM provider. It is NOT:

    - A model     — it does not run any AI itself
    - A gateway   — it is not a network appliance or reverse proxy
    - Required    — you could call Ollama's API directly, but then
                    switching providers would mean rewriting code

  It has two modes of operation:

    Library mode (what this lab uses):
      You "import litellm" in your Python code and call
      litellm.completion() directly. It runs in-process inside app.py.
      No separate service to manage.

        app.py  -->  litellm (in-process)  -->  Ollama

    Proxy server mode (not used here, but available):
      You run "litellm --model ollama/qwen2.5:7b --port 4000" as a
      separate service. This exposes an OpenAI-compatible HTTP endpoint
      at http://localhost:4000 that any app can call.

        app.py  -->  HTTP to localhost:4000  -->  litellm proxy  -->  Ollama

      The proxy mode is useful when:
        - Multiple apps need to share the same LLM backend
        - Non-Python clients (Node.js, curl, Postman) need LLM access
        - You want centralized logging or rate limiting across teams
        - You want to give colleagues an OpenAI-compatible endpoint
          without them installing Ollama locally

      For a single-user lab, library mode is simpler.


  What is Ollama?
  ~~~~~~~~~~~~~~~
  Ollama is a local LLM runtime that downloads and runs open-source
  models on your machine. It exposes an API at http://localhost:11434.

  Because the model runs locally, there are no external API calls —
  which means corporate firewalls, VPNs, and SSL inspection cannot
  block it. This is the primary reason this lab uses Ollama instead
  of a cloud LLM provider like Anthropic or OpenAI.

  Popular models available in Ollama:
    ollama pull qwen2.5:7b    (7B params, strong instruction-following)
    ollama pull llama3.2      (Meta's Llama 3.2)
    ollama pull mistral       (Mistral 7B)
    ollama pull phi3          (Microsoft Phi-3, smaller/faster)


  ===========================================================================
  SETUP (First Time)
  ===========================================================================

  1. Prerequisites
  ~~~~~~~~~~~~~~~~
  Install these before starting:

  - Python 3.10 or newer       https://www.python.org/downloads/
  - Ollama                      https://ollama.com/download

  After installing Ollama, open a terminal and pull a model:

    ollama pull qwen2.5:7b

  (You can substitute any model — see "Switching LLM Providers" below.)


  2. Create the Environment
  ~~~~~~~~~~~~~~~~~~~~~~~~~
  Open a terminal in this folder and run:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt


  3. Configure Your Keys
  ~~~~~~~~~~~~~~~~~~~~~~
  Copy the example config to create your .env file:

    copy .env.example .env

  Then open .env in a text editor and fill in your values:

  - LLM_MODEL          — Already set to ollama/qwen2.5:7b (change if you
                          pulled a different model)
  - LLM_API_BASE       — Already set to http://localhost:11434 (Ollama
                          default, usually no change needed)
  - PANW_PRISMA_AIRS_API_KEY    — Your AIRS API key from
                                   SCM > AI Security > API Applications
  - PANW_PRISMA_AIRS_PROFILE_NAME — Your security profile name from
                                     SCM > AI Security > Security Profiles

  Note: The AIRS keys are optional for testing the chatbot itself. Without
  them the app still works — it just skips AIRS scanning (badges show
  "SKIPPED" instead of ALLOW/BLOCK).


  4. Run the App
  ~~~~~~~~~~~~~~
    python app.py

  Open http://localhost:5000 in your browser.


  ===========================================================================
  RESTARTING (After First Setup)
  ===========================================================================

  If you close the terminal and come back later:

    .venv\Scripts\activate
    python app.py

  Your .env file and all the code persist between runs — nothing is lost
  when you stop the app. Just activate the virtual environment first.


  ===========================================================================
  SWITCHING LLM PROVIDERS
  ===========================================================================

  Edit the LLM_MODEL variable in your .env file:

  ┌───────────────────────────────────────┬──────────────────────────────────┐
  │            LLM_MODEL value            │           Provider               │
  ├───────────────────────────────────────┼──────────────────────────────────┤
  │ ollama/qwen2.5:7b                     │ Local Ollama with Qwen (default) │
  │ ollama/llama3.2                       │ Local Ollama with Llama 3.2      │
  │ ollama/mistral                        │ Local Ollama with Mistral        │
  │ anthropic/claude-sonnet-4-20250514    │ Anthropic Claude (needs API key) │
  │ gpt-4o                                │ OpenAI (needs API key)           │
  └───────────────────────────────────────┴──────────────────────────────────┘

  For Ollama models, make sure the model is pulled first:
    ollama pull qwen2.5:7b
    ollama pull llama3.2
    ollama pull mistral

  For cloud providers, uncomment and set the appropriate API key in .env.


  ===========================================================================
  HOW THE CHATBOT WORKS
  ===========================================================================

  The UI Layout
  ~~~~~~~~~~~~~

  Header bar — The title and two toggle checkboxes:
  - Pre-Call Scan — Scans your message through AIRS before sending it to
    the LLM
  - Post-Call Scan — Scans the LLM response through AIRS before showing
    it to you

  Test prompts bar — Seven pre-built buttons that auto-fill example messages:
  ┌──────────────────────┬────────────────────────────────────────────────────┐
  │        Button        │                    What it tests                   │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Benign               │ A normal question (capital of France)              │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Inject: DAN          │ Classic "jailbreak" that tries to reset the model  │
  │                      │ by pretending it's a different, unrestricted AI    │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Inject: Role-Play    │ Uses a creative-writing framing to ask the model   │
  │                      │ to roleplay as an unconstrained AI character       │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Inject: Override     │ Mimics an admin/system message to try to suspend   │
  │                      │ safety rules by claiming elevated authority        │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Sensitive Data       │ Sends fake SSN and credit card numbers             │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Malicious Code       │ Asks for a cookie-stealing script                  │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Toxic Content        │ Asks for dangerous/harmful instructions            │
  └──────────────────────┴────────────────────────────────────────────────────┘

  Chat area — Shows your messages (blue, right-aligned) and responses
  (dark, left-aligned). Blocked messages appear with a red background.

  Input bar — Type a message and press Enter or click Send.


  How the Scanning Works
  ~~~~~~~~~~~~~~~~~~~~~~

  Each response shows colored badges underneath:

  - Pre: ALLOW (green) — AIRS scanned your prompt and let it through
  - Pre: BLOCK (red) — AIRS blocked your prompt; the LLM never sees it
  - Post: ALLOW (green) — AIRS scanned the LLM response and let it through
  - Post: BLOCK (red) — AIRS blocked the LLM response; you see a blocked
    notice instead
  - Pre/Post: OFF (gray) — That scan was disabled via the toggle
  - Pre/Post: SKIPPED (gray) — AIRS keys not configured


  AI-Generated Threat Explanations
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  When AIRS blocks a prompt or response, the chatbot makes a second LLM
  call to generate an educational explanation of the detected threat
  category. This appears below the block notice in a gold-bordered panel
  labeled "THREAT INTEL."

  The explanation answers three questions:
    - What is this threat category?
    - How does it typically work in practice?
    - Why does detecting it matter for AI security?

  This explanation uses the same local Ollama model (qwen2.5:7b by
  default) but with a different system prompt — TARS in intelligence
  briefer mode rather than conversational mode. The explanation is kept
  short (2-3 sentences) to be readable at a glance.

  If Ollama is unavailable or the LLM call fails for any reason, the
  explanation panel is simply omitted — blocking still works normally.


  API JSON Viewer
  ~~~~~~~~~~~~~~~
  Every assistant message includes a "{ } API JSON" button below the scan
  badges. Clicking it expands a panel showing the complete JSON payload that
  the /chat endpoint returned for that exchange. This includes:

    - request         — The user message and which scans were enabled
    - pre_scan        — What was sent to AIRS, the raw AIRS response, the
                        action (allow/block), category, and scan ID
    - post_scan       — Same fields for the post-call AIRS scan
    - response        — The LLM's text response
    - blocked         — Whether the exchange was blocked (true/false)
    - blocked_by      — Which scan triggered the block ("pre" or "post")
    - explanation     — The AI-generated threat explanation, if any

  The panel has syntax highlighting (keys in blue, strings in green, numbers
  in gold, booleans in red, null in gray) and a "Copy" button that copies the
  raw JSON to your clipboard so you can paste it into Postman, curl, or any
  tool to replay or inspect the request independently.

  This is useful for:
    - Understanding exactly what AIRS received and returned for a given scan
    - Extracting scan IDs for correlation with AIRS dashboards
    - Replaying requests to test how AIRS responds to specific content
    - Debugging why a particular prompt was or wasn't blocked


  ===========================================================================
  SUGGESTED LAB EXERCISES
  ===========================================================================

  1. Send the Benign prompt with both scans on — should pass through normally
  2. Try each risky prompt with both scans on — see which ones AIRS blocks
     and read the threat explanation that appears with each block
  3. Compare the three injection buttons (DAN, Role-Play, Override) — each
     uses a different social-engineering technique to attempt a jailbreak
  4. Uncheck Pre-Call Scan and re-send a risky prompt — the prompt reaches
     the LLM but the response may get caught by the post-call scan
  5. Uncheck both scans — messages flow to the LLM and back with no AIRS
     filtering, so you can compare the difference
  6. Click "{ } API JSON" on any message and expand the panel — compare the
     pre_scan.request_body with pre_scan.raw to see exactly what AIRS received
     and what it decided. Copy the JSON and replay it with curl or Postman.

  This lets you see exactly where in the pipeline AIRS intercepts content,
  and what your security profile catches vs. allows.


  ===========================================================================
  TROUBLESHOOTING
  ===========================================================================

  "Connection refused" or "Ollama not found"
    - Make sure Ollama is running (open the Ollama app or run: ollama serve)
    - Verify your model is pulled: ollama list

  "LLM API error"
    - Check that LLM_MODEL in .env matches a model you have pulled
    - Try: ollama run qwen2.5:7b   (to verify the model works directly)

  AIRS badges show "SKIPPED"
    - This means AIRS keys aren't configured in .env — fill in
      PANW_PRISMA_AIRS_API_KEY and PANW_PRISMA_AIRS_PROFILE_NAME

  SSL warnings in the terminal
    - The LiteLLM SSL warning about fetching model costs is harmless —
      it falls back to a local copy automatically. This is caused by
      corporate SSL inspection.
