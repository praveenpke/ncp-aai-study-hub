# Hands-On Labs & Platform Setup

> **What this page is.** The other domain notes teach the *concepts*; this page is the **runnable onboarding track** — the minimum NVIDIA path from zero to a guarded, deployable agent, each step with a verified command and an expected result. The exam assumes you have *actually touched* build.nvidia.com, `nat run`, and `nemoguardrails chat`; this is the fastest way to earn that.
>
> **Core principle: hosted first, local second, self-hosted last.** Every numbered step below runs on a **CPU laptop with an internet connection and one API key**. GPUs, Docker, NGC, and Kubernetes only appear in the clearly-marked *optional* steps (5) and the *study path* of step (6). Do not let "I don't have an H100" stop you — 90% of the platform is reachable from a hosted API key.
>
> **Version honesty.** The NeMo Agent Toolkit moves fast (latest is **v1.7**; package extras and field names shift between minor versions). Where a fact is version-dependent it is marked. Always re-verify exact extra names against [docs.nvidia.com/nemo/agent-toolkit](https://docs.nvidia.com/nemo/agent-toolkit/latest/) before installing.

---

## 1. Why this matters (exam + building)

NVIDIA's certification is deliberately *practitioner-shaped*: the question stems assume you know which command produces which artifact, what an `nvapi-` key looks like, why `integrate.api.nvidia.com` is the API host but `build.nvidia.com` is the web UI, and what the difference is between the **hosted** NIM endpoint and a **self-hosted** NIM container that serves the identical API. Those are not concepts you can fully absorb by reading — they're muscle memory you get by running the track once.

The track also de-risks every other domain's Code Companion. Domain 2's NAT snippets, Domain 3's `nat eval`, Domain 5/6's RAG, Domain 9's Guardrails — all of them assume `NVIDIA_API_KEY` is set and `nvidia-nat` / `nemoguardrails` import cleanly. **Do steps 1-4 once and every other page becomes copy-paste runnable.**

**The six-step minimum path (mental model — a pipeline of flowing nodes):**

```
(1) Account + API key  →  (2) First NIM call  →  (3) First NAT workflow
                                                        ↓
        (6) Blueprint  ←  (5) [opt] self-hosted NIM  ←  (4) First Guardrails run
            walkthrough        (GPU/Docker)               (Colang rail)
                ↓
          deploy / monitor
```

Steps 1-4 are mandatory and CPU-only. Step 5 is optional and needs a GPU. Step 6 is "study + partial reproduction" on CPU, with optional full deployment on GPU.

---

## 2. Mental model — the onboarding pipeline

**Analogy: getting badged into a building, then commissioning a production line.** Step 1 is **getting your badge** (account + key) — without it nothing else swipes. Step 2 is **swiping the badge on the simplest door** (one chat completion) to prove it works and to learn the building's address (`integrate.api.nvidia.com/v1`). Step 3 is **wiring up your first machine on the line** (a NAT agent: a model + a tool + a YAML that says how they connect). Step 4 is **bolting on the safety guard** (a Guardrails rail that inspects what goes in and comes out). Step 5 is the optional move of **bringing the machine in-house** instead of renting it (self-hosted NIM — same API, your GPU). Step 6 is **studying the reference assembly line NVIDIA published** (a Blueprint) and rebuilding a small version of it from the parts you already have.

The deep insight that ties it together — and a favorite exam point — is **API stability**: the OpenAI-compatible request you send in step 2 to the hosted endpoint is *byte-for-byte the same* request you'd send to a self-hosted NIM in step 5. You change one thing: the `base_url`. Write once, deploy anywhere.

---

## 3. The numbered track

Each step lists: **goal · one verified command/snippet · expected result · the artifact it produces**. Run them in order.

### Step 1 — Make a build.nvidia.com account + API key

**Goal:** a working `NVIDIA_API_KEY` exported in your shell.

1. Go to **[build.nvidia.com](https://build.nvidia.com)** (the NVIDIA API Catalog — the hub for hosted NIM models). Sign in with any existing NVIDIA account (NGC, GeForce) or create one and verify the email.
2. Open any model card, e.g. **`meta/llama-3.3-70b-instruct`**, and click **Get API Key** (a.k.a. "Build with this NIM" / "Generate Key"). Agree to terms.
3. Copy the key immediately — it is shown once. It looks like `nvapi-xxxxxxxx…`.
4. Store it as an environment variable (never commit it; add `.env` to `.gitignore`):

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"      # add to ~/.bashrc or ~/.zshrc, then: source it
echo "$NVIDIA_API_KEY"                            # must print your key, starting nvapi-
```

**Expected result:** `echo $NVIDIA_API_KEY` prints `nvapi-…` (not blank, no trailing newline/space).
**Artifact:** the exported key. **Hardware:** none. **Free tier:** new accounts get trial credits; staying within the labs is effectively free.

> **Two different keys, do not confuse them.** The **`NVIDIA_API_KEY`** from build.nvidia.com authenticates *hosted API* calls. An **NGC API key** (from [ngc.nvidia.com](https://ngc.nvidia.com)) authenticates *pulling NIM containers* from `nvcr.io` — only needed for the optional self-hosted step 5. Different keys, different jobs.

> **Which model to pick (June 2026).** These labs default to **`meta/llama-3.3-70b-instruct`** because it is stable, OpenAI-tool-calling-capable, and present on the hosted catalog — the least likely to break a copy-paste command. Just know the *current NVIDIA headline family is **Nemotron 3*** (debuted Dec 2025, expanded through H1 2026, showcased at Computex June 2026): **Nano** (~30B total / ~3B active, e.g. `nvidia/nemotron-3-nano-30b-a3b`), **Super** (~100B/10B), **Ultra** (~500–550B/50–55B, NVIDIA's largest open model, positioned for reasoning/planning/agentic), plus **Nemotron 3 Nano Omni** (multimodal). Any of them is a drop-in swap for the labs — change only `model_name` / `model`. Don't hard-code "best model"; pick by capability lane.
>
> **Catalog id vs Hugging Face repo — don't confuse them.** The string that goes in the `model` field is the **hosted-catalog id** — lowercase, e.g. `nvidia/nemotron-3-nano-30b-a3b`. On Hugging Face the *weight repos* carry a different, longer name with a precision suffix — `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` / `-FP8` / `-NVFP4`. Use the lowercase catalog id for hosted API calls; the HF repo names only matter when you download weights to self-host. Pasting an HF repo name into a hosted call → 404 "model not found."

### Step 2 — First NIM API call

**Goal:** prove the key works and learn the endpoint shape. NIM endpoints are **OpenAI-API compatible** — the single most leverageable fact on the platform.

The verified base URL is `https://integrate.api.nvidia.com/v1`. The chat path is `/chat/completions`.

```bash
curl -s https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -d '{
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [{"role":"user","content":"What is NVIDIA NIM in one sentence?"}],
    "max_tokens": 80, "temperature": 0.2
  }'
```

The same call from Python, using the **`openai` client** (proves drop-in compatibility — only `base_url` + `api_key` change vs. talking to OpenAI):

```python
import os
from openai import OpenAI

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",   # the ONE NVIDIA-specific line
                api_key=os.environ["NVIDIA_API_KEY"])
r = client.chat.completions.create(
    model="meta/llama-3.3-70b-instruct",
    messages=[{"role":"user","content":"What is NVIDIA NIM in one sentence?"}],
    max_tokens=80, temperature=0.2)
print(r.choices[0].message.content)
print("finish:", r.choices[0].finish_reason, "| tokens:", r.usage.total_tokens)
```

**Expected result:** a JSON chat completion with `choices[0].message.content` (a one-sentence answer), `finish_reason: "stop"`, and a `usage` token count. `finish_reason: "length"` instead means you hit `max_tokens` (truncated).
**Artifact:** a saved response + token count (this is Milestone 1). **Hardware:** none.

> **Key field meanings (exam-relevant):** `finish_reason` = `stop` (natural end) vs `length` (hit `max_tokens`); `usage.total_tokens` = `prompt + completion` = **what you're billed for**; `model` in the response confirms which model actually served you. The `langchain-nvidia-ai-endpoints` package's `ChatNVIDIA` wraps the same endpoint for LangChain/RAG/Guardrails use and reads `NVIDIA_API_KEY` automatically.

### Step 3 — First NAT workflow (YAML + `nat run`)

**Goal:** a NeMo Agent Toolkit agent that calls a tool, defined entirely in YAML.

Install the toolkit (the import name is **`nat`**; the PyPI package is **`nvidia-nat`**; the CLI is **`nat`**). NAT v1.7 requires **Python 3.11, 3.12, or 3.13**:

```bash
python -m venv nat-env && source nat-env/bin/activate   # Windows: nat-env\Scripts\activate
pip install nvidia-nat
nat --version            # logs the help banner + version (e.g. 1.7.x). Verifies the install.
```

Create `workflow.yml`. Note the **exact current schema** (this is where the most exam-trap detail lives — see §5): every block keys an entity by name; entities declare their kind with **`_type`**; the LLM uses **`_type: nim`** with the field **`model_name`**; the agent goes under **`workflow:`** with **`_type: tool_calling_agent`**, lists tools by **`tool_names`**, and names its model with **`llm_name`**.

```yaml
# workflow.yml — a minimal NAT tool-calling agent on a hosted NIM
functions:
  current_datetime:                 # NAT ships built-in example tools; this returns the time
    _type: current_datetime
llms:
  nim_llm:
    _type: nim                      # NVIDIA NIM provider (reads NVIDIA_API_KEY from env)
    model_name: meta/llama-3.3-70b-instruct
    temperature: 0.0
workflow:
  _type: tool_calling_agent         # native function-calling agent (uses the LLM's tool API)
  tool_names: [current_datetime]    # tools the agent may call, by name
  llm_name: nim_llm                 # which LLM (defined above) it reasons with
  verbose: true                     # log inputs/outputs/intermediate steps
  handle_tool_errors: true          # default: a tool exception becomes a ToolMessage; agent self-corrects
  max_iterations: 15                # default cap on tool-call rounds
```

Run it:

```bash
nat run --config_file workflow.yml --input "What time is it right now?"
```

**Expected result:** the agent recognizes it needs the tool, calls `current_datetime`, and answers in natural language with the time. With `verbose: true` you see the intermediate **tool call → observation → final answer** in the log.
**Artifact:** `workflow.yml` + the trace (Milestone 2). **Hardware:** none (NAT runs locally; inference is hosted).

> **Swap the agent type to feel the difference (Milestone 3).** Change `_type: tool_calling_agent` to **`_type: react_agent`** and re-run. The ReAct agent emits explicit `Thought → Action → Observation` cycles (transparent, prompt-driven), whereas the Tool Calling agent uses the model's *native* function-calling API (more implicit, fewer tokens). Other built-in agent types worth naming: **`reasoning_agent`**, **`rewoo_agent`**, and the multi-agent **router**/orchestration patterns. Comparing 3 agent types on one task — token use, latency, trace length — is exactly Milestone 3.

> **Register your own tool** when a built-in won't do: a Python function decorated with `@register_function(config_type=...)` whose `name=` becomes the `_type` the YAML references. See **Domain 2's Code Companion** for the full `tools.py` + `config.yml` pair — the YAML assembles the agent; the Python only registers parts.

### Step 4 — First NeMo Guardrails run (a Colang rail)

**Goal:** a guardrails config that **blocks off-topic input before the LLM is called** and a refusal you can demonstrate.

NeMo Guardrails has two deployment tiers. **Library mode** (this step, all labs) is `pip install` + your API key, CPU-only. **Microservice mode** (production) is an NGC container, optionally GPU-backed by safety NIMs — not needed here. Install both packages; the second is what lets Guardrails actually reach NVIDIA-hosted models:

```bash
pip install nemoguardrails langchain-nvidia-ai-endpoints
nemoguardrails --version
```

Create the config directory `config/` with two files. **`config/config.yml`:**

```yaml
colang_version: "2.x"             # use Colang 2 flow syntax in the .co files below
models:
  - type: main
    engine: nim                   # NVIDIA-hosted NIM backend (needs langchain-nvidia-ai-endpoints + NVIDIA_API_KEY)
    model: meta/llama-3.3-70b-instruct
rails:
  input:
    flows:
      - check topic               # run this flow on every user message, BEFORE the LLM
```

**`config/topics.co`** (Colang 2 — example-based intent matching):

```colang
define user ask about technology
  "How does a neural network work?"
  "What is NVIDIA NIM?"
  "Explain cloud computing."

define user ask off topic
  "What's a good chocolate cake recipe?"
  "Who won the game last night?"
  "Write me a love poem."

define flow check topic
  user ask off topic
  bot refuse off topic

define bot refuse off topic
  "I can only help with technology questions — programming, AI, and software engineering."
```

Run the interactive chat with your rail active:

```bash
nemoguardrails chat --config config/
# then type an off-topic line, e.g.:  What's a good chocolate cake recipe?
```

**Expected result:** an **on-topic** question ("How does a neural net learn?") passes through to the LLM and gets a real answer; an **off-topic** question is intercepted by the input rail, the LLM is **never called**, and the bot returns the refusal string.
**Artifact:** the `config/` directory + a before/after (guarded vs unguarded) transcript (Milestone 5). **Hardware:** none.

> **Engine name version note.** Current docs use **`engine: nim`**. Older Guardrails versions used **`engine: nvidia_ai_endpoints`** — both reach the same NVIDIA-hosted backend and both require `langchain-nvidia-ai-endpoints`. If `engine: nim` isn't recognized, your version predates the rename; switch to `nvidia_ai_endpoints`.
>
> **Input vs output rails.** Input rails (`rails.input.flows`) run *before* the LLM — cheap, block bad requests, e.g. topic/jailbreak checks. Output rails (`rails.output.flows`) run *after* the LLM on its response — e.g. PII redaction, hallucination/groundedness checks. NVIDIA ships hardened safety NIMs you can name as extra `models` here: **Llama 3.1 NemoGuard 8B TopicControl**, **content-safety / Safety Guard**, and **JailbreakDetect**. See **Domain 9's Code Companion** for output rails and the full rail taxonomy.

### Step 5 — [Optional] Self-hosted NIM

> **Entirely optional. Requires a GPU.** No lab needs this. Do it only if you have the hardware or want to understand the production self-hosting path (Domain 4/10). The payoff concept: **the API is identical to step 2** — only the host changes.

Requirements: Linux, NVIDIA driver **535+** (`nvidia-smi`), Docker **24+**, NVIDIA Container Toolkit, and an **NGC API key** (separate from `NVIDIA_API_KEY`).

```bash
# 1) Authenticate to the NGC container registry (username is the LITERAL string $oauthtoken)
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

# 2) Pull + run a small NIM (8B fits modest GPUs). Persist the model cache across restarts.
docker run -d --name nim-llm --gpus all -p 8000:8000 \
  -v nim-cache:/opt/nim/.cache \
  -e NGC_API_KEY="$NGC_API_KEY" \
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest

# 3) Wait for readiness (first start loads weights into GPU memory — minutes)
curl -s http://localhost:8000/v1/health/ready        # -> {"status":"ready"} (or HTTP 200) when up
```

Then call it with **exactly the step-2 code, one line changed:**

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed-locally")
# ...identical create() call as step 2...
```

**Expected result:** `/v1/health/ready` returns ready, and `/v1/chat/completions` on `localhost:8000` returns the same OpenAI-shaped JSON as the hosted endpoint.
**Artifact:** a running local NIM + a response proving API parity. **Hardware:** GPU (≈1× A100/H100-class for small models; 2× H100 for 70B). **Cleanup:** `docker stop nim-llm && docker rm nim-llm`; the `-v nim-cache` volume keeps weights for next time.

> **Scaling beyond one container** is the **NIM Operator** for Kubernetes (a `NIMService` custom resource with `replicas`/autoscaling) — Domain 10 territory; architectural awareness only.

### Step 6 — A Blueprint walkthrough (AI-Q Research Assistant)

**Goal:** read an NVIDIA reference architecture, then reproduce a *minimal* version of its RAG pipeline on CPU using hosted endpoints.

**Blueprints** are production-grade *reference architectures* (not tutorials) at **[build.nvidia.com/blueprints](https://build.nvidia.com/blueprints)**. The **AI-Q Research Assistant** is an enterprise research agent built on the **NeMo Agent Toolkit** (with LangChain Deep Agents): an orchestrator classifies intent, a shallow agent gives quick cited answers, a deep agent writes long-form cited reports. Its repo is **[github.com/NVIDIA-AI-Blueprints/aiq](https://github.com/NVIDIA-AI-Blueprints/aiq)** (build card: build.nvidia.com/nvidia/aiq). As of June 2026 the Blueprint is on the **v2.x** line and its *default* models have moved to the **Nemotron 3** family — `nvidia/nemotron-3-nano-30b-a3b` for the researcher/classifier (Nemotron 3 Super optional) — with the renamed NeMo Retriever embedder **`nvidia/llama-nemotron-embed-vl-1b-v2`** (the `llama-3.2-nv-embedqa/-nemoretriever` names were rebranded to the Nemotron line in NeMo Retriever NIM 1.13.0). The CPU reproduction below deliberately keeps the simpler, still-live `meta/llama-3.3-70b-instruct` + `nv-embedqa-e5-v5` so it runs anywhere — the architecture lesson is identical.

Most learners **cannot** run the full stack (it wants ~2× H100 to serve LLM + embedding + reranker NIMs at once). That's fine — the learning is in the **architecture map** and a **partial reproduction**. Map its components and which are NVIDIA-specific vs generic:

| Component | Role | In the full Blueprint | Your CPU reproduction (hosted API) |
|---|---|---|---|
| LLM | reasoning / generation | NIM on GPU (v2.x default: **Nemotron 3 Nano 30B**) | NIM hosted API (`integrate.api.nvidia.com`) |
| Embedding | text → vectors | NIM on GPU (**`llama-nemotron-embed-vl-1b-v2`**, renamed in Retriever NIM 1.13.0) | NIM hosted: **`nvidia/nv-embedqa-e5-v5`** (1024-dim) |
| Reranker | re-score passages | NIM on GPU (**`llama-nemotron-rerank-1b-v2`**) | NIM hosted: **`nvidia/llama-nemotron-rerank-1b-v2`** (the older `nv-rerankqa-mistral-4b-v3` is now *deprecated* on the catalog) |
| Vector store | similarity search | **Milvus** (containerized) | **FAISS** (`faiss-cpu`, in-memory) |
| Doc parsing | PDF/table extraction | **NeMo Retriever** | PyPDF / hardcoded text |
| Orchestration | the agent | **NeMo Agent Toolkit** | a Python script |

Minimal reproduction (embed → FAISS retrieve → NIM generate, all CPU):

```python
import os, numpy as np, faiss
from openai import OpenAI

c = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])
docs = ["NVIDIA NIM serves optimized inference microservices.",
        "RAG grounds LLM answers in retrieved documents to reduce hallucination.",
        "Milvus is an open-source vector database; FAISS is a local CPU alternative."]

emb = lambda texts: np.array([d.embedding for d in
        c.embeddings.create(model="nvidia/nv-embedqa-e5-v5", input=texts).data], dtype="float32")
X = emb(docs); faiss.normalize_L2(X)
idx = faiss.IndexFlatIP(X.shape[1]); idx.add(X)          # cosine via inner product on normalized vecs

q = emb(["How does RAG reduce hallucination?"]); faiss.normalize_L2(q)
top = [docs[i] for i in idx.search(q, 2)[1][0]]          # retrieve top-2
ans = c.chat.completions.create(model="meta/llama-3.3-70b-instruct",
        messages=[{"role":"system","content":"Answer ONLY from the context; cite it."},
                  {"role":"user","content":f"Context:\n{chr(10).join(top)}\n\nQ: How does RAG reduce hallucination?"}])
print(ans.choices[0].message.content)
```

**Expected result:** the embedding call reports a 1024-dim vector per doc, FAISS retrieves the RAG/grounding docs as top hits, and the LLM produces a grounded, citation-style answer. The architecture comparison table above is Milestone 6.
**Artifact:** the reproduction script + an architecture/gap-analysis writeup. **Hardware:** none for the reproduction; GPUs only for full deployment.

> **The exam trap to avoid:** *don't* assume everything in a Blueprint is NVIDIA-proprietary. The NVIDIA-specific layer is **NIM** (inference) and **NeMo Retriever** (doc processing); **Milvus** and **FAISS** are open source and swappable. And a Blueprint is a *reference implementation* to study and adapt — not a step-by-step tutorial.

---

## 4. Feature → install matrix

What to `pip install` for each thing you'll build. The base `nvidia-nat` package is **not** enough for eval/profiling/integrations — those need extras (which are version-dependent; verify names against current docs). For NAT, an extra `nvidia-nat[xyz]` is equivalent to the standalone distribution `nvidia-nat-xyz`.

| Feature | Install | Credentials | Hardware | Where it's used |
|---|---|---|---|---|
| **NIM hosted call** (OpenAI client) | `pip install openai` | `NVIDIA_API_KEY` | CPU | Step 2; every domain |
| **LangChain ↔ NVIDIA endpoints** | `pip install langchain-nvidia-ai-endpoints` | `NVIDIA_API_KEY` | CPU | RAG, Guardrails, D2/D5/D6 |
| **NAT core** (agents, YAML, functions) | `pip install nvidia-nat` | `NVIDIA_API_KEY` | CPU | Step 3; D2 |
| **NAT + LangChain** | `pip install nvidia-nat[langchain]` *(verify name)* | `NVIDIA_API_KEY` | CPU | D2/D6 |
| **NAT evaluation** (ragas, trajectory, red-team) | `pip install nvidia-nat[eval]` *(or `nvidia-nat-eval`)* | `NVIDIA_API_KEY` | CPU | D3 |
| **NAT profiler / sizing calc** | `pip install nvidia-nat[profiler]` *(or `nvidia-nat-profiler`)* | `NVIDIA_API_KEY` | CPU | D3/D4 |
| **NAT observability (Phoenix)** | `pip install nvidia-nat arize-phoenix` | `NVIDIA_API_KEY` | CPU | D8 |
| **NeMo Guardrails — library mode** | `pip install nemoguardrails langchain-nvidia-ai-endpoints` | `NVIDIA_API_KEY` | CPU | Step 4; D9 |
| **NeMo Guardrails — server mode** | `pip install nemoguardrails[server]` | `NVIDIA_API_KEY` | CPU | D9/D10 (HTTP API) |
| **Vector store — FAISS** | `pip install faiss-cpu` | none | CPU | Step 6; D6 |
| **Vector store — Milvus** | `pip install pymilvus` + Milvus via Docker | none | Docker (GPU for cuVS) | D6 (advanced) |
| **Memory persistence — Redis** | `pip install redis` + `docker run -p 6379:6379 redis` | none | Docker | D2/D5 (memory) |
| **Self-hosted NIM** | `docker pull nvcr.io/nim/...` | **NGC API key** | **GPU** | Step 5; D4/D10 |

> **`langchain-nvidia-ai-endpoints` is the silent dependency.** Guardrails' `engine: nim` config *loads fine* without it, then fails the instant it tries to reach the model. If a guardrails run dies with "provider not found" / an import error, this is almost always why.

---

## 5. Exam traps & gotchas

1. **`build.nvidia.com` is the web UI; `integrate.api.nvidia.com/v1` is the API host.** Pointing curl at `build.nvidia.com` → 404 / HTML. The API base is always `https://integrate.api.nvidia.com/v1` (then `/chat/completions`, `/embeddings`, etc.).

2. **Two keys: `NVIDIA_API_KEY` ≠ NGC API key.** The build.nvidia.com key authenticates *hosted API calls*. The NGC key (username literally `$oauthtoken`) authenticates *pulling containers from `nvcr.io`*. Mixing them up is the classic step-5 failure.

3. **NAT YAML field names are exact and have changed across versions.** Current schema: LLM is **`_type: nim`** with **`model_name`** (not `type:`/`model:`); the agent lives under **`workflow:`** with **`_type: tool_calling_agent`**, **`tool_names:`** (a list), and **`llm_name:`**. There is no `agents:`/`entry_agent:` block in current NAT. Validate parsing with `python -c "import yaml,sys;yaml.safe_load(open('workflow.yml'))"`.

4. **YAML is indentation-sensitive — spaces only, never tabs.** A file that parses but mis-nests (e.g. `tool_names` not a list, empty `system`/instructions block) gives an agent that "has no tools" at runtime. Dump the parsed structure to check it.

5. **Model names are exact and namespaced.** `meta/llama-3.3-70b-instruct` — the `meta/` provider prefix and `-instruct` suffix are mandatory; dropping either → 404 "model not found". Always copy the exact id from the model card.

6. **OpenAI-compatibility is the whole point — and a frequent question.** A NIM endpoint speaks the OpenAI Chat Completions API, so the `openai` client works by changing only `base_url` + `api_key`. The *same* request hits hosted (`integrate.api.nvidia.com/v1`) or self-hosted (`localhost:8000/v1`) NIM unchanged.

7. **Guardrails: `engine: nim` (current) vs `engine: nvidia_ai_endpoints` (older).** Same backend; pick by version. Both need `langchain-nvidia-ai-endpoints`. And set `colang_version: "2.x"` or your Colang 2 `.co` flows silently parse as Colang 1.

8. **Input rails run before the LLM, output rails after.** "Block this off-topic question without paying for an LLM call" = **input** rail. "Redact PII from the model's answer" = **output** rail. Knowing which side a rail sits on is a recurring distinction.

9. **`nemoguardrails server` needs the `[server]` extra.** The base package has no FastAPI/uvicorn; `nemoguardrails server` fails with `ModuleNotFoundError: fastapi` until you `pip install nemoguardrails[server]`.

10. **A Blueprint is a reference architecture, not a tutorial, and not all-NVIDIA.** You study and adapt it; you don't expect hand-holding. NIM + NeMo Retriever are the NVIDIA-specific layers; Milvus/FAISS are open source.

11. **`429 Too Many Requests` = free-tier rate limit, not a broken key.** Back off and retry; don't regenerate the key. (`401` = bad/missing key; `404` = wrong model or wrong host.)

---

## 6. The 3 / 7 / 14-day paths (which steps, which milestones)

NVIDIA-aligned study tracks map onto these steps. Use them to scope how far to go.

| Path | Days | Steps covered | Milestones hit | New tooling touched |
|---|---|---|---|---|
| **3-day — "First Contact"** | 1-3 | Steps 1-4 | 1, 2, 3, 5 (4 of 8) | build.nvidia.com, NIM, NAT (3 agent types), Guardrails + Colang 2 |
| **7-day — "Working Developer"** | 1-7 | Steps 1-4 + RAG + NAT eval + Blueprint study | 1-6 (6 of 8) | + NIM embedding/reranking, FAISS, `nat eval`/profiler, AI-Q Blueprint, Phoenix |
| **14-day — "Production-Ready"** | 1-14 | all + multi-agent + deploy + observability | 1-8 (all) | + NAT Router, red-teaming, Docker Compose deploy, dashboards, HITL approval rails |

The **8 mandatory milestones** (each proves *real* platform interaction, not theory): (1) first NIM call, (2) first NAT workflow, (3) agent-pattern comparison, (4) RAG on the NVIDIA stack, (5) Guardrails in action, (6) Blueprint reproduction, (7) observable agent system, (8) integrated capstone. **Days 1-10 of the 14-day path are CPU-only**; Docker enters only for deployment labs; self-hosted NIM and full Blueprint deployment stay clearly optional throughout.

---

## 7. Troubleshooting matrix (symptom → cause → fix)

| # | Symptom | Likely cause | Fix |
|---|---|---|---|
| 1 | `401 Unauthorized` from the API | Key missing / malformed / has whitespace | `echo $NVIDIA_API_KEY` (starts `nvapi-`, no trailing newline); header must be `Bearer <key>`; regenerate at build.nvidia.com if needed |
| 2 | `404 Not Found` from the API | Wrong host or model name | Host is `integrate.api.nvidia.com` (not `build…`); model id exact incl. `meta/` + `-instruct` |
| 3 | `429 Too Many Requests` | Free-tier RPM/TPM limit | Wait ~60 s, slow request rate; check remaining credits — **don't** regenerate the key |
| 4 | New key returns `401` for a few minutes | Key still propagating | Wait 2-5 min; confirm no copy/paste whitespace |
| 5 | `ModuleNotFoundError: No module named 'nat'` | `nvidia-nat` not installed / wrong venv | `pip install nvidia-nat`; check `which python` matches the install; Python 3.11–3.13 |
| 6 | NAT eval/profiler/langchain import fails | Extra not installed | `pip install nvidia-nat[eval]` / `[profiler]` / `[langchain]` (or the `nvidia-nat-*` distribution); verify name in current docs |
| 7 | NAT config "invalid key" / "unknown field" | Old/new field-name mismatch or YAML typo | Use current schema (`_type`, `model_name`, `tool_names`, `llm_name`); validate with `yaml.safe_load` |
| 8 | NAT agent runs but never calls a tool | Tool not registered, or wrong agent type, or prompt doesn't invite tool use | Confirm the `_type` matches the registered function name; ensure agent type supports tool calling; add a tool-use hint |
| 9 | `nemoguardrails chat` hangs / errors on start | model id wrong, key unset, `langchain-nvidia-ai-endpoints` missing, or `colang_version` missing | Verify all four: model on build.nvidia.com, `NVIDIA_API_KEY` exported, the langchain pkg installed, `colang_version: "2.x"` set |
| 10 | `nemoguardrails server` → `No module named 'fastapi'` | Server deps not installed | `pip install nemoguardrails[server]` (or `pip install fastapi uvicorn`) |
| 11 | Colang rail never triggers | Too few/narrow examples or flow-name mismatch | Add more `define user ask off topic` examples; flow name in `config.yml` must match the `.co` flow name exactly |
| 12 | Colang rail blocks *everything* | Off-topic examples too broad / overlap on-topic | Add more on-topic positive examples; make off-topic examples more distinct |
| 13 | Embedding "dimension mismatch" | Index dim ≠ model output dim, or different embed model at index vs query | `nv-embedqa-e5-v5` is **1024**-dim; use the *same* embed model for index and query |
| 14 | FAISS `ModuleNotFoundError` | Wrong package | `pip install faiss-cpu` (not `faiss-gpu` unless you have CUDA) |
| 15 | `docker pull nvcr.io/...` unauthorized | NGC login missing/expired | `echo "$NGC_API_KEY" \| docker login nvcr.io --username '$oauthtoken' --password-stdin` |
| 16 | NIM container OOM / `CUDA out of memory` | Model too big for the GPU | Use a smaller model (7-8B); check `nvidia-smi`; close other GPU procs |
| 17 | Self-hosted NIM runs but inference is minutes-slow | GPU not seen by container, or insufficient VRAM | Verify `nvidia-smi` *inside* the container; confirm NVIDIA Container Toolkit on host; smaller model |
| 18 | `Address already in use` on a server/console | Port 8000/8090/6006 taken | Use another port (`--port`), or stop the stale process/container (`docker stop …`) |
| 19 | Blueprint Docker Compose wants GPUs you lack | Attempting full deployment without the hardware | Use the "study + partial reproduction" path (step 6) — CPU + hosted APIs |

---

## 8. Builder's corner — habits that make this real

- **Set `NVIDIA_API_KEY` once, in your shell profile, and verify with `echo` before every session.** Half of all "it's broken" moments are an unset or stale key in a fresh terminal or a freshly-activated venv.
- **Keep one scratch `workflow.yml`** you can mutate: swap `model_name`, flip `tool_calling_agent`→`react_agent`, add a tool. The fastest way to internalize the NAT schema is to break it and read the error.
- **Save every artifact.** The milestones are artifact-based on purpose: a saved curl response, a `workflow.yml`, a guarded-vs-unguarded transcript, an architecture comparison. They're also the receipts you'd want before sitting the exam.
- **Pin `temperature: 0`** while learning so outputs are reproducible and you can tell a real change from sampling noise.
- **Reach for hosted endpoints first, always.** Self-hosting is a *latency/privacy/cost-at-scale/offline* decision, not a learning prerequisite. If you're reaching for a GPU to do a lab, you've probably over-built.

---

## 9. Cross-links to the Code Companions

Each domain page has a "Code Companion" section with deeper runnable snippets. This track is the on-ramp; those are the destinations:

- **Domain 2 — Agent Development:** the full NAT `tools.py` + `config.yml` (custom function registration, MCP client/server, streaming). The authoritative NAT YAML shape lives here.
- **Domain 3 — Evaluation & Tuning:** `nat eval` config, ragas + trajectory evaluators, the CI eval gate, NAT Profiler + Sizing Calculator, red-teaming evaluator.
- **Domain 5/6 — Cognition / Knowledge Integration:** the full RAG pipeline (chunking, NIM embeddings, reranking, vector store) that step 6 previews.
- **Domain 9 — Safety, Ethics & Compliance:** output rails, PII redaction, the NemoGuard safety NIMs, and microservice-mode Guardrails that step 4 previews.
- **Domain 4 / 10 — Deployment / Run-Monitor-Maintain:** the self-hosted NIM, NIM Operator, and Docker/K8s deployment that step 5 previews.

---

## 10. Sources

- NVIDIA API Catalog (hosted NIM + key) — https://build.nvidia.com · model card example: https://build.nvidia.com/meta/llama-3.3-70b-instruct
- NeMo Agent Toolkit docs (install, run, workflow config, agents, LLMs) — https://docs.nvidia.com/nemo/agent-toolkit/latest/ · GitHub: https://github.com/NVIDIA/NeMo-Agent-Toolkit · PyPI: https://pypi.org/project/nvidia-nat/
- NAT Tool Calling Agent config — https://docs.nvidia.com/nemo/agent-toolkit/latest/components/agents/tool-calling-agent/tool-calling-agent.html
- NeMo Guardrails (model config, rails, Colang 2, safety NIMs) — https://docs.nvidia.com/nemo/guardrails/latest/
- NeMo Retriever embedding / reranking NIMs — https://build.nvidia.com/nvidia/nv-embedqa-e5-v5 · https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/
- NIM for LLMs (self-host: docker, `/v1/health/ready`, NGC login) — https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html · https://ngc.nvidia.com
- AI-Q Research Assistant Blueprint — https://build.nvidia.com/nvidia/aiq · https://github.com/NVIDIA-AI-Blueprints/aiq · catalog: https://build.nvidia.com/blueprints
- Verified against current NVIDIA docs, June 2026. NAT facts cross-checked with the Domain 2 notes' verified config schema.
