# Mock Exam & Practice Questions — NCP-AAI

> **What this page is.** A timed-condition practice set plus the exam-craft you wrap around it. NVIDIA does **not** publish official sample questions, so every item here is *synthesized* from the published domain descriptions, competency statements, and current NVIDIA documentation (NAT/NeMo/NIM, Blueprints). Treat these as *study reps*, not a leaked exam. The point is pattern recognition: map a scenario → the right NVIDIA primitive → the right architectural call, fast.
>
> **Every NVIDIA fact below was web-verified against current docs.** Where the third-party reference we mined was version-stale, this page corrects it and flags it. The two biggest corrections worth internalizing:
> 1. **NeMo Guardrails has five rail types: input, dialog, retrieval, output, execution** — *not* a standalone "topical" rail. **Topic control is a *use* of dialog rails** (or the Topic Control NIM). If an answer choice says "topical rail," it is using loose language for a dialog rail.
> 2. **NAT agent/executor types** as of current docs (1.7) include a **Parallel Executor** alongside Sequential Executor, plus ReAct, Reasoning, ReWOO, Router, and Tool Calling. The Responses-API surface exists but the "exactly seven named types" framing is version-dependent — reason about *behavior*, not a memorized count.

---

## §1 · How the exam is shaped, and how to take it

**The format (verify on the official page before you sit — NVIDIA changes these):** ~60–70 questions, 120 minutes, online remote-proctored, $200, 2-year validity, "Professional" level. That is roughly **~2 minutes per question** — about 100–110 seconds if you want a review pass. You do **not** have time to derive answers from first principles; you need instant scenario→tool→decision mapping. That is what this page drills.

### The 10-domain weight map (memorize the shape, not the decimals)

| # | Domain | Weight | ~Q (of 65) | Risk where notes are thin |
|---|--------|-------:|-----------:|---------------------------|
| 1 | Agent Architecture & Design | 15% | ~10 | agent-type selection traps |
| 2 | Agent Development | 15% | ~10 | NAT YAML, function groups, MCP |
| 3 | Evaluation & Tuning | 13% | ~8–9 | metric-diagnosis, profiler vs eval |
| 4 | Deployment & Scaling | 13% | ~8–9 | server choice, NIM Operator, Dynamo |
| 5 | Cognition, Planning & Memory | 10% | ~6–7 | object stores, ReWOO vs ReAct |
| 6 | Knowledge Integration & Data | 10% | ~6–7 | AI-Q models, chunking, hybrid |
| 7 | NVIDIA Platform Implementation | 7% | ~4–5 | NeMo microservice roles |
| 8 | Run, Monitor & Maintain | 5% | ~3–4 | trace levels, redaction |
| 9 | Safety, Ethics & Compliance | 5% | ~3–4 | 5 rail types, safety NIMs |
| 10 | Human-AI Interaction & Oversight | 5% | ~3–4 | approval gates, escalation |

**Tier reading:** Domains 1+2 are **30%** of the exam — master agent types, YAML, tools, and architecture tradeoffs and you've anchored a third of the test. Domains 3+4 add **26%** (dev→prod). The four 5–7% domains together are **22%** — skipping them can cost you 12–15 questions, which is the gap between pass and fail. "Each is only 5%" is a trap; *two* 5% domains is 10%.

### Timing discipline (worked plan for a 65-question, 120-minute sitting)

- **Budget:** 65 × 110s ≈ 119 min. Hold ~10 minutes back for a review pass → target **~100s/question** on the first pass.
- **First pass:** answer everything you're confident on; **flag-and-move** on anything you'd spend >2 min on. Never sink 4 minutes into one 1.5%-of-score question while three easy ones wait.
- **Second pass:** return to flags. Often a *later* question jogs the concept you needed for an earlier one.
- **Guessing:** there is no negative marking on multiple-choice — **never leave a blank.** Eliminate, then pick.

### The three question *traps* the exam leans on

These are wording traps, not knowledge traps. They flip the answer with a single word.

- **NOT / EXCEPT / LEAST** — you're picking the **wrong / worst** option, the odd one out. Three options will be correct facts; the answer is the false one. *Underline the negative word.* Most "I knew this and still missed it" errors are here.
- **BEST / MOST appropriate / FIRST** — **multiple options work**; you want the one *purpose-built* for the scenario. "Could technically work" loses to "designed for exactly this." On **FIRST** questions (e.g., "error rate jumped 2%→12%, what FIRST?"), the answer is almost always **observe before you act** — read the traces / diagnose, *then* change prompt/model. Changing the model first is the classic wrong answer.
- **Distractor that's true-but-irrelevant** — a statement that is factually correct but doesn't answer *this* question. Re-read the stem; correctness ≠ relevance.

### The NVIDIA-first elimination heuristic

When a "which tool / how would you" question gives you 4 plausible options, eliminate by **layer** and **lifecycle**, then prefer the **most specific NVIDIA primitive**:

1. **Wrong layer?** NIM = *inference serving*. NAT = *orchestration*. They are not interchangeable — "NIM orchestrates the agent" is always wrong; so is "NAT runs the model weights."
2. **Wrong lifecycle phase?** Eval tools (NAT Evaluation Framework, Profiler, Sizing Calculator) are not deployment tools; deployment servers are not safety tools.
3. **Two still standing?** Prefer the option that names a **specific NVIDIA primitive** over a generic best practice. The exam rewards "execution rail" over "add validation logic," "NeMo Curator" over "preprocess the data," "NIM Operator" over "write Kubernetes YAML."
4. **General-knowledge tie-break:** when an inferred best-practice answer competes with a documented NVIDIA feature, choose the documented feature.

**The scenario→tool cheat table** (commit to memory — it answers the rapid-fire "which tool" cluster):

| Scenario | NVIDIA answer |
|----------|---------------|
| Orchestrate the agent / workflow logic | **NAT** (NeMo Agent Toolkit) |
| Serve the LLM (inference) | **NIM** microservice |
| Automate NIM lifecycle on Kubernetes | **NIM Operator** |
| Accelerate/optimize inference (batching, prefix cache, disaggregated serve) | **NVIDIA Dynamo** / TensorRT-LLM |
| Safety rails / content filtering | **NeMo Guardrails** (+ Safety NIMs) |
| Expose an agent as an MCP tool | **NAT MCP / FastMCP server** |
| Distributed multi-agent communication | **NAT A2A server** |
| Human approval before a tool fires | **Guardrails execution rail** (+ NAT WebSocket) |
| GPU-accelerated data prep at corpus scale | **NeMo Curator** |
| GPU-accelerated vector search | **Milvus + cuVS** |
| Trace visualization | **Phoenix / Weave / Langfuse** (via OpenTelemetry) |
| Strip PII from telemetry | **NAT redaction processors** |
| Capacity planning (GPUs for N users) | **NAT Sizing Calculator** |
| Find the latency/token hotspot | **NAT Profiler** |
| Continuous improvement loop on prod logs | **Data Flywheel Blueprint** (Curator + Customizer + Evaluator) |

---

## §2 · Question bank (50 items, weighted to the exam)

Items are distributed in proportion to exam weight: **D1 ×8, D2 ×8, D3 ×6, D4 ×6, D5 ×5, D6 ×5, D7 ×4, D8 ×2, D9 ×3, D10 ×3** = 50. Each item carries the **correct answer**, **why the distractors are wrong** (this is where the learning is), a **domain tag**, and a **one-line concept link** back to the relevant domain page.

> Format note: in the HTML these are click-to-reveal cards. Here, the answer follows each question. Try to answer before reading on.

### Domain 1 — Agent Architecture & Design (×8)

**Q1.** A support system handles 5 distinct request categories, each needing different tools and a different system prompt. You're choosing between (A) one ReAct agent with all tools registered, or (B) a Router agent dispatching to 5 specialized Tool Calling agents. Which, and why?
**→ B (Router + specialists).** One agent holding all tools bloats the prompt (every tool description competes for attention) and tool-selection accuracy degrades past ~8–12 tools. A Router classifies intent **once** (cheap) and dispatches to a focused specialist; each specialist scales and is evaluated independently. *Distractors:* "A is simpler" is true but loses on accuracy at scale (BEST trap); "merge into a Sequential Executor" is wrong because requests are *alternatives*, not a fixed pipeline.
*Concept link: D1 — Router pattern & single-vs-multi-agent.*

**Q2.** You need to minimize total LLM calls on a **predictable** multi-tool task where mid-run adaptation is rarely needed. Which agent type, and what do you give up?
**→ ReWOO.** It plans *all* tool calls up front in one reasoning pass, then executes the plan (executing independent tools in parallel within a dependency level) without re-reasoning between steps — far fewer LLM calls than ReAct's per-step loop. **You give up adaptivity:** if a tool returns something unexpected, the fixed plan can't course-correct. *Distractors:* ReAct minimizes *errors on unpredictable tasks*, not LLM calls; Reasoning agent plans-ahead but still centers a single function; Sequential Executor isn't an LLM-reasoning pattern.
*Concept link: D1/D5 — ReWOO plan-then-execute vs ReAct interleave.*

**Q3.** A research agent queries flaky databases where ~10% of calls fail. ReAct or ReWOO?
**→ ReAct.** Reasoning *after each step* lets it retry, reformulate, or switch sources when a call fails. ReWOO's pre-baked plan breaks on the first unexpected failure with no recovery path. *Distractor:* "ReWOO is cheaper" is the deliberate bait — cheaper-but-brittle loses when 1-in-10 steps fail (BEST trap).
*Concept link: D1 — adaptivity vs efficiency tradeoff.*

**Q4.** Which is the **architectural** difference between a Sequential Executor and a Router agent?
**→ The Sequential Executor runs a fixed ordered pipeline (output of step N → input of N+1, every step always runs, no decision logic). The Router examines input, classifies it, and dispatches to one of several agents (a routing decision).** Use Sequential for "extract→transform→validate→store"; use Router for "this type goes here, that type goes there." *Distractor:* "both make routing decisions" is false — that's the trap; only the Router decides.
*Concept link: D1 — executor vs router.*

**Q5.** The AI-Q Blueprint specifies a two-stage retrieval (embed → rerank). Why is the **rerank** stage worth it when you already have a good embedding model? (This is a BEST/why item.)
**→ Bi-encoder embedding retrieval scores query and document independently — fast over millions of docs but coarse. The reranker is a cross-encoder that processes the query and each candidate *together*, capturing fine-grained interaction the bi-encoder misses.** Two-stage = bi-encoder speed for recall + cross-encoder precision on the top-N. *Distractor:* "rerank replaces the embedder" is wrong — it *follows* it on a small candidate set.
*Concept link: D6 — bi-encoder vs cross-encoder, AI-Q architecture.*

**Q6.** Which of the following is **NOT** a reason to prefer multi-agent over single-agent? (NOT trap.)
Options resemble: (a) tool count is large and tools are easily confused, (b) sub-tasks need independent scaling, (c) sub-tasks need different system prompts/models, (d) **the task is a simple linear pipeline with one tool.**
**→ (d).** A simple single-tool linear task is the textbook *single-agent* (or Sequential Executor) case; multi-agent there is over-engineering and adds coordination cost. (a)(b)(c) are all valid multi-agent drivers.
*Concept link: D1 — when multi-agent earns its complexity.*

**Q7.** Name the four phases of the agent loop and where the LLM "thinks."
**→ Perceive → Reason → Act → Observe**, looping. The LLM does the **Reason** (decide next action) and interprets the **Observe** (tool result) to decide whether to loop or finish. *Distractor:* "the LLM executes the tool" is wrong — the *runtime* executes the tool; the LLM only chooses it and reads the result.
*Concept link: D1 — perceive/reason/act/observe.*

**Q8.** You're picking an LLM size per agent in a multi-agent system: a Router that only classifies intent, and a Code-Analysis agent that finds subtle security bugs. **Best** allocation?
**→ Small model (e.g. 8B) for the Router, large model (e.g. 70B) for Code Analysis.** Classification is easy and high-traffic — a big model there wastes GPU and adds latency. Nuanced reasoning needs the big model. Mixed sizing cuts cost and latency where it's safe. *Distractor:* "use 70B everywhere for consistency" is the trap — uniformity is not the goal; fit-for-purpose is.
*Concept link: D1/D4 — model-size-per-agent.*

### Domain 2 — Agent Development (×8)

**Q9.** In NAT, what is a **Function Group**, and what do **Per-User Functions** add?
**→ A Function Group is a named, reusable bundle of functions (tools) you assign to an agent as a unit instead of listing tools individually.** Per-User Functions make specific tools available based on the **authenticated user** — e.g. an admin gets `delete_record`, a regular user doesn't — enabling role-based tool access without forking the agent config per role. *Distractor:* "function groups are middleware" is wrong (different concept).
*Concept link: D2 — function registry, function groups, per-user functions.*

**Q10.** What problem does NAT's **MCP client** solve that a static tool registry does not?
**→ Dynamic, runtime tool discovery across organizational boundaries.** A static registry binds all tools at build time; an MCP client connects to an MCP server *at runtime*, discovers tools, and invokes them — even tools added after the agent deployed. *Distractor:* "MCP makes tools faster" is irrelevant (true-but-not-the-point trap). Note the inverse direction too: NAT can also be an **MCP/FastMCP server**, publishing its workflow *as* an MCP tool for other agents.
*Concept link: D2/D7 — MCP client vs server.*

**Q11.** Name the main NAT middleware uses (the function pre/post-invoke hook interface) with a one-line use each.
**→** NAT exposes a function-middleware interface (pre/post-invoke hooks) with several documented uses: **Caching** (reuse responses for identical/semantically-similar inputs — FAQ agents), **Defense** (input validation / attack detection before the LLM — prompt-injection screening), **Logging** (structured audit of inputs/outputs/tool-calls/timing — compliance trail), **Red-teaming** (auto-generate adversarial inputs to stress the agent pre-deploy). Reason about each *purpose* rather than memorizing a fixed count — the exact catalog evolves across releases. *Distractor:* "rate-limiting" / "load-balancing" are not the documented agent-middleware uses (generic-infra bait).
*Concept link: D2 — middleware stack.*

**Q12.** A team has a working **LangChain** agent and wants NVIDIA evaluation, guardrails, and production deployment **without rewriting** the agent. Fastest path?
**→ Wrap it with the NAT plugin for LangChain.** The plugin adapts the framework's agent interface into a NAT workflow, so it gains NAT's deployment servers (FastAPI/MCP/A2A), Evaluation Framework, Profiler, middleware, and Guardrails — zero logic rewrite. Plugins exist for LlamaIndex, CrewAI, AutoGen, Semantic Kernel, etc. *Distractor:* "rebuild it as a native NAT ReAct agent" ignores the explicit "without rewriting" constraint.
*Concept link: D2/D7 — plugin/framework interop.*

**Q13.** What does the NAT **Agent Hyperparameter Optimizer** (Parameter Optimizer) tune, and how?
**→ It systematically searches YAML-configurable parameters — system-prompt wording, temperature, top-p, max tokens, number of few-shot examples, tool descriptions — by running the agent against an evaluation dataset and keeping the config that maximizes a target metric** (e.g. faithfulness, task success). It replaces manual prompt fiddling with measured search. *Distractor:* "it fine-tunes model weights" is wrong — it tunes *config*, not weights.
*Concept link: D2/D3 — config optimization vs fine-tuning.*

**Q14.** Which NAT authentication mechanism is **NOT** standard? (NOT trap.) Options resemble API Key, OAuth2, Bearer, HTTP Basic, MCP accounts, and "SAML federation."
**→ "SAML federation"** is not the documented NAT auth set; the documented mechanisms are **API Key, OAuth2, Bearer, HTTP Basic, and MCP accounts.** *(Always confirm against current docs — auth surfaces evolve.)*
*Concept link: D2 — auth mechanisms.*

**Q15.** A developer wants tool calls to always have well-formed parameters. What's the **generation-time** lever (vs the runtime guardrail)?
**→ NAT structured outputs / function-calling with JSON schemas** constrain the LLM to emit schema-valid tool calls at generation time. The complementary **runtime** lever is a Guardrails **execution rail** that validates the call before it fires. Defense in depth = both. *Distractor:* picking only the execution rail misses that the question asks for the *generation-time* control.
*Concept link: D2/D9 — structured output + execution rail.*

**Q16.** Sketch the minimum a NAT agent YAML must declare.
**→ Agent type, model/LLM endpoint, tools (functions or a function group), and a system prompt** — optionally middleware, parameters (temperature/max-tokens), memory, and guardrails. The YAML is the **declarative** spec of *what the agent does*; `nat` builds the runnable workflow from it. *Distractor:* "the YAML contains the model weights" is nonsense bait.
*Concept link: D2 — YAML workflow structure.*

### Domain 3 — Evaluation & Tuning (×6)

**Q17.** A RAG system shows **faithfulness 0.95, answer-relevance 0.60.** Where is the fault?
**→ Retrieval.** High faithfulness = the answer faithfully reflects the retrieved passages; low relevance = those passages don't address the question. The generator is honestly summarizing the *wrong* context. Fix retrieval (embeddings, reranking, query reformulation), not the generator. *Distractor:* "raise the temperature / change the prompt" attacks the generator and would make it *worse*.
*Concept link: D3 — faithfulness vs relevance diagnosis.*

**Q18.** Profiler shows: retrieval 0.5s, rerank 0.8s, **LLM generation 9.0s**, guardrails 1.5s (p95 ≈ 12s). Where do you optimize **first**?
**→ LLM generation (75% of latency).** Options: smaller NIM if quality allows, enable Dynamo prefix-caching (system prompt reuse), trim max output tokens, or add NIM replicas if it's queuing. *Distractor:* "optimize retrieval" — it's already fast; touching it is wasted effort (FIRST/BEST trap). Guardrails (1.5s) is the secondary target (parallelize the safety NIMs).
*Concept link: D3 — Profiler hotspot → fix.*

**Q19.** What does **trajectory evaluation** catch that output-only evaluation misses?
**→ Process errors:** wrong tool, wrong order, or wasteful tool calls — even when the final answer happens to be right ("right answer via the wrong path"), or right but via 10 calls when 3 would do. You define expected trajectories and compare actual ones. *Distractor:* "trajectory eval measures answer correctness" — that's output eval; trajectory eval scores the *path*.
*Concept link: D3 — trajectory vs output eval.*

**Q20.** Profiler vs Evaluation Framework — which answers which question?
**→ Profiler = "how fast / how expensive?"** (token counts and latency at workflow/step/tool level, cost estimates). **Evaluation Framework = "how good?"** (faithfulness, relevance, accuracy, trajectory correctness). Use both: eval guards quality, profiler guards resources. *Distractor:* "Profiler measures faithfulness" conflates the two (classic swap trap).
*Concept link: D3 — profiler vs eval split.*

**Q21.** No labeled test set exists for a RAG agent. **Best** practical evaluation bootstrap?
**→ (1)** Generate synthetic Q&A from the corpus (give the LLM a passage, have it write questions it answers). **(2)** Use the **Evaluation Framework with LLM-as-Judge** for reference-free metrics — faithfulness and relevance need no ground-truth labels. **(3)** Use synthetic pairs as pseudo-ground-truth for recall@k. **(4)** Later, have experts label a real-query subset. *Distractor:* "wait until you have human labels" stalls forever (cold-start trap).
*Concept link: D3 — LLM-as-judge, synthetic eval.*

**Q22.** Leadership wants to cut inference cost on a high-volume tool-calling agent using **traffic you already log**, without losing accuracy. Which NVIDIA pattern?
**→ The Data Flywheel Blueprint.** It mines tagged production logs, builds eval/finetune datasets, finds a smaller distilled model that still clears the accuracy bar (NeMo Evaluator + Customizer), and promotes it after human review. *Distractor:* "just switch to a smaller model" skips the *measure-it-still-passes* step the flywheel exists to provide.
*Concept link: D3 — Data Flywheel loop.*

### Domain 4 — Deployment & Scaling (×6)

**Q23.** NAT offers FastAPI, MCP, FastMCP, and A2A servers. When **A2A over FastAPI**?
**→ When agents must talk to *other agents* via the Agent-to-Agent protocol** — typed inter-agent messages, task delegation/lifecycle, agent discovery, automatic trace propagation. FastAPI is for *external* clients (humans/apps) over REST/WebSocket. Typical multi-agent shape: FastAPI gateway on the Router (external), A2A between specialists (internal). *Distractor:* "A2A is just FastAPI with extra steps" ignores the typed-contract/discovery/lifecycle features.
*Concept link: D4 — server type selection.*

**Q24.** What does the **NIM Operator** automate that raw Kubernetes YAML does not?
**→ GPU discovery/allocation, model caching across pod restarts (PVCs), LLM-aware health probes (model-load/inference readiness, not just TCP), and inference-load-based scaling.** It encapsulates NVIDIA's operational know-how for serving on K8s. *Distractor:* "it orchestrates the agent logic" — no, that's NAT; the Operator manages *NIM* lifecycle.
*Concept link: D4 — NIM Operator.*

**Q25.** Which optimization makes **NVIDIA Dynamo** especially valuable for *agents*?
**→ Prefix caching** — agents reuse the same long system prompt on every request, so caching the KV-cache for that shared prefix avoids recomputation. Dynamo also does dynamic batching, disaggregated prefill/decode serving, and multi-node inference. *Distractor:* "Dynamo replaces NAT orchestration" is a layer error.
*Concept link: D4 — Dynamo for agents.*

**Q26.** A 70B model on 2×H100 handles ~20–40 concurrent requests. You have **8×H100** and need **500** concurrent users on a 70B model. Feasible? (NOT-feasible recognition.)
**→ No.** Do the GPU math: a 70B model in FP16 (~140 GB weights) needs ~2×H100 (80 GB each) **per NIM instance**, so 8×H100 ÷ 2 = **only 4 instances** → 4 × (20–40) ≈ **80–160 concurrent**, far short of 500. Options: drop to an 8B/13B model (one instance fits on a single H100, so 8×H100 could clear 500+), raise the GPU budget to ~24–32×H100, add aggressive caching, or queue with timeouts. **Validate with the NAT Sizing Calculator.** *Distractor:* "yes, just add replicas" ignores that you have no more GPUs to give them — replicas need GPUs you don't have.
*Concept link: D4 — sizing, GPU planning.*

**Q27.** Why is **request-count autoscaling** insufficient for agent endpoints? (Concept trap.)
**→ Agent requests have wildly variable resource cost** — one complex multi-step request can consume 100× the tokens/GPU of a simple one — so request count doesn't track load. Scale on **token throughput / GPU queue depth / utilization** instead. *Distractor:* "agents don't get requests" / "K8s can't count requests" are nonsense baits.
*Concept link: D4 — variable-length scaling.*

**Q28.** Which NAT deployment server exposes your agent **as a callable MCP tool for other MCP-compatible agents**?
**→ The MCP server (or FastMCP server runtime).** FastAPI serves humans/apps; A2A is agent-to-agent; the MCP server publishes the workflow as an MCP tool. *Distractor:* "A2A" is the near-miss — A2A is for agent *peers*, MCP is for *tool* exposure.
*Concept link: D4/D7 — MCP/FastMCP publishing.*

### Domain 5 — Cognition, Planning & Memory (×5)

**Q29.** What is **Test-Time Compute** in the NAT Reasoning Agent, and what does it cost?
**→ Spending extra inference compute to improve quality: generate multiple candidates, plan/deliberate over them, and select the best** — a deliberation step instead of single-pass generation. It trades **latency and tokens (often 3–5×)** for accuracy on hard reasoning. *Distractor:* "it fine-tunes at inference" — no weights change; it's extra forward passes.
*Concept link: D5 — test-time compute.*

**Q30.** Match each NAT **Object Store** backend to its best fit: in-memory, Redis, MySQL, S3.
**→ In-memory:** dev/test, ephemeral sessions (fast, volatile). **Redis:** production session memory needing low-latency reads + survival across restarts (+ TTL). **MySQL:** long-term, queryable, auditable, backed-up memory (compliance). **S3:** large artifacts (documents, reports) not needing low-latency random access (archival). *Distractor:* "use S3 for hot session reads" — wrong latency profile.
*Concept link: D5 — object store selection.*

**Q31.** What is the **Automatic Memory Wrapper**, and when should you **NOT** use it? (NOT angle.)
**→ It transparently adds memory to any agent** — intercepting I/O, storing relevant info, and injecting retrieved memories into context — with no logic change. **Don't** use it when: memory needs are highly specific/structured; injected memories would blow the context budget and evict important context; you need fine-grained retain/forget control; or the agent is intentionally stateless. *Distractor:* "never use it in production" overstates — it's fine when its automatic policy matches your needs.
*Concept link: D5 — automatic memory wrapper.*

**Q32.** ReWOO's three phases?
**→ Planning (create the full plan + all tool calls), Execution (run each step; independent tools can run in parallel within a dependency level), Solution (synthesize gathered evidence into the final answer).** *Distractor:* "Perceive/Reason/Act" is the *generic* loop, not ReWOO's named phases.
*Concept link: D5 — ReWOO internals.*

**Q33.** A conversational agent must remember user preferences (language, style, past topics) **across sessions** with sub-millisecond reads. Which memory config?
**→ NAT memory on the Redis object store + the Automatic Memory Wrapper.** Redis persists across restarts, reads in <1ms, and supports TTL for aging out stale memories; the wrapper handles store-on-turn and retrieve-into-context. Add MySQL as a secondary store only if you need a queryable audit trail. *Distractor:* "in-memory store" loses everything on restart (cross-session requirement fails).
*Concept link: D5 — cross-session memory.*

### Domain 6 — Knowledge Integration & Data (×5)

**Q34.** Name the AI-Q / NVIDIA RAG Blueprint's retrieval stack (embed, rerank, vector store) and the default generation model class.
**→ Embed: `llama-nemotron-embed-1b-v2`. Rerank: `llama-nemotron-rerank-1b-v2`. Vector store: Milvus (GPU-accelerated via cuVS).** **(Currency:** NeMo Retriever NIM **1.13.0** renamed these to the *Nemotron* brand — older docs/strings call them `llama-3.2-nv-embedqa-1b-v2` and `llama-3.2-nv-rerankqa-1b-v2`; either name may appear on the exam, but the current model strings are the `llama-nemotron-*` ones. The embed model keeps 8192-token context, multilingual, Matryoshka dynamic-dim.**)** Default generation in the current AI-Q Blueprint is a **Nemotron Super**–class model (e.g. `Llama-3.3-Nemotron-Super-49B-v1.5`), swappable to any NIM-compatible model — verify on the specific Blueprint page, as it changes. *Distractor:* "FAISS / OpenAI embeddings" — not the NVIDIA stack.
*Concept link: D6 — AI-Q components.*

**Q35.** When does **sparse** retrieval (BM25) beat **dense** (embeddings)?
**→ On rare exact tokens** — error codes, identifiers, SKUs ("error NV-4231") the embedder never learned good vectors for; very short 1–2 word queries with little semantic signal; and specialized vocabulary outside the embedder's training. Production answer for technical corpora: **hybrid** (dense + sparse, fused) usually beats either alone. *Distractor:* "dense always wins" is the over-generalization trap.
*Concept link: D6 — dense/sparse/hybrid.*

**Q36.** What does **cuVS** do for Milvus, and when does the GPU pay off?
**→ cuVS is NVIDIA's CUDA library for GPU-accelerated approximate-nearest-neighbor search;** in Milvus it replaces CPU ANN (CPU-HNSW/IVF), parallelizing distance computation across many concurrent queries. It pays off at **high concurrency** (e.g. >50 QPS with tight latency SLAs) or very large indexes; for single-digit-QPS dev, CPU-HNSW is fine. *Distractor:* "cuVS makes single queries asymptotically faster" — the win is *throughput under concurrency*, not single-query big-O.
*Concept link: D6 — Milvus + cuVS.*

**Q37.** 5M documents; CPU preprocessing takes 72 hours. Which NVIDIA tool, and what does it accelerate?
**→ NeMo Curator** (GPU-accelerated): fuzzy dedup (MinHash LSH), language ID/filtering, quality scoring/classification, extraction/cleaning — typically 10–100× over CPU, turning overnight jobs into interactive ones. Pair with **NIM embedding** endpoints (batch) and **Milvus+cuVS** indexing for an end-to-end GPU ingestion pipeline. *Distractor:* "NeMo Customizer" is *fine-tuning*, not data prep (layer/lifecycle trap).
*Concept link: D6 — NeMo Curator.*

**Q38.** Users report **outdated** answers from a RAG policy agent. **Most likely** cause? (FIRST/diagnosis.)
**→ The index wasn't refreshed with recent policy updates** — stale index is the most common cause of stale answers. *Distractors:* "the LLM is outdated," "embedding model too small," "wrong chunking" are all *possible* but far less likely than an unrefreshed index (MOST-LIKELY trap). Fix: re-run ingestion; in the long run, automate freshness.
*Concept link: D6 — index freshness / operations.*

### Domain 7 — NVIDIA Platform Implementation (×4)

**Q39.** The four phases of the end-to-end NAT workflow?
**→ Configure (YAML: type, model, tools, prompt, middleware, guardrails) → Build (instantiate, register functions, connect NIM, test locally, tune with the optimizer) → Evaluate (RAG/trajectory/red-team eval, profile, size) → Deploy (pick FastAPI/MCP/A2A, containerize, K8s via NIM Operator or Docker Compose, wire observability).** *Distractor:* "Train → Serve" skips the eval gate the exam emphasizes.
*Concept link: D7 — NAT lifecycle.*

**Q40.** Map each NeMo microservice to its role: Retriever, Curator, Customizer, Evaluator, Guardrails.
**→ Retriever:** embedding + reranking + extraction for RAG. **Curator:** GPU data prep/curation. **Customizer:** fine-tuning (SFT/LoRA/DPO). **Evaluator:** benchmark/LLM-as-judge evaluation. **Guardrails:** runtime safety rails. *Distractor:* swapping Curator↔Customizer is the deliberate near-homophone trap — **Curator = data, Customizer = training.**
*Concept link: D7 — NeMo microservice ecosystem.*

**Q41.** In the NAT+NIM split, which component holds the **orchestration/tool-routing/state** logic? (Layer trap.)
**→ NAT.** NIM only serves inference. "NIM orchestrates the agent" is always wrong. *Distractor:* "NVIDIA Dynamo orchestrates" — Dynamo optimizes *inference*, not orchestration.
*Concept link: D7 — NAT vs NIM responsibilities.*

**Q42.** Trace a request end-to-end through the platform, naming each layer it touches.
**→ User → NAT (input guardrails → retrieval/tools → NIM inference → output guardrails) → response,** with NAT exporting OpenTelemetry traces to the observability backend throughout. *Distractor:* "User → NIM → NAT → response" inverts the layers (NAT fronts NIM, not the reverse).
*Concept link: D7 — request path.*

### Domain 8 — Run, Monitor & Maintain (×2)

**Q43.** Name NAT's three tracing levels and the redaction-processor's job.
**→ Workflow level** (whole request: total latency, total tokens, status), **Step level** (each reasoning step/ReAct iteration), **Tool level** (each tool call: name, params, response, time) — letting you drill from "request took 8s" to "the DB query tool took 5s." A **redaction processor** masks/removes sensitive data (API keys, PII, proprietary content) from traces *before* export so dashboards don't leak it. *Distractor:* "two levels (request + tool)" omits the step level (count trap).
*Concept link: D8 — trace granularity + redaction.*

**Q44.** A production agent occasionally loops, calling the same tool repeatedly. Which metric detects it, and which observability backends does NAT export to?
**→ A sudden spike in *steps-per-execution*** flags looping (token cost rises too, but as a secondary signal). NAT exports OpenTelemetry traces to **Phoenix, Weave, Langfuse** (and raw OTel). *Distractor:* "CPU utilization" doesn't isolate a reasoning loop.
*Concept link: D8 — loop detection, exporters.*

### Domain 9 — Safety, Ethics & Compliance (×3)

**Q45.** Name the **five** NeMo Guardrails rail types and place "topic control" correctly. (The most common exam-trap area.)
**→ The five are: input, dialog, retrieval, output, execution.** **Topic control is a *use* of dialog rails** (or the Topic Control NIM) — there is **no separate "topical rail" type** in the docs. Input rails screen/alter user input (content safety, jailbreak); retrieval rails filter RAG chunks; output rails screen/alter the response (fact-check, PII redaction); execution rails validate tool/action calls. *Distractor:* any option that lists "topical" as one of the five canonical types is wrong by current docs.
*Concept link: D9 — five rail types.*

**Q46.** How do **execution rails** provide agentic security? Give a concrete blocked action.
**→ They intercept a tool call *before* it executes and validate name, parameters, and context against policy — so even a prompt-injected or buggy LLM can't fire an unauthorized/dangerous call.** Concrete: an agent with `delete_record(id)` — an injection makes the LLM emit `delete_record("*")`; the execution rail validates the id is a real UUID (not a wildcard), checks the user's permission, and/or requires human approval, blocking the mass delete. *Distractor:* "an output rail would catch it" — output rails act on text, not on the *action*; the action layer is the execution rail.
*Concept link: D9 — execution rails.*

**Q47.** Public-facing agent facing adversarial users: Colang-only rules, or Colang **plus** Safety NIMs? (BEST/defense-in-depth.)
**→ Colang rules + Safety NIMs (Content Safety, Jailbreak Detection, Topic Control).** Colang patterns handle topic/flow/simple filtering; the ML-based Safety NIMs catch novel phrasings, encoding tricks, and multi-turn manipulation that pattern-matching misses. Colang-only is acceptable only for internal, trusted, structured-input agents. *Distractor:* "Safety NIMs alone, drop Colang" loses dialog/flow control.
*Concept link: D9 — safety NIMs + rails.*

### Domain 10 — Human-AI Interaction & Oversight (×3)

**Q48.** An execution rail decides a tool call needs human approval. Using NAT, how is the request delivered to the human in real time?
**→ A NAT WebSocket interactive workflow** sends the pending action to the connected client and waits for approve/reject (bidirectional, real-time). On approve the rail lets the call proceed; on reject it blocks and returns feedback. *Distractor:* "email notification" / "write to a DB and poll" aren't the NAT real-time primitive (specific-primitive trap).
*Concept link: D10 — WebSocket approval gate.*

**Q49.** An agent does both **low-risk** password resets and **high-risk** Active Directory edits. **Best** oversight architecture?
**→ Risk-proportional: auto-execute the low-risk resets; gate the AD edits behind a human approval via a Guardrails execution rail (+ WebSocket).** Log everything via logging middleware for the audit trail. *Distractors:* "approve everything" kills throughput on safe actions; "approve nothing" is reckless on AD; "log but never gate" provides no real control (all three are BEST-trap foils).
*Concept link: D10 — risk-proportional oversight.*

**Q50.** Which is **NOT** a NVIDIA-documented HITL *primitive* (as opposed to an inferred design pattern)? (NOT trap; honesty about coverage.)
Options resemble: WebSocket interactive workflows, execution-rail approval gates, per-user functions/auth, and **"a built-in escalation-routing dashboard."**
**→ "A built-in escalation-routing dashboard."** NVIDIA provides the *primitives* (interactive workflows, execution rails, per-user auth, audit middleware); escalation paths, approval queues, and feedback loops are **design patterns you build on those primitives**, not shipped components. On the exam, prefer the answer anchored to a documented primitive over an inferred pattern.
*Concept link: D10 — primitives vs patterns (the exam's weakest-coverage domain).*

---

## §3 · Integrative challenges (the two capstones, summarized)

The exam's hardest questions cross domain boundaries. These two integrative scenarios — distilled from the course capstones — are the shape of those questions. For each, the drill is: walk **every domain** and name the NVIDIA primitive that implements your choice. If you can do both end-to-end, you can answer any cross-domain item.

### Challenge A — Production RAG Agent (single-agent, knowledge-heavy)

*Build a citation-grounded document research assistant over 500–2,000 technical docs that never hallucinates, refuses off-domain queries, and evaluates itself on every change.* Exercises **D6, D2, D3, D9, D7**.

| Domain | Decision | NVIDIA primitive |
|--------|----------|------------------|
| Knowledge (D6) | GPU ingest → adaptive chunk (~512 tok / 64 overlap) → embed → index → two-stage retrieve | NeMo Curator → `llama-nemotron-embed-1b-v2` (NIM) → Milvus+cuVS → `llama-nemotron-rerank-1b-v2` |
| Development (D2) | Citation-required system prompt; tool registry; FastAPI front | NAT YAML + FastAPI server |
| Evaluation (D3) | Retriever (recall@5≥0.85, MRR≥0.80) + RAG (faithfulness≥0.90) + adversarial + latency; CI gate | NAT Evaluation Framework + Profiler |
| Safety (D9) | **input** (Content Safety + Jailbreak NIM + topic-control dialog rail), **retrieval** (filter chunks), **output** (fact-check + PII redaction), **execution** (validate tool calls) | NeMo Guardrails (Colang) + Safety NIMs |
| Platform (D7) | Configure→Build→Evaluate→Deploy; OTel traces | NAT lifecycle + Phoenix/Langfuse |

**The integrative trap to expect:** *"The grounding/fact-check rail is blocking correct answers — why?"* → the faithfulness threshold is too strict **or** retrieval returned thin context. The fix is a *retrieval/threshold* tune, not loosening safety blindly. This is a D3+D6+D9 cross-question.

### Challenge B — Multi-Agent Code Review (distributed, governance-heavy)

*Router + Code-Analysis + Documentation + Review agents as separate A2A services; distributed tracing; secrets never leak; publication needs human approval; low-confidence findings escalate.* Exercises **D1, D2, D4, D8, D10, D9**.

| Domain | Decision | NVIDIA primitive |
|--------|----------|------------------|
| Architecture (D1) | Router classifies + dispatches; specialists are focused Tool Calling agents | NAT Router + Tool Calling agents |
| Development (D2) | Per-agent YAML, domain-restricted prompts, shared state | NAT YAML + Redis object store |
| Deployment (D4) | Each agent a separate service for independent scaling/fault-isolation; shared NIM | NAT **A2A** servers + NIM (+ Docker Compose) |
| Monitor (D8) | One `trace_id` from the Router propagated across all hops → one Phoenix view | NAT OpenTelemetry, trace correlation |
| Oversight (D10) | Publication → human approval; confidence <0.7 → escalate; auto-approve low-risk | Execution rail + WebSocket + logging middleware |
| Safety (D9) | Secret-scan execution rail on generated docs; domain isolation per agent | Guardrails execution + dialog rails |

**The integrative traps to expect:** *"Why A2A and not FastAPI everywhere?"* → typed inter-agent contracts, task lifecycle, agent discovery, automatic trace propagation (D1+D4). *"90% of traffic hits one specialist — what now?"* → scale that specialist's replicas independently (the payoff of A2A service separation), and verify the router with **trajectory evaluation** before assuming it's a bug (D1+D3+D4). *"Where's the real bottleneck under a doc-spike?"* → not a technical layer — the **human approval queue** (D10); technical scaling can't fix a human gate, only batching/delegation can.

---

## §4 · Last-mile checklist

Before you sit, confirm you can do each of these *cold*:

- Recite the **scenario→tool cheat table** (§1) without peeking.
- Name the **5 Guardrails rail types** (input/dialog/retrieval/output/execution) and explain why "topical" isn't one.
- Distinguish **ReAct vs ReWOO** and pick correctly for predictable-vs-flaky tasks.
- Diagnose a RAG failure from **faithfulness vs relevance** numbers.
- State what **Profiler** answers vs what the **Evaluation Framework** answers.
- Pick a **NAT server** (FastAPI / MCP / FastMCP / A2A) for a given scenario.
- Match the **four object stores** to their use cases.
- Map the **five NeMo microservices** (Curator=data, Customizer=training, Retriever, Evaluator, Guardrails) without swapping Curator/Customizer.
- Apply the **NVIDIA-first elimination heuristic** and the **NOT/EXCEPT** and **BEST/FIRST** trap discipline.
- Walk **both capstones** domain-by-domain, naming the primitive each time.

*Reminder: these are synthesized study reps, not official questions. Verify any number (question count, time, price, exact model names) against the live NVIDIA certification and docs pages before exam day — those values change.*
