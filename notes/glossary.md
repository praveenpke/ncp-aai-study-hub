# NCP-AAI Glossary

> A single-source reference for every term the NCP-AAI exam reaches for. Each entry is written to be *self-sufficient*: a plain-language definition, the **NVIDIA-specific context** (exact product / tool / config name where one exists), **why it matters for the exam**, and the **domain(s)** it lives under. This is a synthesis of our 11 domain notes plus the course glossary — facts have been web-verified against current NVIDIA docs (NeMo Agent Toolkit, NeMo microservices, NeMo Guardrails, build.nvidia.com) and corrected where the source was version-stale.
>
> **How to read the domain tags:** D1 Agent Architecture · D2 Agent Development · D3 Evaluation & Tuning · D4 Deployment & Scaling · D5 Cognition/Planning/Memory · D6 Knowledge Integration · D7 NVIDIA Platform · D8 Run/Monitor/Maintain · D9 Safety/Ethics/Compliance · D10 Human-AI Interaction.
>
> **Verification notes (where the third-party source was wrong/stale — know these, they are classic distractors):**
> - NAT supports **8 agent types** in current docs (1.7), not 7: ReAct, Reasoning, ReWOO, Responses API Agent, **Router**, **Parallel Executor**, **Sequential Executor**, Tool Calling. The "Parallel Executor" is the one most often left off lists.
> - NeMo Guardrails has **five rail categories: input, dialog, output, retrieval, execution** — *not* "topical." **Topical control is a behavior built on dialog rails** (or enforced by the Topic Control NIM), not a top-level rail category. The "five rails = input/output/dialog/topical/execution" framing is a common but wrong distractor.
> - The Safety NIMs have **exact product names**: *Llama 3.1 NemoGuard 8B Content Safety*, *Llama 3.1 NemoGuard 8B Topic Control*, *NemoGuard Jailbreak Detect*. NVIDIA's red-team tool is **garak** (Generative AI Red-teaming and Assessment Kit), and there is a current **Safety for Agentic AI Blueprint** (the old standalone "Safety Blueprint" is superseded).

---

## How this maps to our domain notes

This glossary does not replace the domain notes — it is the fast lookup layer. Every entry points back to the domain note(s) where the concept is taught in depth with worked examples. Sections below group terms by category rather than alphabetically, because on the exam terms cluster by *job to be done* (you reach for "the eval metrics" or "the safety rails" as a set).

- §1 Agent patterns & agent types
- §2 NVIDIA platform: NAT, NeMo, NIM
- §3 Knowledge & retrieval (RAG)
- §4 Evaluation & metrics
- §5 Tuning & customization
- §6 Safety, guardrails & compliance
- §7 Deployment, serving & inference
- §8 Protocols & interoperability
- §9 Observability & operations
- §10 Cognition, memory & planning

---

## §1 — Agent patterns & agent types

### Agent
A software system that **perceives** input, **reasons** about what to do, **acts** via tools/APIs, and **observes** the result to inform the next step — distinguished from a fixed pipeline by its ability to *change its behavior based on intermediate observations*.
**NVIDIA context:** In the NVIDIA stack, agents are authored in **NeMo Agent Toolkit (NAT)** via YAML config, choosing one of NAT's agent-type implementations.
**Why it matters:** The exam constantly contrasts "agent" (adaptive, observation-driven) vs "workflow/pipeline" (fixed graph). If the task can change course mid-run based on a tool result, it wants an agent.
**Domains:** D1, D2.

### Agent loop (perceive → reason → act → observe)
The four-phase execution cycle every agent runs until a termination condition is met.
**NVIDIA context:** NAT's agent types implement *different shapes* of this loop — ReAct interleaves reason/act/observe every step; ReWOO plans all steps first then executes; Tool Calling runs essentially one act→observe.
**Why it matters:** "Which agent type?" questions are really "which loop shape fits this task?" Memorize the loop and you can derive the right agent type.
**Domains:** D1, D2.

### ReAct (Reasoning + Acting)
An agent pattern that **interleaves** a reasoning step ("Thought") with a tool call ("Action") and its result ("Observation"), looping until done. The agent re-plans after *every* observation.
**NVIDIA context:** One of NAT's agent types; routes to tools using their names/descriptions in the prompt.
**Why it matters:** Best for **adaptive, multi-step** tasks where the plan may change with intermediate results. Highest token/latency cost of the loop-based agents because it calls the LLM between every step. The exam's default "general autonomous agent."
**Domains:** D1, D2, D5.

### ReWOO (Reasoning Without Observation)
An agent pattern that generates the **entire plan** (all tool calls + dependencies) in a *single* up-front LLM call, then executes the plan with no further LLM reasoning between steps.
**NVIDIA context:** A NAT agent type; decouples planning from execution for **token efficiency**.
**Why it matters:** Cheaper and faster than ReAct for **predictable** tasks, but **cannot adapt** mid-execution if a step's result invalidates the plan. The classic "ReAct vs ReWOO" trade is *adaptivity vs cost*.
**Domains:** D1, D2, D5.

### Reasoning Agent
A NAT agent type that performs **planning/reasoning ahead of time** and invokes an underlying function/agent — reasoning *through planning* rather than *between every step* (the distinction from ReAct). Associated with **test-time compute** (deliberate more, sometimes generating/evaluating multiple candidates).
**NVIDIA context:** NAT's vehicle for test-time-compute-style deliberation; wraps another agent/function.
**Why it matters:** Reach for it when answer *quality on hard reasoning* justifies extra inference cost. Don't confuse it with ReAct — Reasoning plans ahead; ReAct reacts each step.
**Domains:** D2, D3, D5.

### Tool Calling Agent
A NAT agent type that makes a **single** tool selection + execution per turn — the LLM picks a tool and arguments, the tool runs, the result returns. No multi-step reasoning loop.
**NVIDIA context:** Lowest latency/token cost of the NAT agent types.
**Why it matters:** Right answer for **simple, well-defined** tasks where the correct tool is obvious from the input. If a question stresses "minimize latency/cost, single clear action," this is it.
**Domains:** D2.

### Router Agent
An agent that **classifies** the input and **dispatches** it to the appropriate specialist agent/handler. It does not do the task itself — it routes.
**NVIDIA context:** A NAT agent type; the typical **entry point of a multi-agent system**. Routes by intent classification, keywords, or LLM-based classification.
**Why it matters:** "Send billing questions to the billing agent, tech questions to the support agent" = Router. Distinguish from Sequential (always runs every stage) and Parallel (fans out to many at once).
**Domains:** D1, D2.

### Sequential Executor
A NAT agent type that runs a **fixed, ordered** sequence of steps where each step's output feeds the next. Every step always executes; there is no routing decision.
**NVIDIA context:** A NAT agent type for deterministic pipelines (e.g., extract → transform → validate → store).
**Why it matters:** The "agent" answer when the process is the *same every time* — no branching. Contrast with Router (branches) and ReAct (re-plans).
**Domains:** D1, D2.

### Parallel Executor
A NAT agent type that runs multiple sub-agents/functions **concurrently** and aggregates their results.
**NVIDIA context:** A NAT agent type — the one most often *omitted* from third-party "list the agent types" cheat sheets (which is exactly why the exam can use it as a discriminator).
**Why it matters:** Use when sub-tasks are **independent** and can run at once to cut wall-clock latency (e.g., query three data sources in parallel, then merge). If you memorized "seven agent types," you'll miss this — there are eight.
**Domains:** D1, D2.

### Responses API Agent
A NAT agent type built around the OpenAI-style **Responses API** interaction shape (stateful, multi-turn, tool-augmented responses).
**NVIDIA context:** One of NAT's eight agent types; aligns NAT workflows with the Responses API surface.
**Why it matters:** Appears in "name all NAT agent types" questions. Know it exists and that it is the Responses-API-shaped agent.
**Domains:** D2.

### Multi-agent system
A system of multiple specialized agents that **coordinate** — via a router/orchestrator, sequential handoff, or peer-to-peer messaging — to solve a task no single agent owns end-to-end.
**NVIDIA context:** Built in NAT by composing agents; cross-process coordination uses the **A2A protocol**. The **Multi-Agent Warehouse Blueprint** is the reference example.
**Why it matters:** Know the coordination *shapes* (orchestrator/router, sequential, hierarchical, A2A peer) and when each applies.
**Domains:** D1, D5.

---

## §2 — NVIDIA platform: NAT, NeMo, NIM

### NAT (NeMo Agent Toolkit)
NVIDIA's open-source toolkit for **building, deploying, evaluating, profiling, and monitoring** agentic systems. YAML-driven config; ships agent types, deployment servers (FastAPI/MCP/A2A), an evaluation framework, a profiler, middleware, and a plugin registry.
**NVIDIA context:** Installed via `pip install nvidia-nat`. Framework-agnostic — connects/optimizes "teams of agents" and integrates LangChain, LlamaIndex, CrewAI, etc. (Formerly "Agent Intelligence Toolkit / AIQ"; the package/CLI prefix `nat` reflects the rename.)
**Why it matters:** The **center of gravity** of the exam. Almost every domain touches NAT. Know that the base install covers core agents + config + functions + middleware, and that **eval/profiling/integration extras** may need extra packages.
**Domains:** D1–D8.

### Function / Function Group (NAT)
A **function** is a callable tool registered with NAT. A **Function Group** is a named bundle of functions assignable to an agent as a unit.
**NVIDIA context:** Part of NAT's function registry; Function Groups make tool sets modular and reusable across agents.
**Why it matters:** Config-shape questions ("how do you give several agents the same toolset cleanly?") → Function Group.
**Domains:** D2.

### Middleware (NAT)
Components that sit in the request pipeline between caller and agent core, processing requests/responses in transit (without changing the agent).
**NVIDIA context:** NAT provides middleware for cross-cutting concerns — caching, logging, defense/security, and red-teaming — configured in the NAT YAML.
**Why it matters:** "Add caching/logging/safety without touching agent logic" → middleware. Know it's a pipeline concern, configured declaratively.
**Domains:** D2, D8, D9.

### Plugin system (NAT)
NAT's extension mechanism: register custom agents, tools/functions, evaluators, front-ends, and integrations as plugins discovered by the toolkit.
**NVIDIA context:** Lets you extend NAT without forking it; the basis for the function/evaluator registries.
**Why it matters:** "How do you add a custom evaluator / custom tool to NAT?" → register it as a plugin.
**Domains:** D2, D3.

### NeMo (microservices family)
NVIDIA's family of microservices for the **model/data lifecycle** around agents: **Customizer** (fine-tune), **Evaluator** (evaluate), **Curator** (data prep), **Retriever** (RAG retrieval), **Guardrails** (safety). Each runs as a deployable microservice with an API.
**NVIDIA context:** These are *distinct products* — don't conflate "NeMo" the platform with any single one. NAT orchestrates *agents*; NeMo microservices supply *customization, evaluation, data, retrieval, and safety*.
**Why it matters:** "Which NeMo microservice does X?" is a recurring question shape. Memorize the five and their one-line jobs.
**Domains:** D3, D4, D6, D7, D8, D9.

### NeMo Customizer
A microservice for **fine-tuning** LLMs — supervised fine-tuning (SFT), parameter-efficient (LoRA), and preference alignment (**DPO**) — via API jobs.
**NVIDIA context:** The fine-tuning engine in the **Data Flywheel** cycle. NAT's finetuning harness can drive Customizer for SFT/DPO (and OpenPipe ART for GRPO/RL).
**Why it matters:** "Run a DPO/SFT job as a managed service" → Customizer. Pair it mentally with Evaluator (close the loop).
**Domains:** D3, D7, D8.

### NeMo Evaluator
A microservice for **systematic evaluation** — academic benchmarks (MMLU, GSM8K, …), **LLM-as-a-judge** (any NIM as the judge), and **retriever/RAG metrics** (Recall@K, NDCG@K, faithfulness) — via eval targets/configs/jobs.
**NVIDIA context:** Complements NAT's in-workflow eval framework; Evaluator is the **standalone microservice**, NAT eval runs *inside* the workflow.
**Why it matters:** "Benchmark a NIM model + judge custom chats + score a retriever, all via API in the NeMo platform" → NeMo Evaluator. Know the NAT-eval vs Evaluator split.
**Domains:** D3, D7.

### NeMo Curator
A GPU-accelerated **data preparation** toolkit/microservice: fuzzy dedup, language/quality filtering, text extraction, PII removal, cleaning — at corpus scale (10–100× CPU speedups).
**NVIDIA context:** The **ingestion/curation** stage feeding embeddings, fine-tuning, and the Data Flywheel.
**Why it matters:** "Prepare millions of documents for indexing/training" → Curator. It's *data prep*, not retrieval and not eval.
**Domains:** D6, D8.

### NeMo Retriever
A microservice family for **RAG retrieval** — embedding, indexing, search, and reranking — served as NIMs.
**NVIDIA context:** Exact current models include **`llama-nemotron-embed-1b-v2`** (multilingual embedding, 26 languages, long-document QA) and **`llama-nemotron-rerank-1b-v2`** (cross-encoder rerank score). *(Note: these supersede the older `llama-3.2-nv-embedqa-1b-v2` / `llama-3.2-nv-rerankqa-1b-v2` NIMs, whose v2 APIs carried a 2026-05-18 deprecation date — model names are version-dependent; verify the current model card before relying on a specific name.)*
**Why it matters:** "Which NVIDIA service provides the embedding + reranking for my RAG pipeline?" → NeMo Retriever (NIMs). Know embedding ≠ reranking (two different model roles).
**Domains:** D6, D7.

### NIM (NVIDIA Inference Microservice)
NVIDIA's standardized **container** for serving an AI model as a microservice with optimized inference (TensorRT-LLM/vLLM backends), an **OpenAI-compatible API**, and health endpoints.
**NVIDIA context:** The serving layer — LLMs, embedders, rerankers, and **safety models** all ship as NIMs. Try cloud-hosted on **build.nvidia.com**, or self-host the container.
**Why it matters:** "Self-hosted, standardized, OpenAI-compatible model endpoint" → NIM. Understand the build.nvidia.com (hosted) vs on-prem (container) split.
**Domains:** D4, D7.

### NIM Operator
A **Kubernetes operator** that automates the lifecycle of NIM deployments — GPU allocation, model caching, health monitoring, and inference-aware autoscaling.
**NVIDIA context:** The recommended way to run NIM on Kubernetes in production.
**Why it matters:** "Deploy/scale NIMs on K8s without hand-managing GPUs and replicas" → NIM Operator. Don't confuse with Dynamo (inference *framework*) — Operator is *orchestration*.
**Domains:** D4.

### Blueprint
A complete, **deployable reference architecture** from NVIDIA showing how to compose NVIDIA tools into a production system for a use case.
**NVIDIA context:** Found at **build.nvidia.com/blueprints**. Key ones: **AI-Q** (RAG research assistant), **Multi-Agent Warehouse** (multi-agent coordination), **Data Flywheel** (continuous improvement), **Safety for Agentic AI** (build/deploy/run-stage safety).
**Why it matters:** Blueprint questions test "which reference architecture solves this end-to-end problem?" Match the use case to the named Blueprint.
**Domains:** D1, D6, D7, D9.

### AI-Q Blueprint
NVIDIA's **RAG research-assistant** Blueprint — an end-to-end reference RAG architecture (ingestion → embedding → Milvus + cuVS → reranker → NIM LLM).
**NVIDIA context:** The canonical "how a production RAG pipeline is wired with NVIDIA parts" example.
**Why it matters:** When a question describes a full RAG stack and asks for the NVIDIA reference, AI-Q is the name.
**Domains:** D6, D7.

### Data Flywheel (Blueprint)
A continuous-improvement loop: **collect** production traffic → **evaluate** → **curate** → **fine-tune** a (often smaller/distilled) model → **redeploy** after human review.
**NVIDIA context:** The Data Flywheel Blueprint wires NAT (collection, tagged by `workload_id`) + NeMo Evaluator + NeMo Curator + NeMo Customizer. Can find a smaller model that still clears the accuracy bar (large cost cuts for tool-calling workloads).
**Why it matters:** "Cut inference cost on a high-volume agent using traffic we already log, without losing accuracy" → Data Flywheel. It's the *systematized* version of the eval→tune loop.
**Domains:** D3, D7, D8.

---

## §3 — Knowledge & retrieval (RAG)

### RAG (Retrieval-Augmented Generation)
An architecture that **grounds** generation in retrieved knowledge: search a knowledge base for relevant chunks, then pass them as context to the LLM to answer.
**NVIDIA context:** NVIDIA's end-to-end RAG = NeMo Curator (prep) → NIM embeddings → Milvus + cuVS (search) → NIM reranker → NIM LLM; AI-Q is the reference Blueprint.
**Why it matters:** RAG is the fix for **knowledge/freshness/private-data** gaps — *not* fine-tuning. The "RAG vs fine-tune" decision is one of the most tested distinctions: **RAG = facts; fine-tuning = skills/behavior.**
**Domains:** D6, D3.

### Embedding
A dense vector representation of text where **semantic similarity = geometric proximity**, enabling similarity search/retrieval/clustering.
**NVIDIA context:** Served via NeMo Retriever embedding NIMs (e.g., `llama-nemotron-embed-1b-v2`). NAT embedder components connect to these endpoints.
**Why it matters:** Embedding quality bounds **recall** — if the right doc isn't retrievable, no downstream fix helps. The embedding model is the first RAG tuning lever after chunking.
**Domains:** D6.

### Reranking
A **second-stage** step that uses a **cross-encoder** to re-score/reorder first-stage candidates, capturing fine-grained query–document interactions bi-encoder embeddings miss.
**NVIDIA context:** NeMo Retriever reranking NIM (`llama-nemotron-rerank-1b-v2`, a cross-encoder logit scorer; supersedes the older `llama-3.2-nv-rerankqa-1b-v2`); sits between retrieval and generation in AI-Q.
**Why it matters:** Improves **precision@k / NDCG** by promoting the most relevant chunk to the top (LLMs weight earlier context more). "Right docs retrieved but ranked too low" → add/tune a reranker.
**Domains:** D6, D3.

### Chunking
Splitting source documents into retrievable units (fixed-size, sentence/semantic, or structure-aware), each embedded and indexed.
**NVIDIA context:** The **first** RAG tuning lever; NVIDIA publishes chunking-strategy studies.
**Why it matters:** Bad chunking caps both recall (answer split across chunks) and precision (noisy chunks). "Low recall@k, answers miss facts" → revisit chunking before anything heavier.
**Domains:** D6, D3.

### Vector database / Milvus
A database purpose-built to **store, index, and search** dense embeddings at scale (index types: HNSW, IVF, flat; metadata filtering; partitioning). **Milvus** is the open-source one NVIDIA standardizes on.
**NVIDIA context:** Milvus is the vector store in AI-Q; integrates **cuVS** for GPU-accelerated search.
**Why it matters:** Know Milvus as *the* NVIDIA-recommended production vector DB, and that the index type (HNSW vs IVF) trades recall vs speed/memory.
**Domains:** D6.

### cuVS (CUDA Vector Search)
NVIDIA's **GPU-accelerated** library for approximate-nearest-neighbor (ANN) search — GPU implementations of HNSW/IVF that exploit parallelism for high-throughput concurrent search.
**NVIDIA context:** Integrated into Milvus to accelerate similarity search.
**Why it matters:** Most beneficial under **high concurrent query load**, where CPU ANN becomes the bottleneck. "Vector search is the bottleneck at high QPS" → cuVS/GPU search.
**Domains:** D6.

### Faithfulness / Groundedness
A RAG generation metric: are the **claims in the answer supported by the retrieved context**? (claims supported ÷ total claims, 0–1).
**NVIDIA context:** Scored via NAT/NeMo Evaluator LLM-as-a-judge; a key production RAG metric (often gated ≥ ~0.8–0.9).
**Why it matters:** **Low faithfulness with good context = a generation problem** (the LLM is hallucinating past correct context) → fix grounding/prompt, not retrieval. This exact diagnosis appears repeatedly.
**Domains:** D3, D6.

### Context relevance / Answer relevance (RAGAS triad)
The other two of the three RAG-generation metrics. **Context relevance:** is the retrieved context relevant to the question (minimal noise)? **Answer relevance:** does the answer actually address the question?
**NVIDIA context:** The **RAGAS triad** (faithfulness + answer relevance + context relevance) covers the three edges of the question↔context↔answer triangle; supported via NAT `ragas` / NeMo Evaluator.
**Why it matters:** Splitting the triad **localizes the fault**: faithful-but-wrong → retrieval; relevant-context-but-unfaithful → generation. This is the RAG diagnosis pattern the exam loves.
**Domains:** D3, D6.

---

## §4 — Evaluation & metrics

### Offline vs Online evaluation
**Offline** = pre-deployment, in CI, on golden/benchmark datasets (repeatable, catches *known* failures). **Online** = post-deployment, on live traffic (A/B tests, canaries, user feedback, drift — catches *novel* failures).
**NVIDIA context:** Offline via NeMo Evaluator / NAT eval gates; online via observability + the Data Flywheel feeding failures back into golden sets.
**Why it matters:** They are **complementary, not alternatives**. "Catch novel real-world failures" → online; "block a bad change before ship" → offline.
**Domains:** D3, D8.

### Golden dataset
A curated, **version-controlled** set of input → expected-output (or expected-**trajectory**) pairs representing the task distribution, including edge cases and past failures.
**NVIDIA context:** The substrate for offline eval and CI gates; **"every production bug becomes a test case."**
**Why it matters:** Eval scores are meaningless without a versioned golden set pinned to the prompt/config version. Quality > quantity.
**Domains:** D3.

### LLM-as-a-Judge
Using a strong LLM to score another model's outputs against a rubric — pointwise (score one), **pairwise** (A vs B), or reference-based (vs a gold answer).
**NVIDIA context:** Supported by NAT eval and NeMo Evaluator with any **NIM** as the judge.
**Why it matters:** The scalable approximation of human judgment — but watch the **three named biases** (below). Prefer pairwise + position-swapping over absolute scoring.
**Domains:** D3.

### Judge biases (position, verbosity, self-preference)
Systematic LLM-judge errors. **Position bias:** favors the answer in a fixed slot (often A) regardless of quality. **Verbosity bias:** prefers longer answers. **Self-preference (self-enhancement):** favors its own model family's outputs.
**NVIDIA context:** Mitigate via **position-swapping** (judge both orderings, keep consistent verdicts), conciseness-aware rubrics, and using a **different** judge model / a jury.
**Why it matters:** "v2 wins 70%, but flipping order makes v1 win 65%" → **position bias**; the named fix is swapping orderings. These three biases are near-guaranteed exam content.
**Domains:** D3.

### Compound error
End-to-end success of a multi-step agent ≈ `p_step ^ n_steps` — per-step accuracies **multiply**.
**NVIDIA context:** The core reason agent eval is hard; drives the push for fewer steps / higher per-step reliability / verification.
**Why it matters:** Classic compute question: 8 steps × 92% → `0.92^8 ≈ 0.51`. The lever is **fewer steps or a hardened weak step**, *not* "a bigger model."
**Domains:** D3, D1.

### Trajectory evaluation
Scoring the **sequence of actions** an agent took (tools, arguments, order, recovery, termination) — not just the final answer.
**NVIDIA context:** NAT eval supports trajectory scoring (exact / in-order / any-order match vs a reference trajectory).
**Why it matters:** Catches "right answer via the wrong tools" and process inefficiency. "Right output but wrong tool sequence" → trajectory eval.
**Domains:** D3, D2.

### Task completion rate (TSR / goal completion)
The headline agent metric: % of tasks where the agent achieved the user's goal end-to-end (task = intent + constraints).
**NVIDIA context:** Common CI gate threshold (e.g., **≥ 85%** for production); measure under normal, degraded, and ambiguous scenarios.
**Why it matters:** The number leadership cares about; many gate questions reference an ≥85% TSR threshold.
**Domains:** D3.

### Tool-call accuracy
Did the agent pick the **right tool** with **schema-valid, correct arguments**? Includes detecting **hallucinated tools** (calling nonexistent tools) and overuse.
**NVIDIA context:** Component of NAT trajectory/agent eval (target often **>90%**).
**Why it matters:** A common failure layer distinct from reasoning — "the plan was right but the tool args were wrong."
**Domains:** D3, D2.

### Retrieval metrics (Recall@k, Precision@k, MRR, NDCG@k)
**Recall@k:** fraction of all relevant docs in the top-k (coverage). **Precision@k:** fraction of top-k that are relevant (noise). **MRR:** how early the *first* relevant doc appears (1/rank). **NDCG@k:** quality of the whole *graded* ranking, position-discounted, normalized to the ideal.
**NVIDIA context:** Computed by NeMo Evaluator retriever eval / NAT.
**Why it matters:** **Recall@k vs the others localizes RAG faults:** low recall → fix chunking/embeddings/k; recall fine but bad answers → fix generation. MRR/NDCG matter because LLMs weight earlier context more.
**Domains:** D3, D6.

### pass@k vs pass^k
**pass@k** = at least one success in k tries (the *capability* metric; HumanEval-style). **pass^k** = all k tries succeed (the *reliability* metric).
**NVIDIA context:** Production gates care about **pass^k** (consistency), not just whether the agent *can* do it once.
**Why it matters:** A stochastic agent can have high pass@k but low pass^k. "Works sometimes but not reliably" → you're measuring the wrong one; gate on pass^k.
**Domains:** D3.

### NAT Profiler
A NAT tool that measures **token counts, latency, and cost** at workflow / step / tool granularity to find bottlenecks.
**NVIDIA context:** Pairs with the **Sizing Calculator** (capacity) — Profiler finds the hot step, Sizing Calculator estimates GPUs.
**Why it matters:** "One LLM step is 40% of latency/tokens" → Profiler found it; fixes = smaller model for that step, fewer retrieved chunks, or caching.
**Domains:** D3, D4, D8.

### Sizing Calculator (NAT)
A NAT planning tool that **estimates compute** (GPU count, replicas) needed to hit a throughput/latency target at a given concurrency.
**NVIDIA context:** `nat sizing calc`; slope-based across concurrency levels.
**Why it matters:** "How many GPUs at 200 concurrent users?" → Sizing Calculator. It's *capacity planning*, distinct from the Profiler's *bottleneck finding*.
**Domains:** D3, D4.

---

## §5 — Tuning & customization

### Tuning ladder (prompt → RAG → fine-tune → align)
The cheapest-fix-first escalation: **(1) prompting** (~free) → **(2) RAG tuning** (facts) → **(3) LoRA/QLoRA** → **(4) full SFT** → **(5) DPO/RLHF** (alignment). Each rung ~10× the prior.
**NVIDIA context:** NAT's Config Optimizer (`nat optimize`) covers rung 1; NeMo Customizer covers 3–5.
**Why it matters:** **Climb, never jump.** Most "we need fine-tuning" is solved by prompting or RAG. **Fine-tuning = skills/behavior; RAG = knowledge/facts.** The single most-tested tuning principle.
**Domains:** D3.

### Prompt engineering / Parameter optimization
Shaping behavior via instructions, few-shot examples, output schemas, and sampling params — and **automating** that search against an eval set.
**NVIDIA context:** NAT's **Config Optimizer** (run via `nat optimize`; install the `[config-optimizer]` extra) does Optuna numeric search + a genetic-algorithm prompt optimizer over prompt wording, temperature, top-p, max tokens, few-shot, and tool descriptions to maximize a target eval metric.
**Why it matters:** Rung 1 of the ladder; "tune the agent without training" → parameter optimization. Replaces manual prompt tweaking with data-driven search.
**Domains:** D2, D3.

### SFT (Supervised Fine-Tuning)
Training a model on labeled input→output pairs to update weights toward a task/style.
**NVIDIA context:** A NeMo Customizer job; full SFT updates all weights (rung 4 — best quality ceiling, most data/GPU).
**Why it matters:** The "skill/behavior/format gap, at scale, with labeled data" answer — *after* prompting and RAG have plateaued.
**Domains:** D3.

### LoRA / QLoRA (PEFT)
**Parameter-efficient fine-tuning:** train small adapter matrices (~0.1–1% of params) instead of all weights. **QLoRA** quantizes the base to 4-bit so larger models fit on small GPUs.
**NVIDIA context:** A NeMo Customizer technique (rung 3); ~100s–10k examples.
**Why it matters:** "Consistent format at scale, plenty of labeled examples, limited GPU" → LoRA/QLoRA. Cheaper than full SFT with most of the benefit.
**Domains:** D3.

### DPO (Direct Preference Optimization)
Preference alignment trained **directly on (chosen, rejected) pairs** — no reward model, no RL loop.
**NVIDIA context:** A NeMo Customizer job (NAT finetuning harness drives it); the **default** preference-alignment method.
**Why it matters:** "Align tone/preference with preference pairs, minimal infra, no reward-model expertise" → DPO. Cheaper/more stable than RLHF.
**Domains:** D3.

### RLHF / GRPO
**RLHF:** classic alignment via a learned **reward model + RL (PPO)** — frontier-scale, complex. **GRPO:** RL that compares a **group of sampled rollouts** against a reward — no preference pairs, no reward model.
**NVIDIA context:** NAT's finetuning harness supports **GRPO via OpenPipe ART**.
**Why it matters:** "We can auto-score each run (a reward) but have no chosen/rejected pairs, and want the agent to improve with experience" → **GRPO**. DPO needs pairs; GRPO needs a reward.
**Domains:** D3.

---

## §6 — Safety, guardrails & compliance

### Guardrail
A **runtime** safety mechanism that monitors/validates/modifies inputs, outputs, or actions to enforce policy and prevent harm.
**NVIDIA context:** **NeMo Guardrails** is NVIDIA's framework, configured in **Colang**.
**Why it matters:** Guardrails are *runtime* controls (vs *pre-deployment* auditing). Know what each rail category catches.
**Domains:** D9.

### NeMo Guardrails
NVIDIA's open-source toolkit for **programmable guardrails** on LLM apps, with **five rail categories** and Colang configuration.
**NVIDIA context:** Two modes — **library mode** (`pip install nemoguardrails`, runs on CPU) and **microservice mode** (container, NGC key, GPU for Safety NIMs). Core API class: **`LLMRails`**, configured by **`RailsConfig`**.
**Why it matters:** The exam tests the rail categories and Colang. **The five categories are input, dialog, output, retrieval, execution** — *not* "topical" (see below).
**Domains:** D9.

### The five rail categories (input, dialog, output, retrieval, execution)
**Input rails:** screen/alter user input (block harmful, mask PII, rephrase). **Dialog rails:** govern conversation flow / how the LLM is prompted (includes *topical* control). **Output rails:** screen/alter the LLM response (fact-check, PII redaction). **Retrieval rails:** screen/alter retrieved RAG chunks. **Execution rails:** validate inputs/outputs of custom actions/tools.
**NVIDIA context:** All implemented as **Colang flows**.
**Why it matters:** **Common distractor:** "topical" is *not* a sixth top-level category — topical control is a **dialog-rail behavior** (or enforced by the Topic Control NIM). And **retrieval rails** are the category people forget. Memorize the real five.
**Domains:** D9, D6.

### Colang
NVIDIA's domain-specific language for defining guardrail **flows, actions, and event handlers** (Colang 2.0 has a Python-like syntax).
**NVIDIA context:** The configuration language for NeMo Guardrails; all rails are Colang flows.
**Why it matters:** "What language configures NeMo Guardrails behaviors?" → Colang. Recognize Colang-style flow syntax.
**Domains:** D9.

### Safety NIMs (NemoGuard family)
NIM microservices that classify content for safety, deployed as input/output rails. Exact names: **Llama 3.1 NemoGuard 8B Content Safety** (toxicity/harm taxonomy), **Llama 3.1 NemoGuard 8B Topic Control** (keep within approved topics), **NemoGuard Jailbreak Detect** (prompt-injection/jailbreak attempts).
**NVIDIA context:** Integrated with NeMo Guardrails as parallel input/output rails; newer reasoning-capable variants exist (e.g., Nemotron Content Safety Reasoning).
**Why it matters:** Use the **exact product names** — generic "Content Safety NIM" can be a less-precise distractor. Map each NIM to its job (harm / topic / jailbreak).
**Domains:** D9.

### Jailbreak / Prompt injection
Adversarial attacks that **bypass safety/system-prompt constraints** — role-play, system-prompt extraction, encoding tricks, injected instructions in tool/RAG content.
**NVIDIA context:** Detected by **NemoGuard Jailbreak Detect** + Guardrails input rails; tested by **garak**.
**Why it matters:** Know jailbreak (bypass safety) vs prompt injection (untrusted content hijacks the agent) and which rail/NIM defends each.
**Domains:** D9, D2.

### PII (Personally Identifiable Information) & redaction
Data identifying an individual (names, emails, SSNs, addresses). **Redaction** detects and masks it before storage/transmission/display.
**NVIDIA context:** NeMo Guardrails does PII detection/redaction via integrations (GLiNER, Presidio, Private AI) on input/output rails; NAT telemetry redaction masks PII in traces.
**Why it matters:** Two redaction contexts — guardrail output rail (user-facing) and observability traces. Know both.
**Domains:** D9, D8.

### garak (Generative AI Red-teaming and Assessment Kit)
NVIDIA's open-source **LLM vulnerability scanner** — probes for jailbreaks, prompt injection, data leakage, toxicity, hallucination, etc. ("nmap for LLMs").
**NVIDIA context:** Open-source (Apache 2.0), integrates with NeMo Guardrails; NAT also ships red-teaming middleware.
**Why it matters:** "Which NVIDIA tool automates LLM red-teaming / vulnerability scanning?" → garak. Don't confuse with runtime guardrails (garak *tests*, guardrails *enforce*).
**Domains:** D9.

### Safety for Agentic AI (Blueprint) / pre-deployment auditing
NVIDIA's Blueprint to improve **safety, security, and privacy at build, deploy, and run** stages — the current home for pre-deployment safety auditing (bias/safety/policy checks before production).
**NVIDIA context:** Supersedes the older standalone "Safety Blueprint"; works with garak + Guardrails + Safety NIMs.
**Why it matters:** Distinguish **pre-deployment auditing** (catch issues before ship) from **runtime guardrails** (catch them live). Both belong to the safety lifecycle.
**Domains:** D9.

---

## §7 — Deployment, serving & inference

### NVIDIA Dynamo
A datacenter-scale, low-latency **distributed inference framework** for serving generative/reasoning models.
**NVIDIA context:** Key techniques: **disaggregated serving** (separate **prefill** and **decode** workers — prefill is compute-bound, decode memory-bound — with RDMA/NIXL KV-cache transfer), **KV-cache-aware request routing** (reuse cache, avoid recompute), dynamic GPU scheduling, and KV offloading. Backends: TensorRT-LLM, vLLM, SGLang.
**Why it matters:** "Scale LLM inference, separate prefill/decode, reuse KV cache across requests" → Dynamo. Prefix/KV reuse especially helps agents that share long system prompts.
**Domains:** D4.

### Disaggregated serving (prefill / decode)
Splitting LLM inference into a compute-bound **prefill** phase (process the prompt) and a memory-bound **decode** phase (generate tokens), run on **independently scaled** workers.
**NVIDIA context:** A core Dynamo capability; KV cache moves prefill→decode over RDMA.
**Why it matters:** Lets you scale the two phases separately for better GPU utilization. The "why separate prefill and decode?" question = different resource profiles.
**Domains:** D4.

### KV cache / Prefix caching
The cached attention key/value tensors for already-processed tokens; **prefix caching** reuses the KV cache for a **shared prompt prefix** across requests to skip recomputation.
**NVIDIA context:** Dynamo's KV-aware routing reuses prefixes; high value for agents that resend the same system prompt every turn.
**Why it matters:** "Many requests share a long system prompt — cut redundant compute" → prefix/KV-cache reuse. A recurring inference-efficiency lever.
**Domains:** D4.

### TensorRT-LLM
NVIDIA's library for **optimized LLM inference** on NVIDIA GPUs (kernel fusion, quantization, in-flight batching) — the engine inside many NIMs.
**NVIDIA context:** A NIM/Dynamo backend; the "why is the NIM fast?" answer.
**Why it matters:** Know it as the optimization layer beneath NIM serving, alongside vLLM/SGLang as alternative backends.
**Domains:** D4, D7.

### Structured / guided decoding (NIM)
Constraining the decoder so output **must** conform to a schema/grammar/regex/choice — guaranteeing a valid parse, unlike "please reply in JSON."
**NVIDIA context:** On NIM, ride the **`nvext`** extension: `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` (xgrammar is the default backend). Pattern: **Pydantic model → JSON schema → `guided_json` → validate**.
**Why it matters:** Three structured-output mechanisms differ — prompt-only (no guarantee), JSON mode (valid JSON, any shape), **guided decoding (schema-guaranteed)**. The exam tests the difference; on NIM it's `nvext.guided_json`.
**Domains:** D2, D4.

---

## §8 — Protocols & interoperability

### MCP (Model Context Protocol)
An open standard ("USB-C for AI tools") for **dynamic tool discovery and invocation**: an MCP **server** exposes tools (plus resources and prompts) with standard descriptions; an MCP **client** connects at runtime to discover and call them.
**NVIDIA context:** NAT can be an **MCP client** (consume remote tools in a workflow) and an **MCP server** (publish workflow functions as tools, including via **FastMCP**). Transports: stdio (local) or streamable HTTP (remote).
**Why it matters:** "Discover tools at runtime across a boundary, standardized" → MCP. Know the three primitives (tools/resources/prompts) and that NAT is both client and server. Third-party MCP servers are an injection/supply-chain surface.
**Domains:** D2, D7.

### A2A (Agent-to-Agent Protocol)
A protocol for **inter-agent** messaging: typed payloads, task delegation, status tracking, and agent **discovery** between independently deployed agent services.
**NVIDIA context:** NAT supports A2A as **client** (delegate to remote agents) and **server** (publish a workflow as a discoverable A2A agent), built on the **official A2A Python SDK**, with authentication support.
**Why it matters:** **MCP = agent↔tools; A2A = agent↔agent.** "Build a team of distributed agents that delegate to each other" → A2A. Don't swap the two.
**Domains:** D1, D7.

---

## §9 — Observability & operations

### OpenTelemetry (OTel)
An open observability standard for collecting distributed **traces, metrics, and logs**.
**NVIDIA context:** NAT exports telemetry via OpenTelemetry to backends like **Phoenix, Weave, Langfuse**, with workflow→step→tool tracing granularity.
**Why it matters:** "Standard, vendor-neutral way to trace an agent's steps" → OpenTelemetry. Know NAT exports OTel and which backends it targets.
**Domains:** D8.

### Tracing / spans
A **trace** is the full record of one agent run; **spans** are nested timed segments (workflow → step → tool call) within it.
**NVIDIA context:** NAT produces spans at workflow/step/tool level (needs stable IDs so trajectories can be reconstructed).
**Why it matters:** Tracing is how you reconstruct a trajectory to debug *where* a multi-step run failed (planning vs tool vs synthesis).
**Domains:** D8, D3.

### Drift detection
Monitoring for **distribution shift** between training/eval data and live traffic (input drift) or degrading output quality over time.
**NVIDIA context:** An online-eval signal; novel drifted failures feed back into the golden set (Data Flywheel).
**Why it matters:** "Quality silently degrades after deploy as inputs change" → drift; the fix loop is online detection → add to golden set → re-tune.
**Domains:** D8, D3.

### Failure-mode distribution
A breakdown of **where** in the trajectory failures occur — planning, tool call, or synthesis.
**NVIDIA context:** A key diagnostic artifact from trajectory eval + tracing.
**Why it matters:** Tells you *which layer* to fix. "Most failures are at the tool-call step" → harden tool schemas/args, not the model.
**Domains:** D3, D8.

---

## §10 — Cognition, memory & planning

### Memory (agent)
An agent's ability to **retain and recall** information across interactions — conversation history, user preferences, learned facts, task context.
**NVIDIA context:** NAT provides a memory system with multiple **Object Store** backends and the **Automatic Memory Wrapper**.
**Why it matters:** Know the split: **short-term** (in-context history) vs **long-term** (persisted store, retrieved on demand). Memory enables continuity/personalization across sessions.
**Domains:** D5.

### Object Store (NAT)
NAT's **storage backend abstraction** for agent memory/state — a unified interface over the underlying tech.
**NVIDIA context:** Backends: **in-memory** (dev), **MySQL** (queryable persistent), **Redis** (low-latency persistent), **S3** (large-object archival).
**Why it matters:** Config question: "low-latency persistent memory" → Redis; "queryable" → MySQL; "dev/throwaway" → in-memory; "big artifacts" → S3.
**Domains:** D5.

### Automatic Memory Wrapper (NAT)
A NAT component that **transparently** adds memory to any agent — intercepts inputs/outputs, stores relevant info in an Object Store, and injects relevant memories into context on later calls.
**NVIDIA context:** Zero-code memory; less control than manual memory management.
**Why it matters:** "Add memory to an agent without rewriting it" → Automatic Memory Wrapper. Trade-off: convenience vs control.
**Domains:** D5.

### Chain-of-Thought (CoT)
Prompting the model to produce **intermediate reasoning steps** before the final answer, improving accuracy on complex reasoning.
**NVIDIA context:** Leveraged by NAT's ReAct and Reasoning agents; **test-time compute** extends it (multiple chains, pick best).
**Why it matters:** Costs tokens/latency; reasoning models do it internally. CoT + strict output schema → put the reasoning field *first* (so the model reasons before committing to the answer).
**Domains:** D5, D2.

### Test-time compute
Spending **extra inference compute** (multiple calls, deliberation, self-verification, candidate selection) to raise output quality — trading latency/cost for accuracy.
**NVIDIA context:** Embodied in NAT's **Reasoning Agent** (plan / generate candidates / select).
**Why it matters:** "Spend more at inference for a better answer on hard tasks" → test-time compute / Reasoning Agent. It's an *inference-time* lever, not training.
**Domains:** D5, D3.

### Hallucination
When an LLM generates content **not supported** by its input context or training data — fabricated facts, citations, or claims stated as real.
**NVIDIA context:** Mitigated by RAG (grounding), Guardrails fact-checking output rail, and **faithfulness** evaluation.
**Why it matters:** In RAG, hallucination = **low faithfulness** → a *generation* fix (grounding/prompt), assuming retrieval was good. The most common safety/quality failure mode.
**Domains:** D5, D3, D6, D9.

---

## One-screen exam cheat lines

- **MCP = agent↔tools; A2A = agent↔agent.** NAT is both client and server for each.
- **8 NAT agent types** (include **Parallel Executor**); ReAct re-plans each step, ReWOO plans once, Tool Calling is single-shot, Router dispatches, Sequential always runs every stage, Reasoning plans ahead (test-time compute).
- **NeMo Guardrails = 5 rail categories: input / dialog / output / retrieval / execution.** "Topical" is a *dialog-rail behavior*, not a category.
- **RAG = facts; fine-tuning = skills.** Tuning ladder: prompt → RAG → LoRA/QLoRA → SFT → DPO/RLHF (climb, never jump).
- **DPO needs (chosen, rejected) pairs; GRPO needs a reward (no pairs); RLHF needs a reward model + RL.**
- **NeMo microservices:** Customizer (fine-tune) · Evaluator (eval/benchmarks/judge) · Curator (data prep) · Retriever (RAG NIMs) · Guardrails (safety).
- **Profiler finds the bottleneck; Sizing Calculator estimates GPUs.**
- **Dynamo:** disaggregated prefill/decode + KV-cache-aware routing. **NIM Operator:** K8s lifecycle for NIMs.
- **Judge biases:** position (swap orderings), verbosity (reward conciseness), self-preference (use a different judge).
- **Compound error:** `p_step ^ n_steps` — fewer steps or harden the weak step, not "a bigger model."
- **Safety NIMs (exact):** NemoGuard Content Safety · Topic Control · Jailbreak Detect. Red-team tool = **garak**.
