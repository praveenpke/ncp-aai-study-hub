# Builder's Pulse — what the field is saying (2025-26)

## Why this file

The NCP-AAI domains test concepts; this file maps those concepts onto the live arguments practitioners are actually having — so when an exam question says "compaction" or "LLM-as-judge bias" or "agent-to-agent protocol," you know not just the definition but *why the industry cares and who pushed it there*. Read it as connective tissue across your six domain notes (03 evaluation/tuning, 04 deployment/scaling, 07 NVIDIA platform, 08 run/monitor/maintain, 09 safety/compliance, 10 human-AI interaction).

---

## The 10 conversations that matter

### 1. Context engineering: the term that ate prompt engineering

**What the debate is.** Mid-2025, the field renamed its core craft. The argument: "prompt engineering" had been captured by a conception of the work — clever phrasing of a single instruction — that no longer matched what production agent systems require, where the prompt is a small fraction of a context window otherwise filled with history, tool outputs, retrieved docs, and state.

**The voices.**
- [Tobi Lütke](https://x.com/tobi/status/1935533422589399127) (Shopify CEO, June 18, 2025) supplied the canonical definition: "the art of providing all the context for the task to be plausibly solvable by the LLM."
- [Andrej Karpathy endorsed it](https://x.com/karpathy/status/1937902205765607626) six days later: "+1 for 'context engineering' over 'prompt engineering'... the delicate art and science of filling the context window with just the right information for the next step."
- [Simon Willison tracked the shift](https://simonwillison.net/2025/Jun/27/context-engineering/) and framed prompt engineering as now a *subset* of context engineering.

**The technical anchor.** Chroma's [Context Rot report](https://research.trychroma.com/context-rot) (Kelly Hong, Anton Troynikov, Jeff Huber, July 2025):
- Tested 18 models (GPT-4.1, Claude 4, Gemini 2.5, Qwen3...) on tasks of controlled input length.
- Finding: models do *not* process context uniformly — the 10,000th token is not handled as reliably as the 100th, and performance grows increasingly unreliable as input grows, even on simple tasks.
- Practical consequence: the naive "just use the 1M window" instinct died; effective context budgets are far below advertised limits.

Anthropic's [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (Sept 2025) made it canon: attention is a finite budget (every token attends to every other — n² relationships), so treat context as a scarce resource. Their three long-horizon techniques:
- **Compaction** — summarize a near-full window with high fidelity, reinitiate a fresh window with the summary.
- **Structured note-taking** — persistent memory written to files (NOTES.md, TODO.txt) *outside* the window, reloaded when needed.
- **Sub-agent architectures** — specialists explore deeply in clean windows and return compact summaries to a coordinating parent.

**Where it lands.** Context curation is now considered *the* core agent-engineering skill — more leverage than model choice for most failures. The default playbook:
- Keep the window lean; curate tool outputs instead of dumping them raw.
- Externalize durable state (plans, findings, intermediate artifacts) to files or a store.
- Isolate exploration in sub-agents that return summaries, not transcripts.
- Compact at thresholds rather than waiting for hard window limits.

**For your exam:** "agent degrades on long multi-step tasks" → context rot; remedies are compaction / external memory / sub-agent isolation. Maps to failure diagnosis in run/monitor/maintain (domain 08) and to KV-cache/context budgeting in deployment (domain 04).

### 2. Anthropic's "Building Effective Agents" lineage: workflows vs agents, simple harnesses

**What the debate is.** How much autonomy should you actually give the model, and how much scaffolding belongs in code? Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) (Dec 2024) is the most-cited agent design doc of the era. Its core distinction:
- **Workflows** — LLM calls and tools orchestrated through *predefined code paths*: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
- **Agents** — the LLM *dynamically directs its own process and tool use*, deciding the path at runtime.

Its core advice: "the most successful implementations use simple, composable patterns rather than complex frameworks." The escalation ladder:
1. Single augmented LLM call (+ retrieval, tools) — try this first.
2. Workflow patterns when the task decomposes predictably (chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer).
3. A true agent loop only when the task genuinely needs open-ended autonomy — and accept the cost/latency/error tradeoff knowingly.

**The harness turn.** Through 2025 the discourse shifted from "which framework" to "what's in your harness." A harness, in current usage, is everything wrapped around the model:
- the agent loop itself (reasoning → tool call → observation, repeated),
- the tool surface (which tools, how described, how results are truncated/curated),
- context management (compaction triggers, memory, sub-agent spawning),
- stop conditions, budgets, and error handling.

Claude Code's design (a simple while-loop + tools + filesystem) became the reference implementation people reverse-engineered, and "harness engineering" is now treated as a first-class systems discipline rather than a thin wrapper. Anthropic's context-engineering post (thread #1) is effectively part two of the same lineage: as models improve, the harness — not the prompt — is where engineering effort concentrates.

**Where it lands.** Practitioner consensus: most production "agents" should be workflows; reserve autonomy for where it pays for its unpredictability; keep the harness simple enough to debug.

**For your exam:** the workflow-pattern taxonomy (chaining, routing, parallelization, orchestrator-worker, evaluator-optimizer) shows up in architecture questions. Note "evaluator-optimizer" is literally an eval loop (domain 03), and "use the simplest pattern that works" is the standing answer to over-engineering scenarios.

### 3. 12-Factor Agents (Dex Horthy / HumanLayer)

**What the debate is.** Dex Horthy's [12-factor-agents](https://github.com/humanlayer/12-factor-agents) (April 2025; a top-trending GitHub repo, plus AI Engineer and [MLOps Community](https://home.mlops.community/public/videos/12-factor-agents-patterns-of-reliable-llm-applications-dexter-horthy-agents-in-production-2025-2025-08-06) talks) adapted Heroku's 12-Factor App methodology to LLM software. The thesis: most products billing themselves as "AI agents" aren't very agentic — the ones that work are mostly deterministic code with LLM decisions sprinkled in at exactly the right points. Greenfield "framework does everything" agents stall at ~70-80% quality; owning the moving parts gets you to production.

**The factors.**
1. Natural language → tool calls
2. Own your prompts
3. Own your context window
4. Tools are just structured outputs
5. Unify execution state and business state
6. Launch/Pause/Resume with simple APIs
7. Contact humans with tool calls
8. Own your control flow
9. Compact errors into context window
10. Small, focused agents
11. Trigger from anywhere, meet users where they are
12. Make your agent a stateless reducer
13. *(appendix)* Pre-fetch all the context you might need

**Mapping to your six domains:**

| Factors | Why | Your notes file |
|---|---|---|
| 2, 3, 12 | Owned prompts/context + stateless reducer → reproducible, testable, version-controlled | 03 evaluation/tuning |
| 6, 11, 12 | Pause/resume APIs, trigger anywhere, statelessness → horizontal scaling, durable execution | 04 deployment/scaling |
| 10 | Small focused agents → model-per-task routing, SLM/NIM microservices | 07 NVIDIA platform |
| 5, 9 | Unified state → debuggable traces; error compaction → bounded self-healing retry loops | 08 run/monitor/maintain |
| 4, 8 | Structured outputs + owned control flow → deterministic gates before side effects | 09 safety/compliance |
| 7 | Human contact as a first-class tool call, not an afterthought UI | 10 human-AI interaction |

**Where it lands.** The closest thing the field has to a shared production checklist — and notably *framework-agnostic*: the factors critique frameworks that hide prompts, control flow, or context assembly behind abstractions you can't inspect. (LangGraph fares well by this standard precisely because it exposes state and control flow — which you already know.) Horthy's "dumb zone" framing — quality degrades well before the window is physically full, so keep utilization low — independently reinforced thread #1.

### 4. The evals movement — and its backlash

**What the debate is.** Whether systematic evaluation is the discipline that separates shipping teams from demo teams, or process theater that slows fast iterators down.

**The movement (pro side).**
- [Hamel Husain](https://hamel.dev/blog/posts/evals/) ("Your AI Product Needs Evals") and [Shreya Shankar](https://www.sh-reya.com/) built the dominant methodology, taught to thousands via their [AI Evals course](https://maven.com/parlance-labs/evals) and distilled in Hamel's [Evals FAQ](https://hamel.dev/blog/posts/evals-faq/).
- Tenet 1 — **error analysis over scores**: read your actual traces, open-code failures, cluster them into a failure taxonomy *before* writing a single metric. "Look at your data" is the mantra; generic metric dashboards without a failure taxonomy are the anti-pattern.
- Tenet 2 — **binary pass/fail over 1-5 Likert scales**: humans and LLM judges are far more consistent on yes/no with a written critique than on numeric scales whose midpoints mean nothing.
- Tenet 3 — **validate the judge against human labels** before trusting it; report judge↔human agreement, not just judge scores.
- Shankar's [Who Validates the Validators?](https://arxiv.org/abs/2404.12272) (UIST 2024) supplied the research backbone: LLM judges inherit the flaws of the models they judge, and graders exhibit **criteria drift** — you need criteria to grade outputs, but grading outputs changes your criteria, so alignment is iterative, never one-shot.
- [Eugene Yan](https://eugeneyan.com/writing/llm-evaluators/) covers judge reliability and built [AlignEval](https://eugeneyan.com/writing/aligneval/) around the loop: label data → align evaluator → build the harness.
- The bull case hardened into "evals are your moat": your labeled failure data and regression suite are the asset competitors can't copy, even when everyone has the same base models.

**The backlash (anti side).** September 2025 brought a wave of "evals are overrated" posts across X and Substack. The recurring arguments:
- Top labs ship on taste, dogfooding, and fast feedback loops rather than formal harnesses.
- Eval suites ossify around stale failure modes while the product and models move on.
- Maintaining judges and golden sets is overhead a fast-moving product can't afford.
- Vibe-checking by people with good judgment outperforms a brittle proxy metric.

**The responses.**
- Shankar, [In Defense of AI Evals, for Everyone](https://www.sh-reya.com/blog/in-defense-ai-evals/) (Sept 2025): defines evals as *systematic measurement of application quality* — rigor dials up or down with stakes; the strawman being attacked (mandatory giant benchmarks for every prototype) is not what anyone teaches.
- Hamel on [Vanishing Gradients ep. 60](https://hugobowne.substack.com/p/episode-60-10-things-i-hate-about-e83): everyone relies on evals — some just consume them upstream (model providers ran them for you); and not every problem needs one (an obvious formatting bug just gets fixed).

**Where it lands.** Synthesis position: error analysis is non-negotiable, formal harnesses are proportional to risk, and the production loop — novel failure → labeled example → regression gate — is how teams actually improve.

**For your exam:** this is domain 03's worldview verbatim — golden datasets that grow from production bugs, LLM-as-judge validated against humans, offline gates + online monitoring as complements. If a question offers "ship and watch dashboards" vs "build a regression suite from observed failures," pick the latter.

### 5. Deep agents / long-horizon agents

**What the debate is.** How to get agents past ~15-minute tasks without quality collapse. LangChain's [Deep Agents post](https://blog.langchain.com/deep-agents/) (mid-2025) named the answer by reverse-engineering Claude Code, Deep Research, and Manus: "shallow" tool-loop agents fail on long horizons; deep agents add four things —
- **A planning tool** (write_todos-style task lists the agent maintains and re-reads)
- **A filesystem** (shared workspace, scratch notes, artifacts — context that survives the window)
- **Sub-agents** (spawned for deep exploration, returning summaries — context quarantine)
- **A detailed system prompt** (Claude Code's prompt is famously long and example-dense)

The [deepagents library](https://github.com/langchain-ai/deepagents) ships this as a batteries-included harness on the LangGraph runtime — pitched as Claude Code's architecture decoupled from Claude, usable with any tool-calling model. (As of LangChain/LangGraph **1.x** in 2026, the stack layers cleanly: LangGraph = the graph runtime, `langchain.agents.create_agent` = the minimal agent harness on top of it — this **superseded** the now-deprecated `create_react_agent` from `langgraph.prebuilt` — and deepagents = the opinionated harness on top of `create_agent` with filesystem, sub-agents, and middleware bundled in.)

**Shallow vs deep, at a glance:**

| | Shallow agent | Deep agent |
|---|---|---|
| Loop | LLM + tools in a while-loop | Same loop + planning, filesystem, sub-agents |
| Horizon | Single-session, minutes | Hours/days, resumable |
| State | All in the context window | Externalized to files/store; window stays lean |
| Failure mode | Context rot, lost goals | Coordination overhead, sub-agent context loss |

**Connecting to what you already know.** You know LangGraph and the deep-agents concepts — the point of this thread is that the pattern *converged independently*: Anthropic's compaction/note-taking/sub-agents (thread #1), Manus's filesystem-as-context and todo recitation (thread #8), and deepagents' four pillars are the same design discovered at least three times. Read the components functionally: planning = attention manipulation (reciting goals keeps them out of the rotted middle of the context); filesystem = unlimited, persistent, agent-operated memory; sub-agents = context isolation, not anthropomorphic role-play.

**For your exam:** long-horizon task questions resolve to this pattern. It also underlies NVIDIA's agent blueprints — orchestrator + specialized workers + externalized state is the same shape (domain 07).

### 6. MCP ecosystem explosion + A2A goes to the Linux Foundation

**What happened.**
- Anthropic launched **MCP** (Nov 2024) as the "USB-C for AI tools" — one open protocol replacing N×M custom integrations between models and data sources.
- OpenAI adopted it in March 2025 (Agents SDK, Responses API, ChatGPT desktop); Google/Gemini, Microsoft Copilot, Cursor, and VS Code followed — rare full-industry convergence on a competitor's standard.
- The official registry hit ~2,000 entries by Nov 2025 — 407% growth from its September launch batch ([one-year retrospective](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/)) — with 10,000+ active public servers in the wild.
- Dec 9, 2025: both major protocols landed under one roof — the Linux Foundation announced the **Agentic AI Foundation (AAIF)**, anchored by three contributed projects: [Anthropic's MCP, Block's goose, and OpenAI's AGENTS.md](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation). Platinum backers include AWS, Anthropic, Block, Bloomberg, Cloudflare, Google, Microsoft, and OpenAI. MCP is now **vendor-neutral / community-governed**, not "Anthropic's protocol."
- Google's **A2A (Agent2Agent)** protocol launched April 2025 with 50+ partners and was [donated to the Linux Foundation in June 2025](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents); it [passed 150 member organizations within a year](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year) (AWS, Microsoft, Salesforce, SAP, ServiceNow...).
- **A2A v1.0 shipped early 2026** ([announcement](https://a2a-protocol.org/latest/announcing-1.0/); the spec crossed 150+ production orgs by April 2026) — the first production-grade release: **Signed Agent Cards** (cryptographic domain verification of an agent's identity), **multi-tenancy** (multiple agents behind one endpoint), and **version negotiation** (v0.3→v1.0 backward-compat). It is now governed under the AAIF and already iterating past 1.0.
- Division of labor: **MCP = agent ↔ tools/data. A2A = agent ↔ agent** (JSON-RPC 2.0 + SSE transport, Agent Cards for capability discovery). Complementary, not competing.

**The security counter-current.**
- [Simon Willison flagged MCP's prompt-injection problem](https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/) (April 2025), then coined the [lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) (June 2025): an agent combining **private data access + exposure to untrusted content + an external communication channel** is exploitable by a single poisoned document — no code vulnerability required.
- **Tool poisoning** — malicious instructions hidden in tool descriptions, visible to the model but not normally shown to users — became its own attack class; benchmarks like MCPTox (45 real MCP servers, 353 tools) showed >70% attack success on some models.
- The composability that makes MCP attractive is exactly what assembles the trifecta by accident: mix community servers and you've granted all three legs without noticing.

**The codified version.** This discourse hardened into a standard: the [OWASP Top 10 for Agentic Applications (2026)](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/), released **Dec 9, 2025** by the OWASP GenAI Security Project (IDs **ASI01–ASI10**: agent goal hijacking, tool misuse, identity/privilege abuse, supply chain, unexpected code execution, memory poisoning, insecure inter-agent comms, cascading failures, human-agent trust exploitation, rogue agents). It sits *alongside* the established OWASP Top 10 for LLM Applications 2025 (`LLM01:2025`…), which still has prompt injection at LLM01. A current agentic guide should cite **both** lists.

**For your exam:** know MCP vs A2A roles cold. The mitigation for the trifecta is *removing one leg* — sandbox egress, allowlist tools, isolate untrusted content from privileged context. That's domain 09 (safety/compliance) plus approval gates on tool use (domain 10). When a question references an "agent-specific" risk catalog, the answer is the **OWASP Agentic Apps Top 10 (ASI01–ASI10)**, not the LLM Top 10.

### 7. Multi-agent: skeptics vs advocates

**What the debate is.** June 2025, dueling posts days apart — the field's sharpest public disagreement.
- **Skeptic:** Cognition's Walden Yan, [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents). Parallel sub-agents make conflicting decisions based on implicit assumptions they can't share; stitching their outputs produces incoherent results. His two principles: *share full context* (every agent sees the full trace, not a summary) and *actions carry implicit decisions* (parallel actors inevitably contradict each other). For deep, stateful work like coding — Devin's domain — use one agent, one continuous context. His line: "At the core of reliability is Context Engineering."
- **Advocate:** Anthropic, [How we built our multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system). An orchestrator (Opus 4) spawning parallel search sub-agents (Sonnet 4) beat single-agent Opus 4 by **90.2%** on internal research evals — at roughly **15× the tokens** of a chat interaction. Token spend explained most of the performance variance; parallelization compresses wall-clock time for breadth-first work.

Engineering lessons buried in the Anthropic post that practitioners quote constantly:
- The orchestrator must give sub-agents *explicit objectives, output formats, and effort budgets* — vague delegation produces duplicated or divergent work.
- Effort should scale with task complexity (simple fact-check = 1 agent with 3-10 tool calls; complex research = many agents with divided responsibilities).
- Debugging multi-agent systems requires full production tracing — non-deterministic failures can't be reproduced locally (ties directly to thread #10).

**Where it lands.** Less a contradiction than a task taxonomy:
- Multi-agent wins on **wide, parallelizable, read-heavy** work (research, gathering, evaluation) where sub-tasks are independent and results merge cleanly.
- Single-agent wins on **deep, sequential, write-heavy** work (codebases, long documents) where every action depends on shared context.
- Both sides agree the bottleneck is context flow between agents — which is why thread #1 is the meta-thread.

**For your exam:** "when do you split into multiple agents?" → when sub-tasks are independent and the token cost is justified. "Why did the multi-agent system produce inconsistent output?" → context fragmentation / conflicting implicit decisions. Maps to architecture choices in domains 04 and 08.

### 8. Production war stories: reliability, token economics, and small models

**The sobering math practitioners repeat.**
- **Errors compound**: 95% per-step accuracy over a 10-step task is ~60% task success (0.95^10). This single fact drives most harness design — checkpoints, validation gates, bounded retries, and the preference for shorter chains.
- **Costs scale superlinearly with autonomy**: agents re-read their whole growing context every step, so cost grows roughly quadratically with trajectory length unless caching is engineered. (Why quadratic: step 1 reads 1 unit, step 2 reads 2, …, step N reads N; the sum 1+2+…+N ≈ N²/2. A 50-step trajectory costs ~25× a 10-step one for the same per-step work — which is exactly why prefix/KV-cache stability below is the lever that pulls it back toward linear.)
- **Loop detection matters**: runaway agent conversations have produced five-figure surprise bills; budget caps and step limits are guardrails, not paranoia.

**The flagship field report.** Manus's [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) (Yichao "Peak" Ji, July 2025), written after four complete framework rebuilds:
- **KV-cache hit rate is the single most important production metric.** Agent input:output token ratio runs ~100:1, so stable prompt prefixes and append-only context dominate both cost and latency. (Cached input tokens are billed at roughly an order of magnitude less than uncached across major providers.)
- **Mask tools, don't remove them** mid-task — preserves the cache prefix and avoids confusing the model with a shifting action space.
- **Use the filesystem as context** — unlimited, persistent, restorable; compress by dropping content but keeping paths/URLs.
- **Recite the todo list** — rewriting objectives at the end of context fights lost-in-the-middle.
- **Keep errors in context** — failed actions and stack traces teach the model within the episode; scrubbing them invites repetition.
- **Avoid few-shot ruts** — uniform action histories make agents overgeneralize a rhythm; inject controlled variation in serialization and phrasing.

**The small-model thread.** NVIDIA Research's position paper [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) (Belcak et al., June 2025; [project page](https://research.nvidia.com/labs/lpr/slm-agents/)):
- Claim: most agent invocations are narrow, repetitive, and format-constrained, so SLMs (<10B params, by their 2025 definition) are "sufficiently powerful, inherently more suitable, and necessarily more economical."
- Economics: serving a ~7B SLM is 10-30× cheaper (latency, energy, FLOPs) than a 70-175B LLM.
- The future they sketch is **heterogeneous**: SLMs handle routine steps, LLMs are escalation paths for genuinely hard reasoning, and logged agent traces feed an LLM→SLM fine-tuning/distillation flywheel.
- The paper explicitly invited public rebuttals (published on the project page) and got real debate — counterpoints include orchestration/routing complexity, and the generalist premium when task distributions drift.

**For your exam:** KV-cache/prefix stability and route-to-cheaper-models are domain 04 staples; cost-per-task tracking and loop guards are domain 08; the SLM paper is NVIDIA's own thesis — expect sympathy for "right-size the model, one NIM microservice per role" answers (domain 07).

### 9. Memory for agents

**What the debate is.** Can agents learn across sessions, not just within one window — and do you need a dedicated memory product for that?

**The lineage and the landscape.**
- **MemGPT** ([arXiv 2310.08560](https://arxiv.org/abs/2310.08560), Oct 2023) framed it durably: treat the LLM like an OS managing memory tiers — in-context "RAM" vs external "disk," with the agent paging data via tools. It became **Letta**, which added research on "sleep-time compute" — background memory consolidation between sessions.
- **Mem0** ([arXiv 2504.19413](https://arxiv.org/abs/2504.19413)) — extraction-based drop-in memory layer (vector + graph + key-value); the community favorite by GitHub stars.
- **Zep / Graphiti** ([arXiv 2501.13956](https://arxiv.org/abs/2501.13956), Jan 2025) — temporal knowledge graph memory; every edge carries both event time (when the fact was true) and ingestion time (when the agent learned it), enabling "what did we know when" reasoning.
- **LangMem** — LangChain's memory SDK (early 2025), riding LangGraph's persistent store.
- The field converged on a cognitive-science taxonomy: **episodic** (what happened), **semantic** (facts and preferences), **procedural** (how to behave — including agents editing their own instructions).

**The skeptical take.** A vocal camp holds that a files-and-notes filesystem plus good retrieval (thread #5) covers most real needs, and a dedicated memory layer adds an extraction pipeline that can hallucinate facts or go stale.

Practical resolution circulating in the discourse:
- Default to filesystem + retrieval for single-agent, project-scoped work.
- Reach for a memory product when you need cross-session personalization, multi-user fact stores, or temporal reasoning ("what did we believe last Tuesday") at scale.
- Whatever the store, the *write policy* (what gets remembered, when, by whom) is the design decision that matters — not the vector database underneath.

**For your exam:** know the three memory types and the in-context vs external-store distinction; memory read/write appears in agent-architecture questions and ties to state management in domains 04 and 08.

### 10. Observability and eval-in-prod tooling consolidation

**What happened.** The trace-everything market matured and consolidated around a handful of platforms:
- **LangSmith** — deepest LangChain/LangGraph integration: node-level state diffs, full execution graphs, trajectory replay against new model versions.
- **Langfuse** — the open-source leader; MIT-licensed its formerly commercial modules (LLM-as-judge evals, annotation queues, playground) in June 2025, reached 26M+ SDK installs/month (6M+ Docker pulls), and was acquired by ClickHouse in January 2026 (alongside ClickHouse's $400M Series D).
- **Arize Phoenix** — open-source, OpenTelemetry-native via the OpenInference conventions; strongest for notebook- and eval-heavy workflows.
- **Braintrust** — closed-source, eval-first (regression harness as the center of gravity) with tracing attached.
- Datadog and Honeycomb pulled LLM traces into general-purpose observability for teams that want infra correlation.

**The unifying trend.** **OpenTelemetry GenAI semantic conventions** as the neutral instrumentation layer — instrument once, ship spans to any backend, switch vendors later. Portability of traces became a purchasing criterion, and most backends (Langfuse, Phoenix, LangSmith) now ingest OTel-format spans regardless of which SDK produced them.

**What "eval-in-prod" concretely means in these platforms:**
- LLM-as-judge evaluators sampling a percentage of live traces against rubrics.
- Annotation queues routing flagged traces to human reviewers (the domain-10 connection).
- Dataset builders that promote interesting production traces into offline regression sets with one click.
- Experiment/replay: re-run a captured trajectory against a new prompt or model version and diff outcomes.

**Where it lands.** The conceptual convergence matters more than vendor choice: traces are the shared substrate for *both* debugging and evaluation. Online monitoring feeds novel failures into offline golden datasets (closing thread #4's loop), and "eval-in-prod" — LLM-as-judge sampling live traffic, human annotation queues — is now table stakes in every platform.

**For your exam:** this is domain 08 (run/monitor/maintain) — tracing, spans, drift detection, feedback loops — plus the offline/online eval complementarity from domain 03. If asked about portable instrumentation, the answer is OpenTelemetry-based.

---

## Thread → domain quick map

Use this as a revision index: when reviewing a domain file, skim the threads listed for it here to recall the "why" behind the concepts.

| # | Thread | Primary domains | The one idea to retain |
|---|---|---|---|
| 1 | Context engineering | 04, 08 | Attention is a finite budget; compaction / notes / sub-agents are the long-horizon toolkit |
| 2 | Building Effective Agents | 03, architecture | Workflows before agents; simplest composable pattern that works |
| 3 | 12-Factor Agents | all six | Own prompts, context, and control flow; agents are mostly software |
| 4 | Evals movement + backlash | 03 | Error analysis first; binary judgments; rigor proportional to risk |
| 5 | Deep agents | 07, architecture | Planning + filesystem + sub-agents + detailed prompt — converged on three times |
| 6 | MCP + A2A | 09, 10 | MCP = agent↔tools, A2A = agent↔agents (both now under AAIF; A2A v1.0 = Signed Agent Cards); break one leg of the lethal trifecta — see OWASP Agentic Top 10 |
| 7 | Multi-agent debate | 04, 08 | Parallel for wide/read-heavy, single for deep/write-heavy; context flow is the bottleneck |
| 8 | War stories + SLMs | 04, 07, 08 | Errors compound; KV-cache hit rate; right-size the model |
| 9 | Memory | 04, 08 | Episodic / semantic / procedural; in-context vs external store |
| 10 | Observability consolidation | 03, 08 | Traces feed both debugging and evals; OpenTelemetry for portability |

## Five meta-takeaways (if you remember nothing else)

1. **Everything converges on context.** Context engineering is the meta-thread: it is the resolution of the multi-agent debate (#7), the substance of harness design (#2), the mechanism behind deep agents (#5), and the top production cost lever (#8). When in doubt on an architecture question, the answer that manages context deliberately is usually right.
2. **Agents are mostly software engineering.** The 12-factor framing, the workflows-first advice, and the war stories all say the same thing: deterministic code with LLM decision points beats maximal autonomy in production. Exam scenarios reward the boring, controllable option.
3. **Measurement is a loop, not a gate.** Look at data → taxonomize failures → binary judges aligned to humans → regression suite → production traces feed new failures back in. Threads #4 and #10 are two halves of one loop.
4. **Composability is also the attack surface.** The same open-protocol explosion that made MCP/A2A useful (#6) created tool poisoning and the lethal trifecta. Capability and exposure grow together; mitigations subtract capability legs.
5. **Token economics shape architecture.** Multi-agent at 15× tokens, 100:1 input:output ratios, cache-hit-rate engineering, SLM routing — cost is not an afterthought, it is a design input (#7, #8).

---

## Reading list

### Anthropic (engineering blog)
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — the Dec 2024 ur-text: workflows vs agents, five composable patterns, "keep it simple."
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — attention as a finite budget; compaction, note-taking, sub-agents.
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system) — orchestrator-workers at ~15× token cost, +90.2% on research evals.
- [Donating MCP / the Agentic AI Foundation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) — MCP becomes vendor-neutral under the Linux Foundation (Dec 2025).

### Hamel Husain / Shreya Shankar / Eugene Yan (the evals school)
- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) — Hamel's foundational post: unit tests, LLM-judge, error analysis as the kernel.
- [LLM Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) — Hamel's living answers file: binary judgments, sample sizes, when to dial rigor down.
- [In Defense of AI Evals, for Everyone](https://www.sh-reya.com/blog/in-defense-ai-evals/) — Shankar's reply to the Sept 2025 anti-evals wave.
- [Who Validates the Validators?](https://arxiv.org/abs/2404.12272) — Shankar et al., UIST 2024: criteria drift; aligning LLM judges with human grades.
- [Evaluating LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/) — Eugene Yan's survey of judge reliability and biases.
- [AlignEval](https://eugeneyan.com/writing/aligneval/) — Yan's label → align → harness loop, built as a working app.
- [AI Evals for Engineers & PMs](https://maven.com/parlance-labs/evals) — the Husain/Shankar course that institutionalized the methodology.
- [Vanishing Gradients ep. 60 with Hamel](https://hugobowne.substack.com/p/episode-60-10-things-i-hate-about-e83) — the backlash discussed head-on.

### Dex Horthy / HumanLayer
- [12-factor-agents (GitHub)](https://github.com/humanlayer/12-factor-agents) — all twelve factors plus factor 13; the production-agents checklist.
- [12-Factor Agents talk, Agents in Production 2025](https://home.mlops.community/public/videos/12-factor-agents-patterns-of-reliable-llm-applications-dexter-horthy-agents-in-production-2025-2025-08-06) — the talk version with the "dumb zone" framing.

### Simon Willison (simonwillison.net)
- [MCP has prompt injection security problems](https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/) — the early warning shot (April 2025).
- [The lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) — private data + untrusted content + exfiltration channel = exploitable agent.
- [Context engineering](https://simonwillison.net/2025/Jun/27/context-engineering/) — tracking the term's takeover in real time.

### LangChain
- [Deep Agents](https://blog.langchain.com/deep-agents/) — planning + filesystem + sub-agents + detailed prompt, named and packaged.
- [deepagents (GitHub)](https://github.com/langchain-ai/deepagents) — the harness as a pip-installable library on the LangGraph runtime.

### Cognition
- [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) — Walden Yan: share full context; actions carry implicit decisions.

### Manus
- [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) — KV-cache hit rate as *the* metric; mask-don't-remove; keep errors in context.

### NVIDIA Research
- [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153) — Belcak et al.: SLMs for routine agent steps, LLM escalation, distillation flywheel ([project page with rebuttals](https://research.nvidia.com/labs/lpr/slm-agents/)).

### Chroma
- [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://research.trychroma.com/context-rot) — 18 models; performance degrades non-uniformly with input length.

### X (the canonical posts)
- [Karpathy on context engineering](https://x.com/karpathy/status/1937902205765607626) — the endorsement that mainstreamed the term (June 2025).
- [Tobi Lütke's definition](https://x.com/tobi/status/1935533422589399127) — "all the context for the task to be plausibly solvable by the LLM."

### Protocols and ecosystem
- [Linux Foundation launches the A2A project](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — Google's donation, June 2025.
- [A2A surpasses 150 organizations](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year) — one-year adoption report.
- [One Year of MCP](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — registry growth, spec evolution, adoption stats.
- [Agentic AI Foundation (AAIF) formation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) — Linux Foundation, Dec 9, 2025: MCP + goose + AGENTS.md under one neutral body.
- [Announcing A2A v1.0](https://a2a-protocol.org/latest/announcing-1.0/) — early 2026 production release: Signed Agent Cards, multi-tenancy, version negotiation.
- [A2A Protocol site](https://a2a-protocol.org/latest/) — spec, Agent Cards, transport details.
- [OWASP Top 10 for Agentic Applications (2026)](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) — ASI01–ASI10; the agent-specific companion to the LLM Top 10 2025, released Dec 9, 2025.

### Memory papers
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) — OS-style memory tiers; became Letta.
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413) — extraction-based hybrid memory layer.
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956) — Graphiti's bitemporal edges.
