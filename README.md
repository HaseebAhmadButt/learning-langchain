# LangChain & LangGraph: Building Agentic AI Applications

This repository contains the code snippets and worked examples from the Udemy course
[LangChain](https://nisum.udemy.com/course/langchain/). It progresses from a single LLM call
all the way to stateful, self-correcting multi-agent systems and standardized tool servers (MCP).

This README doubles as a **presentation outline** for the course. Each heading is a candidate
slide (or slide group) with a brief description to speak to.

---

## Table of Contents

1. [Introduction & Course Roadmap](#1-introduction--course-roadmap)
2. [Why LangChain? The Core Architecture](#2-why-langchain-the-core-architecture)
3. [Module 1 — Basic Model Integration](#module-1--basic-model-integration)
4. [Module 2 — Building Agents](#module-2--building-agents)
5. [Module 3 — Retrieval-Augmented Generation (RAG)](#module-3--retrieval-augmented-generation-rag)
6. [Module 4 — Documentation Assistant](#module-4--real-world-project-documentation-assistant)
7. [Module 5 — LangGraph](#module-5--langgraph-stateful-multi-step-agents)
8. [Module 6 — Model Context Protocol (MCP)](#module-6--model-context-protocol-mcp)
9. [Cross-Cutting Tools & Ecosystem](#23-cross-cutting-tools--ecosystem)
10. [Key Takeaways](#24-key-takeaways)
11. [Demo & Q&A](#25-demo--qa)

---

## 1. Introduction & Course Roadmap
A quick framing of the journey: from a single LLM call → tool-using agents → retrieval (RAG) →
stateful multi-step graphs → standardized tool servers (MCP). Sets expectations that the course
builds incrementally, each module layering on the previous.

## 2. Why LangChain? The Core Architecture
LangChain is built around **standard interfaces with interchangeable implementations** — you code
against abstractions (`BaseChatModel`, `Embeddings`, `VectorStore`, `Retriever`, `Tool`), not
against a specific vendor. The three layers:

- **Core** (`langchain_core`) — interfaces, base classes, schemas
- **Integrations** (`langchain_openai`, `langchain_ollama`, …) — providers implementing those interfaces
- **Orchestration** — chains, agents, execution logic

---

## Module 1 — Basic Model Integration

### 3. Prompts, Models, and LCEL Chains
The "Hello World" of LangChain: `PromptTemplate` + a chat model composed with the pipe operator
(`prompt | llm`). Introduce **LCEL (LangChain Expression Language)** and the **Runnable**
interface — the composable building block behind `.invoke()`, `.stream()`, and `.batch()`.

### 4. Swapping Models: Cloud vs. Local
Same code, different model — `ChatOpenAI` (GPT) vs. `ChatOllama` (local models like `qwen3-coder`).
Demonstrates the plug-and-play promise of standard interfaces. Touch on `temperature` (creativity
vs. determinism) and managing keys via `.env` / `dotenv`.

### 5. Observability with LangSmith
Tracing every prompt, chain, and tool call. Why observability matters for debugging and cost
control. (`@traceable` shows up throughout the agent modules.)

---

## Module 2 — Building Agents

### 6. What Is an Agent? The Agent Loop
Core idea: load tools into the LLM → LLM decides *which* tool to call and *with what arguments* →
framework executes the tool → result feeds back to the LLM → repeat until a final answer. The LLM
controls the flow.

### 7. Tools & Structured Output
Defining tools with the `@tool` decorator, using prebuilt tools (`TavilySearch`), and forcing typed
responses with **Pydantic** schemas (e.g., an answer + list of sources).

### 8. Four Ways to Implement the Loop (Increasing Abstraction)
Same task, four implementations:

1. **`create_agent`** — highest-level, batteries included
2. **`.bind_tools()`** — LangChain tool-calling loop, manually managed messages
3. **Raw vendor function-calling** — hand-written JSON tool schemas via the Ollama SDK
4. **Manual ReAct prompt** — Thought/Action/Observation parsed by regex, with a scratchpad

Key takeaway: the framework hides a lot — seeing the raw version reveals what's really happening.

---

## Module 3 — Retrieval-Augmented Generation (RAG)

### 9. RAG Architectures Compared
- **2-Step RAG** — "retrieve once, then answer" (simple, low latency)
- **Agentic RAG** — "think, search, refine, repeat" (handles multi-hop, costlier)
- **Hybrid RAG** — combine dense + sparse retrieval / multiple sources, then fuse/rerank

### 10. Ingestion: Loaders, Splitting & Chunking
The indexing pipeline: document loaders → `RecursiveCharacterTextSplitter` → why **chunk size and
overlap** matter for preserving context across boundaries.

### 11. Embeddings & Vector Stores
Turning text into vectors (`OpenAIEmbeddings`) and storing/querying them in **Pinecone**. Introduce
the `Retriever` abstraction and `top-k` similarity search.

### 12. Building the Retrieval Chain (With vs. Without LCEL)
Contrast the verbose manual approach (retrieve → format → prompt → LLM) with the declarative LCEL
chain using `RunnablePassthrough.assign`. Highlights LCEL's streaming, async, batching, and
composability benefits.

---

## Module 4 — Real-World Project: Documentation Assistant

### 13. End-to-End RAG Application
A practical capstone for the RAG section: a Q&A assistant over LangChain's own documentation, with
**source citation**.

### 14. Web Crawling & Async Ingestion at Scale
Using **Tavily** (`TavilyCrawl`, `TavilyMap`, `TavilyExtract`) to scrape docs, then **async batch
indexing** into the vector store for performance. RAG packaged as a tool the agent calls
automatically.

---

## Module 5 — LangGraph (Stateful, Multi-Step Agents)

### 15. Why LangGraph? Graphs over Chains
When linear chains aren't enough: model agents as a **graph** of nodes and edges with shared
**state**, **conditional edges** for routing, and loops. The foundation for everything in this
module.

### 16. Reflection Agents
The simplest improvement loop: **generate → self-critique → revise**, iterating until quality
improves. Great for reasoning, code, and complex Q&A.

### 17. Reflexion Agents
A more structured architecture: a **Responder** produces an initial answer + self-critique + search
queries → a **Tool Execution** node gathers info → a **Revisor** produces an improved, cited answer.
Self-improvement grounded in external evidence.

### 18. ReAct Agent in LangGraph
Rebuilding the agent loop as a graph: a **reasoning node**, a prebuilt **`ToolNode`**, and a
`should_continue` conditional edge that routes between "act" and "end." Shows the same ReAct pattern
from Module 2, now as explicit, inspectable graph wiring.

### 19. Advanced RAG Patterns: Agentic, Self-RAG, Adaptive RAG
The payoff of the course — graphs that grade and self-correct their own retrieval:

- **Agentic RAG** — agent decides when/what to retrieve
- **Self-RAG** — adds **graders**: document relevance, hallucination check (is the answer
  grounded?), and answer usefulness
- **Adaptive RAG** — a **router** picks retrieval vs. web search, with grading-driven fallbacks and
  retries

Emphasize the reusable building blocks: retrieval grader, hallucination grader, answer grader,
web-search fallback.

---

## Module 6 — Model Context Protocol (MCP)

### 20. What Is MCP and Why It Matters
A standardized protocol for exposing tools/data to LLMs across applications — decoupling tool
servers from agent code (the "USB-C of AI tools").

### 21. Building MCP Servers
Using **`FastMCP`** and the `@mcp.tool()` decorator to build servers (Math, Weather), and the
difference between **`stdio`** and **`sse`** transports.

### 22. MCP Clients, Adapters & Tooling
Connecting agents to MCP servers via LangChain's MCP adapter, plus the supporting tools — `mcpdoc`
for serving docs and the **MCP Inspector** for discovering/testing tools.

---

## 23. Cross-Cutting Tools & Ecosystem
A recap of the supporting cast used throughout:

- **OpenAI / Ollama** — models
- **Pinecone / Chroma** — vector stores
- **Tavily** — search & crawl
- **LangSmith** — tracing
- **Pydantic** — structured output
- **dotenv** — configuration

## 24. Key Takeaways
- One mental model — *prompt → reason → act → observe → repeat* — scales from a single tool call to
  self-correcting RAG graphs.
- Abstractions are convenient, but understanding the raw layer underneath makes you a better builder.
- Add structure (state, grading, reflection) to move from demos to reliable, production-grade agents.

## 25. Demo & Q&A
Suggested live demos:

- The 4-implementation agent loop (Module 2)
- The documentation assistant (Module 4)
- An Adaptive RAG graph run showing the grading/routing in action

---

## Repository Structure

| Folder | Topic |
| --- | --- |
| `1_BasicModelIntegration/` | Prompts, LCEL chains, cloud vs. local models |
| `2_BuildingAgents/` | Agent loop in four implementations (create_agent → raw ReAct) |
| `3_RAG/` | RAG architectures, ingestion, retrieval chains |
| `4_DocumentationAssistant/` | End-to-end RAG app with Tavily crawling |
| `5_LangGraph/` | Reflection, Reflexion, ReAct, Agentic/Self/Adaptive RAG |
| `6_MCPServers/` | MCP servers, clients, and tooling |

> **Note:** Flow diagrams live alongside the code in each LangGraph subfolder
> (e.g., `AdaptiveRAG_Flow.png`, `ReflexionAgent.png`, `GraphExecutionFlow.png`) — drop these
> straight into the relevant slides.

## Presentation Tips
- **If time is short**, the highest-impact arc is: #6–#8 (the four-way agent loop) → #9–#12 (RAG)
  → #15–#19 (LangGraph). Those carry the core ideas; MCP (#20–#22) can be a brief "what's next."