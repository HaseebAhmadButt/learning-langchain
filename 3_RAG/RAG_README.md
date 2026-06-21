# Retrieval-Augmented Generation (RAG) — End to End

This package walks through a complete RAG system, one file at a time. RAG lets a
Large Language Model (LLM) answer questions using **your own documents** instead
of relying only on what it memorized during training.

The big idea:

```
Your documents  ──►  stored as vectors  ──►  search by meaning  ──►  LLM answers using what it found
                       (ingestion)            (retrieval)              (generation)
```

There are two phases, each in its own file:

| Phase | File | What it does |
|-------|------|--------------|
| 0. Concepts | `RAGArchitectures.py` | The mental models for *how* RAG can be wired |
| 1. Ingestion (offline) | `1_rag_ingestion.py` | Load → split → embed → store documents in a vector DB |
| 2. Retrieval + Generation (online) | `2_rag_gist.py` | Take a question → fetch relevant chunks → feed to LLM → answer |

---

## The core components of RAG

Before looking at the code, here are the moving parts that appear throughout:

- **Document Loader** — reads raw source files (`.txt`, PDF, HTML, Markdown…) into LangChain `Document` objects.
- **Text Splitter** — cuts large documents into smaller **chunks** so they fit the embedding model and improve retrieval precision.
- **Embeddings model** — turns each chunk of text into a **vector** (a list of numbers that captures meaning).
- **Vector Store** — a database (here, **Pinecone**) that stores those vectors and can find the *closest* ones to a query vector.
- **Retriever** — the search interface over the vector store; given a question, it returns the top-k most relevant chunks.
- **Prompt Template** — a reusable text scaffold that injects the retrieved context and the question into a single instruction for the LLM.
- **LLM** — the model (here, `ChatOpenAI`) that reads the context + question and writes the final answer.
- **Output Parser** — converts the LLM's raw message object into a plain string.
- **Chain (LCEL)** — LangChain Expression Language wires the above together with the pipe (`|`) operator into one runnable pipeline.

---

## File 0 — `RAGArchitectures.py`: the three architectures

This file is pure conceptual notes. It defines three ways to build a RAG system,
from simplest to most advanced.

### 1. 2-Step RAG — *"retrieve once, then answer"*

A fixed, two-phase pipeline:

```
query -> retriever -> top_k_documents -> LLM -> final_answer
```

- One-shot retrieval (no iteration), same strategy for every query.
- **Strengths:** simple, easy to debug, low latency (single retrieval call).
- **Weaknesses:** can't refine queries; weak on multi-hop or ambiguous questions; quality depends heavily on how the initial query is phrased.

> This is the architecture the code in this package actually implements.

### 2. Agentic RAG — *"think, search, refine, repeat"*

The LLM acts as an agent that controls retrieval dynamically:

```
query -> LLM agent -> decide if search needed -> generate search query
      -> retrieve -> evaluate results -> optionally repeat -> synthesize answer
```

- Iterative reasoning loop with multiple retrieval calls; supports query rewriting/decomposition (ReAct-style).
- **Strengths:** handles complex multi-step reasoning; adapts per query; recovers from weak initial results.
- **Weaknesses:** higher latency and cost; harder to debug; less deterministic.

### 3. Hybrid RAG — *"use multiple retrieval systems together"*

Combine multiple retrieval strategies or sources before calling the LLM:

```
query -> [dense_retriever] + [sparse_retriever] + [other_sources]
      -> fusion/rerank -> unified_context -> LLM -> answer
```

- Common forms: **dense + sparse fusion** (vector search + BM25/keyword, merged with e.g. Reciprocal Rank Fusion), or **multi-source** retrieval (vector DB + SQL + APIs).
- **Strengths:** strong recall and robustness; great for production search.
- **Weaknesses:** more complexity; needs fusion/ranking tuning; heavier infrastructure.

**Mental model summary:**

| Architecture | One-liner |
|--------------|-----------|
| 2-Step RAG | retrieve once, then answer |
| Agentic RAG | think, search, refine, repeat |
| Hybrid RAG | use multiple retrieval systems together |

---

## File 1 — `1_rag_ingestion.py`: building the knowledge base (offline phase)

This is **data indexing**: storing your documents in vector space so they can
later be retrieved by meaning. It runs *once* (or whenever your data changes),
not on every user question.

### Step-by-step

**1. Load the document**

```python
loader = TextLoader("/SampleDocument.txt", encoding="utf-8")
documents = loader.load()
```

`TextLoader` reads a text file into `Document` objects. Swap the loader
(`PyPDFLoader`, `WebBaseLoader`, etc.) to ingest other formats — the rest of the
pipeline stays the same.

**2. Split into chunks**

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
docs = text_splitter.split_documents(documents)
```

- `chunk_size=1000` — each chunk is ~1000 characters.
- `chunk_overlap=0` — no repeated text between consecutive chunks.

Why split at all? Embedding models and LLMs have size limits, and smaller chunks
give *more precise* retrieval. `RecursiveCharacterTextSplitter` tries a list of
separators in order (`\n\n` → `\n` → spaces) and keeps related text together as
much as possible.

> **Tip — overlap:** Setting `chunk_overlap=200` repeats the last 200 characters
> of one chunk at the start of the next. This preserves context that would
> otherwise be cut at a chunk boundary, improving retrieval accuracy for
> information that spans the split.

**3. Create embeddings**

```python
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
```

`OpenAIEmbeddings` is the model that converts each text chunk into a vector. The
*same* embedding model must be used for ingestion and for querying, so the
vectors live in the same space.

**4. Store in the vector database**

```python
PineconeVectorStore.from_documents(docs, embeddings, index_name=os.environ["INDEX_NAME"])
```

This embeds every chunk and upserts the vectors into the Pinecone index named by
`INDEX_NAME`. After this runs, your knowledge base is ready to be searched.

### Ingestion flow

```
SampleDocument.txt
   └─ TextLoader.load()              → Document(s)
        └─ RecursiveCharacterTextSplitter → chunks
             └─ OpenAIEmbeddings          → vectors
                  └─ PineconeVectorStore  → stored in Pinecone (INDEX_NAME)
```

---

## File 2 — `2_rag_gist.py`: building the retrieval chain (online phase)

This is where a user's question gets answered. It first sets up the shared
components, then shows **three implementations** of increasing quality.

### Shared setup

```python
embeddings  = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
llm         = ChatOpenAI(model="gpt-5.2", openai_api_key=os.environ["OPENAI_API_KEY"])
vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
retriever   = vectorStore.as_retriever(search_kwargs={"k": 3})   # top 3 matches
```

- We reconnect to the *same* Pinecone index using the *same* embeddings model.
- `as_retriever(search_kwargs={"k": 3})` turns the store into a **retriever** that returns the top 3 most relevant chunks for any query.

The **prompt template** constrains the LLM to answer *only* from retrieved context — this is what keeps RAG grounded and reduces hallucination:

```python
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:
    """
)
```

A small helper joins retrieved documents into one context string:

```python
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
```

### The 5 logical steps of any retrieval chain

Every RAG answer — no matter the implementation style — performs these steps:

```
1. retrieve   query     → retriever          → relevant docs
2. format     docs      → format_docs        → context string
3. prompt     context+Q → prompt_template    → filled prompt
4. generate   prompt    → llm                → answer message
5. parse      message   → .content / parser  → final string
```

### Implementation 0 — Raw LLM (no RAG, the baseline)

```python
result_raw = llm.invoke([HumanMessage(content=query)])
```

The LLM answers from its own training knowledge only — no retrieval. This is the
baseline to compare against: it may be outdated or hallucinate on
domain-specific questions.

### Implementation 1 — Without LCEL (manual, functional)

```python
def retrieval_chain_without_lcel(query: str):
    doc     = retriever.invoke(query)                                 # 1. retrieve
    context = format_docs(doc)                                        # 2. format
    prompt  = prompt_template.format_messages(context=context, question=query)  # 3. prompt
    response = llm.invoke(prompt)                                     # 4. generate
    return response.content                                          # 5. parse
```

This makes the 5 steps explicit and is easy to follow. Its **limitations**:
manual step-by-step execution, no built-in streaming or async, harder to compose
with other chains, more verbose and error-prone.

### Implementation 2 — With LCEL (the recommended way)

LCEL (LangChain Expression Language) wires the steps with the pipe operator. The
*intuitive* version would be:

```python
# retriever | format_docs | prompt_template | llm | StrOutputParser()
```

…but that breaks, because `prompt_template` needs **two** inputs (`context`
*and* `question`), while a straight pipe only passes one value through. The fix
is `RunnablePassthrough.assign`, which adds a `context` key while keeping the
original `question`:

```python
def create_retrieval_chain():
    retrieval_chain = (
        RunnablePassthrough.assign(
            # take "question" → retrieve → format into a "context" field
            context=itemgetter("question") | retriever | format_docs,
        )
        | prompt_template     # gets {question, context}
        | llm                 # generate
        | StrOutputParser()   # parse to plain string
    )
    return retrieval_chain
```

Invoke it with a dict:

```python
chain = create_retrieval_chain()
answer = chain.invoke({"question": "What is Pinecone in Machine Learning?"})
```

**How the data flows through the chain:**

```
{"question": "..."}
   └─ RunnablePassthrough.assign(context = question | retriever | format_docs)
        → {"question": "...", "context": "...retrieved text..."}
            └─ prompt_template  → filled ChatPromptTemplate
                 └─ llm         → AI message
                      └─ StrOutputParser → final answer string
```

> **Why `format_docs` works inside the pipe:** it's a plain function, not a
> Runnable. LangChain automatically wraps it as `RunnableLambda(format_docs)` so
> it composes with `|` like everything else.

**Why LCEL is better:** declarative and composable, built-in `stream()` /
`ainvoke()` / `batch()`, better type integration, less code, reusable, and
better observability/debugging. Prefer it for production.

---

## Putting it all together

```
                         ┌────────────────────── OFFLINE (run once) ──────────────────────┐
   SampleDocument.txt ──►│ Loader ─► Splitter ─► Embeddings ─► Pinecone (INDEX_NAME)        │
                         └─────────────────────────────────────────────────────────────────┘
                                                       │  (vectors stored)
                                                       ▼
   user question  ──► ┌──────────────────────── ONLINE (per query) ──────────────────────┐
                      │ retriever (top-k) ─► format_docs ─► prompt_template ─► llm ─► parser│ ──► answer
                      └────────────────────────────────────────────────────────────────────┘
```

- **`RAGArchitectures.py`** tells you *which* shape to build — this package builds the **2-Step RAG**.
- **`1_rag_ingestion.py`** builds the knowledge base (offline).
- **`2_rag_gist.py`** builds the retrieval chain that answers questions (online), best expressed with **LCEL**.

### Environment variables required

Set these (e.g. in a `.env` file loaded by `load_dotenv()`):

- `OPENAI_API_KEY` — for embeddings and the chat model
- `INDEX_NAME` — the Pinecone index to write to / read from
- Pinecone credentials (`PINECONE_API_KEY`) as required by `langchain_pinecone`

### Run order

```bash
python 1_rag_ingestion.py   # 1) build the index (only when data changes)
python 2_rag_gist.py        # 2) ask questions against the index
```