# 2-Step 3_RAG
# -------------------------
# Core idea:
#   A fixed pipeline with exactly two phases: retrieval and generation.
#
# Flow:
#   query -> retriever -> top_k_documents -> LLM -> final_answer
#
# Behavior:
#   - One-shot retrieval (no iteration)
#   - Same retrieval strategy for every query
#   - LLM only sees retrieved context, then answers
#
# Strengths:
#   - Simple to implement and maintain
#   - Easy to debug (clear separation of concerns)
#   - Low latency (single retrieval call)
#
# Weaknesses:
#   - Cannot refine queries dynamically
#   - Poor for multi-hop or ambiguous questions
#   - Retrieval quality heavily depends on initial query formulation


# Agentic 3_RAG
# -------------------------
# Core idea:
#   The LLM acts as an agent that controls the retrieval process dynamically.
#
# Flow:
#   query -> LLM agent
#            -> decide need for search?
#            -> generate search query
#            -> retrieve documents
#            -> evaluate results
#            -> optionally repeat retrieval
#            -> synthesize final answer
#
# Behavior:
#   - Iterative reasoning loop
#   - Multiple retrieval calls possible
#   - Query rewriting and decomposition supported
#   - Tool usage is LLM-driven (ReAct-style behavior)
#
# Strengths:
#   - Handles complex, multi-step reasoning tasks
#   - Adapts retrieval strategy per query
#   - Can recover from incomplete or weak initial results
#
# Weaknesses:
#   - Higher latency (multiple cycles)
#   - More cost due to repeated tool calls
#   - Harder to debug and control
#   - Output can be less deterministic


# Hybrid 3_RAG
# -------------------------
# Core idea:
#   Combine multiple retrieval strategies or data sources before passing context to the LLM.
#
# Common forms:
#
#   (1) Dense + Sparse retrieval fusion:
#       - Dense (vector search / embeddings)
#       - Sparse (BM25 / keyword search)
#       - Merge results using ranking or fusion logic (e.g., RRF)
#
#   (2) Multi-source retrieval:
#       - Vector DB (unstructured docs)
#       - SQL DB (structured data)
#       - External APIs (live or domain-specific data)
#
# Flow:
#   query -> [dense_retriever]
#        -> [sparse_retriever]
#        -> [other_sources]
#        -> fusion/rerank
#        -> unified_context -> LLM -> answer
#
# Behavior:
#   - Retrieval is multi-channel but typically not iterative
#   - Focus on coverage and recall across systems
#
# Strengths:
#   - Strong recall and robustness
#   - Works well in production search systems
#   - Handles diverse query types effectively
#
# Weaknesses:
#   - More system complexity
#   - Requires tuning of fusion/ranking strategy
#   - Harder infrastructure setup than basic 3_RAG


# Summary Mental Model
# -------------------------
# 2-Step 3_RAG    -> "retrieve once, then answer"
# Agentic 3_RAG   -> "think, search, refine, repeat"
# Hybrid 3_RAG    -> "use multiple retrieval systems together"