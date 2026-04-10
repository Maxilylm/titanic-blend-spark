# RAG Chatbot — Lessons Learned

**Project:** Titanic Survival Prediction Dashboard  
**Component:** AI Data Insights Chatbot (Tab 5)  
**Stack:** Groq API (Llama 3.3 70B) + Sentence-Transformers + Streamlit  
**Date:** 2026-04-02

---

## 1. ChromaDB Breaks in Streamlit — Use In-Memory for Small Knowledge Bases

**Problem:** ChromaDB's `PersistentClient` holds an internal SQLite connection. When cached via `@st.cache_resource`, the connection object survives across Streamlit reruns, but the underlying OS file descriptor becomes stale — causing `[Errno 32] Broken pipe` on every query.

**Root cause:** Streamlit re-executes the entire script on each user interaction. `@st.cache_resource` preserves the Python object, but SQLite connections are tied to the thread/process that created them. When Streamlit's execution context changes, the cached connection is dead.

**Solution:** For knowledge bases under ~100 chunks, skip the vector database entirely. A numpy dot product on L2-normalized embeddings (cosine similarity) is instantaneous and has zero connection state:

```python
scores = (embeddings @ query_embedding.T).flatten()
top_indices = np.argsort(scores)[::-1][:k]
```

**Takeaway:** Only introduce a vector database when the dataset exceeds what fits in memory (~10K+ chunks). For analytics projects with structured knowledge bases, in-memory is simpler and more reliable.

---

## 2. Structured Knowledge Chunks Beat Document Chunking for Analytics

**Problem:** The initial version missed basic questions like "What is the overall survival rate?" even though the information was embedded in a large `dataset_overview` chunk. The retrieval scored it lower than more specific chunks about age or sex survival.

**Root cause:** A single chunk covering many topics (survival rate, sex distribution, age range, fare range) dilutes its embedding — it's "about everything" so it matches nothing strongly. Standard document chunking (splitting by character count) makes this worse by creating arbitrary boundaries.

**Solution:** Build domain-aware chunks where each chunk answers one category of question:
- `dataset_overview` — overall stats and survival rate
- `key_survival_factors` — ranked list of predictors
- `survival_by_sex` — sex-specific analysis
- `correlations` — feature correlations
- `column_reference` — schema documentation

**Takeaway:** For data/ML projects, hand-craft chunks around the questions stakeholders will ask, not around document structure. One chunk = one topic = one type of question answered well.

---

## 3. Include Raw Counts Alongside Percentages

**Problem:** Early chunks only had percentages (e.g., "74.2% survival rate for females"). The LLM would sometimes hedge with "the context doesn't give exact numbers" when asked about counts.

**Solution:** Every statistical chunk now includes both forms: "74.2% survival rate (233 of 314 female passengers survived)". This gives the LLM concrete numbers to cite and eliminates hedging.

**Takeaway:** LLMs are more confident and accurate when the context provides multiple representations of the same fact. Redundancy in the knowledge base is a feature, not a bug.

---

## 4. Code Sandbox Bridges the Gap Between Static RAG and Ad-Hoc Questions

**Problem:** Pre-built knowledge chunks handle common questions well, but real stakeholder questions are often specific and unpredictable: "What is the average age of 1st class female survivors?" or "How many passengers paid more than 100 GBP?"

**Solution:** A two-pass architecture:
1. LLM receives context + question and decides if it needs computation
2. If yes, it generates Python code in a fenced block
3. Code runs in a sandboxed `exec()` with only `df`, `pd`, `np` available
4. Results go back to the LLM for a stakeholder-friendly final answer

**Key design decisions:**
- Sandbox restricts `__builtins__` to safe operations (no `open`, `import`, `eval`, `exec`)
- The DataFrame is copied before execution (`df.copy()`) so code can't corrupt state
- Two LLM calls cost ~2x latency but Groq's speed (~500 tok/s) keeps it under 3 seconds total

**Takeaway:** RAG alone is insufficient for data projects. The ability to compute on demand makes the chatbot genuinely useful rather than just a search interface over pre-written summaries.

---

## 5. Streamlit Chat State Management Requires Explicit Pending State

**Problem:** Suggested question buttons and `st.chat_input` both need to trigger response generation, but Streamlit's execution model makes this tricky. Button clicks trigger a rerun before the chat input is processed, leading to duplicate or missed messages.

**Solution:** Use a `pending_question` pattern in `st.session_state`:

```python
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Both buttons and chat_input write to the same state
if st.button("Suggested Q"):
    st.session_state.pending_question = question
if user_input := st.chat_input("Ask..."):
    st.session_state.pending_question = user_input

# Single processing point
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    # ... generate response
```

**Takeaway:** Never append directly to chat history from multiple input sources. Funnel everything through a single pending state variable, then process it once.

---

## 6. Cache the Embedding Model, Not the Database Connection

**Problem:** `@st.cache_resource` is essential for avoiding re-loading the SentenceTransformer model (80MB, ~3s cold start) on every interaction. But caching database connections or clients causes the broken pipe issue above.

**Rule of thumb for Streamlit caching:**
- **Safe to cache:** ML models, numpy arrays, plain Python objects, configuration dicts
- **Unsafe to cache:** database connections, HTTP clients with persistent sessions, file handles, anything with OS-level state

**Takeaway:** `@st.cache_resource` caches the Python object but not the OS resources it holds. If an object wraps a file descriptor, socket, or connection pool, it will break across reruns.

---

## 7. Groq API Is Fast Enough for Two-Pass Generation

**Concern:** The code sandbox requires two sequential LLM calls (generate code, then generate answer from results). With OpenAI or Anthropic APIs, this would add 4-8 seconds of latency.

**Result:** Groq's inference speed (~500 tokens/sec for Llama 3.3 70B) keeps the total round-trip under 3 seconds for both calls combined. This makes the two-pass pattern viable for interactive use.

**Takeaway:** When choosing an LLM provider for interactive RAG, inference speed matters more than marginal quality differences. Groq's speed enables architectural patterns (multi-turn, code execution, chain-of-thought) that would feel sluggish on slower providers.

---

## 8. System Prompt Wording Directly Affects Answer Quality

**Problem:** The initial system prompt said "If the context doesn't contain the answer, say so honestly." The LLM interpreted this too aggressively — it would say "the context doesn't contain the overall survival rate" even when the chunk literally included "The overall survival rate is 38.4%."

**Fix:** Changed to "If the context contains the answer, use it directly. Do not say 'the context doesn't contain' when it does." This eliminated the false negatives.

**Takeaway:** LLMs take instructions literally. A well-intentioned "be honest about limitations" instruction can cause the model to be overly cautious. Be specific about what NOT to do, not just what to do.

---

## Summary of Architecture Decisions

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Vector store | In-memory numpy | ChromaDB | Avoids SQLite connection issues in Streamlit; 20 chunks don't need a DB |
| Embeddings | all-MiniLM-L6-v2 | OpenAI ada-002 | Local, free, 80MB, fast on CPU; quality is sufficient for 20 chunks |
| LLM | Groq (Llama 3.3 70B) | OpenAI GPT-4 | Fast enough for two-pass code execution; free tier available |
| Chunking | Domain-structured | Fixed-size recursive | Each chunk maps to one question type; better retrieval precision |
| Code execution | Sandboxed exec() | No code / full Jupyter | Restricted builtins prevent abuse; `df.copy()` prevents state corruption |
| Chat state | pending_question pattern | Direct append | Prevents duplicate messages from multiple input sources |

---

*Written after building and debugging the RAG chatbot across 2 sessions.*
