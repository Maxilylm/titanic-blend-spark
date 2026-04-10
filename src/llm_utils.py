"""
LLM utilities for the ml-automation-llm extension plugin.

Requires ml_utils.py from the ml-automation core plugin to be present
in the same directory (copied via Stage 0 of LLM commands).
"""

import os
import json
import re
import math
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any

from ml_utils import save_agent_report, load_agent_report


# --- Relevance Detection ---

LLM_INDICATORS = {
    "openai",
    "anthropic",
    "langchain",
    "llama_index",
    "transformers",
    "huggingface",
    "chromadb",
    "pinecone",
    "faiss",
    "sentence_transformers",
    "tiktoken",
    "vllm",
    "ollama",
}

LLM_MODEL_PATTERNS = [
    r"gpt-[34]",
    r"claude-",
    r"llama-",
    r"mistral-",
    r"gemini-",
    r"command-r",
    r"text-embedding-",
]


def detect_llm_relevance(project_path="."):
    """Check if project has LLM/GenAI indicators for relevance gating.

    Checks: LLM library imports, prompt files, .jsonl datasets with
    prompt/completion fields, model name references, RAG artifacts.

    Args:
        project_path: root directory of the project

    Returns:
        dict with 'is_llm': bool, 'indicators': list of found indicators
    """
    indicators = []
    project = Path(project_path)

    # Check for prompts directory
    prompts_dir = project / "prompts"
    if prompts_dir.is_dir():
        prompt_files = list(prompts_dir.glob("*"))
        if prompt_files:
            indicators.append(f"prompts/ directory with {len(prompt_files)} files")

    # Check for .prompt files anywhere
    prompt_files = list(project.glob("**/*.prompt"))
    if prompt_files:
        indicators.append(f"{len(prompt_files)} .prompt files found")

    # Check requirements for LLM packages
    for req_file in ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]:
        req_path = project / req_file
        if req_path.exists():
            content = req_path.read_text().lower()
            for pkg in LLM_INDICATORS:
                if pkg in content:
                    indicators.append(f"{pkg} in {req_file}")

    # Check Python files for LLM imports
    py_files = list(project.glob("**/*.py"))[:50]  # limit scan
    for py_file in py_files:
        try:
            content = py_file.read_text()
            for pkg in LLM_INDICATORS:
                if f"import {pkg}" in content or f"from {pkg}" in content:
                    indicators.append(f"{pkg} import in {py_file.name}")
                    break
        except (UnicodeDecodeError, PermissionError):
            continue

    # Check for .jsonl files with prompt/completion fields
    jsonl_files = list(project.glob("**/*.jsonl"))[:10]
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    obj = json.loads(first_line)
                    if any(k in obj for k in ["prompt", "completion", "messages",
                                               "instruction", "response"]):
                        indicators.append(f"LLM dataset: {jsonl_file.name}")
        except (json.JSONDecodeError, UnicodeDecodeError, PermissionError):
            continue

    # Check for vector store artifacts
    for vs_dir in ["vector_stores", "chroma_db", ".chroma"]:
        if (project / vs_dir).is_dir():
            indicators.append(f"Vector store directory: {vs_dir}/")

    # Check for FAISS index files
    faiss_files = list(project.glob("**/*.faiss"))
    if faiss_files:
        indicators.append(f"{len(faiss_files)} FAISS index files")

    return {
        "is_llm": len(indicators) > 0,
        "indicators": indicators,
    }


# --- Text Generation Metrics ---

def _ngrams(tokens, n):
    """Generate n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(predictions: List[str], references: List[str],
                 max_n: int = 4) -> Dict[str, float]:
    """Compute BLEU score (corpus-level and per-sample).

    Implements smoothed BLEU with brevity penalty.

    Args:
        predictions: list of predicted text strings
        references: list of reference text strings
        max_n: maximum n-gram order (default: 4)

    Returns:
        dict with 'corpus_bleu', 'per_sample' (list), 'mean', 'min', 'max'
    """
    assert len(predictions) == len(references), "Prediction/reference count mismatch"

    per_sample = []
    corpus_match_counts = [0] * max_n
    corpus_pred_counts = [0] * max_n
    total_ref_len = 0
    total_pred_len = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        total_ref_len += len(ref_tokens)
        total_pred_len += len(pred_tokens)

        sample_scores = []
        for n in range(1, max_n + 1):
            pred_ngrams = _ngrams(pred_tokens, n)
            ref_ngrams = _ngrams(ref_tokens, n)

            ref_counts = Counter(ref_ngrams)
            match_count = 0
            for ng in pred_ngrams:
                if ref_counts[ng] > 0:
                    match_count += 1
                    ref_counts[ng] -= 1

            corpus_match_counts[n - 1] += match_count
            corpus_pred_counts[n - 1] += len(pred_ngrams)

            # Smoothed precision for per-sample
            precision = (match_count + 1) / (len(pred_ngrams) + 1) if pred_ngrams else 0
            sample_scores.append(precision)

        # Geometric mean of n-gram precisions
        if all(s > 0 for s in sample_scores):
            log_avg = sum(math.log(s) for s in sample_scores) / max_n
            geo_mean = math.exp(log_avg)
        else:
            geo_mean = 0.0

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
        per_sample.append(bp * geo_mean)

    # Corpus-level BLEU
    corpus_precisions = []
    for n in range(max_n):
        if corpus_pred_counts[n] > 0:
            corpus_precisions.append(corpus_match_counts[n] / corpus_pred_counts[n])
        else:
            corpus_precisions.append(0.0)

    if all(p > 0 for p in corpus_precisions):
        log_avg = sum(math.log(p) for p in corpus_precisions) / max_n
        corpus_geo_mean = math.exp(log_avg)
    else:
        corpus_geo_mean = 0.0

    bp = min(1.0, math.exp(1 - total_ref_len / max(total_pred_len, 1)))
    corpus_bleu = bp * corpus_geo_mean

    return {
        "corpus_bleu": round(corpus_bleu, 4),
        "mean": round(sum(per_sample) / len(per_sample), 4) if per_sample else 0.0,
        "min": round(min(per_sample), 4) if per_sample else 0.0,
        "max": round(max(per_sample), 4) if per_sample else 0.0,
        "per_sample": [round(s, 4) for s in per_sample],
    }


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: list of predicted text strings
        references: list of reference text strings

    Returns:
        dict with 'rouge1', 'rouge2', 'rougeL' each containing
        'precision', 'recall', 'f1' (corpus-level averages)
    """
    assert len(predictions) == len(references), "Prediction/reference count mismatch"

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        # ROUGE-1
        pred_unigrams = Counter(pred_tokens)
        ref_unigrams = Counter(ref_tokens)
        overlap = sum((pred_unigrams & ref_unigrams).values())
        p1 = overlap / max(len(pred_tokens), 1)
        r1 = overlap / max(len(ref_tokens), 1)
        f1_1 = 2 * p1 * r1 / max(p1 + r1, 1e-8)
        scores["rouge1"].append({"precision": p1, "recall": r1, "f1": f1_1})

        # ROUGE-2
        pred_bigrams = Counter(_ngrams(pred_tokens, 2))
        ref_bigrams = Counter(_ngrams(ref_tokens, 2))
        overlap2 = sum((pred_bigrams & ref_bigrams).values())
        p2 = overlap2 / max(len(pred_bigrams), 1)
        r2 = overlap2 / max(len(ref_bigrams), 1)
        f1_2 = 2 * p2 * r2 / max(p2 + r2, 1e-8)
        scores["rouge2"].append({"precision": p2, "recall": r2, "f1": f1_2})

        # ROUGE-L (longest common subsequence)
        lcs_len = _lcs_length(pred_tokens, ref_tokens)
        pl = lcs_len / max(len(pred_tokens), 1)
        rl = lcs_len / max(len(ref_tokens), 1)
        f1_l = 2 * pl * rl / max(pl + rl, 1e-8)
        scores["rougeL"].append({"precision": pl, "recall": rl, "f1": f1_l})

    # Average across samples
    result = {}
    for key in scores:
        n = len(scores[key])
        result[key] = {
            "precision": round(sum(s["precision"] for s in scores[key]) / n, 4),
            "recall": round(sum(s["recall"] for s in scores[key]) / n, 4),
            "f1": round(sum(s["f1"] for s in scores[key]) / n, 4),
        }
    return result


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    # Space-optimized DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


# --- Document Chunking ---

def chunk_documents(documents: List[str], chunk_size: int = 512,
                    overlap: int = 50, method: str = "recursive") -> List[Dict[str, Any]]:
    """Split documents into chunks for RAG pipelines.

    Args:
        documents: list of document text strings
        chunk_size: target chunk size in characters
        overlap: overlap between consecutive chunks in characters
        method: chunking method ('fixed', 'recursive', 'sentence')

    Returns:
        list of dicts with 'text', 'doc_index', 'chunk_index', 'char_count'
    """
    chunks = []

    for doc_idx, doc in enumerate(documents):
        if method == "fixed":
            doc_chunks = _chunk_fixed(doc, chunk_size, overlap)
        elif method == "recursive":
            doc_chunks = _chunk_recursive(doc, chunk_size, overlap)
        elif method == "sentence":
            doc_chunks = _chunk_by_sentence(doc, chunk_size, overlap)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

        for chunk_idx, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "text": chunk_text,
                "doc_index": doc_idx,
                "chunk_index": chunk_idx,
                "char_count": len(chunk_text),
            })

    return chunks


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fixed-size chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def _chunk_recursive(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Recursive character splitting (paragraph > sentence > word)."""
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text, separators):
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else " "
        remaining_seps = separators[1:] if len(separators) > 1 else []

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size and remaining_seps:
                    chunks.extend(_split(part, remaining_seps))
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    return _split(text, separators)


def _chunk_by_sentence(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sentence-based chunking."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    current_sentences = []

    for sentence in sentences:
        candidate = current + " " + sentence if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            current_sentences.append(sentence)
        else:
            if current:
                chunks.append(current)
            # Overlap: keep last N sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = " ".join(overlap_sentences + [sentence])
            current_sentences = overlap_sentences + [sentence]

    if current:
        chunks.append(current)

    return chunks


# --- Embedding Index ---

def create_embedding_index(chunks: List[str], model: str = "all-MiniLM-L6-v2",
                           backend: str = "faiss") -> Dict[str, Any]:
    """Create a vector index from text chunks.

    Args:
        chunks: list of text strings to embed
        model: embedding model name (sentence-transformers or OpenAI)
        backend: 'faiss' or 'chroma'

    Returns:
        dict with 'index' (the index object), 'embeddings' (numpy array),
        'model': model name, 'dimension': embedding dimension, 'count': chunk count
    """
    embeddings = _compute_embeddings(chunks, model)

    if backend == "faiss":
        index = _build_faiss_index(embeddings)
    elif backend == "chroma":
        index = _build_chroma_index(chunks, embeddings, model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return {
        "index": index,
        "embeddings": embeddings,
        "model": model,
        "dimension": embeddings.shape[1] if hasattr(embeddings, "shape") else len(embeddings[0]),
        "count": len(chunks),
    }


def _compute_embeddings(texts: List[str], model: str):
    """Compute embeddings using sentence-transformers or OpenAI."""
    if model.startswith("text-embedding-"):
        # OpenAI embeddings
        try:
            import openai
            client = openai.OpenAI()
            response = client.embeddings.create(input=texts, model=model)
            import numpy as np
            return np.array([item.embedding for item in response.data])
        except ImportError:
            raise ImportError("openai package required for OpenAI embeddings")
    else:
        # Sentence Transformers
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(model)
            return st_model.encode(texts, show_progress_bar=True,
                                   normalize_embeddings=True)
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )


def _build_faiss_index(embeddings):
    """Build a FAISS HNSW index."""
    try:
        import faiss
        import numpy as np
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        index.add(embeddings)
        return index
    except ImportError:
        raise ImportError(
            "faiss-cpu required. Install with: pip install faiss-cpu"
        )


def _build_chroma_index(chunks, embeddings, model):
    """Build a ChromaDB collection."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="vector_stores/chroma")
        collection = client.get_or_create_collection(
            name="documents",
            metadata={"embedding_model": model}
        )
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
        )
        return collection
    except ImportError:
        raise ImportError(
            "chromadb required. Install with: pip install chromadb"
        )


# --- Retrieval Evaluation ---

def evaluate_retrieval(queries: List[str], retrieved: List[List[int]],
                       relevant: List[List[int]],
                       k_values: List[int] = None) -> Dict[str, float]:
    """Evaluate retrieval quality with standard IR metrics.

    Args:
        queries: list of query strings (for reference)
        retrieved: list of lists of retrieved document indices (ranked)
        relevant: list of lists of relevant document indices (ground truth)
        k_values: list of k values for recall@k (default: [5, 10, 20])

    Returns:
        dict with recall@k, MRR, NDCG@10
    """
    if k_values is None:
        k_values = [5, 10, 20]

    assert len(retrieved) == len(relevant), "Retrieved/relevant count mismatch"
    n = len(queries)

    # Recall@k
    recall_at_k = {}
    for k in k_values:
        recalls = []
        for ret, rel in zip(retrieved, relevant):
            ret_set = set(ret[:k])
            rel_set = set(rel)
            if rel_set:
                recalls.append(len(ret_set & rel_set) / len(rel_set))
            else:
                recalls.append(1.0)  # no relevant docs = perfect recall
        recall_at_k[f"recall_at_{k}"] = round(sum(recalls) / n, 4)

    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for ret, rel in zip(retrieved, relevant):
        rel_set = set(rel)
        rr = 0.0
        for rank, doc_id in enumerate(ret, 1):
            if doc_id in rel_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    mrr = round(sum(reciprocal_ranks) / n, 4)

    # NDCG@10
    ndcg_scores = []
    for ret, rel in zip(retrieved, relevant):
        rel_set = set(rel)
        dcg = 0.0
        for rank, doc_id in enumerate(ret[:10], 1):
            if doc_id in rel_set:
                dcg += 1.0 / math.log2(rank + 1)
        # Ideal DCG
        ideal_relevant = min(len(rel_set), 10)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_relevant + 1))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    ndcg_at_10 = round(sum(ndcg_scores) / n, 4)

    result = {**recall_at_k, "mrr": mrr, "ndcg_at_10": ndcg_at_10}
    return result


# --- Prompt Utilities ---

def load_prompt_template(path: str) -> Dict[str, str]:
    """Load a prompt template file.

    Supports formats:
    - Plain text: entire file is the user prompt
    - Markdown with ---SYSTEM--- and ---USER--- sections
    - JSON with 'system' and 'user' keys

    Args:
        path: path to prompt template file

    Returns:
        dict with 'system' and 'user' keys
    """
    content = Path(path).read_text()

    # JSON format
    if path.endswith(".json"):
        data = json.loads(content)
        return {
            "system": data.get("system", ""),
            "user": data.get("user", content),
        }

    # Markdown sections
    if "---SYSTEM---" in content:
        parts = content.split("---SYSTEM---")
        after_system = parts[1] if len(parts) > 1 else ""
        if "---USER---" in after_system:
            system_part, user_part = after_system.split("---USER---", 1)
            return {
                "system": system_part.strip(),
                "user": user_part.strip(),
            }
        return {"system": after_system.strip(), "user": ""}

    # Plain text
    return {"system": "", "user": content.strip()}


def format_prompt(template: Dict[str, str], **variables) -> Dict[str, str]:
    """Fill variables in a prompt template.

    Variables are referenced as {variable_name} in the template.

    Args:
        template: dict with 'system' and 'user' keys
        **variables: key-value pairs to substitute

    Returns:
        dict with 'system' and 'user' keys, variables substituted
    """
    return {
        "system": template["system"].format(**variables) if template["system"] else "",
        "user": template["user"].format(**variables) if template["user"] else "",
    }


# --- Data Utilities ---

def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: path to JSONL file

    Returns:
        list of dicts (one per line)
    """
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON on line {line_num}, skipping")
    return records


def save_jsonl(records: List[Dict], path: str):
    """Save a list of dicts to a JSONL file.

    Args:
        records: list of dicts
        path: output file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def validate_finetune_dataset(path: str) -> Dict[str, Any]:
    """Validate a fine-tuning dataset for common issues.

    Checks format, duplicates, empty fields, token lengths.

    Args:
        path: path to JSONL dataset

    Returns:
        dict with 'valid': bool, 'sample_count', 'issues': list, 'stats': dict
    """
    records = load_jsonl(path)
    issues = []
    prompt_lengths = []
    completion_lengths = []
    seen_hashes = set()
    duplicates = 0

    for i, record in enumerate(records):
        # Check format
        if "messages" in record:
            # Chat format
            messages = record["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                issues.append(f"Line {i + 1}: messages must be a list with >= 2 entries")
                continue
            for msg in messages:
                if "role" not in msg or "content" not in msg:
                    issues.append(f"Line {i + 1}: message missing role or content")
            content = " ".join(m.get("content", "") for m in messages)
            prompt_lengths.append(len(content.split()))
        elif "prompt" in record and "completion" in record:
            # Completion format
            if not record["prompt"].strip():
                issues.append(f"Line {i + 1}: empty prompt")
            if not record["completion"].strip():
                issues.append(f"Line {i + 1}: empty completion")
            prompt_lengths.append(len(record["prompt"].split()))
            completion_lengths.append(len(record["completion"].split()))
            content = record["prompt"] + record["completion"]
        else:
            issues.append(f"Line {i + 1}: missing 'messages' or 'prompt'/'completion' fields")
            continue

        # Check duplicates
        content_hash = hash(content)
        if content_hash in seen_hashes:
            duplicates += 1
        seen_hashes.add(content_hash)

    stats = {
        "sample_count": len(records),
        "duplicates": duplicates,
        "avg_prompt_words": round(sum(prompt_lengths) / max(len(prompt_lengths), 1), 1),
        "avg_completion_words": round(
            sum(completion_lengths) / max(len(completion_lengths), 1), 1
        ) if completion_lengths else None,
    }

    return {
        "valid": len(issues) == 0,
        "sample_count": len(records),
        "issues": issues[:20],  # cap at 20 issues
        "stats": stats,
    }
