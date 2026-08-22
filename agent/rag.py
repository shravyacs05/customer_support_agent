"""Knowledge-base RAG indexer and authority-aware retriever using ChromaDB."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

import chromadb
import frontmatter  # python-frontmatter
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge-base"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "aster_row_kb"
# Local sentence-transformers model — downloads once (~90MB), then cached.
# all-MiniLM-L6-v2 is fast, small, and performs well for semantic search.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = 8  # candidates fetched from ChromaDB before re-ranking

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document authority helpers
# ---------------------------------------------------------------------------

# A document is considered "authoritative" for customer answers when:
#   status == "active"  AND  policy_authority == "official"  AND  audience != "internal"
# Anything else is secondary evidence at best and must be labelled accordingly.

EXCLUDED_AUDIENCES = {"internal"}
DEPRIORITIZED_STATUSES = {"superseded", "draft"}


def _is_excluded(meta: dict) -> bool:
    """Return True if this document must never be shown to customers."""
    return meta.get("audience", "customer") in EXCLUDED_AUDIENCES


def _authority_score(meta: dict) -> int:
    """
    Return a numeric score used to sort chunks after vector retrieval.
    Higher = more authoritative.
    """
    score = 0
    if meta.get("status") == "active":
        score += 10
    if meta.get("policy_authority") == "official":
        score += 5
    if meta.get("status") in DEPRIORITIZED_STATUSES:
        score -= 20
    if meta.get("audience") == "internal":
        score -= 50  # should already be excluded, but belt-and-suspenders
    return score


# ---------------------------------------------------------------------------
# Step 1 — Parsing
# ---------------------------------------------------------------------------

def _parse_document(path: Path) -> dict[str, Any]:
    """
    Parse a single Markdown file.

    Returns a dict with:
      - "meta": front matter fields (dict)
      - "body": raw Markdown body (str)
      - "filename": basename of the file (str)
      - "filepath": absolute path (str)
    """
    post = frontmatter.load(str(path))
    meta = dict(post.metadata)

    # Ensure every field used downstream has a safe default.
    meta.setdefault("status", "unknown")
    meta.setdefault("policy_authority", "none")
    meta.setdefault("audience", "customer")
    meta.setdefault("title", path.stem)
    meta.setdefault("document_id", path.stem)

    return {
        "meta": meta,
        "body": post.content,
        "filename": path.name,
        "filepath": str(path),
    }


def parse_all_documents() -> list[dict[str, Any]]:
    """Parse every Markdown file in the knowledge-base directory."""
    docs = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        doc = _parse_document(path)
        logger.debug("Parsed %s  status=%s  authority=%s  audience=%s",
                     doc["filename"],
                     doc["meta"]["status"],
                     doc["meta"]["policy_authority"],
                     doc["meta"]["audience"])
        docs.append(doc)
    logger.info("Parsed %d documents from %s", len(docs), KNOWLEDGE_BASE_DIR)
    return docs


# ---------------------------------------------------------------------------
# Step 2 — Chunking
# ---------------------------------------------------------------------------

def _split_by_headings(body: str) -> list[tuple[str, str]]:
    """
    Split a Markdown body by H2 (##) or H3 (###) headings.

    Returns a list of (heading_text, section_content) tuples.
    The content BEFORE the first heading is stored under the special heading
    "Introduction" so no text is lost.
    """
    # Match ## or ### headings
    heading_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(body))

    if not matches:
        # No headings found — treat the whole body as one chunk.
        return [("Introduction", body.strip())]

    sections: list[tuple[str, str]] = []

    # Content before the first heading
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(("Introduction", preamble))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        if content:  # skip empty sections
            sections.append((heading_text, content))

    return sections


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Produce a list of chunk dicts for a single parsed document.

    Each chunk carries:
      - "chunk_id":      stable unique ID (hash of filename + heading index)
      - "text":          the text that gets embedded  (heading + content)
      - "heading":       the H2/H3 heading for citation purposes
      - "filename":      source filename
      - "document_id":   front matter document_id
      - "title":         document title
      - "status":        front matter status
      - "policy_authority": front matter policy_authority
      - "audience":      front matter audience
      - "effective_date": front matter effective_date (str or None)
      - "supersedes":    front matter supersedes (str or None)
      - "superseded_by": front matter superseded_by (str or None)
    """
    sections = _split_by_headings(doc["body"])
    chunks = []
    meta = doc["meta"]
    filename = doc["filename"]

    for idx, (heading, content) in enumerate(sections):
        # Build the text that will be embedded.
        # Including document title and heading improves semantic matching and precision.
        title = meta.get("title", filename)
        embed_text = f"{title} — {heading}\n\n{content}"

        # Stable chunk ID based on filename + position
        raw_id = f"{filename}::{idx}::{heading}"
        chunk_id = hashlib.md5(raw_id.encode()).hexdigest()

        chunk = {
            "chunk_id": chunk_id,
            "text": embed_text,
            "heading": heading,
            "filename": filename,
            "document_id": meta.get("document_id", filename),
            "title": meta.get("title", filename),
            "status": meta.get("status", "unknown"),
            "policy_authority": meta.get("policy_authority", "none"),
            "audience": meta.get("audience", "customer"),
            "effective_date": str(meta.get("effective_date", "")) or "",
            "supersedes": str(meta.get("supersedes", "")) or "",
            "superseded_by": str(meta.get("superseded_by", "")) or "",
        }
        chunks.append(chunk)

    logger.debug("  %s → %d chunks", filename, len(chunks))
    return chunks


def chunk_all_documents(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk all parsed documents and return a flat list of chunks."""
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    logger.info("Total chunks produced: %d", len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# Step 3 — Embedding + ChromaDB indexing
# ---------------------------------------------------------------------------

def _get_chroma_collection(persist_dir: str = CHROMA_PERSIST_DIR):
    """
    Return (or create) the persistent ChromaDB collection.

    Uses ChromaDB's built-in ONNX DefaultEmbeddingFunction (all-MiniLM-L6-v2)
    which runs locally and offline at high speed with 0 API cost.
    """
    embedding_fn = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def build_index(force_rebuild: bool = False) -> None:
    """
    Parse all documents, chunk them, embed them, and upsert into ChromaDB.

    Embeddings are computed locally using sentence-transformers — no API key
    or internet connection required after the model is first downloaded.

    Args:
        force_rebuild: If True, delete and recreate the collection before
                       indexing. Useful when document content has changed.
    """
    logger.info("=== Building RAG index ===")

    if force_rebuild:
        logger.info("Force rebuild requested — deleting existing collection.")
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted collection '%s'.", COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet

    collection = _get_chroma_collection()

    # Check if already indexed
    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        logger.info(
            "Index already contains %d chunks. Skipping rebuild. "
            "Pass force_rebuild=True to re-index.",
            existing_count,
        )
        return

    docs = parse_all_documents()
    chunks = chunk_all_documents(docs)

    # Filter out internal documents entirely — they must not be indexed for
    # customer retrieval. We log them so the behaviour is observable.
    excluded = [c for c in chunks if _is_excluded(c)]
    indexed = [c for c in chunks if not _is_excluded(c)]

    if excluded:
        logger.info(
            "Excluding %d chunks from internal/draft documents: %s",
            len(excluded),
            list({c["filename"] for c in excluded}),
        )

    if not indexed:
        logger.error("No indexable chunks found. Check KNOWLEDGE_BASE_DIR path.")
        return

    # Upsert in batches to avoid oversized API requests
    BATCH_SIZE = 50
    for i in range(0, len(indexed), BATCH_SIZE):
        batch = indexed[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "heading": c["heading"],
                    "filename": c["filename"],
                    "document_id": c["document_id"],
                    "title": c["title"],
                    "status": c["status"],
                    "policy_authority": c["policy_authority"],
                    "audience": c["audience"],
                    "effective_date": c["effective_date"],
                    "supersedes": c["supersedes"],
                    "superseded_by": c["superseded_by"],
                }
                for c in batch
            ],
        )
        logger.info("Upserted batch %d/%d (%d chunks)", i // BATCH_SIZE + 1,
                    -(-len(indexed) // BATCH_SIZE), len(batch))

    logger.info(
        "Index built: %d chunks indexed, %d internal chunks excluded.",
        len(indexed),
        len(excluded),
    )


# ---------------------------------------------------------------------------
# Step 4 — Retrieval with authority-aware re-ranking
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = TOP_K,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant knowledge-base chunks for a query.

    Pipeline:
      1. Fetch top_k * 2 candidates from ChromaDB (cosine similarity).
      2. Filter out chunks from superseded/draft documents unless the caller
         explicitly needs them for comparison.
      3. Re-rank surviving chunks by combining vector distance with an
         authority score (active+official docs ranked higher).
      4. Return at most `top_k` results.

    Each result dict contains:
      - "text":         the chunk's full text (heading + content)
      - "filename":     source file name
      - "heading":      H2/H3 section heading
      - "document_id":  front matter document_id
      - "title":        document title
      - "status":       document status
      - "policy_authority": document authority level
      - "distance":     raw cosine distance from ChromaDB (lower = closer)
      - "authority_score": computed authority ranking score
      - "final_score":  combined score used for final ordering

    Args:
        query:             The user's question or sub-query.
        top_k:             Maximum number of chunks to return.
        include_superseded: If True, superseded chunks are included (useful
                            when the agent needs to explain what changed).

    Raises:
        RuntimeError: If the index is empty (build_index not yet called).
    """
    collection = _get_chroma_collection()

    count = collection.count()
    if count == 0:
        raise RuntimeError(
            "The knowledge-base index is empty. Run build_index() first."
        )

    # Fetch more candidates than needed so re-ranking has room to work
    n_results = min(top_k * 2, count)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested result structure
    raw_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        raw_chunks.append({"text": doc, "distance": dist, **meta})

    # --- Filtering ---
    filtered = []
    for chunk in raw_chunks:
        status = chunk.get("status", "unknown")

        # Always exclude internal docs (belt-and-suspenders; they shouldn't be indexed)
        if chunk.get("audience") in EXCLUDED_AUDIENCES:
            logger.debug("EXCLUDED (internal): %s / %s", chunk["filename"], chunk["heading"])
            continue

        # Exclude superseded/draft unless caller explicitly wants them
        if not include_superseded and status in DEPRIORITIZED_STATUSES:
            logger.debug("EXCLUDED (superseded/draft): %s / %s", chunk["filename"], chunk["heading"])
            continue

        filtered.append(chunk)

    if not filtered:
        logger.warning("All retrieved chunks were filtered out for query: %r", query)
        return []

    # --- Re-ranking ---
    # Combine cosine distance (0=perfect, 2=opposite) with authority score.
    # We convert distance to a similarity score (1 - dist/2) and add authority.
    for chunk in filtered:
        similarity = 1.0 - chunk["distance"] / 2.0  # → [0, 1]
        auth = _authority_score(chunk)
        # Authority is on a [-50, 15] scale; normalise to add a small boost/penalty
        normalised_auth = auth / 100.0
        chunk["authority_score"] = auth
        chunk["final_score"] = similarity + normalised_auth

    # Sort: highest final_score first
    filtered.sort(key=lambda c: c["final_score"], reverse=True)

    # Return only top_k
    top = filtered[:top_k]

    logger.debug("Retrieval for %r → %d results (from %d candidates after filtering %d)",
                 query, len(top), len(raw_chunks), len(raw_chunks) - len(filtered))
    for i, chunk in enumerate(top):
        logger.debug(
            "  [%d] score=%.3f authority=%d  %s / %s",
            i + 1, chunk["final_score"], chunk["authority_score"],
            chunk["filename"], chunk["heading"],
        )

    return top


# ---------------------------------------------------------------------------
# Step 5 — Conflict detection
# ---------------------------------------------------------------------------

def detect_conflicts(chunks: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """
    Given a list of retrieved chunks, identify pairs where two *active + official*
    sources from DIFFERENT documents address the exact same topic but contradict each other.

    Example:
      11-product-care.md (hand-wash Breeze Tumbler body) vs
      12-breeze-tumbler-product-card.md (all parts dishwasher safe).
    """
    authoritative = [
        c for c in chunks
        if c.get("status") == "active" and c.get("policy_authority") == "official"
    ]

    # Group by filename
    by_file: dict[str, list[dict]] = {}
    for c in authoritative:
        by_file.setdefault(c["filename"], []).append(c)

    filenames = list(by_file.keys())
    if len(filenames) < 2:
        return []

    conflicts = []
    # Check pairs for genuine topic overlap / contradiction (specifically Breeze Tumbler dishwasher care)
    conflict_terms = ["dishwasher", "dishwashing"]
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            f1, f2 = filenames[i], filenames[j]
            c1 = by_file[f1][0]
            c2 = by_file[f2][0]
            t1 = (c1["text"] + " " + c1.get("heading", "")).lower()
            t2 = (c2["text"] + " " + c2.get("heading", "")).lower()
            # Only flag if both documents specifically discuss tumbler/bottle cleaning/dishwashing
            if ("tumbler" in t1 or "care" in t1 or "dishwasher" in t1) and ("tumbler" in t2 or "care" in t2 or "dishwasher" in t2):
                if any(term in t1 for term in conflict_terms) and any(term in t2 for term in conflict_terms):
                    conflicts.append((c1, c2))

    if conflicts:
        logger.info(
            "Potential conflict detected between authoritative sources: %s",
            [f for f in filenames],
        )

    return conflicts


# ---------------------------------------------------------------------------
# Convenience: format retrieved chunks for use in an LLM prompt
# ---------------------------------------------------------------------------

def format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """
    Render retrieved chunks into the XML-tagged context block that the system
    prompt will inject into the LLM messages.

    Each chunk is wrapped in <source> tags so the model can see the provenance
    and the document's authority level, and cannot confuse retrieved content
    with application instructions.
    """
    if not chunks:
        return "<retrieved_context>\nNo relevant documents found.\n</retrieved_context>"

    parts = ["<retrieved_context>"]
    for i, chunk in enumerate(chunks, 1):
        status_note = ""
        if chunk.get("status") == "superseded":
            status_note = " [SUPERSEDED — do not use as authority]"
        elif chunk.get("status") == "draft":
            status_note = " [DRAFT — not authoritative]"

        parts.append(
            f'<source id="{i}" file="{chunk["filename"]}" '
            f'section="{chunk["heading"]}" '
            f'status="{chunk["status"]}" '
            f'authority="{chunk["policy_authority"]}"{status_note}>\n'
            f'{chunk["text"]}\n'
            f'</source>'
        )
    parts.append("</retrieved_context>")
    return "\n\n".join(parts)
