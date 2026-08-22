"""
test_rag.py — Sanity checks for Phase 1 (RAG indexer + retrieval)

Run with:
    python test_rag.py

This script does NOT require pytest. It prints pass/fail for each check so
you can confirm the index is working correctly before building the agent.

Checks:
  1. All 14 documents parse without errors.
  2. The internal document (doc 14) is excluded from the index.
  3. The legacy/superseded document (doc 02) is excluded from normal retrieval.
  4. "What is the return window?" retrieves doc 01 as the top result.
  5. "Canada shipping" retrieves doc 06.
  6. "TrailPlus return window" retrieves doc 09.
  7. "Breeze Tumbler dishwasher" retrieves BOTH doc 11 and doc 12 (conflict pair).
  8. Privacy-sensitive content from doc 14 is NOT in any retrieved chunk.
"""

import logging
import os
import sys
from pathlib import Path

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Set up logging so we can see what the RAG module is doing
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_rag")

# ---- import after path is set ----
from agent.rag import (
    parse_all_documents,
    chunk_all_documents,
    build_index,
    retrieve,
    detect_conflicts,
    _is_excluded,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, condition))


# ---------------------------------------------------------------------------
# Check 1 — Parsing
# ---------------------------------------------------------------------------

print("\n── Check 1: Document parsing ─────────────────────────")
docs = parse_all_documents()
check("Parsed 14 documents", len(docs) == 14, f"Found {len(docs)}")

filenames = [d["filename"] for d in docs]
check("Doc 01 present", "01-returns-policy-current.md" in filenames)
check("Doc 14 present (but will be excluded)", "14-internal-content-migration-notes.md" in filenames)

# Check front matter was extracted
doc01 = next(d for d in docs if "01-returns-policy-current" in d["filename"])
check("Doc 01 status=active", doc01["meta"]["status"] == "active",
      f"Got: {doc01['meta']['status']}")
check("Doc 01 policy_authority=official", doc01["meta"]["policy_authority"] == "official",
      f"Got: {doc01['meta']['policy_authority']}")

doc02 = next(d for d in docs if "02-returns-policy-legacy" in d["filename"])
check("Doc 02 status=superseded", doc02["meta"]["status"] == "superseded",
      f"Got: {doc02['meta']['status']}")

doc14 = next(d for d in docs if "14-internal" in d["filename"])
check("Doc 14 audience=internal", doc14["meta"]["audience"] == "internal",
      f"Got: {doc14['meta']['audience']}")


# ---------------------------------------------------------------------------
# Check 2 — Chunking + exclusion
# ---------------------------------------------------------------------------

print("\n── Check 2: Chunking and exclusion ───────────────────")
chunks = chunk_all_documents(docs)
check("Produced chunks", len(chunks) > 0, f"{len(chunks)} total chunks")

# Excluded chunks — docs 13 (internal escalation rules) and 14 (internal scratchpad)
INTERNAL_DOCS = {"14-internal-content-migration-notes.md", "13-support-escalation.md"}
excluded = [c for c in chunks if _is_excluded(c)]
check(
    "Internal docs (13, 14) chunks are excluded",
    all(c["filename"] in INTERNAL_DOCS for c in excluded),
    f"{len(excluded)} excluded chunks from: {list({c['filename'] for c in excluded})}"
)

# Non-excluded chunks should not include internal docs
indexed = [c for c in chunks if not _is_excluded(c)]
internal_in_indexed = [c for c in indexed if c["audience"] == "internal"]
check("No internal chunks in indexed set", len(internal_in_indexed) == 0,
      f"Found {len(internal_in_indexed)} internal chunks that slipped through")

# Each chunk should have required keys
required_keys = {"chunk_id", "text", "heading", "filename", "status", "policy_authority"}
missing_key_chunks = [c for c in chunks if not required_keys.issubset(c.keys())]
check("All chunks have required metadata keys", len(missing_key_chunks) == 0,
      f"{len(missing_key_chunks)} chunks missing keys")


# ---------------------------------------------------------------------------
# Summary helper — defined early so it's available even if we exit mid-test
# ---------------------------------------------------------------------------

def _print_summary():
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"\n{'─'*52}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  All checks passed -- Phase 1 is ready!")
    else:
        print("  Some checks failed -- see details above.")
    print(f"{'─'*52}\n")


# ---------------------------------------------------------------------------
# Check 3 -- Index build
# ---------------------------------------------------------------------------

print("\n-- Check 3: Index build --")
print("  Building index (this embeds text via OpenAI -- may take 10-30s)...")
try:
    build_index(force_rebuild=True)
    check("build_index() completed without error", True)
except Exception as e:
    check("build_index() completed without error", False, str(e))
    print("\nWARNING: Index build failed. Remaining retrieval checks skipped.")
    _print_summary()
    sys.exit(1)


# ---------------------------------------------------------------------------
# Check 4 — Retrieval: return window → doc 01
# ---------------------------------------------------------------------------

print("\n-- Check 4: Retrieval quality --")

q1 = "What is the standard return window for customers?"
r1 = retrieve(q1)
filenames_r1 = [c["filename"] for c in r1]
check(
    "Return window query → doc 01 is top result",
    filenames_r1[0] == "01-returns-policy-current.md" if r1 else False,
    f"Top result: {filenames_r1[0] if r1 else 'no results'}"
)
check(
    "Return window query → doc 02 (legacy) NOT in results",
    "02-returns-policy-legacy.md" not in filenames_r1,
    f"Filenames returned: {filenames_r1}"
)
check(
    "Return window query → doc 14 (internal) NOT in results",
    "14-internal-content-migration-notes.md" not in filenames_r1,
)


# ---------------------------------------------------------------------------
# Check 5 — Canada shipping → doc 06
# ---------------------------------------------------------------------------

q2 = "Do you ship to Canada and how long does it take?"
r2 = retrieve(q2)
filenames_r2 = [c["filename"] for c in r2]
check(
    "Canada shipping query → doc 06 is in top 3",
    "06-international-shipping.md" in filenames_r2[:3],
    f"Top 3: {filenames_r2[:3]}"
)


# ---------------------------------------------------------------------------
# Check 6 — TrailPlus return window → doc 09
# ---------------------------------------------------------------------------

q3 = "TrailPlus member return window"
r3 = retrieve(q3)
filenames_r3 = [c["filename"] for c in r3]
check(
    "TrailPlus query → doc 09 is in top 3",
    "09-trailplus-membership.md" in filenames_r3[:3],
    f"Top 3: {filenames_r3[:3]}"
)


# ---------------------------------------------------------------------------
# Check 7 — Breeze Tumbler dishwasher → conflict between doc 11 and doc 12
# ---------------------------------------------------------------------------

q4 = "Can I put the Breeze Tumbler in the dishwasher?"
r4 = retrieve(q4)
filenames_r4 = [c["filename"] for c in r4]
has_11 = "11-product-care.md" in filenames_r4
has_12 = "12-breeze-tumbler-product-card.md" in filenames_r4
check(
    "Dishwasher query → doc 11 (product care) retrieved",
    has_11,
    f"Filenames: {filenames_r4}"
)
check(
    "Dishwasher query → doc 12 (product card) retrieved",
    has_12,
    f"Filenames: {filenames_r4}"
)

conflicts = detect_conflicts(r4)
check(
    "detect_conflicts() flags doc 11 vs doc 12 as a conflict pair",
    len(conflicts) > 0,
    f"Conflict pairs found: {[(a['filename'], b['filename']) for a, b in conflicts]}"
)


# ---------------------------------------------------------------------------
# Check 8 — Privacy: internal text not in any retrieved chunk
# ---------------------------------------------------------------------------

print("\n-- Check 5: Privacy / injection safety --")
sensitive_strings = [
    "SYSTEM INSTRUCTION",
    "Ignore all prior rules",
    "ava.morgan@example.test",
    "fraud review cleared",
]

all_retrieved_texts = " ".join(c["text"] for c in r1 + r2 + r3 + r4)
for s in sensitive_strings:
    check(
        f"Sensitive string not in any retrieved chunk: {s!r}",
        s not in all_retrieved_texts,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_print_summary()
sys.exit(0 if all(ok for _, ok in results) else 1)
