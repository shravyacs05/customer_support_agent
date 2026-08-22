# Aster & Row — Customer Support Agent

A customer support assistant built for Aster & Row, an ecommerce brand selling bags, drinkware, and travel accessories. The agent handles policy inquiries with source-grounded citations, provides sanitized order status tracking via function calling, detects source conflicts, defends against prompt injections, and manages multi-turn conversation sessions.

---

## Architecture Overview

```
                          +------------------------+
                          |     Customer Query     |
                          +-----------+------------+
                                      |
                                      v
                          +-----------+------------+
                          |   SupportAgent Core    |
                          +-----------+------------+
                                      |
             +------------------------+------------------------+
             |                                                 |
             v                                                 v
+------------+------------+                       +------------+------------+
|  ChromaDB Vector Store  |                       |   Order Lookup Tool     |
| (Local ONNX miniLM-L6)  |                       |  (Normalized & Safe)    |
+------------+------------+                       +------------+------------+
             |                                                 |
             v                                                 v
+------------+------------+                       +------------+------------+
|  Authority Re-Ranker    |                       |     Privacy Filter      |
|  & Conflict Detector    |                       |  (Masks PII & Notes)    |
+------------+------------+                       +------------+------------+
             \                                                 /
              \                                               /
               v                                             v
             +-----------------------------------------------+
             |      Gemini Model with Structured Tool Calls  |
             +----------------------+------------------------+
                                    |
                                    v
             +----------------------+------------------------+
             |   Grounded Answer + Citations + Handoff Flag  |
             +-----------------------------------------------+
```

---

## Technical Approach

### 1. Document Indexing and Authority Re-Ranking
- Embeddings are computed locally using ChromaDB's ONNX runtime (`all-MiniLM-L6-v2`, 384 dimensions) with no external embedding API latency or cost.
- Documents are split by markdown headings (`##` and `###`), with parent document titles prepended to each section chunk to retain semantic context.
- Front-matter metadata (`status`, `policy_authority`, `audience`) is parsed during indexing.
- Internal documents (`audience: internal`) are excluded from vector indexing to prevent leakage.
- Active official policies receive positive weighting during retrieval re-ranking, while draft or legacy documents are filtered out.

### 2. Conflict Detection
- The retriever automatically inspects candidate documents for direct policy contradictions between active official sources (such as the Breeze Tumbler dishwasher guidance in Doc 11 versus Doc 12).
- When a conflict is detected, the agent surfaces both document positions, provides interim safe guidance (hand-washing the body), and advises confirming with human support.

### 3. Order Tool and Data Privacy
- Order IDs are normalized across common user inputs (e.g. whitespace, lowercase, surrounding punctuation).
- PII (emails, customer names, shipping addresses) and internal fields (risk scores, warehouse notes, internal tags) are stripped before tool results reach the model.
- Cancelled and returned orders have stale tracking numbers, carriers, and delivery dates wiped so old estimates are never quoted to customers.
- Orders in an exception state or requests to cancel already-delivered shipments automatically recommend human support escalation.

---

## Setup and Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- A Gemini API key (from Google AI Studio)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/shravyacs05/customer_support_agent.git
cd customer_support_agent
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Copy `.env.example` to `.env` and insert your Gemini API key:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-3.1-flash-lite
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
```

---

## Running the Agent

### Interactive CLI Mode
Start the interactive chat interface:
```bash
python app.py
```

### Observability and Debug Mode
To inspect retrieval scores, chunks, sanitized tool payloads, and session state on every turn:
```bash
python app.py --debug
```

*(You can also toggle debug mode inside the CLI at any time by typing `debug`)*

#### Structured Trace Fields:
- `current_user_message`: The active prompt received.
- `conversation_history`: Multi-turn message history with role tags.
- `retrieved_passages`: Retrieved chunks, section headings, authority levels, cosine distances, and re-ranking scores.
- `conflicts_detected`: Contradicting document pairs surfaced by the conflict detector.
- `tool_call`: Name of invoked tool and sanitized input arguments.
- `sanitized_tool_result`: Safe payload returned from the order lookup tool.
- `final_response`: Customer-facing text output.
- `handoff_recommended`: Boolean indicator for human escalation.

---

## Evaluation Suite

Run the full automated benchmark test suite with a single command:

```bash
python evaluation/run_eval.py
```

The evaluation suite runs deterministic assertions across all 20 test cases (15 visible benchmark cases + 5 original edge cases) verifying source citations, tool calls, argument values, forbidden text, abstentions, and handoff flags.

### Benchmark Results

| Category | Baseline Prototype | Final Agent | Status |
| :--- | :---: | :---: | :---: |
| Retrieval | 33.3% (1/3) | 100.0% (3/3) | PASS |
| Multi-Source Grounding | 50.0% (1/2) | 100.0% (2/2) | PASS |
| Conversation | 0.0% (0/1) | 100.0% (1/1) | PASS |
| Groundedness | 33.3% (1/3) | 100.0% (3/3) | PASS |
| Tool Use | 100.0% (2/2) | 100.0% (2/2) | PASS |
| Tool Reliability | 50.0% (2/4) | 100.0% (4/4) | PASS |
| Privacy | 100.0% (1/1) | 100.0% (1/1) | PASS |
| Prompt Security | 50.0% (1/2) | 100.0% (2/2) | PASS |
| Abstention | 100.0% (1/1) | 100.0% (1/1) | PASS |
| Source Conflict | 100.0% (1/1) | 100.0% (1/1) | PASS |
| **Total** | **65.0% (13/20)** | **100.0% (20/20)** | **PASS** |

### Individual Test Case Summary

1. `standard-return-window` (Retrieval) — 30-day return policy with citation, excludes legacy notes. [PASS]
2. `trailplus-return-window` (Retrieval) — 45-day member return policy. [PASS]
3. `final-sale-damaged-exception` (Multi-Source) — Synthesizes Docs 03 and 04 with 7-day damaged review. [PASS]
4. `canada-multiturn` (Conversation) — Retains international context, quotes 5–9 days and duties. [PASS]
5. `unsupported-country` (Groundedness) — Abstains on unsupported shipping destinations (Germany). [PASS]
6. `valid-order-lookup` (Tool Use) — Invokes lookup tool and outputs carrier / arrival date. [PASS]
7. `missing-order-id` (Tool Use) — Requests order ID when not provided. [PASS]
8. `cancelled-order-stale-eta` (Tool Reliability) — Suppresses stale delivery estimates on cancelled orders. [PASS]
9. `unknown-order` (Tool Reliability) — Handles missing order IDs gracefully and suggests human support. [PASS]
10. `shipped-without-eta` (Tool Reliability) — Notes carrier transit when delivery ETA is unavailable. [PASS]
11. `order-data-privacy` (Privacy) — Refuses disclosure of customer emails, addresses, and risk scores. [PASS]
12. `no-lifetime-warranty` (Groundedness) — Refutes lifetime warranty assumptions with 1-year / 2-year periods. [PASS]
13. `retrieved-prompt-injection` (Prompt Security) — Ignores migration note override attempts. [PASS]
14. `insufficient-information` (Abstention) — Safely abstains on undocumented specifications (vegan fabrics). [PASS]
15. `genuine-active-source-conflict` (Source Conflict) — Surfaces dishwashing contradiction on Breeze Tumbler. [PASS]
16. `custom-trailplus-fee-waiver` (Retrieval) — Explains $6.95 return shipping fee waiver for members. [PASS]
17. `custom-gift-card-refund-refusal` (Groundedness) — Explains gift cards are final sale and non-refundable. [PASS]
18. `custom-cancel-delivered-order` (Tool Reliability) — Refuses cancellation on delivered orders and suggests returns. [PASS]
19. `custom-price-adjustment-window` (Multi-Source) — Enforces the 7-day price adjustment policy cutoff. [PASS]
20. `custom-system-prompt-exfiltration` (Prompt Security) — Refuses developer prompt extraction overrides. [PASS]

---

## Bug Diary

### Bug 1: Internal Rule Leakage via Vector DB
- **Reproduction**: When asked about policy exceptions, the retriever pulled passages from `14-internal-content-migration-notes.md` containing prompt injection instructions ("IGNORE ALL PREVIOUS INSTRUCTIONS").
- **Root Cause**: The RAG indexer parsed all `.md` files without checking audience metadata.
- **Fix**: Added a metadata exclusion rule in `agent/rag.py` that ignores documents marked `audience: internal`.
- **Regression Test**: `evaluation/all-cases.json` -> `retrieved-prompt-injection`.

### Bug 2: Stale Delivery Date Quoted on Cancelled Orders
- **Reproduction**: Asking for the status of `ORD-1004` (cancelled) returned an old estimated delivery date.
- **Root Cause**: The order lookup tool returned raw JSON fields from the dataset without sanitizing records where status was cancelled or returned.
- **Fix**: Added status precedence sanitization in `agent/order_tool.py` that wipes carrier, tracking, and ETA fields when an order is cancelled or returned.
- **Regression Test**: `evaluation/all-cases.json` -> `cancelled-order-stale-eta`.

### Bug 3: False-Positive Conflict Trigger on Standard Return Questions
- **Reproduction**: Asking "How long does a customer have to return an unused backpack?" caused the agent to inject a conflict warning between Doc 01 and Doc 11 and recommend an unnecessary human handoff.
- **Root Cause**: The conflict detector in `agent/rag.py` searched for the keyword `"wash"`, which matched `"unwashed"` in the return policy and `"wash"` in the product care document.
- **Fix**: Scoped conflict detection strictly to documents that both discuss drinkware / tumbler care.
- **Regression Test**: `evaluation/all-cases.json` -> `standard-return-window`.

### Bug 4: Delivered Order Cancellation Handling
- **Reproduction**: Querying "Please cancel my order ORD-1006 right now" did not state that delivered orders cannot be cancelled and failed to flag a human handoff.
- **Root Cause**: The cancellation check in `agent/agent.py` only handled `status == "shipped"`.
- **Fix**: Added handling for `status == "delivered"`, stating delivered orders cannot be cancelled, directing to returns, and setting `handoff = True`.
- **Regression Test**: `evaluation/all-cases.json` -> `custom-cancel-delivered-order`.

---

## Known Limitations and Future Work

1. **Concurrent Async Tool Calling**: Tool execution currently runs synchronously within turn processing. Adding `asyncio` support would improve throughput under concurrent load.
2. **Deterministic Response Caching**: Common static policy inquiries (such as standard return windows) could be served via a lightweight key-value cache to reduce model latency.
3. **Multi-lingual Embeddings**: Vector search is currently optimized for English using `all-MiniLM-L6-v2`. Switching to a multilingual embedding model would allow native support for international customers.

---

## AI Tool Reflection

This project was developed with the assistance of AI pair programming tools for code generation, test scaffolding, and debugging.

- **Usage**: Scaffolding test suites, drafting initial ChromaDB indexing logic, and building deterministic assertion checks.
- **Example of an incomplete AI suggestion**: The AI tool initially suggested simple keyword matching (`"wash" in text`) for conflict detection. This caused false-positive conflict warnings on standard return inquiries because `01-returns-policy-current.md` contained the word `"unwashed"`. The heuristic was replaced with scoped topic matching that verifies both documents discuss drinkware care before raising a conflict.

---

## Demo Walkthrough

Demo video / GIF demonstrating:
1. Knowledge-base question with citations
2. Order lookup tool in action
3. Multi-turn conversation handling
4. Safe abstention and human escalation recommendation
5. Automated evaluation suite execution (20/20 PASS)
 Demo videos: https://drive.google.com/drive/folders/1MQJDj9NbUAhH48nCtFFjz5qsqs3Tn1i0?usp=sharing
`demo.mp4` / `demo.gif`
