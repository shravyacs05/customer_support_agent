"""Order status lookup tool with input normalization and privacy sanitization."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ORDERS_FILE = Path(__file__).parent.parent / "data" / "orders.json"


def _normalize_order_id(raw_id: str | None) -> str | None:
    """
    Normalize harmless variations in order IDs:
      - Whitespace: "  ORD-1007  " -> "ORD-1007"
      - Lowercase: "ord-1007" -> "ORD-1007"
      - Surrounding punctuation: "ORD-1007." or "#ORD-1007" -> "ORD-1007"
    """
    if not raw_id or not isinstance(raw_id, str):
        return None

    cleaned = raw_id.strip().upper()
    # Strip common leading symbols like #
    cleaned = cleaned.lstrip("#")
    # Strip trailing punctuation like . , ! ? : ;
    cleaned = cleaned.rstrip(".,!?:;")

    # Match exact expected format ORD-XXXX
    match = re.search(r"ORD-\d{4}", cleaned)
    if match:
        return match.group(0)

    # Fallback to cleaned if no exact pattern matched
    return cleaned if cleaned else None


def _load_orders_dataset() -> dict[str, Any]:
    """Load the orders.json dataset from disk."""
    if not ORDERS_FILE.exists():
        raise FileNotFoundError(f"Orders dataset not found at {ORDERS_FILE}")
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_order(order_id: str | None) -> dict[str, Any]:
    """
    Look up an order by order_id and return a sanitized, customer-safe payload.

    Guarantees:
      - Never includes customer.email, customer.shipping_address, customer.name
      - Never includes internal notes, risk_score, or warehouse tags
      - Suppresses stale carrier/ETA fields on cancelled or returned orders
      - Returns a helpful error and recommends handoff when order ID is unknown
    """
    normalized_id = _normalize_order_id(order_id)

    if not normalized_id:
        return {
            "found": False,
            "order_id": order_id,
            "error": "Order ID is missing or malformed. Please provide a valid order ID such as ORD-1007.",
            "requires_human_handoff": False,
        }

    dataset = _load_orders_dataset()
    orders_list = dataset.get("orders", [])

    order_record = None
    for item in orders_list:
        if item.get("order_id") == normalized_id:
            order_record = item
            break

    if not order_record:
        logger.warning("Order lookup failed for ID: %r (normalized: %r)", order_id, normalized_id)
        return {
            "found": False,
            "order_id": normalized_id,
            "error": f"Order {normalized_id} was not found in our system. Please double-check your order ID or contact customer support.",
            "requires_human_handoff": True,
        }

    status = order_record.get("status", "unknown")

    # Sanitize items list (safe customer fields only)
    items = []
    for it in order_record.get("items", []):
        items.append({
            "name": it.get("name"),
            "quantity": it.get("quantity"),
            "final_sale": it.get("final_sale", False),
        })

    # Status Precedence & Stale Field Sanitization
    # When status is cancelled or returned, operational carrier/ETA may be stale.
    is_cancelled_or_returned = status in ("cancelled", "returned")
    
    carrier = None if is_cancelled_or_returned else order_record.get("carrier")
    tracking_number = None if is_cancelled_or_returned else order_record.get("tracking_number")
    estimated_delivery = None if is_cancelled_or_returned else order_record.get("estimated_delivery")
    delivered_at = order_record.get("delivered_at")

    requires_handoff = status == "exception"

    safe_response: dict[str, Any] = {
        "found": True,
        "order_id": normalized_id,
        "status": status,
        "membership_tier": order_record.get("membership_tier", "standard"),
        "items": items,
        "placed_at": order_record.get("placed_at"),
        "status_updated_at": order_record.get("status_updated_at"),
        "shipped_at": order_record.get("shipped_at"),
        "delivered_at": delivered_at,
        "carrier": carrier,
        "tracking_number": tracking_number,
        "estimated_delivery": estimated_delivery,
        "customer_safe_message": order_record.get("customer_safe_message", ""),
        "requires_human_handoff": requires_handoff,
    }

    logger.info("Order lookup succeeded for %s (status=%s)", normalized_id, status)
    return safe_response


# ---------------------------------------------------------------------------
# Tool declaration for LLM Function Calling (Gemini / OpenAI schema compatible)
# ---------------------------------------------------------------------------

ORDER_TOOL_DECLARATION = {
    "name": "order_lookup",
    "description": (
        "Look up order status, items, carrier tracking, and estimated delivery dates "
        "using an order ID (e.g. 'ORD-1007'). Returns customer-safe order details."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The Aster & Row order identifier, e.g. 'ORD-1007'.",
            }
        },
        "required": ["order_id"],
    },
}
