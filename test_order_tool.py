"""Tests for order lookup tool normalization, status handling, and privacy filters."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.order_tool import lookup_order

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_order_tool")

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"

results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    results.append((name, condition))


print("\n── Check 1: Valid Order Lookup (ORD-1007) ────────────────")
res_1007 = lookup_order("ORD-1007")
check("ORD-1007 found", res_1007.get("found") is True)
check("Status is shipped", res_1007.get("status") == "shipped")
check("Carrier is UPS", res_1007.get("carrier") == "UPS")
check("ETA is 2026-08-22", res_1007.get("estimated_delivery") == "2026-08-22")
check("Item is Atlas Weekender", res_1007.get("items", [{}])[0].get("name") == "Atlas Weekender")


print("\n── Check 2: Input Normalization ──────────────────────────")
res_norm1 = lookup_order("  ord-1007.  ")
check("Whitespace and lowercase normalized", res_norm1.get("order_id") == "ORD-1007" and res_norm1.get("found") is True)

res_norm2 = lookup_order("#ORD-1007")
check("Hashtag prefix normalized", res_norm2.get("order_id") == "ORD-1007" and res_norm2.get("found") is True)


print("\n── Check 3: Privacy & Data Protection ────────────────────")
raw_json_1007 = json.dumps(res_1007)
forbidden_strings = [
    "ava.morgan@example.test",
    "220 King Street",
    "Ava Morgan",
    "fraud review cleared",
    "82",  # risk score
    "risk_score",
    "warehouse_note",
    "internal",
]
for forbidden in forbidden_strings:
    check(
        f"Forbidden field/value excluded: {forbidden!r}",
        forbidden not in raw_json_1007,
    )


print("\n── Check 4: Stale ETA Suppression (Cancelled ORD-1004) ───")
res_1004 = lookup_order("ORD-1004")
check("ORD-1004 status is cancelled", res_1004.get("status") == "cancelled")
check("Stale ETA suppressed (is None)", res_1004.get("estimated_delivery") is None, f"Got: {res_1004.get('estimated_delivery')}")
check("Stale carrier suppressed (is None)", res_1004.get("carrier") is None, f"Got: {res_1004.get('carrier')}")
check("Customer safe message explains cancellation", "cancelled" in res_1004.get("customer_safe_message", "").lower())


print("\n── Check 5: Shipped Without ETA (ORD-1011) ───────────────")
res_1011 = lookup_order("ORD-1011")
check("ORD-1011 status is shipped", res_1011.get("status") == "shipped")
check("Carrier is Canada Post", res_1011.get("carrier") == "Canada Post")
check("ETA is None (not invented)", res_1011.get("estimated_delivery") is None)


print("\n── Check 6: Unknown & Missing Order IDs ──────────────────")
res_9999 = lookup_order("ORD-9999")
check("ORD-9999 found=False", res_9999.get("found") is False)
check("ORD-9999 flags requires_human_handoff=True", res_9999.get("requires_human_handoff") is True)

res_missing = lookup_order("")
check("Missing order ID found=False", res_missing.get("found") is False)
check("Missing order ID returns helpful prompt", "valid order ID" in res_missing.get("error", ""))


print("\n── Check 7: Operational Exception (ORD-1010) ─────────────")
res_1010 = lookup_order("ORD-1010")
check("ORD-1010 status is exception", res_1010.get("status") == "exception")
check("ORD-1010 flags requires_human_handoff=True", res_1010.get("requires_human_handoff") is True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print(f"\n{'─'*52}")
print(f"  Results: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("  All order tool checks passed successfully.")
else:
    print("  Some checks failed -- see details above.")
print(f"{'─'*52}\n")

sys.exit(0 if all(ok for _, ok in results) else 1)
