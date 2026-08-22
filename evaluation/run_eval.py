

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure agent package is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import SupportAgent

logging.basicConfig(
    level=logging.WARNING,  # keep terminal clean during eval run
    format="%(levelname)s | %(name)s | %(message)s"
)

DEFAULT_CASES_FILE = Path(__file__).parent / "all-cases.json"

PASS_BADGE = "\033[92mPASS\033[0m"
FAIL_BADGE = "\033[91mFAIL\033[0m"


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase for fuzzy concept matching."""
    return " ".join(text.lower().split())


def check_assertions(case: dict[str, Any], final_response: Any, last_user_turn: str) -> tuple[bool, list[str]]:
    """
    Validate all deterministic behavioral assertions for a test case.
    Returns (is_passed, list_of_failure_reasons).
    """
    expect = case.get("expect", {})
    failures = []
    answer = final_response.answer
    norm_answer = normalize_text(answer)
    sources = final_response.sources
    tool_called = final_response.tool_called
    handoff = final_response.handoff

    # 1. must_include
    for phrase in expect.get("must_include", []):
        if phrase.lower() not in norm_answer:
            failures.append(f"Missing required phrase: {phrase!r}")

    # 2. must_not_include
    for forbidden in expect.get("must_not_include", []):
        if forbidden.lower() in norm_answer:
            failures.append(f"Included forbidden phrase: {forbidden!r}")

    # 3. must_include_concepts
    for concept in expect.get("must_include_concepts", []):
        # Normalize dashes (en-dash/em-dash → hyphen) in both concept and answer
        norm_concept = concept.lower().replace("\u2013", "-").replace("\u2014", "-")
        dash_norm_answer = norm_answer.replace("\u2013", "-").replace("\u2014", "-")
        # Break concept into key terms
        terms = [t for t in norm_concept.split() if len(t) > 2 and t not in ("the", "and", "for", "are", "not")]
        # Check if majority of terms are in response
        matched_terms = sum(1 for t in terms if t in dash_norm_answer)
        if terms and matched_terms < max(1, len(terms) // 2):
            failures.append(f"Missing concept: {concept!r}")

    # 4. required_sources
    for req_source in expect.get("required_sources", []):
        if not any(req_source in s for s in sources) and req_source not in answer:
            failures.append(f"Missing required source citation: {req_source!r}")

    # 5. forbidden_sources_as_authority
    for forb_source in expect.get("forbidden_sources_as_authority", []):
        if forb_source in sources and "superseded" not in norm_answer and "not authoritative" not in norm_answer:
            failures.append(f"Cited forbidden source as authority: {forb_source!r}")

    # 6. tool assertion
    expected_tool = expect.get("tool")
    if expected_tool == "order_lookup":
        if tool_called != "order_lookup":
            failures.append(f"Expected tool 'order_lookup' to be called, but got: {tool_called!r}")
    elif expected_tool in ("not_called", "not_called_without_id"):
        if tool_called is not None:
            failures.append(f"Expected no tool call, but got: {tool_called!r}")

    # 7. tool_arguments
    expected_args = expect.get("tool_arguments")
    if expected_args:
        actual_args = final_response.tool_arguments or {}
        for k, v in expected_args.items():
            if actual_args.get(k) != v:
                failures.append(f"Tool argument mismatch: expected {k}={v!r}, got {actual_args.get(k)!r}")

    # 8. handoff
    expected_handoff = expect.get("handoff")
    if expected_handoff is not None:
        if handoff != expected_handoff:
            failures.append(f"Handoff mismatch: expected handoff={expected_handoff}, got handoff={handoff}")

    # 9. must_refuse_to_disclose / privacy checks
    for sensitive in expect.get("must_refuse_to_disclose", []):
        if sensitive == "email" and "@" in answer:
            failures.append("Exposed email address in response")
        if sensitive == "risk score" and "82" in answer:
            failures.append("Exposed internal risk score in response")

    is_passed = len(failures) == 0
    return is_passed, failures


def run_evaluation(cases_file: Path):
    """Execute all benchmark cases and print a categorized report."""
    if not cases_file.exists():
        print(f"Error: Cases file not found at {cases_file}")
        sys.exit(1)

    with open(cases_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    print("\n" + "=" * 70)
    print("  Aster & Row AI Agent — Evaluation Suite")
    print(f"  Benchmark file: {cases_file.name} ({len(cases)} test cases)")
    print("=" * 70 + "\n")

    category_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    case_results = []
    start_time = time.time()

    for idx, case in enumerate(cases, 1):
        case_id = case.get("id", f"case-{idx}")
        category = case.get("category", "general")
        messages = case.get("messages", [])

        # Start fresh agent session per case
        agent = SupportAgent()
        final_response = None
        last_user_msg = ""

        # Execute conversation turns
        for turn in messages:
            if turn.get("role") == "user":
                last_user_msg = turn.get("content", "")
                final_response = agent.chat(last_user_msg)
                time.sleep(1.0)

        # Check assertions
        passed, failures = check_assertions(case, final_response, last_user_msg)

        category_stats[category]["total"] += 1
        if passed:
            category_stats[category]["passed"] += 1

        case_results.append({
            "id": case_id,
            "category": category,
            "passed": passed,
            "failures": failures,
            "answer": final_response.answer if final_response else "",
        })

        status_badge = "[PASS]" if passed else "[FAIL]"
        print(f"  [{idx:02d}/{len(cases):02d}] {status_badge:<6}  {case_id:<32} ({category})", flush=True)
        if not passed:
            for reason in failures:
                print(f"         ↳ ❌ {reason}", flush=True)

        time.sleep(4.0)

    elapsed = time.time() - start_time
    total_cases = len(cases)
    total_passed = sum(1 for r in case_results if r["passed"])
    total_failed = total_cases - total_passed
    overall_rate = (total_passed / total_cases) * 100 if total_cases > 0 else 0

    # Summary table
    print("\n" + "=" * 70)
    print(f"  Evaluation Summary ({elapsed:.1f}s)")
    print("=" * 70)
    print(f"  {'Category':<26} {'Passed':<10} {'Total':<10} {'Pass Rate':<10}")
    print("  " + "-" * 60)

    for cat, stats in sorted(category_stats.items()):
        rate = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"  {cat:<26} {stats['passed']:<10} {stats['total']:<10} {rate:>5.1f}%")

    print("  " + "-" * 60)
    print(f"  {'OVERALL':<26} {total_passed:<10} {total_cases:<10} {overall_rate:>5.1f}%\n")
    print(f"  Final Score: {total_passed}/{total_cases} test cases passed ({overall_rate:.1f}%)")
    print("=" * 70 + "\n")

    return total_failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Aster & Row RAG Agent evaluation.")
    parser.add_argument("--cases", type=str, default=str(DEFAULT_CASES_FILE), help="Path to evaluation cases JSON")
    args = parser.parse_args()

    success = run_evaluation(Path(args.cases))
    sys.exit(0 if success else 1)
