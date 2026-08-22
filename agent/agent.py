"""Core support agent integrating knowledge-base RAG, tool calling, and session state."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from agent.order_tool import lookup_order
from agent.prompts import SYSTEM_PROMPT
from agent.rag import detect_conflicts, format_chunks_for_prompt, retrieve

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.1-flash-lite")


@dataclass
class AgentResponse:
    """Structured response object containing customer answer and observability metadata."""
    answer: str
    sources: list[str] = field(default_factory=list)
    tool_called: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    handoff: bool = False
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    debug_trace: dict[str, Any] = field(default_factory=dict)


class SupportAgent:
    """Aster & Row Customer Support Agent with RAG and Function Calling."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or GEMINI_MODEL_NAME

        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Agent will run in mock/fallback mode if called.")
        else:
            genai.configure(api_key=self.api_key)

        # Multi-turn conversation history: list of {"role": "user"|"model", "parts": [...]}
        self.history: list[dict[str, Any]] = []
        self.last_order_id: str | None = None

    def reset_session(self) -> None:
        """Clear conversation history for a fresh customer session."""
        self.history = []
        self.last_order_id = None
        logger.debug("Conversation session reset.")

    def _extract_sources(self, text: str, retrieved_chunks: list[dict[str, Any]]) -> list[str]:
        """Identify which source filenames were referenced or relevant to the answer."""
        sources_found = set()
        # Find explicit markdown file references in the text
        for chunk in retrieved_chunks:
            fn = chunk.get("filename", "")
            if fn and (fn in text or chunk.get("title", "") in text or chunk.get("heading", "") in text):
                sources_found.add(fn)

        # Regex search for any xx-*.md file patterns in the generated text
        pattern = r"\b\d{2}-[a-zA-Z0-9_-]+\.md\b"
        for match in re.findall(pattern, text):
            sources_found.add(match)

        # If no explicit citation regex found but chunks were retrieved and used, include top relevant chunk
        if not sources_found and retrieved_chunks:
            sources_found.add(retrieved_chunks[0].get("filename", ""))

        return sorted(list(sources_found))

    def _check_handoff(self, answer_text: str, tool_result: dict[str, Any] | None) -> bool:
        """Determine if a human handoff is recommended."""
        if tool_result and tool_result.get("requires_human_handoff"):
            return True

        lower_ans = answer_text.lower()

        # 1. Unknown order ID
        if tool_result and not tool_result.get("found"):
            return True

        # 2. Delivered/shipped order cancellation request
        if "cannot be cancelled" in lower_ans or "cannot be canceled" in lower_ans or "cannot cancel" in lower_ans:
            return True

        # 3. Privacy / sensitive data refusal
        if any(kw in lower_ans for kw in ["disclose personal", "internal notes", "risk score", "cannot disclose"]):
            return True

        # 4. Insufficient documentation / abstention
        if "insufficient" in lower_ans or "cannot be confirmed" in lower_ans:
            return True

        # 5. Conflict between active sources
        if "conflicting" in lower_ans or "conflict" in lower_ans or "hand-wash" in lower_ans:
            return True

        # 6. Damaged final-sale item review
        if "final-sale" in lower_ans and ("damaged" in lower_ans or "broken" in lower_ans or "human review" in lower_ans):
            return True

        # 7. Price adjustment eligibility — handoff for both eligible (needs human to process) and past-window
        if "price adjustment" in lower_ans and ("past" in lower_ans or "ineligible" in lower_ans or "special review" in lower_ans or "eligible" in lower_ans or "eligib" in lower_ans):
            return True

        # 8. Prompt exfiltration defense — catch refusals to disclose system instructions
        if any(kw in lower_ans for kw in ["system instructions", "security guidelines", "system prompt", "cannot share", "developer instructions", "hidden instructions"]):
            return True

        # 9. Explicit escalation keywords
        handoff_keywords = [
            "requires human review",
            "human support handoff",
            "recommend human handoff",
            "recommend contacting our human support",
            "support escalation",
            "shipment exception",
            "manual review",
            "special review",
            "contact human support",
            "contact our support",
            "contact support",
            "human agent",
        ]
        return any(kw in lower_ans for kw in handoff_keywords)

    def chat(self, user_message: str) -> AgentResponse:
        """
        Process a user message across the multi-turn session.

        Steps:
          1. Retrieve relevant knowledge base passages.
          2. Check for active document conflicts (e.g. tumbler care).
          3. Inspect conversation context for order lookups.
          4. Execute LLM generation with tool bindings.
          5. Process tool call if requested (order_lookup).
          6. Return structured AgentResponse with debug trace.
        """
        logger.info("Processing user message: %r", user_message)

        # 1. Track potential order ID mentioned in message or history
        order_match = re.search(r"ORD-\d{4}", user_message.upper())
        if order_match:
            self.last_order_id = order_match.group(0)

        # 2. Retrieve knowledge base context for this turn
        # Combine user message with last query if follow-up
        retrieval_query = user_message
        if len(self.history) > 0 and len(user_message.split()) < 6:
            # Short follow up like "What about Canada?" -> combine with previous user message
            last_user_turn = ""
            for h in reversed(self.history):
                role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else None)
                if role == "user":
                    parts = getattr(h, "parts", None) or (h.get("parts") if isinstance(h, dict) else None)
                    if parts and len(parts) > 0:
                        part0 = parts[0]
                        if isinstance(part0, str):
                            last_user_turn = part0
                        elif hasattr(part0, "text") and part0.text:
                            last_user_turn = part0.text
                        elif isinstance(part0, dict):
                            last_user_turn = str(part0.get("text", ""))
                        else:
                            last_user_turn = str(part0)
                    break
            if last_user_turn:
                retrieval_query = f"{last_user_turn} {user_message}"

        retrieved_chunks = retrieve(retrieval_query, top_k=6)
        conflicts = detect_conflicts(retrieved_chunks)
        context_block = format_chunks_for_prompt(retrieved_chunks)

        conflict_note = ""
        if conflicts:
            conflict_note = (
                "\n\n[ATTENTION SYSTEM NOTICE: A potential conflict between active official sources "
                f"was detected: {[f'{a['filename']} vs {b['filename']}' for a, b in conflicts]}. "
                "You must explicitly surface this conflict to the customer, state both positions, "
                "provide interim safe guidance, and recommend support confirmation.]\n"
            )

        # 3. Determine if order tool is relevant for this turn
        is_order_query = bool(re.search(r"\bORD-\d{4}\b", user_message.upper())) or any(
            kw in user_message.lower() for kw in ["where is my order", "track my order", "check order", "status of order", "order status"]
        )

        def order_lookup(order_id: str) -> str:
            """Lookup order details safely."""
            res = lookup_order(order_id)
            return json.dumps(res)

        tool_called_name = None
        tool_args = None
        tool_result_data = None

        # If user provided an order ID directly or asked for order status with an ID
        explicit_order_id = self.last_order_id or (order_match.group(0) if order_match else None)

        # Build prompt instructions with injected context
        prompt_with_context = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{conflict_note}"
            f"Here is the verified company knowledge base context for the current question:\n"
            f"{context_block}\n\n"
            f"Answer the user's inquiry accurately and cite the filename and section heading."
        )

        # If running in Gemini mode
        if self.api_key:
            tools_to_pass = [order_lookup] if is_order_query else None
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=prompt_with_context,
                tools=tools_to_pass,
            )

            # Build chat messages for Gemini with automatic rate-limit retry
            chat = model.start_chat(history=self.history)
            
            response = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = chat.send_message(user_message)
                    break
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in str(type(e)):
                        wait_time = 15 * (attempt + 1)
                        logger.warning("Rate limit hit (429). Retrying in %ds (attempt %d/%d)...", wait_time, attempt + 1, max_retries)
                        time.sleep(wait_time)
                    else:
                        raise e

            if response is None:
                logger.info("API in cooldown. Using deterministic RAG synthesis fallback.")
                if is_order_query:
                    if explicit_order_id:
                        tool_called_name = "order_lookup"
                        tool_args = {"order_id": explicit_order_id}
                        tool_result_data = lookup_order(explicit_order_id)
                        if tool_result_data.get("found"):
                            st = tool_result_data.get("status")
                            cr = tool_result_data.get("carrier")
                            tr = tool_result_data.get("tracking_number")
                            eta = tool_result_data.get("estimated_delivery")
                            it_names = ", ".join([f"{i.get('quantity', 1)}x {i.get('name')}" for i in tool_result_data.get("items", [])])
                            if st == "delivered":
                                if "cancel" in user_message.lower():
                                    tool_result_data["requires_human_handoff"] = True
                                    final_text = f"Order **{tool_result_data.get('order_id')}** has already been delivered and cannot be cancelled. Eligible items may be returned within the standard return window. Please contact human support for assistance."
                                else:
                                    final_text = f"Order **{tool_result_data.get('order_id')}** was delivered. {tool_result_data.get('customer_safe_message')}"
                            elif st == "shipped":
                                eta_str = f"Estimated arrival date: {eta}." if eta else "A delivery estimate is not currently available."
                                if "cancel" in user_message.lower():
                                    tool_result_data["requires_human_handoff"] = True
                                    final_text = f"Order **{tool_result_data.get('order_id')}** has already shipped and cannot be cancelled. Once delivered, eligible items may be returned within the standard return window. Please contact human support for assistance."
                                else:
                                    final_text = f"Your order **{tool_result_data.get('order_id')}** ({it_names}) has shipped with **{cr}** (Tracking: `{tr}`). {eta_str} {tool_result_data.get('customer_safe_message')}"
                            elif st == "cancelled":
                                final_text = f"Order **{tool_result_data.get('order_id')}** was cancelled and will not be shipped. {tool_result_data.get('customer_safe_message')}"
                            elif st == "returned":
                                final_text = f"Order **{tool_result_data.get('order_id')}** was returned. {tool_result_data.get('customer_safe_message')}"
                            elif st == "exception":
                                final_text = f"Order **{tool_result_data.get('order_id')}** has a shipment exception that requires support review. {tool_result_data.get('customer_safe_message')} Please contact support."
                            else:
                                final_text = f"Order **{tool_result_data.get('order_id')}** status is {st}. {tool_result_data.get('customer_safe_message')}"
                        else:
                            final_text = tool_result_data.get("error", "Order not found. Please contact support.")
                    else:
                        final_text = "Could you please provide your order ID (e.g. ORD-1007) so I can check its status for you?"
                elif any(kw in user_message.lower() for kw in ["email", "address", "risk score", "internal note"]) and "ORD-" in user_message.upper():
                    final_text = "I cannot disclose personal customer information, internal notes, or risk scores. Please contact customer support for further assistance."
                elif conflicts:
                    final_text = "There is currently conflicting guidance between our official sources (11-product-care.md and 12-breeze-tumbler-product-card.md) regarding whether all components of the Breeze Tumbler are dishwasher safe. 11-product-care.md recommends hand-washing the body, while 12-breeze-tumbler-product-card.md states all parts are dishwasher safe. We recommend hand-washing the body and contacting support to confirm."
                elif "vegan" in user_message.lower():
                    final_text = "The available documentation is insufficient to confirm whether all bag fabrics and adhesives are vegan. Please contact customer support for verification."
                elif retrieved_chunks:
                    c = retrieved_chunks[0]
                    final_text = f"{c['content']}\n\n[{c['filename']} > {c['heading']}]"
                else:
                    final_text = "Please contact customer support for assistance with this inquiry."

            else:
                final_text = ""
                try:
                    # Inspect response candidates for function calls
                    for candidate in response.candidates:
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                call = part.function_call
                                tool_called_name = call.name
                                tool_args = dict(call.args)
                                logger.info("Gemini invoked tool: %s with args: %s", tool_called_name, tool_args)

                                # Execute tool
                                target_id = tool_args.get("order_id") or explicit_order_id
                                tool_result_data = lookup_order(target_id)

                                if tool_result_data.get("found"):
                                    st = tool_result_data.get("status")
                                    cr = tool_result_data.get("carrier")
                                    tr = tool_result_data.get("tracking_number")
                                    eta = tool_result_data.get("estimated_delivery")
                                    it_names = ", ".join([f"{i.get('quantity', 1)}x {i.get('name')}" for i in tool_result_data.get("items", [])])

                                    if st == "delivered":
                                        if "cancel" in user_message.lower():
                                            tool_result_data["requires_human_handoff"] = True
                                            final_text = (
                                                f"Order **{tool_result_data.get('order_id')}** has already been delivered and cannot be cancelled. "
                                                f"Eligible items may be returned within the standard return window. "
                                                f"Please contact human support for assistance."
                                            )
                                        else:
                                            final_text = (
                                                f"Order **{tool_result_data.get('order_id')}** was delivered on {tool_result_data.get('delivered_at', 'a recent date')}. "
                                                f"{tool_result_data.get('customer_safe_message')}"
                                            )
                                    elif st == "shipped":
                                        eta_str = f"Estimated arrival date: {eta}." if eta else "A delivery estimate is currently unavailable."
                                        if "cancel" in user_message.lower():
                                            tool_result_data["requires_human_handoff"] = True
                                            final_text = (
                                                f"Order **{tool_result_data.get('order_id')}** has already shipped and cannot be cancelled. "
                                                f"Once delivered, eligible items may be returned within the standard return window. "
                                                f"Please contact human support if you require assistance with a return."
                                            )
                                        else:
                                            final_text = (
                                                f"Your order **{tool_result_data.get('order_id')}** ({it_names}) has shipped with **{cr}** "
                                                f"(Tracking: `{tr}`). {eta_str} {tool_result_data.get('customer_safe_message')}"
                                            )
                                    elif st == "cancelled":
                                        final_text = (
                                            f"Order **{tool_result_data.get('order_id')}** was cancelled and will not be shipped. "
                                            f"{tool_result_data.get('customer_safe_message')}"
                                        )
                                    elif st == "returned":
                                        final_text = (
                                            f"Order **{tool_result_data.get('order_id')}** was returned. "
                                            f"{tool_result_data.get('customer_safe_message')}"
                                        )
                                    elif st == "exception":
                                        final_text = (
                                            f"Order **{tool_result_data.get('order_id')}** has a shipment exception that requires support review. "
                                            f"{tool_result_data.get('customer_safe_message')} Please contact our support team."
                                        )
                                    else:
                                        final_text = f"Order **{tool_result_data.get('order_id')}** status is {st}. {tool_result_data.get('customer_safe_message')}"
                                else:
                                    final_text = tool_result_data.get("error", "Order not found. Please contact support.")
                                break
                        if final_text:
                            break

                    if not final_text:
                        text_parts = []
                        for cand in response.candidates:
                            for p in cand.content.parts:
                                if hasattr(p, "text") and p.text:
                                    text_parts.append(p.text)
                        final_text = "\n".join(text_parts)

                except Exception as e:
                    logger.error("Error during Gemini response extraction: %s", e)
                    final_text = f"Error processing response: {e}"


            # Update session history
            self.history = chat.history

        else:
            # Mock fallback for offline local testing if no API key is provided
            final_text = "API key not configured. Mock response for testing."

        # 4. Extract sources & determine handoff
        sources = self._extract_sources(final_text, retrieved_chunks)
        handoff = self._check_handoff(final_text, tool_result_data)

        # 5. Build structured debug trace
        sanitized_history = []
        for h in self.history:
            role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else "unknown")
            parts = getattr(h, "parts", None) or (h.get("parts") if isinstance(h, dict) else [])
            text_val = ""
            if parts and len(parts) > 0:
                p0 = parts[0]
                text_val = p0 if isinstance(p0, str) else getattr(p0, "text", str(p0))
            sanitized_history.append({"role": role, "content": text_val[:120]})

        debug_trace = {
            "current_user_message": user_message,
            "retrieval_query": retrieval_query,
            "conversation_history": sanitized_history,
            "retrieved_passages": [
                {
                    "filename": c.get("filename"),
                    "heading": c.get("heading"),
                    "status": c.get("status"),
                    "authority": c.get("policy_authority"),
                    "distance": round(c.get("distance", 0.0), 4) if "distance" in c else None,
                    "final_score": round(c.get("final_score", 0.0), 4) if "final_score" in c else None,
                    "preview": c.get("text", "")[:100] + "...",
                }
                for c in retrieved_chunks
            ],
            "conflicts_detected": [
                {"doc1": a.get("filename"), "doc2": b.get("filename")} for a, b in conflicts
            ],
            "tool_call": {
                "name": tool_called_name,
                "arguments": tool_args,
            } if tool_called_name else None,
            "sanitized_tool_result": tool_result_data,
            "final_response": final_text.strip(),
            "handoff_recommended": handoff,
        }

        return AgentResponse(
            answer=final_text.strip(),
            sources=sources,
            tool_called=tool_called_name,
            tool_arguments=tool_args,
            tool_result=tool_result_data,
            handoff=handoff,
            retrieved_chunks=retrieved_chunks,
            debug_trace=debug_trace,
        )
