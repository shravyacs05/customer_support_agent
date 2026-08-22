import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from agent.agent import SupportAgent


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent CLI")
    parser.add_argument("--debug", action="store_true", help="Print structured debug trace for each turn")
    args = parser.parse_args()

    debug_mode = args.debug

    print("\n" + "=" * 65)
    print("  Aster & Row Support Assistant (CLI)")
    print("=" * 65)
    print("  Type your question below (e.g. 'What is the return window?')")
    print("  Type 'reset' for a new customer session.")
    print("  Type 'debug' to toggle debug trace mode on/off.")
    print("  Type 'exit' or 'quit' to leave.")
    print("=" * 65 + "\n")

    agent = SupportAgent()

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("\nSession ended. Have a great day.\n")
                break

            if user_input.lower() == "reset":
                agent.reset_session()
                print("\n[Session Reset] Started a new customer session.")
                continue

            if user_input.lower() == "debug":
                debug_mode = not debug_mode
                status = "ENABLED" if debug_mode else "DISABLED"
                print(f"\n[Debug Mode {status}]")
                continue

            print("\nProcessing inquiry...")
            response = agent.chat(user_input)

            # Format answer cleanly for terminal
            clean_answer = response.answer.replace("**", "")

            print("\n" + "-" * 65)
            print(f"Agent:\n{clean_answer}")
            print("-" * 65)

            if response.sources:
                print(f"Sources: {', '.join(response.sources)}")
            if response.tool_called:
                print(f"Tool Used: {response.tool_called} (args: {response.tool_arguments})")
            if response.handoff:
                print("Human Escalation: Recommended")
            print("-" * 65)

            if debug_mode:
                print("\n--- [DEBUG TRACE] ---")
                print(json.dumps(response.debug_trace, indent=2, ensure_ascii=False))
                print("---------------------\n")

        except KeyboardInterrupt:
            print("\n\nSession terminated.")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
