"""
Example demonstrating OpenAI responses.compact session functionality.

This example shows how to use OpenAIResponsesCompactionSession to automatically
compact conversation history when it grows too large, reducing token usage
while preserving context.

Exercise additions: the same conversation also runs on a plain session so you can
compare the item count and the input tokens sent per turn, and the compaction
threshold is a constant you can tweak.
"""

import asyncio

from agents import Agent, OpenAIResponsesCompactionSession, Runner, SQLiteSession
from agents.items import TResponseInputItem
from agents.memory import Session

# Exercise: change this value (for example to 2 or 8) to see when compaction triggers.
COMPACTION_THRESHOLD = 4

PROMPTS = [
    "What is the tallest mountain in the world?",
    "How tall is it in feet?",
    "When was it first climbed?",
    "Who was on that expedition?",
    "What country is the mountain in?",
]


def print_items(title: str, items: list[TResponseInputItem]) -> None:
    print(f"=== {title} ===")
    print(f"Total items: {len(items)}")
    for item in items:
        # Some inputs are stored as easy messages (only `role` and `content`).
        item_type = item.get("type") or ("message" if "role" in item else "unknown")
        if item_type == "compaction":
            print("  - compaction (encrypted content)")
        elif item_type == "message":
            role = item.get("role", "unknown")
            print(f"  - message ({role})")
        else:
            print(f"  - {item_type}")
    print()


async def run_conversation(agent: Agent, session: Session, label: str) -> list[int]:
    """Run all prompts on the session and return the input tokens used by each turn."""
    print(f"=== Conversation ({label}) ===\n")
    input_tokens_per_turn = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"Turn {i}:")
        print(f"User: {prompt}")
        result = await Runner.run(agent, prompt, session=session)
        # Exercise: the usage of a turn includes the compaction request when it runs.
        input_tokens = result.context_wrapper.usage.input_tokens
        input_tokens_per_turn.append(input_tokens)
        print(f"Assistant: {result.final_output}")
        print(f"[input tokens this turn: {input_tokens}]\n")
    return input_tokens_per_turn


async def main():
    agent = Agent(
        name="Assistant",
        instructions="Reply concisely. Keep answers to 1-2 sentences.",
    )

    print("=== Compaction Session Example ===\n")

    # Create an underlying session for storage.
    underlying = SQLiteSession(":memory:")

    # Wrap with compaction session - will automatically compact when threshold hit.
    session = OpenAIResponsesCompactionSession(
        session_id="demo-session",
        underlying_session=underlying,
        # Custom compaction trigger (default is 10 candidates).
        should_trigger_compaction=lambda ctx: (
            len(ctx["compaction_candidate_items"]) >= COMPACTION_THRESHOLD
        ),
    )

    compacted_tokens = await run_conversation(agent, session, "with compaction")

    # Show session state after automatic compaction (if triggered).
    print_items("Session State (Auto Compaction)", await session.get_items())

    # Manual compaction after inspecting the auto-compacted state.
    print("=== Manual Compaction ===")
    await session.run_compaction({"force": True})
    print("Done\n")

    # Show final session state after manual compaction.
    print_items("Session State (Manual Compaction)", await session.get_items())

    # Exercise: run the same conversation on a plain session without compaction.
    plain_session = SQLiteSession("plain-session")
    plain_tokens = await run_conversation(agent, plain_session, "without compaction")
    print_items("Session State (No Compaction)", await plain_session.get_items())

    # Exercise: compare the input tokens sent on each turn.
    print("=== Input Tokens per Turn ===")
    print(f"{'Turn':<6}{'With compaction':>18}{'Without compaction':>21}")
    for i, (with_c, without_c) in enumerate(zip(compacted_tokens, plain_tokens, strict=True), 1):
        print(f"{i:<6}{with_c:>18}{without_c:>21}")
    print(f"{'Total':<6}{sum(compacted_tokens):>18}{sum(plain_tokens):>21}")


if __name__ == "__main__":
    asyncio.run(main())
