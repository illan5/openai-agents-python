"""
Example demonstrating session memory functionality.

This example shows how to use session memory to maintain conversation history
across multiple agent runs without manually handling .to_input_list().
"""

import asyncio

from agents import Agent, Runner, SQLiteSession

# Exercise: file where the sessions are persisted. It is ignored by git.
DB_PATH = "conversations.db"


async def main():
    # Create an agent
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
    )

    # Create a session instance that will persist across runs.
    # Exercise: store the session in a file so it survives between executions.
    # Run the script twice and the total item count will double.
    session_id = "conversation_123"
    session = SQLiteSession(session_id, DB_PATH)

    print("=== Session Example ===")
    print("The agent will remember previous messages automatically.\n")

    # First turn
    print("First turn:")
    print("User: What city is the Golden Gate Bridge in?")
    result = await Runner.run(
        agent,
        "What city is the Golden Gate Bridge in?",
        session=session,
    )
    print(f"Assistant: {result.final_output}")
    print()

    # Second turn - the agent will remember the previous conversation
    print("Second turn:")
    print("User: What state is it in?")
    result = await Runner.run(agent, "What state is it in?", session=session)
    print(f"Assistant: {result.final_output}")
    print()

    # Third turn - continuing the conversation
    print("Third turn:")
    print("User: What's the population of that state?")
    result = await Runner.run(
        agent,
        "What's the population of that state?",
        session=session,
    )
    print(f"Assistant: {result.final_output}")
    print()

    print("=== Conversation Complete ===")
    print("Notice how the agent remembered the context from previous turns!")
    print("Sessions automatically handles conversation history.")

    # Demonstrate the limit parameter - get only the latest 2 items
    print("\n=== Latest Items Demo ===")
    latest_items = await session.get_items(limit=2)
    print("Latest 2 items:")
    for i, msg in enumerate(latest_items, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"  {i}. {role}: {content}")

    print(f"\nFetched {len(latest_items)} out of total conversation history.")

    # Get all items to show the difference
    all_items = await session.get_items()
    print(f"Total items in session: {len(all_items)}")

    # Exercise: undo the last question and everything after it with pop_item().
    # The turn can also contain reasoning or tool items, so pop until the user message is removed.
    print("\n=== Undo Demo ===")
    while (item := await session.pop_item()) is not None:
        print(f"Removed: {item.get('role') or item.get('type')}")
        if item.get("role") == "user":
            break
    result = await Runner.run(agent, "What was my last question?", session=session)
    print(f"Assistant (after undo): {result.final_output}")

    # Exercise: two sessions with different IDs do not share history.
    print("\n=== Separate Sessions Demo ===")
    session_a = SQLiteSession("user_a", DB_PATH)
    session_b = SQLiteSession("user_b", DB_PATH)
    await session_a.clear_session()
    await session_b.clear_session()

    await Runner.run(agent, "My name is Jose.", session=session_a)
    result_a = await Runner.run(agent, "What's my name?", session=session_a)
    result_b = await Runner.run(agent, "What's my name?", session=session_b)
    print(f"user_a -> {result_a.final_output}")
    print(f"user_b -> {result_b.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
