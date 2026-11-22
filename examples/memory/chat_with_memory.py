"""
Exercise: an interactive chat whose memory survives between executions.

Tell the agent your name, quit with Ctrl+C, run the script again and ask for your name.
"""

import asyncio

from agents import Agent, Runner, SQLiteSession

DB_PATH = "conversations.db"


async def main():
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
    )
    session = SQLiteSession("my_chat", DB_PATH)

    previous_items = await session.get_items()
    print(f"Loaded {len(previous_items)} items from previous executions. Ctrl+C to quit.\n")

    while True:
        msg = input("You: ")
        result = await Runner.run(agent, msg, session=session)
        print(f"Agent: {result.final_output}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
