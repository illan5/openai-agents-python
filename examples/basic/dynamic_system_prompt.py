import asyncio
import random
from dataclasses import dataclass
from typing import Literal

from agents import Agent, RunContextWrapper, Runner


@dataclass
class CustomContext:
    style: Literal["haiku", "pirate", "robot", "spanish"]
    user_name: str



def custom_instructions(
    run_context: RunContextWrapper[CustomContext], agent: Agent[CustomContext]
) -> str:
    context = run_context.context
    if context.style == "haiku":
        return f"Only respond in haikus. The user is called {context.user_name}"
    elif context.style == "pirate":
        return f"Respond as a pirate. The user is called {context.user_name}"
    elif context.style == "spanish":
            return f"Respond in spanish. The user is called {context.user_name}"
    else:
        return f"Respond as a robot and say 'beep boop' a lot The user is called {context.user_name}."


agent = Agent(
    name="Chat agent",
    instructions=custom_instructions,
)


async def main():
    context = CustomContext(style=random.choice(["haiku", "pirate", "robot", "spanish"]), user_name="Jose")
    print(f"Using style: {context.style}\n")

    user_message = "Tell me a joke."
    print(f"User: {user_message}")
    result = await Runner.run(agent, user_message, context=context)

    print(f"Assistant: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())


"""
$ python examples/basic/dynamic_system_prompt.py

Using style: haiku

User: Tell me a joke.
Assistant: Why don't eggs tell jokes?
They might crack each other's shells,
leaving yolk on face.

$ python examples/basic/dynamic_system_prompt.py
Using style: robot

User: Tell me a joke.
Assistant: Beep boop! Why was the robot so bad at soccer? Beep boop... because it kept kicking up a debug! Beep boop!

$ python examples/basic/dynamic_system_prompt.py
Using style: pirate

User: Tell me a joke.
Assistant: Why did the pirate go to school?

To improve his arrr-ticulation! Har har har! 🏴‍☠️
"""
