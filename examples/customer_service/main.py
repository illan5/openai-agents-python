"""Airline customer service with triage, FAQ, and seat booking agents.

Exercise improvements over the original demo:

1. `update_seat` validates the confirmation number and the seat, and asks the customer to
   approve the change before it runs (`needs_approval`).
2. Tools return error messages instead of crashing with `assert`.
3. The conversation history is stored in a persistent SQLite session with compaction.
4. The customer context and the current agent are saved to disk, so restarting the script
   with the same `--conversation-id` continues the conversation.
5. An input guardrail rejects messages unrelated to the airline.
6. Responses are streamed as they are generated.
"""

from __future__ import annotations as _annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OpenAIResponsesCompactionSession,
    RunContextWrapper,
    Runner,
    RunResultStreaming,
    SQLiteSession,
    TResponseInputItem,
    handoff,
    trace,
)
from agents.decorators import input_guardrail, tool
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from examples.auto_mode import confirm_with_fallback, input_with_fallback, is_auto_mode

# Exercise: files where the conversation history and the customer state are persisted.
# Both are ignored by git.
DB_PATH = "customer_service.db"
STATE_PATH = Path("customer_service_state.json")

OFF_TOPIC_REFUSAL = "Sorry, I can only help with questions about your flight and our airline."

### CONTEXT


class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None


### FAKE BOOKING DATABASE

# Exercise: in a real system this would be a database or an API. The plane has rows 1-30
# with seats A-D, which matches the 120 seats described by the FAQ.
BOOKINGS: dict[str, dict[str, str]] = {
    "ABC123": {"passenger_name": "Alex Smith", "flight_number": "FLT-101", "seat": "12A"},
    "XYZ789": {"passenger_name": "Sam Lee", "flight_number": "FLT-202", "seat": "4C"},
}
SEAT_PATTERN = re.compile(r"^([1-9]|[12][0-9]|30)[A-D]$")
TAKEN_SEATS: dict[str, set[str]] = {
    "FLT-101": {"12A", "1A", "1B"},
    "FLT-202": {"4C", "16D"},
}

### TOOLS


@tool(name_override="faq_lookup_tool", description_override="Lookup frequently asked questions.")
async def faq_lookup_tool(question: str) -> str:
    question_lower = question.lower()
    if any(
        keyword in question_lower
        for keyword in ["bag", "baggage", "luggage", "carry-on", "hand luggage", "hand carry"]
    ):
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    elif any(keyword in question_lower for keyword in ["seat", "seats", "seating", "plane"]):
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom. "
        )
    elif any(
        keyword in question_lower
        for keyword in ["wifi", "internet", "wireless", "connectivity", "network", "online"]
    ):
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."


# Exercise: changing a seat has side effects, so the customer must approve each call.
@tool(needs_approval=True)
async def update_seat(
    context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str
) -> str:
    """
    Update the seat for a given confirmation number.

    Args:
        confirmation_number: The confirmation number for the flight.
        new_seat: The new seat to update to, for example 12A.
    """
    # Exercise: validate the arguments chosen by the model instead of trusting them.
    confirmation_number = confirmation_number.strip().upper()
    new_seat = new_seat.strip().upper()

    booking = BOOKINGS.get(confirmation_number)
    if booking is None:
        return f"Error: confirmation number {confirmation_number} was not found."

    # Exercise: once a booking is linked to the conversation, other bookings cannot be changed.
    linked = context.context.confirmation_number
    if linked is not None and linked != confirmation_number:
        return f"Error: this conversation is linked to booking {linked}, not {confirmation_number}."

    if not SEAT_PATTERN.match(new_seat):
        return f"Error: {new_seat} is not a valid seat. Seats go from 1A to 30D."

    flight_number = booking["flight_number"]
    if new_seat in TAKEN_SEATS.get(flight_number, set()):
        return f"Error: seat {new_seat} is already taken on flight {flight_number}."

    # Exercise: return an error the model can explain instead of crashing with an assert.
    if not flight_number:
        return "Error: the flight for this booking could not be found."

    TAKEN_SEATS.setdefault(flight_number, set()).discard(booking["seat"])
    TAKEN_SEATS[flight_number].add(new_seat)
    booking["seat"] = new_seat

    context.context.passenger_name = booking["passenger_name"]
    context.context.confirmation_number = confirmation_number
    context.context.flight_number = flight_number
    context.context.seat_number = new_seat
    return f"Updated seat to {new_seat} on flight {flight_number} for {confirmation_number}."


### HOOKS


async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    # Exercise: load the flight from the booking instead of inventing a random one.
    confirmation_number = context.context.confirmation_number
    if confirmation_number and confirmation_number in BOOKINGS:
        context.context.flight_number = BOOKINGS[confirmation_number]["flight_number"]


### GUARDRAILS


class OffTopicOutput(BaseModel):
    reasoning: str
    is_off_topic: bool


off_topic_guardrail_agent = Agent(
    name="Off-topic check",
    instructions=(
        "You review messages sent to an airline customer service chat. "
        "Judge only the latest user message, using earlier messages as context. "
        "It is on-topic if it is about flights, bookings, seats, baggage, onboard services, "
        "or the airline. Greetings, thanks, confirmation numbers, seat numbers, and short "
        "answers such as 'yes' or 'no' are on-topic. Anything else is off-topic."
    ),
    output_type=OffTopicOutput,
)


# Exercise: run the check before the agent starts, so off-topic messages never reach the agent
# and no partial answer is streamed.
@input_guardrail(run_in_parallel=False)
async def off_topic_guardrail(
    context: RunContextWrapper[AirlineAgentContext],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    result = await Runner.run(off_topic_guardrail_agent, input)
    output = result.final_output_as(OffTopicOutput)
    return GuardrailFunctionOutput(output_info=output, tripwire_triggered=output.is_off_topic)


### AGENTS

faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
)

seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.
    4. If the tool returns an error, explain it to the customer and ask for corrected details.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent. """,
    tools=[update_seat],
)

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        handoff(agent=faq_agent, tool_name_override="transfer_to_faq_agent"),
        handoff(
            agent=seat_booking_agent,
            on_handoff=on_seat_booking_handoff,
            tool_name_override="transfer_to_seat_booking_agent",
        ),
    ],
)

faq_agent.handoffs.append(
    handoff(agent=triage_agent, tool_name_override="transfer_to_triage_agent")
)
seat_booking_agent.handoffs.append(
    handoff(agent=triage_agent, tool_name_override="transfer_to_triage_agent")
)

AGENTS_BY_NAME: dict[str, Agent[AirlineAgentContext]] = {
    agent.name: agent for agent in (triage_agent, faq_agent, seat_booking_agent)
}

# Exercise: input guardrails only run for the first agent of each run. The current agent can be
# any of the three, so all of them get the guardrail.
for _agent in AGENTS_BY_NAME.values():
    _agent.input_guardrails = [off_topic_guardrail]


### PERSISTENCE


def load_state(conversation_id: str) -> tuple[AirlineAgentContext, Agent[AirlineAgentContext]]:
    """Load the saved context and current agent, or start fresh."""
    try:
        saved = json.loads(STATE_PATH.read_text())[conversation_id]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return AirlineAgentContext(), triage_agent
    context = AirlineAgentContext.model_validate(saved["context"])
    agent = AGENTS_BY_NAME.get(saved["agent"], triage_agent)
    return context, agent


def save_state(
    conversation_id: str, context: AirlineAgentContext, agent: Agent[AirlineAgentContext]
) -> None:
    try:
        all_states = json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        all_states = {}
    all_states[conversation_id] = {"context": context.model_dump(), "agent": agent.name}
    STATE_PATH.write_text(json.dumps(all_states, indent=2))


### RUN


async def print_stream(result: RunResultStreaming) -> None:
    """Print text as it streams, plus handoffs and tool activity."""
    line_open = False
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            if not line_open:
                print(f"{result.current_agent.name}: ", end="", flush=True)
                line_open = True
            print(event.data.delta, end="", flush=True)
        elif event.type == "run_item_stream_event":
            item = event.item
            if item.type == "message_output_item" and line_open:
                print()
                line_open = False
            elif item.type == "handoff_output_item":
                print(f"[Handed off from {item.source_agent.name} to {item.target_agent.name}]")
            elif item.type == "tool_call_item":
                tool_name = getattr(item.raw_item, "name", "unknown tool")
                print(f"[{item.agent.name}: calling {tool_name}]")
            elif item.type == "tool_call_output_item":
                print(f"[{item.agent.name}: tool output: {item.output}]")
    if line_open:
        print()


async def run_turn(
    agent: Agent[AirlineAgentContext],
    user_input: str,
    context: AirlineAgentContext,
    session: OpenAIResponsesCompactionSession,
) -> RunResultStreaming:
    """Run one streamed turn, asking the customer to approve tool calls that need it."""
    result = Runner.run_streamed(agent, user_input, context=context, session=session)
    await print_stream(result)

    # Exercise: update_seat pauses the run until the customer approves or rejects it.
    while result.interruptions:
        state = result.to_state()
        for interruption in result.interruptions:
            approved = confirm_with_fallback(
                f"Approve {interruption.name} with {interruption.arguments}? (y/n): ",
                default=True,
            )
            if approved:
                state.approve(interruption)
            else:
                state.reject(interruption)
        result = Runner.run_streamed(agent, state, session=session)
        await print_stream(result)
    return result


async def record_refusal(session: OpenAIResponsesCompactionSession, user_input: str) -> None:
    """Keep the history coherent when the guardrail blocks a message."""
    items = await session.get_items(limit=1)
    last = items[-1] if items else None
    if not (last and last.get("role") == "user" and last.get("content") == user_input):
        await session.add_items([{"role": "user", "content": user_input}])
    await session.add_items([{"role": "assistant", "content": OFF_TOPIC_REFUSAL}])


async def main(conversation_id: str) -> None:
    context, current_agent = load_state(conversation_id)
    session = OpenAIResponsesCompactionSession(
        session_id=conversation_id,
        underlying_session=SQLiteSession(conversation_id, DB_PATH),
    )
    auto_mode = is_auto_mode()

    history = await session.get_items()
    print(
        f"Conversation {conversation_id}: {len(history)} items loaded, "
        f"talking to {current_agent.name}. Ctrl+C to quit.\n"
    )

    while True:
        user_input = input_with_fallback("Enter your message: ", "What are your store hours?")
        with trace("Customer service", group_id=conversation_id):
            try:
                result = await run_turn(current_agent, user_input, context, session)
            except InputGuardrailTripwireTriggered:
                print(OFF_TOPIC_REFUSAL)
                await record_refusal(session, user_input)
            else:
                context = result.context_wrapper.context
                current_agent = result.last_agent
                save_state(conversation_id, context, current_agent)
        if auto_mode:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conversation-id",
        default="demo-customer",
        help="Use the same ID to continue a previous conversation.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(args.conversation_id))
    except KeyboardInterrupt:
        print("\nBye!")
