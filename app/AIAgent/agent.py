from dotenv import load_dotenv
from uuid_utils import uuid7
load_dotenv()

import os
from langchain.agents import create_agent
from app.AIAgent.BookLeave.book_leave_tool import book_leave_tool
from app.AIAgent.RAG.rag_tool import ask_rag
from app.AIAgent.SeatBooking.seat_booking_tool import seat_booking
from app.AIAgent.CabBooking.cab_booking_tool import cab_booking
import json
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.agents.middleware import HumanInTheLoopMiddleware

# langchain-google-genai expects GOOGLE_API_KEY
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

tools = [ask_rag,seat_booking, cab_booking, book_leave_tool]
config = {"configurable": {"thread_id": str(uuid7())}}

def get_agent(tools):
    agent = create_agent(
        tools=tools,
        model=os.getenv("GOOGLE_GENAI_MODEL"),
        checkpointer=InMemorySaver(), 
        middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_leave_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "cab_booking": False,
                "seat_booking": False,
            }
        ),
    ],

    )
    return agent


def get_pending_interrupts(agent, config):
    """Check if the agent is paused waiting for human input."""
    state = agent.get_state(config)
    if state.tasks:
        for task in state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                return task.interrupts
    return None


def display_hitl_request(hitl_request):
    """Display the pending action(s) for human review."""
    print("\n" + "=" * 50)
    print("APPROVAL REQUIRED")
    print("=" * 50)
    for i, action in enumerate(hitl_request["action_requests"]):
        print(f"\nAction {i + 1}: {action['name']}")
        print(f"  Args: {json.dumps(action['args'], indent=4)}")
        if "description" in action:
            print(f"  Description: {action['description']}")
    print("\nAllowed decisions:")
    for rc in hitl_request["review_configs"]:
        print(f"  {rc['action_name']}: {rc['allowed_decisions']}")
    print("=" * 50)


def collect_decisions(hitl_request):
    """Prompt the user for a decision on each pending action."""
    decisions = []
    for i, action in enumerate(hitl_request["action_requests"]):
        allowed = hitl_request["review_configs"][i]["allowed_decisions"]
        prompt = f"\nDecision for '{action['name']}' ({'/'.join(allowed)}): "
        while True:
            choice = input(prompt).strip().lower()
            if choice in allowed:
                break
            print(f"  Invalid choice. Pick one of: {allowed}")

        if choice == "approve":
            decisions.append({"type": "approve"})
        elif choice == "reject":
            reason = input("  Reason (optional, press Enter to skip): ").strip()
            decision = {"type": "reject"}
            if reason:
                decision["message"] = reason
            decisions.append(decision)
        elif choice == "edit":
            print(f"  Current args: {json.dumps(action['args'], indent=4)}")
            print("  Enter edited args as JSON (e.g. {\"start_date\": \"2026-07-01\"}):")
            while True:
                try:
                    edited_args = json.loads(input("  > "))
                    break
                except json.JSONDecodeError:
                    print("  Invalid JSON. Try again.")
            merged_args = {**action["args"], **edited_args}
            decisions.append({
                "type": "edit",
                "edited_action": {"name": action["name"], "args": merged_args}
            })
        elif choice == "respond":
            msg = input("  Your response message: ").strip()
            decisions.append({"type": "respond", "message": msg})

    return {"decisions": decisions}


if __name__ == "__main__":
    agent = get_agent(tools)
    print("Agent is ready to assist you. Type your queries below:")
    while True:
        user_input = input("\nUser: ")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Check for HITL interrupts and handle them in a loop
        while True:
            interrupts = get_pending_interrupts(agent, config)
            if not interrupts:
                break

            hitl_request = interrupts[0].value
            display_hitl_request(hitl_request)
            hitl_response = collect_decisions(hitl_request)

            # Resume the agent with the human's decisions
            response = agent.invoke(
                Command(resume=hitl_response),
                config=config,
            )

        # Print the final agent response
        last_msg = response["messages"][-1]
        if hasattr(last_msg, "content"):
            content = last_msg.content
            if isinstance(content, list) and content:
                print(f"\nAgent: {content[0].get('text', content[0])}")
            else:
                print(f"\nAgent: {content}")
        else:
            print(f"\nAgent: {last_msg}")