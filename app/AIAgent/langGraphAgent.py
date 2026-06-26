import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../..')))

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage, RemoveMessage

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv


load_dotenv()

# ── Import tools ──────────────────────────────────────────────────────────────
from app.AIAgent.BookLeave.book_leave_tool import book_leave_tool
from app.AIAgent.CabBooking.cab_booking_tool import cab_booking
from app.AIAgent.SeatBooking.seat_booking_tool import seat_booking
from app.AIAgent.RAG.rag_tool import ask_rag

from langgraph.checkpoint.memory import InMemorySaver

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str  # rolling summary of older messages

# ── Constants ─────────────────────────────────────────────────────────────────
# Max number of recent messages passed to the LLM on each turn.
# Older messages beyond this will be summarised and removed from live context.
MAX_MESSAGES = 10

# ── LLM with tools bound ──────────────────────────────────────────────────────
tools = [book_leave_tool, cab_booking, seat_booking, ask_rag]

llm = init_chat_model(
    model=os.getenv("OPEN_AI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)

# ── Node functions ────────────────────────────────────────────────────────────
def summarise_node(state: State) -> dict:
    """Condense messages older than MAX_MESSAGES into a running summary.
    This node only runs when the conversation exceeds MAX_MESSAGES turns."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # Ask the LLM to extend the existing summary with the older messages
    messages_to_summarise = messages[:-MAX_MESSAGES]
    summary_prompt = (
        f"Extend the conversation summary below with the new messages.\n\n"
        f"Existing summary:\n{existing_summary}\n\n"
        f"New messages to add:\n"
        + "\n".join(
            f"{m.__class__.__name__}: {m.content}"
            for m in messages_to_summarise
        )
    )
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    new_summary = response.content

    # Delete the old messages that are now in the summary
    delete_ops = [RemoveMessage(id=m.id) for m in messages_to_summarise]
    return {"summary": new_summary, "messages": delete_ops}


def llm_node(state: State) -> dict:
    """Invoke the LLM, prepending the rolling summary as a system message."""
    messages = state["messages"]
    summary = state.get("summary", "")

    if summary:
        messages = [SystemMessage(content=f"Conversation summary so far:\n{summary}")] + messages

    return {"messages": [llm_with_tools.invoke(messages)]}

# ── Routing function ──────────────────────────────────────────────────────────
def route_tools(state: State) -> str:
    """Route to the appropriate tool node based on the last tool call, or END."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        routing = {
            "book_leave_tool": "leave_booking_node",
            "cab_booking":     "cab_booking_node",
            "seat_booking":    "seat_booking_node",
            "ask_rag":         "rag_node",
        }
        return routing.get(tool_name, END)
    return END


def should_summarise(state: State) -> str:
    """Trigger summarisation when message history exceeds MAX_MESSAGES."""
    if len(state["messages"]) > MAX_MESSAGES:
        return "summarise_node"
    return "llm_node"

# ── Individual ToolNodes ──────────────────────────────────────────────────────
leave_booking_node = ToolNode([book_leave_tool])
cab_booking_node   = ToolNode([cab_booking])
seat_booking_node  = ToolNode([seat_booking])
rag_node           = ToolNode([ask_rag])

# ── Build graph ───────────────────────────────────────────────────────────────
graph = StateGraph(State)

graph.add_node("summarise_node",     summarise_node)
graph.add_node("llm_node",          llm_node)
graph.add_node("leave_booking_node", leave_booking_node)
graph.add_node("cab_booking_node",   cab_booking_node)
graph.add_node("seat_booking_node",  seat_booking_node)
graph.add_node("rag_node",           rag_node)

# Entry: check if we need to summarise first, then call LLM
graph.add_conditional_edges(START, should_summarise)
graph.add_edge("summarise_node", "llm_node")

# Conditional routing: LLM → appropriate tool node or END
graph.add_conditional_edges(
    "llm_node",
    route_tools,
    {
        "leave_booking_node": "leave_booking_node",
        "cab_booking_node": "cab_booking_node",
        "seat_booking_node": "seat_booking_node",
        "rag_node": "rag_node",
        END: END,
    },)

# Each tool node loops back to the LLM for a final answer
graph.add_edge("leave_booking_node", "llm_node")
graph.add_edge("cab_booking_node",   "llm_node")
graph.add_edge("seat_booking_node",  "llm_node")
graph.add_edge("rag_node",           "llm_node")

app = graph.compile(checkpointer=InMemorySaver())
print("Graph compiled successfully.")

config = {"configurable": {"thread_id": "test_thread"}}

if __name__ == "__main__":
    # Example user query that may require multiple tools
    while True:
        user_query = input("\nEnter your query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break   
        if not user_query.strip():
            print("Please enter a valid query.")
            continue
        
        final_response = app.invoke({"messages": [user_query]}, config=config)
       
         # Print the final agent response
        last_msg = final_response["messages"][-1]
        if hasattr(last_msg, "content"):
            content = last_msg.content
            if isinstance(content, list) and content:
                print(f"\nAgent: {content[0].get('text', content[0])}")
            else:
                print(f"\nAgent: {content}")
        else:
            print(f"\nAgent: {last_msg}")
