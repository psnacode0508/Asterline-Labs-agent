"""LangGraph orchestration for the Asterline multi-agent research assistant."""

from typing import Dict, List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

from agents import research_node, synthesis_node, triage_node


load_dotenv()


class AgentState(TypedDict, total=False):
    """Shared state passed between Asterline's specialized functional agents."""

    user_id: str
    user_input: str
    user_memory: Dict[str, str]
    intent: str
    memory_payload: Dict[str, str]
    retrieved_docs: List[Document]
    response: str


def route_after_triage(state: AgentState) -> str:
    """Route unsafe and memory-only requests without unnecessary retrieval."""
    if state["intent"] == "injection_blocked":
        return "end"
    if state["memory_payload"] or state["intent"] == "chat":
        return "synthesis"
    return "research"


workflow = StateGraph(AgentState)
workflow.add_node("triage", triage_node)
workflow.add_node("research", research_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_edge(START, "triage")
workflow.add_conditional_edges(
    "triage",
    route_after_triage,
    {"end": END, "research": "research", "synthesis": "synthesis"},
)
workflow.add_edge("research", "synthesis")
workflow.add_edge("synthesis", END)
app = workflow.compile()


if __name__ == "__main__":
    test_result = app.invoke(
        {
            "user_id": "alice",
            "user_input": "What is the weekly stipend for on-call engineers?",
            "user_memory": {},
            "intent": "",
            "memory_payload": {},
            "retrieved_docs": [],
            "response": "",
        }
    )
    response = test_result["response"]
    print("\n" + "="*40)
    print("ACTUAL OUTPUT FROM SYNTHESIS:")
    print(repr(response))
    print("="*40 + "\n")
    assert "$175" in response
    assert "[handbook:1]" in response
    print("Graph self-test passed: triage -> research -> synthesis.")
