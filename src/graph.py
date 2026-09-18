"""Minimal LangGraph workflow example.

This module demonstrates a simple two-node graph:
    1. A node that generates a response.
    2. A node that formats the response.

The graph is compiled and invoked from `main.py`.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import settings


class GraphState(TypedDict, total=False):
    """State schema shared by graph nodes."""

    question: str
    answer: str
    formatted: str


def build_graph() -> StateGraph:
    """Build and compile a small LangGraph workflow."""
    llm = ChatOpenAI(**settings.get_llm_kwargs())

    def generate_node(state: GraphState) -> GraphState:
        """Call the LLM to produce an answer for the user's question."""
        question = state.get("question", "")
        response = llm.invoke(question)
        answer = response.content if isinstance(response, AIMessage) else str(response)
        return {"answer": answer}

    def format_node(state: GraphState) -> GraphState:
        """Wrap the generated answer into a human-friendly format."""
        answer = state.get("answer", "")
        return {"formatted": f"AI: {answer}"}

    graph = StateGraph(GraphState)
    graph.add_node("generate", generate_node)
    graph.add_node("format", format_node)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "format")
    graph.add_edge("format", END)
    return graph.compile()


async def run_graph(question: str) -> str:
    """Run the compiled graph with a user question."""
    app = build_graph()
    result = await app.ainvoke({"question": question})
    return result.get("formatted", "")
