from typing import TypedDict, List
import requests
import os
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


# -------------------------
# Agent State
# -------------------------
class AgentState(TypedDict):
    user_question: str
    need_docs: bool
    retrieved_docs: List[str]
    final_answer: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------
# Decision Node
# -------------------------
def decide_need_docs(state: AgentState) -> AgentState:
    prompt = f"""
        You are an AI documentation assistant.

        You must decide whether to use the system's internal documentation
        to answer the user's question.

        Rules:
        - If the question is about technical concepts, definitions, or explanations,
        ALWAYS use documentation.
        - If the question is casual, conversational, or about greetings,
        DO NOT use documentation.

        User question:
        "{state['user_question']}"

        Answer ONLY with:
        YES or NO
    """

    response = llm.invoke(prompt).content.strip().upper()
    need_docs = response == "YES"

    return {
        **state,
        "need_docs": need_docs,
        "retrieved_docs": [],
        "final_answer": ""
    }


# -------------------------
# MCP Tool Call Node
# -------------------------
def call_mcp_retrieve(state: AgentState) -> AgentState:
    response = requests.post(
        "http://127.0.0.1:8000/retrieve_docs",
        json={"query": state["user_question"]}
    )

    docs = response.json()["chunks"]

    return {
        **state,
        "retrieved_docs": docs
    }


# -------------------------
# Answer Generation Node
# -------------------------
def generate_answer(state: AgentState) -> AgentState:
    if state["need_docs"]:
        context = "\n\n".join(state["retrieved_docs"])

        prompt = f"""
            Answer the question using ONLY the documentation below.

            Documentation:
            {context}

            Question:
            {state['user_question']}
        """
    else:
        prompt = f"""
            Answer the following question clearly:

            {state['user_question']}
        """

    answer = llm.invoke(prompt).content.strip()

    return {
        **state,
        "final_answer": answer
    }


# -------------------------
# Routing Logic
# -------------------------
def route_after_decision(state: AgentState):
    if state["need_docs"]:
        return "call_mcp"
    return "answer"


# -------------------------
# Graph Builder
# -------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decide", decide_need_docs)
    graph.add_node("call_mcp", call_mcp_retrieve)
    graph.add_node("answer", generate_answer)

    graph.set_entry_point("decide")

    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {
            "call_mcp": "call_mcp",
            "answer": "answer"
        }
    )

    graph.add_edge("call_mcp", "answer")
    graph.add_edge("answer", END)

    return graph.compile()
