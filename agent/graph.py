from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os


class AgentState(TypedDict):
    user_question: str
    need_docs: bool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

def decide_need_docs(state: AgentState) -> AgentState:
    prompt = f"""
        You are an AI assistant.

        User question:
        "{state['user_question']}"

        Decide whether answering this question requires
        looking up documentation.

        Answer ONLY with:
        YES or NO
    """

    response = llm.invoke(prompt).content.strip().upper()
    need_docs = True if response == "YES" else False

    return {
        **state,
        "need_docs": need_docs
    }

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("decide", decide_need_docs)
    graph.set_entry_point("decide")
    graph.add_edge("decide", END)
    return graph.compile()