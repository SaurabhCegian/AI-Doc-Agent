from dotenv import load_dotenv
load_dotenv()

from agent.graph import build_graph

agent = build_graph()

result = agent.invoke({
    "user_question": "What is LangGraph?"
})

print("Need docs:", result["need_docs"])
print("Final answer:\n", result["final_answer"])