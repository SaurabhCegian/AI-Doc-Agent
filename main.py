from dotenv import load_dotenv
load_dotenv()

from rag.vector_store import create_vector_store, retrieve_docs
from agent.graph import build_graph

# 1️⃣ Build vector store (one-time or startup)
create_vector_store("docs")

# 2️⃣ Test retrieval
# results = retrieve_docs("What is Transformer?")
# print(results)

# 3️⃣ Test agent decision
agent = build_graph()

result = agent.invoke({
    "user_question": "What is Transformer?"
})

print(result)
