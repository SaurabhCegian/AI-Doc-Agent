import os
from dotenv import load_dotenv
from rag.loader import load_and_chunk_docs
from rag.vector_store import create_vector_store, retrieve_docs

# Load environment variables
load_dotenv()

chunks = load_and_chunk_docs("docs")
print(f"Loaded {len(chunks)} chunks from documents.")
print(chunks[0])  # Print first chunk for verification

create_vector_store(chunks)

results = retrieve_docs("What is Transformer?")
print(results)