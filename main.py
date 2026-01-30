from rag.loader import load_and_chunk_docs

chunks = load_and_chunk_docs("docs")
print(f"Loaded {len(chunks)} chunks from documents.")
print(chunks[:2])  # Print first 2 chunks for verification