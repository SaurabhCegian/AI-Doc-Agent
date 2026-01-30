from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rag.loader import load_and_chunk_docs

def create_vector_store(chunks: List[str], persist_dir: str = "chroma_db") -> Chroma:
    """Create and persist a Chroma vector store from documents in the specified directory."""
    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(
        texts = chunks, 
        embedding = embedding_model, 
        persist_directory = persist_dir
    )
    # vector_store.persist()
    return vector_store

def load_vector_store(persist_dir: str = "chroma_db") -> Chroma:
    """Load documents from a directory, chunk them, and create a vector store."""
    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma(
        embedding_function = embedding_model, 
        persist_directory = persist_dir
    )
    return vector_store

def retrieve_docs(query: str, k: int = 3) -> List[str]:
    """Retrieve similar documents from the vector store based on the query."""
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]