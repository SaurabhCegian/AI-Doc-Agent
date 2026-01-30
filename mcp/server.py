import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from rag.vector_store import retrieve_docs

# Load environment variables
load_dotenv()

app = FastAPI()

class RetrieveDocsRequest(BaseModel):
    query: str

class RetrieveDocsResponse(BaseModel):
    chunks: List[str]

@app.post("/retrieve_docs", response_model=RetrieveDocsResponse)
def retrieve_docs_tool(request: RetrieveDocsRequest):
    chunks = retrieve_docs(request.query)
    return RetrieveDocsResponse(chunks=chunks)