import os
from typing import TypedDict, List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END

# Define the local storage path for ChromaDB
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../../data/chroma_db")

# 1. Define the State
class IngestionState(TypedDict):
    file_path: str
    documents: List[any]
    status: str

# 2. Node: Load and Chunk the Resume
def load_and_chunk(state: IngestionState):
    loader = PyPDFLoader(state["file_path"])
    docs = loader.load()
    
    # Split the resume into logical chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    
    return {"documents": chunks, "status": "Chunked successfully"}

# 3. Node: Embed and Store in ChromaDB
def embed_and_store(state: IngestionState):
    # Initialize the Ollama embeddings with the base_url
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434" # <--- You set it right here
    )
    
    # Store chunks in ChromaDB locally
    Chroma.from_documents(
        documents=state["documents"],
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return {"status": "Embeddings stored in VectorDB via local Ollama"}

# 4. Build the Graph
workflow = StateGraph(IngestionState)

workflow.add_node("load_and_chunk", load_and_chunk)
workflow.add_node("embed_and_store", embed_and_store)

workflow.set_entry_point("load_and_chunk")
workflow.add_edge("load_and_chunk", "embed_and_store")
workflow.add_edge("embed_and_store", END)

# Compile the ingestion graph
ingestion_app = workflow.compile()