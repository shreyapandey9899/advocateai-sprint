# main.py
import os
import uuid
import json
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI client
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except TypeError:
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    exit()


# Initialize the FastAPI application
app = FastAPI()

# --- FIX 1: Create a single, shared ChromaDB client for the application ---
client = chromadb.Client()

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This is CRITICAL for allowing your HTML file (frontend) to communicate with this backend server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity in a hackathon
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def serve_frontend():
    """This endpoint serves your main HTML file."""
    return FileResponse('index.html')

# --- Pydantic Models: Define the structure of API requests ---
class CreateSessionRequest(BaseModel):
    full_text: str

class AskRequest(BaseModel):
    session_id: str
    question: str

# --- API Endpoints ---

@app.post("/simulator/create_session")
async def create_simulator_session(request: CreateSessionRequest):
    """
    Takes a full document text, processes it, and sets up a new chat session.
    """
    session_id = str(uuid.uuid4())
    
    # 1. Split the document into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([request.full_text])
    
    # 2. Get the embedding model
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Note: switched to a standard embedding model name
    
    # 3. Create a unique, in-memory vector database collection for this session
    #    This now uses the single, shared client
    db = client.create_collection(name=session_id)
    
    # 4. Add the document chunks to the collection
    db.add(
        documents=[doc.page_content for doc in docs],
        ids=[str(i) for i in range(len(docs))]
    )
    
    # 5. Return the unique session ID to the client
    return {"session_id": session_id}


@app.post("/simulator/ask")
async def ask_question(request: AskRequest):
    """
    Answers a user's question based on the document context of a specific session.
    """
    # 1. Get the specific collection for this session using the shared client
    try:
        db = client.get_collection(name=request.session_id)
    except Exception: # Catching a broader exception for robustness
        return {"error": "Invalid or expired session ID. Please start a new session."}

    # 2. Query the database to find the most relevant document chunks
    results = db.query(
        query_texts=[request.question],
        n_results=4  # Retrieve the top 4 most relevant chunks
    )
    
    # --- FIX 2: Add a safety check for empty results ---
    if not results or not results['documents'] or not results['documents'][0]:
        return {"answer": "I'm sorry, I couldn't find any relevant information in the document to answer that question."}

    retrieved_context = "\n---\n".join(results['documents'][0])
    
    # 3. Prepare the model and the prompt
    llm = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are a helpful AI assistant. Your task is to answer the user's question based *exclusively* on the provided context from a legal document. Do not use any external knowledge.

    **Provided Context from the Document:**
    {retrieved_context}

    **User's Question:**
    {request.question}

    **Your Answer:**
    If the context provides an answer, state it clearly. If the context does not contain the answer, you MUST respond with: "The document does not provide specific information on this topic."
    """
    
    # 4. Generate the final answer
    response = llm.generate_content(prompt)
    return {"answer": response.text}


# --- Main entry point to run the server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
