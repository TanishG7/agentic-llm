import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from typing import Optional
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import requests

# Disable internal LLM
Settings.llm = None

# Config
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "tinyllama"
GEMINI_API_KEY = "AIzaSyBWA3hPu5iidk15on-LrbeCBLnAlVeNw9w" # Set your Gemini API key in environment variables
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
PERSIST_DIR = "./storage"

print("🔄 Loading index...")

start_load = time.time()
try:
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    print(f"✅ Index loaded successfully | Time taken: {time.time() - start_load:.2f} seconds")
except Exception as e:
    print(f"❌ Failed to load index: {e}")
    raise RuntimeError("Could not initialize search system")

app = FastAPI(title="Document Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows your React app's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
class SearchRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class SearchResponse(BaseModel):
    answer: str
    context: str
    success: bool

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Add proper error handling
    except WebSocketDisconnect:
        print("Client disconnected")

def query_documents(question: str, top_k: int = 3) -> str:
    print(f"🔍 Querying index for top {top_k} results...")
    start_query = time.time()
    try:
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(question)
        duration = time.time() - start_query
        print(f"✅ Document chunks retrieved | Time taken: {duration:.2f} seconds | Context length: {len(response.response)} characters")
        return response.response
    except Exception as e:
        print(f"❌ Document query failed: {e}")
        raise

def ask_ollama(context: str, question: str) -> str:
    prompt = f"""You are a helpful assistant.
Answer the question using the following context only.

### Context:
{context}

### Question:
{question}

INSTRUCTIONS:
1. Answer in 6-7 complete sentences
2. Include all relevant details from the context
3. If multiple items exist, list them clearly
4. Never make up information

### Answer:"""

    print(f"🤖 Sending prompt to Ollama... | Context length: {len(context)} characters")
    start_ollama = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180
        )
        duration = time.time() - start_ollama
        if response.status_code == 200:
            print(f"✅ Response received from Ollama | Time taken: {duration:.2f} seconds")
            return response.json()["response"]
        else:
            print(f"❌ Ollama API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="LLM service error")
    except Exception as e:
        print(f"❌ Ollama request failed: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

def ask_gemini(context: str, question: str) -> str:
    prompt = f"""You are a helpful assistant.
Answer the question using the following context only.

### Context:
{context}

### Question:
{question}

INSTRUCTIONS:
1. Answer in 6-7 complete sentences
2. Include all relevant details from the context
3. If multiple items exist, list them clearly
4. Never make up information

### Answer:"""

    print(f"💎 Sending prompt to Gemini... | Context length: {len(context)} characters")
    start_gemini = time.time()
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "maxOutputTokens": 1000  # Adjust as needed
                }
            },
            timeout=30  # Timeout in seconds
        )
        
        if response.status_code != 200:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=500, detail="Gemini API error")
        
        answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Response received from Gemini | Time taken: {time.time() - start_gemini:.2f} seconds")
        return answer
        
    except Exception as e:
        print(f"Gemini request failed: {e}")
        raise HTTPException(status_code=503, detail="Gemini service unavailable")

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    print(f"🚀 Starting search for: '{request.question}'")
    overall_start = time.time()

    try:
        context = query_documents(request.question, request.top_k)
        final_answer = ask_ollama(context, request.question)
        total_time = time.time() - overall_start
        print(f"🎯 Search completed | Total time: {total_time:.2f} seconds")

        return SearchResponse(
            answer=final_answer,
            context=context,
            success=True
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-gemini", response_model=SearchResponse)
async def search_gemini(request: SearchRequest):
    print(f"🚀 Starting Gemini search for: '{request.question}'")
    overall_start = time.time()

    try:
        context = query_documents(request.question, request.top_k)
        final_answer = ask_gemini(context, request.question)
        total_time = time.time() - overall_start
        print(f"💎 Gemini search completed | Total time: {total_time:.2f} seconds")

        return SearchResponse(
            answer=final_answer,
            context=context,
            success=True
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Gemini search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if index and embed_model:
        return {"status": "healthy", "index_loaded": True}
    return {"status": "unhealthy", "index_loaded": False}

@app.get("/")
async def root():
    return {
        "message": "Document Search API",
        "description": "Search documents using semantic search and LLM generation",
        "endpoints": {
            "search": "POST /search",
            "search-gemini": "POST /search-gemini",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)