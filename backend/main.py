import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from backend.services.ingestion import ingestion_app

app = FastAPI(title="Agentic Job Matcher API")

# Ensure data directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(DATA_DIR, file.filename)
    
    # Save the uploaded file locally
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Trigger the LangGraph ingestion workflow
        initial_state = {"file_path": file_path, "documents": [], "status": "Starting"}
        result = ingestion_app.invoke(initial_state)
        
        # Clean up the PDF after processing (optional)
        os.remove(file_path)
        
        return {"filename": file.filename, "status": result["status"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))