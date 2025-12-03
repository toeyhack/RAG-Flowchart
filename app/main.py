import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from .llm_client import OpenRouterClient
from .processor import Processor
from .models import IngestResponse, JobStatus, ForwardTarget
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Multimodal RAG Ingest API")
OR_CLIENT = OpenRouterClient()
PROCESSOR = Processor(OR_CLIENT)

# simple in-memory job store (sqlite or proper DB recommended)
JOB_STORE = {}

def save_upload(file: UploadFile) -> str:
    doc_id = str(uuid.uuid4())
    folder = os.path.join("uploads", doc_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return doc_id, path

@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(file: UploadFile = File(...), model_id: Optional[str] = None, output_format: str = "md", background_tasks: BackgroundTasks = None, forward_to: Optional[str] = None):
    doc_id, path = save_upload(file)
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {"status":"queued","doc_id":doc_id,"path":path}
    background_tasks.add_task(_process_job, job_id, doc_id, path, model_id or os.getenv("MODEL_DEFAULT","qwen3-vl-235b"), output_format, forward_to)
    return {"job_id": job_id, "status_url": f"/status/{job_id}"}

def _process_job(job_id, doc_id, path, model_id, output_format, forward_to):
    JOB_STORE[job_id]["status"] = "processing"
    try:
        nodes = PROCESSOR.process_pdf(doc_id, path, model_id)
        # compose result (simplified)
        result_path = os.path.join("uploads", doc_id, f"result.{output_format}")
        with open(result_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Document {doc_id}\n\n")
            for n in nodes:
                fh.write(f"- node: {n['node_id']} type: {n['type']}\n")
        JOB_STORE[job_id]["status"] = "done"
        JOB_STORE[job_id]["result_path"] = result_path
    except Exception as e:
        JOB_STORE[job_id]["status"] = "error"
        JOB_STORE[job_id]["message"] = str(e)

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="job not found")
    st = JOB_STORE[job_id]
    return {"job_id": job_id, "status": st.get("status"), "result_path": st.get("result_path"), "message": st.get("message")}

@app.get("/result/{job_id}")
async def get_result(job_id: str, format: str = "md"):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="job not found")
    st = JOB_STORE[job_id]
    if st.get("status") != "done":
        raise HTTPException(status_code=409, detail="job not ready")
    return FileResponse(st["result_path"], media_type="text/markdown" if format=="md" else "text/plain", filename=os.path.basename(st["result_path"]))
