from pydantic import BaseModel
from typing import Optional, Dict, Any

class IngestResponse(BaseModel):
    job_id: str
    status_url: str

class ForwardTarget(BaseModel):
    url: str
    api_key: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    result_path: Optional[str] = None
    message: Optional[str] = None