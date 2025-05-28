from pydantic import BaseModel
from app.ai.rag.models.MedicalRecord import MedicalRecord

class MedicalRecordSearchInputDTO(BaseModel):
    search: str
    medical_records_source_list: list[MedicalRecord]
