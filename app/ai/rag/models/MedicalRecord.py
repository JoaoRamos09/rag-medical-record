from pydantic import BaseModel

class MedicalRecord(BaseModel):
    id: int
    medical_record: str
