from pydantic import BaseModel

class MedicalRecordFormatInputDTO(BaseModel):
    medical_record: str
    medical_record_template: str
