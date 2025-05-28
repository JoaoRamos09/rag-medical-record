from pydantic import BaseModel

class SummaryMedicalRecordInputDTO(BaseModel):
    medical_record_markdown: str
    summary_max_words: int
    
