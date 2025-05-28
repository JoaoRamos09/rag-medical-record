from typing_extensions import TypedDict

class MedicalRecordSummaryState(TypedDict):
    medical_record_markdown: str
    medical_record: str
    medical_record_data: str
    summary_max_words: int
    summary: str
    
