from typing_extensions import TypedDict
from app.ai.rag.models.MedicalRecord import MedicalRecord
from langchain_core.documents import Document

class PatientMedicalRecordsRagState(TypedDict):
    user_question: str
    source_medical_records: list[MedicalRecord]
    chunks: list[Document]
    generated_answer: str
    selected_medical_records: list[MedicalRecord]
    most_relevant_chunks: list[tuple]
    relevant_question: str
    protocol_id: str


