from app.ai.rag.models.PatientMedicalRecordsRagState import PatientMedicalRecordsRagState
from app.ai.rag.models.MedicalRecord import MedicalRecord
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

from langsmith import traceable

@traceable(run_type="chain", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "rag"})
def chunking_medical_records_node(state: PatientMedicalRecordsRagState) -> PatientMedicalRecordsRagState:
    protocol_id = generate_protocol_id()
    chunks = process_medical_records(state["source_medical_records"], protocol_id)
    
    return update_state(state, chunks, protocol_id)

def generate_protocol_id() -> str:
    return str(uuid.uuid4())

def process_medical_records(medical_records: list[MedicalRecord], protocol_id: str) -> list[Document]:
    return [chunk for record in medical_records for chunk in format_chunks(create_chunks(record), record.id, protocol_id)]

def update_state(state: PatientMedicalRecordsRagState, chunks: list[Document], protocol_id: str) -> PatientMedicalRecordsRagState:
    state["chunks"] = chunks
    state["protocol_id"] = protocol_id
    return state

def create_chunks(medical_record: MedicalRecord):
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=120,
        chunk_overlap=25
    )
    return text_splitter.split_text(medical_record.medical_record)

def format_chunks(chunks: list[str], medical_record_id: int, protocol_id: int):
    
    return [Document(id=str(uuid.uuid4()), page_content=chunk, metadata={"id_medical_record": medical_record_id, "protocol_id": protocol_id}) for chunk in chunks]

