from langchain_chroma import Chroma
from app.ai.rag.models.PatientMedicalRecordsRagState import PatientMedicalRecordsRagState
from app.ai.utils.openai_utils import get_embeddings
from app.ai.utils.chromaDB_utils import create_chroma_db, cleanup_chroma_db, find_most_relevant_chunks
from langsmith import traceable

chroma_db = create_chroma_db(get_embeddings())

@traceable(run_type="retriever", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "rag"})
async def retriever_medical_record_node(state: PatientMedicalRecordsRagState) -> PatientMedicalRecordsRagState:
    protocol_id = state["protocol_id"]
    chunks = state["chunks"]
    user_question = state["user_question"]

    most_relevant_chunks = await find_most_relevant_chunks(
            chroma_db=chroma_db, 
            chunks=chunks, 
            user_question=user_question, 
            protocol_id=protocol_id, 
            k=3)
    
    await cleanup_chroma_db(chroma_db, [chunk.id for chunk in chunks])    
    selected_medical_records = get_selected_medical_records_from_chunks(state["source_medical_records"], most_relevant_chunks)
    return update_state(state, selected_medical_records, most_relevant_chunks)


def update_state(state: PatientMedicalRecordsRagState, selected_medical_records: list, most_relevant_chunks: list[tuple]) -> PatientMedicalRecordsRagState:
    state["selected_medical_records"] = selected_medical_records
    state["most_relevant_chunks"] = most_relevant_chunks
    return state

def get_selected_medical_records_from_chunks(source_medical_records: list, chunks: list[tuple]) -> list:
    chunks_ids = set([chunk[0].metadata["id_medical_record"] for chunk in chunks])
    return ([medical_record for medical_record in source_medical_records if medical_record.id in chunks_ids])
            
