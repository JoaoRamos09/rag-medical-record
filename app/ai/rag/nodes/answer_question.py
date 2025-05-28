from langchain_chroma import Chroma
from app.ai.rag.models.PatientMedicalRecordsRagState import PatientMedicalRecordsRagState
from app.ai.utils.openai_utils import invoking_model
from app.ai.rag.models.MedicalRecord import MedicalRecord
from langsmith import traceable

@traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "rag"})
async def answer_question_node(state: PatientMedicalRecordsRagState) -> PatientMedicalRecordsRagState:
    medical_context = state["selected_medical_records"]
    user_question = state["user_question"]
    
    system_prompt = create_medical_assistant_prompt(medical_context)
    formatted_question = format_user_question(user_question)
    
    response = await invoking_model(
        user_input=formatted_question,
        prompt_system=system_prompt
    )
    
    state["generated_answer"] = response.content
    return state

def create_medical_assistant_prompt(medical_context: str) -> str:
    return f"""
    Você é um assistente médico especializado. Sua tarefa é responder perguntas médicas com base nos prontuários médicos fornecidos.
    
    {format_context(medical_context)}
    
    **Instruções**:
    - Use os prontuários médicos fornecidos para formular sua resposta.
    - Seja conciso e direto.
    - Se os prontuários médicos não fornecerem informações suficientes, indique que mais informações são necessárias.
    """

def format_user_question(question: str) -> str:
    return f"""
    **Pergunta do Usuário**:
    {question}
    
    **Instruções**:
    - Responda com base nos prontuários médicos fornecidos.
    """
    
def format_context(context: list[MedicalRecord]) -> str:
    return "\n".join([f"Prontuário Médico {i+1}: {doc.medical_record}" for i, doc in enumerate(context)])
