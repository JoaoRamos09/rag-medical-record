from app.ai.rag.models.PatientMedicalRecordsRagState import PatientMedicalRecordsRagState
from app.ai.utils.openai_utils import invoking_model
from langsmith import traceable

@traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "rag"})
async def generate_generic_answer_node(state: PatientMedicalRecordsRagState) -> PatientMedicalRecordsRagState:
    prompt = create_prompt_generic_answer()
    response = await invoking_model(user_input=state["user_question"], prompt_system= prompt, temperature=0.7)
    state["generated_answer"] = response.content
    state["selected_medical_records"] = []
    state["most_relevant_chunks"] = []
    return state

def create_prompt_generic_answer() -> str:
    return f"""
    Você é um assistente especializado em prontuários médicos, projetado para auxiliar com questões médico-hospitalares. Mas no entanto, não foi possível identificar uma relação clara entre sua pergunta e algum dos prontuários médicos disponíveis.
    
    Sempre que for acionado, é porque não foi possível identificar uma relação clara entre a pergunta do usuário e os prontuários médicos disponíveis.
    
    Avalie a pergunta do usuário e formule uma resposta adequada informando que não foi possível identificar uma relação clara entre a pergunta e os prontuários médicos disponíveis.

    Leve em consideração os seguintes aspectos:
    
    - Questões sobre procedimentos médicos
    - Dúvidas sobre consultas ou tratamentos
    - Perguntas sobre medicamentos ou terapias
    - Informações sobre exames ou diagnósticos
    - Dúvidas sobre histórico médico ou prontuário
    """
