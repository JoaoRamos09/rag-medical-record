from app.ai.rag.models.PatientMedicalRecordsRagState import PatientMedicalRecordsRagState
from app.ai.utils.openai_utils import invoking_model
from langsmith import traceable

@traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "rag"})
async def validates_question_node(state: PatientMedicalRecordsRagState) -> PatientMedicalRecordsRagState:
    prompt = create_prompt_validate_question()
    response = await invoking_model(user_input=state["user_question"], prompt_system= prompt)
    state["relevant_question"] = response.content
    return state

def create_prompt_validate_question() -> str:
    return f"""Você é um especialista em validação e filtragem de conteúdo médico, com amplo conhecimento em terminologia médica, procedimentos hospitalares, consultas médicas, pesquisa clínica, consultório médico e prontuários médicos.

    Sua função é analisar perguntas e determinar se elas são relevantes para o contexto de prontuários médicos.
  
    1. Considere dentro do contexto de prontuários médicos, clinicas médicas, consultas médicas, pesquisa clínica, consultório médico e saúde, perguntas relacionadas a:
       - Consultas médicas (agendamento, histórico, procedimentos)
       - Tratamentos e terapias
       - Medicações e prescrições
       - Exames e diagnósticos
       - Histórico médico e prontuários
       - Procedimentos hospitalares
       - Questões de saúde preventiva
       - Pesquisa médica e estudos clínicos

    3. Responda APENAS com:
       - "True" se a pergunta for ao contexto de prontuários médicos, clinicas médicas, consultas médicas, pesquisa clínica, consultório médico e saúde
       - "False" se a pergunta não tiver relação com prontuários médicos, clinicas médicas, consultas médicas, pesquisa clínica, consultório médico e saúde

    Lembre-se: Seja rigoroso na análise. Perguntas ambíguas ou sem clara relação com saúde devem ser classificadas como "False".
    """
