from app.ai.medical_record_formatter.models.MedicalRecordFormatterState import MedicalRecordFormatterState
from app.ai.utils.openai_utils import invoking_model
from langsmith import traceable

@traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "medical_record_formatter"})
async def get_medical_record_template_fields_node(state: MedicalRecordFormatterState) -> MedicalRecordFormatterState:
    medical_record_template = "Modelo de prontuário médico: {medical_record_template}".format(medical_record_template=state["medical_record_template"])
    
    medical_record_template_fields = await invoking_model(user_input=medical_record_template, prompt_system=get_system_prompt())
    
    state["medical_record_template_fields"] = medical_record_template_fields.content
    
    return state


def get_system_prompt() -> str:
    return """Analise o modelo de prontuário médico e identifique quais são as informações que 
    precisamos coletar para criar o prontuário médico, se baseando no modelo de prontuário médico
    fornecido. O modelo será fornecido em markdown. Retorne somente as informações que precisamos
    coletar."""
