from app.ai.medical_record_formatter.models.MedicalRecordFormatterState import MedicalRecordFormatterState
from app.ai.utils.openai_utils import invoking_model
from langsmith import traceable

@traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "medical_record_formatter"})
async def format_medical_record_node(state: MedicalRecordFormatterState) -> MedicalRecordFormatterState:
    medical_record_input = get_input(state)
    
    formatted_medical_record = await invoking_model(user_input=medical_record_input, prompt_system = get_system_prompt())
    
    state["formatted_medical_record"] = formatted_medical_record.content
    
    return state


def get_input(state: MedicalRecordFormatterState) -> str:
    return f"""
        MODELO DO PRONTUÁRIO MÉDICO: {state["medical_record_template"]} \n\n
        PRONTUÁRIO A SER FORMATADO: {state["input"]}
    """


def get_system_prompt() -> str:
    return """
    Você é um experiente profissional da área de saúde, sua principal função é formatar prontuários médicos baseado nas informações fornecidas.
    
    Será fornecido um modelo de prontuário médico e um rascunho de anotações do paciente ou então outro prontuário.
    
    Você deve adequar o rascunho/o outro prontuário para o modelo fornecido, preenchendo os campos do modelo com as informações do rascunho/outro prontuário.
    
    Você não pode alterar o modelo do prontuário médico nem inventar informações. Caso não seja possível coletar alguma informação, você deve deixar o campo em branco.
    
    As informações que sobrarem, insira ao final do prontuário.
    
    O prontuário médico deve ser retornado em plain text.
    
    Caso receba um prontuário sem nada escrito, apenas retorne o modelo do prontuário.
    """
