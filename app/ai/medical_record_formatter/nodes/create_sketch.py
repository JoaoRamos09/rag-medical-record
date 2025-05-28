from app.ai.medical_record_formatter.models.MedicalRecordFormatterState import MedicalRecordFormatterState
from app.ai.utils.openai_utils import invoking_model
from langsmith import traceable

@traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "medical_record_formatter"})
async def create_sketch_node(state: MedicalRecordFormatterState) -> MedicalRecordFormatterState:
    sketch = await invoking_model(user_input= state["input"], prompt_system= get_system_prompt())
    state["sketch"] = sketch.content
    return state

def get_system_prompt() -> str:
    return """Você é assistente médico experiente, sua principal função é analisar as informações 
    informadas e organiza-lás em um rascunho de um prontuário médico de uma maneira clara e concisa.
    Não invente nenhuma informação e retorne somente as informações organizadas."""
