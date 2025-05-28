from langgraph.graph import StateGraph, END
from app.ai.medical_record_formatter.models.MedicalRecordFormatterState import MedicalRecordFormatterState
from app.ai.medical_record_formatter.nodes.create_sketch import create_sketch_node
from app.ai.medical_record_formatter.nodes.get_medical_record_template_fields import get_medical_record_template_fields_node
from app.ai.medical_record_formatter.nodes.format_medical_record import format_medical_record_node

async def graph():
    graph = StateGraph(MedicalRecordFormatterState)
    graph.add_node("format_medical_record", format_medical_record_node)
    
    graph.set_entry_point("format_medical_record")
    
    graph.add_edge("format_medical_record", END)
    return graph.compile()