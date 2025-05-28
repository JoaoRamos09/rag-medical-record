from langgraph.graph import StateGraph, END
from app.ai.summary_medical_record.models.MedicalRecordSummaryState import MedicalRecordSummaryState
from app.ai.summary_medical_record.nodes.create_summary import create_summary_node

async def graph():
    graph = StateGraph(MedicalRecordSummaryState)
    
    graph.add_node("create_summary", create_summary_node)
    graph.set_entry_point("create_summary")
    graph.add_edge("create_summary", END)
    
    return graph.compile()