from langgraph.graph import StateGraph, END
from app.ai.rag.models.PatientMedicalRecordsRagState import PatientMedicalRecordsRagState
from app.ai.rag.nodes.chunking_medical_records import chunking_medical_records_node
from app.ai.rag.nodes.retriever_medical_record import retriever_medical_record_node
from app.ai.rag.nodes.answer_question import answer_question_node
from app.ai.rag.nodes.generate_generic_answer import generate_generic_answer_node
from app.ai.rag.nodes.validates_question import validates_question_node

async def graph():
    graph = StateGraph(PatientMedicalRecordsRagState)
    
    graph.add_node("chunking_medical_records", chunking_medical_records_node)
    graph.add_node("retriever_medical_record", retriever_medical_record_node)
    graph.add_node("validates_question", validates_question_node)
    graph.add_node("answer_question", answer_question_node)
    graph.add_node("generate_generic_answer", generate_generic_answer_node)

    graph.set_entry_point("validates_question")
    graph.add_conditional_edges("validates_question", lambda state: state["relevant_question"], {"False": "generate_generic_answer", "True": "chunking_medical_records"})
    graph.add_edge("chunking_medical_records", "retriever_medical_record")
    graph.add_edge("retriever_medical_record", "answer_question")
    graph.add_edge("answer_question", END)
    graph.add_edge("generate_generic_answer", END)
    return graph.compile()