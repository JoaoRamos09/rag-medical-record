from langgraph.errors import GraphRecursionError
import logging
from app.ai.medical_record_formatter import medical_record_formatter_graph
from app.ai.summary_medical_record import summary_medical_record_graph
from app.ai.rag import rag_graph
from app.exceptions.medical_record_errors import MedicalRecordFormatterError, MedicalRecordSummaryError, RagError
from app.exceptions.graph_errors import GraphRecursionLimitError
from app.ai.rag.models.MedicalRecord import MedicalRecord

async def invoke_ai_formatter(medical_record: str, medical_record_template: str):
    try:   
        ai = await medical_record_formatter_graph.graph()
        
        logging.info("[AI] Invoking medical record formatter")
        result = await ai.ainvoke({"input": medical_record, "medical_record_template": medical_record_template})
        
        logging.info("[AI] Success on medical record formatter")
        return result
    except GraphRecursionError as e:
        error_msg = "[AI] Graph recursion has reached the allowed limit"
        logging.error(error_msg)
        raise GraphRecursionLimitError(error_msg) from e
    except Exception as e:
        error_msg = f"[AI] Failed to format medical record: {str(e)}"
        logging.error(error_msg)
        raise MedicalRecordFormatterError(error_msg) from e
    

async def invoke_ai_summary(medical_record_markdown: str, summary_max_words: int):
    try:   
        ai = await summary_medical_record_graph.graph()
        
        logging.info("[AI] Invoking medical record summary")
        result = await ai.ainvoke({"medical_record_markdown": medical_record_markdown, "summary_max_words": summary_max_words})
        
        logging.info("[AI] Success on medical record summary")
        return result
    except GraphRecursionError as e:
        error_msg = "[AI] Graph recursion has reached the allowed limit"
        logging.error(error_msg)
        raise GraphRecursionLimitError(error_msg) from e
    except Exception as e:
        error_msg = f"[AI] Failed to summarize medical record: {str(e)}"
        logging.error(error_msg)
        raise MedicalRecordSummaryError(error_msg) from e
    
async def invoke_ai_rag(user_question: str, source_medical_records: list[MedicalRecord]):
    try:
        ai = await rag_graph.graph()
        logging.info("[AI] Invoking RAG")
        result = await ai.ainvoke({"user_question": user_question, "source_medical_records": source_medical_records, "relevant_question": "False"})
        logging.info("[AI] Success on RAG")
        return result
    except GraphRecursionError as e:
        error_msg = "[AI] Graph recursion has reached the allowed limit"
        logging.error(error_msg)
        raise GraphRecursionLimitError(error_msg) from e
    except Exception as e:
        error_msg = f"[AI] Failed to answer question: {str(e)}"
        logging.error(error_msg)
        raise RagError(error_msg) from e