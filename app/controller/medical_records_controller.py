from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.dto.MedicalRecordFormatInputDTO import MedicalRecordFormatInputDTO
from app.dto.SummaryMedicalRecordInputDTO import SummaryMedicalRecordInputDTO
from app.dto.MedicalRecordSearchInputDTO import MedicalRecordSearchInputDTO
from app.services.ai_service import invoke_ai_formatter, invoke_ai_summary, invoke_ai_rag
from app.dto.SimiliarityChunkDTO import SimiliarityChunkDTO
from app.dto.MedicalRecordSearchResponseDTO import MedicalRecordSearchResponseDTO
import logging

medical_records_router = APIRouter(prefix="/medical-records")

@medical_records_router.post("/format")
async def format_medical_record(request: MedicalRecordFormatInputDTO):
    try:
        result = await invoke_ai_formatter(medical_record=request.medical_record, medical_record_template=request.medical_record_template)
        return JSONResponse(content=result["formatted_medical_record"], status_code=200)
    except ValueError as e:
        logging.error(f"[AI] Request invalid: {e}")
        return JSONResponse(content={"status": "error", "message": "Validation Error"}, status_code=400)
    except TypeError as e:
        logging.error(f"[AI] Request invalid: {e}")
        return JSONResponse(content={"status": "error", "message": "Validation Error"}, status_code=400)
    except Exception as e:
        logging.error(f"[AI] Internal server error: {e}")
        return JSONResponse(content={"status": "error", "message": "Internal server error"}, status_code=500)
    
@medical_records_router.post("/summary")
async def format_medical_record(request: SummaryMedicalRecordInputDTO):
    try:
        result = await invoke_ai_summary(medical_record_markdown=request.medical_record_markdown, summary_max_words=request.summary_max_words)
        return JSONResponse(content=result["summary"], status_code=200)
    except ValueError as e:
        logging.error(f"[AI] Request invalid: {e}")
        return JSONResponse(content={"status": "error", "message": "Validation Error"}, status_code=400)
    except TypeError as e:
        logging.error(f"[AI] Request invalid: {e}")
        return JSONResponse(content={"status": "error", "message": "Validation Error"}, status_code=400)
    except Exception as e:
        logging.error(f"[AI] Internal server error: {e}")
        return JSONResponse(content={"status": "error", "message": "Internal server error"}, status_code=500)

@medical_records_router.post("/search", response_model=MedicalRecordSearchResponseDTO, status_code=status.HTTP_200_OK)
async def searchInPatientMedicalRecords(request: MedicalRecordSearchInputDTO):
    try:
        result = await invoke_ai_rag(user_question=request.search, source_medical_records=request.medical_records_source_list)
       
        similarity_results = [SimiliarityChunkDTO(medical_record_id=chunk[0].metadata["id_medical_record"], content=chunk[0].page_content, similarity_score=chunk[1]) for chunk in result["most_relevant_chunks"]]
        
        #todo: criar um DTO para essa response
        return MedicalRecordSearchResponseDTO(answer=result["generated_answer"], similarity_results=similarity_results)
    except ValueError as e:
        logging.error(f"[AI] Request invalid: {e}")
        raise HTTPException(status_code=400, detail="Validation Error")
    except TypeError as e:
        logging.error(f"[AI] Request invalid: {e}")
        raise HTTPException(status_code=400, detail="Validation Error")
    except Exception as e:
        logging.error(f"[AI] Internal server error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

