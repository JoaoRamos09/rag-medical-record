from http import HTTPStatus
from typing import Dict, Type

from fastapi import Request
from fastapi.responses import JSONResponse

from app.exceptions.ai_service_errors import AIServiceError

class ErrorHandler:    
    ERROR_MAPPING: Dict[Type[Exception], int] = {
        AIServiceError: HTTPStatus.INTERNAL_SERVER_ERROR,
    }

    @staticmethod
    async def handle_exception(request: Request, exc: Exception) -> JSONResponse:
        error_code = ErrorHandler.ERROR_MAPPING.get(
            type(exc), 
            HTTPStatus.INTERNAL_SERVER_ERROR
        )

        error_response = {
            "success": False,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "cause": str(exc.__cause__) if exc.__cause__ else None,
                "code": error_code,
            }
        }

        return JSONResponse(
            status_code=error_code,
            content=error_response
        )