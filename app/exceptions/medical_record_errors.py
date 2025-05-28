from app.exceptions.ai_service_errors import AIServiceError

class MedicalRecordFormatterError(AIServiceError):
    """Raised when there is an error formatting medical records"""
    pass

class MedicalRecordSummaryError(AIServiceError):
    """Raised when there is an error summarizing medical records"""
    pass

class RagError(AIServiceError):
    """Raised when there is an error on RAG"""
    pass

