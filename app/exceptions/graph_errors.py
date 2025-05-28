from app.exceptions.ai_service_errors import AIServiceError

class GraphRecursionLimitError(AIServiceError):
    """Raised when graph recursion reaches the allowed limit"""
    pass