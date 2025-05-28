from pydantic import BaseModel
from app.dto.SimiliarityChunkDTO import SimiliarityChunkDTO

class MedicalRecordSearchResponseDTO(BaseModel):
    answer: str
    similarity_results: list[SimiliarityChunkDTO]

