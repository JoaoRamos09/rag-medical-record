from pydantic import BaseModel

class SimiliarityChunkDTO(BaseModel):
    medical_record_id: int
    content: str
    similarity_score: float

