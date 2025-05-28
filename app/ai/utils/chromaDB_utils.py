from langchain_chroma import Chroma
from chromadb.config import Settings
from app.exceptions.chroma_errors import ChromaErrorCreateCollection, ChromaErrorRetrieveDocuments

def create_chroma_db(embedding_function):
    try:
        return Chroma(embedding_function=embedding_function, collection_name="collection_name", client_settings=Settings(anonymized_telemetry=False))
    except Exception as e:
        raise ChromaErrorCreateCollection(f"Error on Chroma creating collection: {e}")
    
async def cleanup_chroma_db(chroma_db: Chroma, ids: list):
    await chroma_db.adelete(ids=ids)
    
async def find_most_relevant_chunks(chroma_db: Chroma, chunks: list, user_question: str, protocol_id: str, k: int):
    await chroma_db.aadd_documents(chunks)
    try:
        results = await chroma_db.asimilarity_search_with_score(
            user_question,
            k=k,
            filter={"protocol_id": protocol_id}
        )
        return results
    except Exception as e:
        raise ChromaErrorRetrieveDocuments(f"Error on Chroma retrieving documents: {e}")