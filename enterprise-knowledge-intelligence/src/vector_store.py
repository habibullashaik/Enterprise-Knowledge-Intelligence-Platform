from langchain_chroma import Chroma
from src.ids import generate_chunk_id
def create_vector_store(chunks, embedding_model):

    vector_store = Chroma(
        collection_name="enterprise_knowledge",
        embedding_function=embedding_model,
        persist_directory="data/chroma_db"
    )

    ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    return vector_store
