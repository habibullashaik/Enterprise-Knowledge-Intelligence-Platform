from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        pairs = []

        for document in documents:

            pairs.append(
                (
                    query,
                    document.page_content
                )
            )

        scores = self.model.predict(
            pairs
        )

        ranked_documents = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_documents[:top_k]
if __name__ == "__main__":

    from ingestion import load_document
    from chunking import split_documents
    from metadata import enrich_metadata
    from embeddings import create_embedding_model
    from vector_store import create_vector_store
    from m25_retriever import BM25Retriever
    from hybrid_retriever import HybridRetriever
    from ids import generate_chunk_id

    documents = load_document(
        "data/documents/infoo.pdf"
    )

    chunks = split_documents(documents)

    chunks = enrich_metadata(chunks)

    for chunk in chunks:

        chunk.metadata["chunk_id"] = (
            generate_chunk_id(chunk)
        )

    embedding_model = create_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    bm25_retriever = BM25Retriever(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        vector_store,
        bm25_retriever
    )

    query = (
        "What is Retrieval-Augmented Generation?"
    )

    candidates = hybrid_retriever.search(
        query,
        candidate_k=10,
        final_k=10
    )

    reranker = Reranker()

    results = reranker.rerank(
        query,
        candidates,
        top_k=5
    )

    print(
        "\n=============================="
    )

    print("RERANKED RESULTS")

    print(
        "=============================="
    )

    for i, (document, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{i}. Page: "
            f"{document.metadata.get('page_label')}"
        )

        print(
            f"Score: {score}"
        )

        print(
            document.page_content[:500]
        )