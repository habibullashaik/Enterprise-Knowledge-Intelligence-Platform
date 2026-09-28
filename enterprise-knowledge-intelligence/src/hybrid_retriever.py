
class HybridRetriever:

    def __init__(
        self,
        vector_store,
        bm25_retriever
    ):

        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    # -----------------------------
    # Dense Retrieval
    # -----------------------------

    def vector_search(
        self,
        query,
        k=10
    ):

        return self.vector_store.similarity_search(
            query,
            k=k
        )

    # -----------------------------
    # Sparse Retrieval - BM25
    # -----------------------------

    def bm25_search(
        self,
        query,
        k=10
    ):

        return self.bm25_retriever.search(
            query,
            k=k
        )

    # -----------------------------
    # Reciprocal Rank Fusion
    # -----------------------------

    def reciprocal_rank_fusion(
        self,
        result_lists,
        rrf_constant=60
    ):

        scores = {}
        documents = {}

        # Process each retriever's result list
        for results in result_lists:

            # rank starts from 1
            for rank, document in enumerate(
                results,
                start=1
            ):

                chunk_id = document.metadata[
                    "chunk_id"
                ]

                # Store document
                documents[chunk_id] = document

                # Initialize score
                if chunk_id not in scores:

                    scores[chunk_id] = 0

                # RRF formula
                scores[chunk_id] += (
                    1 / (
                        rrf_constant + rank
                    )
                )

        # Sort chunks by RRF score
        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True
        )

        # Convert chunk IDs back to Documents
        return [
            documents[chunk_id]
            for chunk_id in ranked_ids
        ]

    # -----------------------------
    # Hybrid Search
    # -----------------------------

    def search(self, query, k=5):

        vector_results = self.vector_search(
            query,
            k=k
        )

        bm25_results = self.bm25_search(
            query,
            k=k
        )

        fused_results = self.reciprocal_rank_fusion(
            [
                vector_results,
                bm25_results
            ]
        )

        return fused_results[:k]