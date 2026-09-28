
from src.vector_store import create_vector_store
from src.ingestion import load_document
from src.chunking import split_documents
from src.metadata import enrich_metadata
from src.embeddings import create_embedding_model
from src.ids import generate_chunk_id
from src.m25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.reranking import Reranker
from src.context_builder import build_context
from src.generation import generate_answer
from src.llm import create_llm


class RAGPipeline:

    def __init__(self):

        print("Initializing RAG pipeline...")

        # -----------------------------------
        # 1. DOCUMENT INGESTION
        # -----------------------------------

        self.documents = load_document(
            "data/documents/infoo.pdf"
        )

        print(
            f"Original documents: "
            f"{len(self.documents)}"
        )

        # -----------------------------------
        # 2. CHUNKING
        # -----------------------------------

        self.chunks = split_documents(
            self.documents
        )

        print(
            f"Total chunks: "
            f"{len(self.chunks)}"
        )

        # -----------------------------------
        # 3. METADATA ENRICHMENT
        # -----------------------------------

        self.chunks = enrich_metadata(
            self.chunks
        )

        # -----------------------------------
        # 4. GENERATE CHUNK IDs
        # -----------------------------------

        for chunk in self.chunks:

            chunk.metadata["chunk_id"] = (
                generate_chunk_id(chunk)
            )

        # -----------------------------------
        # 5. EMBEDDING MODEL
        # -----------------------------------

        self.embedding_model = (
            create_embedding_model()
        )

        # -----------------------------------
        # 6. CHROMA VECTOR STORE
        # -----------------------------------

        self.vector_store = create_vector_store(
            self.chunks,
            self.embedding_model
        )

        # -----------------------------------
        # 7. BM25 RETRIEVER
        # -----------------------------------

        self.bm25_retriever = BM25Retriever(
            self.chunks
        )

        # -----------------------------------
        # 8. HYBRID RETRIEVER
        # -----------------------------------

        self.hybrid_retriever = HybridRetriever(
            self.vector_store,
            self.bm25_retriever
        )

        # -----------------------------------
        # 9. RERANKER
        # -----------------------------------

        self.reranker = Reranker()

        # -----------------------------------
        # 10. LLM
        # -----------------------------------

        self.llm = create_llm()

        print("RAG pipeline initialized.")

    # =================================================
    # DENSE RETRIEVAL
    # =================================================

    def dense_retrieve(
        self,
        query,
        k=5
    ):

        return self.vector_store.similarity_search(
            query,
            k=k
        )

    # =================================================
    # SPARSE RETRIEVAL
    # =================================================

    def sparse_retrieve(
        self,
        query,
        k=5
    ):

        return self.bm25_retriever.search(
            query,
            k=k
        )

    # =================================================
    # HYBRID RETRIEVAL
    # =================================================

    def hybrid_retrieve(
        self,
        query,
        k=10
    ):

        return self.hybrid_retriever.search(
            query,
            k=k
        )

    # =================================================
    # RERANKING
    # =================================================

    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        reranked_results = self.reranker.rerank(
            query,
            documents,
            top_k=top_k
        )

        return reranked_results

    # =================================================
    # FINAL RETRIEVAL PIPELINE
    # =================================================

    def retrieve(
        self,
        query,
        candidate_k=10,
        final_k=5
    ):

        # -----------------------------------
        # Step 1: Hybrid Retrieval
        # -----------------------------------

        candidates = self.hybrid_retrieve(
            query,
            k=candidate_k
        )

        # -----------------------------------
        # Step 2: Reranking
        # -----------------------------------

        reranked_results = self.rerank(
            query,
            candidates,
            top_k=final_k
        )

        # -----------------------------------
        # Extract documents
        # -----------------------------------

        documents = [
            document
            for document, score
            in reranked_results
        ]

        return documents

    # =================================================
    # BUILD ANSWER
    # =================================================

    def build_answer(
        self,
        query
    ):

        # -----------------------------------
        # RETRIEVAL
        # -----------------------------------

        documents = self.retrieve(
            query,
            candidate_k=10,
            final_k=5
        )

        # -----------------------------------
        # CONTEXT BUILDING
        # -----------------------------------

        context = build_context(
            documents
        )

        # -----------------------------------
        # GENERATION
        # -----------------------------------

        answer = generate_answer(
            self.llm,
            query,
            context
        )

        # -----------------------------------
        # RETURN RESULT
        # -----------------------------------

        return {
            "answer": answer,
            "documents": documents,
            "context": context
        }

    # =================================================
    # PUBLIC ASK METHOD
    # =================================================

    def ask(
        self,
        query
    ):

        return self.build_answer(
            query
        )

    # =================================================
    # RETRIEVAL COMPARISON
    # =================================================

    def compare_retrieval(
        self,
        query
    ):

        dense_results = self.dense_retrieve(
            query,
            k=5
        )

        sparse_results = self.sparse_retrieve(
            query,
            k=5
        )

        hybrid_results = self.hybrid_retrieve(
            query,
            k=5
        )

        print(
            "\n=============================="
        )

        print(
            "DENSE RETRIEVAL"
        )

        print(
            "=============================="
        )

        for i, document in enumerate(
            dense_results,
            start=1
        ):

            print(
                f"\n{i}. "
                f"Page: "
                f"{document.metadata.get('page_label')}"
            )

            print(
                document.page_content[:300]
            )

        print(
            "\n=============================="
        )

        print(
            "BM25 RETRIEVAL"
        )

        print(
            "=============================="
        )

        for i, document in enumerate(
            sparse_results,
            start=1
        ):

            print(
                f"\n{i}. "
                f"Page: "
                f"{document.metadata.get('page_label')}"
            )

            print(
                document.page_content[:300]
            )

        print(
            "\n=============================="
        )

        print(
            "HYBRID RETRIEVAL"
        )

        print(
            "=============================="
        )

        for i, document in enumerate(
            hybrid_results,
            start=1
        ):

            print(
                f"\n{i}. "
                f"Page: "
                f"{document.metadata.get('page_label')}"
            )

            print(
                f"Chunk ID: "
                f"{document.metadata.get('chunk_id')}"
            )

            print(
                document.page_content[:300]
            )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    pipeline = RAGPipeline()

    query = (
        "What is Retrieval-Augmented Generation?"
    )

    result = pipeline.ask(
        query
    )

    print(
        "\n=============================="
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=============================="
    )

    print(
        result["answer"]
    )

    print(
        "\n=============================="
    )

    print(
        "SOURCES"
    )

    print(
        "=============================="
    )

    for i, document in enumerate(
        result["documents"],
        start=1
    ):

        print(
            f"\n[SOURCE {i}]"
        )

        print(
            f"File: "
            f"{document.metadata.get('file_name')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page_label')}"
        )

        print(
            f"Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

