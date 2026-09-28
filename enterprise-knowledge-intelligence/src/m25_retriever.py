from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        tokenized_documents = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def search(self, query, k=3):

        tokenized_query = query.lower().split()

        results = self.bm25.get_top_n(
            tokenized_query,
            self.chunks,
            n=k
        )

        return results

if __name__ == "__main__":

    from ingestion import load_document
    from chunking import split_documents
    from metadata import enrich_metadata

    documents = load_document(
        "data/documents/infoo.pdf"
    )

    chunks = split_documents(documents)

    chunks = enrich_metadata(chunks)

    retriever = BM25Retriever(chunks)

    query = "What is Retrieval Augmented Generation?"

    results = retriever.search(
        query,
        k=3
    )

    for i, result in enumerate(results):

        print(f"\n--- RESULT {i + 1} ---")

        print(result.page_content)

        print("\nMetadata:")

        print(result.metadata)