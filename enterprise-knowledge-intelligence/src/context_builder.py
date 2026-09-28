def build_context(documents):

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        metadata = document.metadata

        source = metadata.get(
            "file_name",
            "Unknown"
        )

        page = metadata.get(
            "page_label",
            metadata.get("page", "Unknown")
        )

        chunk_id = metadata.get(
            "chunk_id",
            "Unknown"
        )

        context_parts.append(
            f"""
[SOURCE {index}]
Document: {source}
Page: {page}
Chunk ID: {chunk_id}

{document.page_content}
"""
        )

    return "\n".join(context_parts)