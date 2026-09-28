from pathlib import Path


def enrich_metadata(chunks):

    for chunk in chunks:

        source = chunk.metadata.get("source")

        if source:
            file_path = Path(source)

            chunk.metadata["file_name"] = file_path.name
            chunk.metadata["file_type"] = file_path.suffix.lower()

    return chunks


