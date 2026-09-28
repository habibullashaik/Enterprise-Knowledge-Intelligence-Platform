import hashlib

def generate_chunk_id(chunk):
    source = chunk.metadata.get("source","")
    page = chunk.metadata.get("page",0)
    content = chunk.page_content
    raw_id = f"{source}_{page}_{content}"
    chunk_id = hashlib.md5(raw_id.encode('utf-8')).hexdigest()
    return chunk_id
