from langchain_community.document_loaders import PyPDFLoader

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents
