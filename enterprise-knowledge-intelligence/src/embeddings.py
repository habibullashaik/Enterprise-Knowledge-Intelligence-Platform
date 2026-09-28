from langchain_huggingface import HuggingFaceEmbeddings

def create_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5"
    )

    return embedding_model
if __name__ == "__main__":

    embedding_model = create_embedding_model()

    text = "Employees can work remotely up to three days per week."

    vector = embedding_model.embed_query(text)

    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])