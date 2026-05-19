from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_vector_store(chunks):
    embedder = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(chunks, embedder)
    return vector_store.as_retriever()