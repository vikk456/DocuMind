from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    loader = PyPDFLoader(file_path)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(document)

    for chunk in chunks:
        chunk.metadata["source"] = file_path

    return chunks