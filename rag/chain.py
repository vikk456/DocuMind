from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

template = """
answer the question based on the context provided

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def execute_chain(retriever, question):
    pipeline = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt | llm | StrOutputParser()
    )
    return pipeline.invoke(question)


def get_sources(retriever, question):
    return retriever.invoke(question)
