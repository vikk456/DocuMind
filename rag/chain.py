from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """
answer the question based on the context provided

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables = ["context" , "question"],
    template = template
)


llm = ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash",
    temperature = 0.2
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def execute_chain(retriever , question):
    pipeline = (
        {
            "context": retriever | format_docs,
            "question" : RunnablePassthrough()
        } 
        | prompt | llm | StrOutputParser()
    )
    return pipeline.invoke(question)