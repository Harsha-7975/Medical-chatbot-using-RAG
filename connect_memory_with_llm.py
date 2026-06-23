from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import storingdata_forllm
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


#llm:
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

#retreiver:
retreiver = storingdata_forllm.db.as_retriever(search_type = "similarity",search_kwargs = {"k":4})


#prompt
prompt = PromptTemplate(
    template = """
        You are a helpful medibot assistant.
        Answer only from the provided context. 
        If the context is insufficient, just say you dont know

        {context}
        Question: {question}
    """,
    input_variables=["context","question"]
)


#function for retreiver:
def format_docs(retreived_docs):
    context_text = "\n\n".join(doc.page_content for doc in retreived_docs)
    return context_text


#creating a chain
parallel_chain = RunnableParallel({
    "context": retreiver | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()
final_chain = RunnableSequence(parallel_chain,prompt,llm,parser)

print(final_chain.invoke("What is an LLM and machine learning?"))