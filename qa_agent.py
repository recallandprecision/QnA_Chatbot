# qa_agent.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into os.environ


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


VECTORSTORE_DIR = "vectorstore"

def create_qa_chain():
    api_key = os.getenv("OPENAI_API_KEY")
    print("api_key = ", api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

def answer_question(question):
    qa_chain = create_qa_chain()
    return qa_chain.run(question)

if __name__ == "__main__":
    print("Ask questions about your project. Type 'exit' to quit.")
    while True:
        query = input("\nYour question: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = answer_question(query)
        print("\nAnswer:", response)
