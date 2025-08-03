import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load vectorstore
VECTORSTORE_DIR = "vectorstore"
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

# Set up retriever and LLM
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")

# Build QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False,
)

# Main chat loop
print("Ask questions about your project. Type 'exit' to quit.\n")
while True:
    query = input("Your question: ")
    if query.lower() in ['exit', 'quit']:
        break

    response = qa_chain.invoke({"question": query})
    print("\nAnswer:", response["answer"])
    print("-" * 50)
