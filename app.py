import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Load environment
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

api_key = st.secrets["OPENAI_API_KEY"]


# Load vectorstore
VECTORSTORE_DIR = "vectorstore"
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

# Setup model
llm = ChatOpenAI(openai_api_key=api_key, temperature=0.3, model="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

# Streamlit UI
st.set_page_config(page_title="Q&A Chatbot")
st.title("📄 Project Q&A Chatbot")
st.write("Ask me questions about the PDF and SQL files you've uploaded!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Your question", placeholder="Ask something about the project...")

if user_input:
    response = qa_chain.run(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Show history
for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")
