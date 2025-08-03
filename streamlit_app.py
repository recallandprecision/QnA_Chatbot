import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API key securely
api_key = st.secrets["OPENAI_API_KEY"]

# Load the FAISS index
VECTORSTORE_DIR = "vectorstore"
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up Conversational Retrieval QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Streamlit UI
st.title("📚 Project Q&A Chatbot")
st.markdown("Ask questions based on your uploaded documents and SQL files.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question...")
if user_input:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", result["answer"]))

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.text(f"**{speaker}:** {message}")
    else:
        st.text(f"> **{speaker}:** {message}")
