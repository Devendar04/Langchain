import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["USER_AGENT"] = "langchain-app/1.0"
GROQ_API_KEY = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama2:7b")
    st.session_state.loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
    st.session_state.vector = FAISS.from_documents(split_docs, st.session_state.embeddings)

st.title("AI Document Retriever")
llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b")

# ✅ Fix: use {question} instead of {input}
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context below,
    providing detailed and informative responses.
    {context}
    
    Question: {question}
    """
)

retriever = st.session_state.vector.as_retriever()

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt}
)

user_question = st.text_input("Enter your question about Artificial Intelligence:")

if user_question:
    start = time.process_time()
    result = retrieval_chain({"question": user_question, "chat_history": []})
    st.write("Response:", result['answer'])
    st.write(f"Time taken: {time.process_time() - start:.2f} seconds")

    with st.expander("Source Documents"):
        for i, doc in enumerate(result['source_documents']):
            st.write(doc.page_content)
            st.write("--------------------------------")
