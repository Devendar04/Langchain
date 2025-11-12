from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps people find information."),
        ('user', "Question: {question}"),
    ]
)
## streamlit app
st.title("Chatbot Application")
input_question = st.text_input("Enter your question:")

llm= Ollama(model='llama2:7b')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
if input_question:
    st.write(chain.invoke({"question": input_question}))