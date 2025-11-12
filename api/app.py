from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

app = FastAPI(
    title="Chatbot API",
    description="An API for a chatbot using Google Generative AI",
    version="1.0.0"
)

add_routes(
    app,
    ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
    path="/chat-google-genai"
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = Ollama(model='llama2:7b')

prompt1 = ChatPromptTemplate.from_template("Explain the following concept in simple terms: {text}")
prompt2 = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")

add_routes(
    app,
    prompt1|model,
    path="/concept-explanation"
)
add_routes(
    app,
    prompt2|llm,
    path="/text-summarization"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)