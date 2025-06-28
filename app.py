from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings 
from pinecone import Pinecone, ServerlessSpec
from src.prompt import system_prompt 
from store_index import text_chunks 
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Pinecone Index Setup 
index_name = "mentalbot"

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        if text_chunks:
            docsearch = PineconeVectorStore.from_documents(
                documents=text_chunks, 
                index_name=index_name,
                embedding=embeddings 
            )
    else:
        print(f"Pinecone index '{index_name}' already exists.")

    # Initialize retriever from the existing Pinecone index (used for querying)
    retriever = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings 
    ).as_retriever(search_kwargs={"k": 3}) 

except Exception as e:
    print(f"ERROR: Failed to set up Pinecone or retriever: {e}")

# Generative AI (LLM) Configuration 
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3, 
    )
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini LLM: {e}")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# Creating chains
try:
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
except Exception as e:
    print(f"ERROR: Failed to create RAG chains: {e}")


# Flask Routes 
@app.route("/")
def index():
    """Serves the main chatbot HTML page."""
    return render_template('bot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handles chatbot queries and returns responses."""
    try:
        msg = request.form["msg"]
        response = rag_chain.invoke({"input": msg})
        
        answer = response["answer"]
        return str(answer)

    except Exception as e:
        print(f"ERROR during RAG chain invocation: {str(e)}")

        if "quota" in str(e).lower() or "resourceexhausted" in str(e).lower():
            return "Apologies, the chatbot is currently experiencing high demand. Please try again later."
        else:
            return "An internal error occurred. Please try again later."

# Running Flask Application 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
