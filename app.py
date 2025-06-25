from flask import Flask, render_template, jsonify, request
import os
import traceback
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

# Enable CORS for development. For production, restrict origins.
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# --- Pinecone Index Setup and Document Upserting ---
index_name = "mentalbot"

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists and create if not
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, # Must match the dimension of 'all-MiniLM-L6-v2' embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1") # Adjust region if desired
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
    print("Pinecone retriever initialized.")

except Exception as e:
    print(f"ERROR: Failed to set up Pinecone or retriever: {e}")
    traceback.print_exc()
    exit("Pinecone setup failed. Cannot proceed.")

# --- Generative AI (LLM) Configuration ---
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3, 
    )
    print("Gemini LLM initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize Gemini LLM: {e}")
    traceback.print_exc()
    exit("Gemini LLM initialization failed. Cannot proceed.")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create the chains
try:
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("RAG chain successfully created.")
except Exception as e:
    print(f"ERROR: Failed to create RAG chains: {e}")
    traceback.print_exc()
    exit("RAG chain creation failed. Cannot proceed.")


# --- Flask Routes ---
@app.route("/")
def index():
    """Serves the main chatbot HTML page."""
    return render_template('bot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handles chatbot queries and returns responses."""
    try:
        msg = request.form["msg"]
        print(f"Received query: '{msg}'")

        # Invoke the RAG chain to get a response
        response = rag_chain.invoke({"input": msg})
        
        chatbot_answer = response["answer"]
        print(f"Sending response: '{chatbot_answer}'")
        return str(chatbot_answer) # Ensure the response is a string

    except Exception as e:
        # Capture and print the full error on the server side for debugging
        print(f"ERROR during RAG chain invocation: {str(e)}")
        traceback.print_exc()

        # Provide a more specific error message to the frontend if it's a quota error
        if "quota" in str(e).lower() or "resourceexhausted" in str(e).lower():
            return "Apologies, the chatbot is currently experiencing high demand. Please try again in a few moments.", 429
        else:
            return "An internal error occurred. Please try again later.", 500

# --- Run the Flask Application ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
