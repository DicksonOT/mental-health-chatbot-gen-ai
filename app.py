from flask import Flask , render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings 
import os
from dotenv import load_dotenv
from src.prompt import *
from store_index import text_chunks
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

app = Flask (__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "mentalbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings
    )
else:
    print("Index already exists.")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    google_api_key=GEMINI_API_KEY,
    temperature=0.3, 
)
   
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


retriever = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
).as_retriever(search_kwargs={"k": 3})


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('bot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        response = rag_chain.invoke({"input": msg})
        return str(response["answer"])
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 8080, debug = True)