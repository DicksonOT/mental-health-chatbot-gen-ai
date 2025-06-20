from src.helper import  load_pdf_file, text_split
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


extracted_data = load_pdf_file(data='C:/Users/Osei Tutu Dickson/Desktop/Gen AI/mental-health-chatbot-gen-ai/Data/')
text_chunks = text_split(extracted_data)
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