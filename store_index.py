# 
# store_index.py
# This file is responsible for loading and splitting PDF documents,
# making the 'text_chunks' available for import.
# It should NOT initialize embeddings or interact with Pinecone directly,
# as that will be handled in app.py to prevent double-loading.

import os
from src.helper import load_pdf_file, text_split # Assuming src/helper.py is correctly set up

# Define the path to your data directory
# IMPORTANT: Ensure your PDF mental health book is located in this directory.
PDF_DATA_DIRECTORY = 'C:/Users/Osei Tutu Dickson/Desktop/Gen AI/mental-health-chatbot-gen-ai/Data/'

# Load and split documents to get text_chunks
# This code will execute only once when store_index.py is first imported
print("Loading and splitting PDF documents for indexing...")
extracted_data = load_pdf_file(data=PDF_DATA_DIRECTORY)
text_chunks = text_split(extracted_data)

if not text_chunks:
    print("WARNING: No text chunks were created in store_index.py. "
          "The RAG system might not have content for retrieval.")

