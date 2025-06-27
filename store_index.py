from src.helper import load_pdf_file, text_split

PDF_DATA_DIRECTORY = 'C:/Users/Osei Tutu Dickson/Desktop/Gen AI/mental-health-chatbot-gen-ai/Data/'

print("Loading and splitting PDF documents for indexing...")
extracted_data = load_pdf_file(data=PDF_DATA_DIRECTORY)
text_chunks = text_split(extracted_data)

if not text_chunks:
    print("WARNING: No text chunks were created in store_index.py. "
          "The RAG system might not have content for retrieval.")

