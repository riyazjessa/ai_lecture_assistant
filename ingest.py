import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

# --- GLOBAL VARIABLES ---
TEXT_FILE_PATH = "lectures.txt"
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

def ingest_data():
    """Reads data, chunks it, creates embeddings, and uploads to Pinecone."""
    print("Starting data ingestion...")

    # 1. Load the document
    with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Loaded document.")

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(raw_text)
    print(f"Split text into {len(text_chunks)} chunks.")

    # 3. Initialize embeddings and Pinecone
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    print(f"Uploading {len(text_chunks)} chunks to Pinecone index '{PINECONE_INDEX_NAME}'...")
    # This command creates embeddings and uploads them to the cloud index.
    # It will replace any existing content in the index.
    PineconeVectorStore.from_texts(text_chunks, embeddings, index_name=PINECONE_INDEX_NAME)
    
    print("✅ Data ingestion complete!")

if __name__ == "__main__":
    ingest_data()