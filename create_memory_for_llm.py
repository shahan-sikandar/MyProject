#from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS
#from langchain.docstore.document import Document

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


# Step 1: Load raw PDF(s)
#DATA_PATH="data/"
#def load_pdf_files(data):
    #loader = DirectoryLoader(data,
                             #glob='*.pdf',
                             #loader_cls=PyPDFLoader)
    
    #documents=loader.load()
    #return documents

#documents=load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ", len(documents))

# Step 2: Create Chunks
#def create_chunks(extracted_data):
    #text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 #chunk_overlap=50)
    #text_chunks=text_splitter.split_documents(extracted_data)
    #return text_chunks

#text_chunks=create_chunks(extracted_data=documents)
#print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings 

#def get_embedding_model():
    #embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #return embedding_model

#embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
#DB_FAISS_PATH="vectorstore/db_faiss"
#db=FAISS.from_documents(text_chunks, embedding_model)
#db.save_local(DB_FAISS_PATH)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import pandas as pd
import os

# ---- Step 1: Load PDF documents ----
def load_pdf_files():
    pdf_files = [
        "data/mhGAP_intervention_Guide.pdf",
        "data/Mental_Health.pdf"
    ]
    documents = []
    for file in pdf_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            docs = loader.load()
            # Tag PDFs
            for d in docs:
                d.metadata["source"] = os.path.basename(file)
            documents.extend(docs)
        else:
            print(f"⚠️ File not found: {file}")
    return documents

pdf_documents = load_pdf_files()

# ---- Step 2: Load CSV as text documents ----
def load_csv_file():
    csv_path = "data/self_care_tips.csv"
    documents = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            text = f"Self-care tip: {row['type']} | Category: {row['text']}"
            documents.append(Document(page_content=text, metadata={"source": "self_care_tips.csv"}))
    else:
        print(f"⚠️ CSV not found: {csv_path}")
    return documents

csv_documents = load_csv_file()

# ---- Step 3: Combine all documents ----
all_documents = pdf_documents + csv_documents
print(f"✅ Loaded {len(pdf_documents)} PDF docs and {len(csv_documents)} CSV docs. Total: {len(all_documents)}")

# ---- Step 4: Create Chunks (only for PDFs, not CSVs) ----
def create_chunks(extracted_data, chunk_pdfs=True):
    pdfs, csvs = [], []
    for doc in extracted_data:
        if doc.metadata.get("source", "").endswith(".pdf"):
            pdfs.append(doc)
        else:
            csvs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    pdf_chunks = text_splitter.split_documents(pdfs)

    return pdf_chunks + csvs  # CSVs are already short, don’t split

text_chunks = create_chunks(all_documents)

# ---- Step 5: Embeddings and FAISS ----
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_FAISS_PATH = "vectorstore/db_faiss"

db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("✅ Vectorstore created and saved at:", DB_FAISS_PATH)
