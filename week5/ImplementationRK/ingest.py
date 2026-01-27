import os
import glob
from pathlib import Path
from typing import Collection
from unittest import loader
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from torch import embedding
load_dotenv(override=True)
MODEL = "gpt-5-mini"
DB_NAME = str(Path(__file__).parent.parent / "vector.db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def fetch_documents():
    documents = []
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder,glob="**/*.md",loader_cls=TextLoader,loader_kwargs={"encoding":"UTF-8"})
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents

def create_chunks(documents):
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    chunks = textsplitter.split_documents(documents)
    return chunks
def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME,embedding_function=embeddings).delete_collection
    vector_store = Chroma.from_documents(persist_directory=DB_NAME,embedding=embeddings,documents=chunks)
    collection = vector_store._collection
    print(f"count of the vector collections for the documents {collection.count()}")
    sample_embedding = collection.get(limit=1,include=["embeddings"])["embeddings"][0]
    print(f"dimentions for the vector db are {sample_embedding.shape}")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print(f"Data ingestion is")