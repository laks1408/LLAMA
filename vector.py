from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("DATASET AIRPORT 2 - Sheet1.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(page_content=row["text"], metadata={"source": row["source"]}, id=str(i))

        ids.append(str(i))
        documents.append(document)

vectorstore = Chroma(
    collection_name="airport",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vectorstore.add_documents(documents=documents, ids=ids)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})