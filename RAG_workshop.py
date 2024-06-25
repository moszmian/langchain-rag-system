import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

example_documents = """
Today I woke up and drank some lemon water with Mati.
A year ago I finished my studies in Warsaw at 8:00.
In two months I am going to Mars to touch some grass.
Later today I will take a train back to Poznan.
I haven't seen another human being in three weeks.
Right now I am on the verge of losing touch with reality. 
"""

with open("my_diary.txt", "w") as file:
    file.write(example_documents)

# Loading and splitting the document
loader = TextLoader("my_diary.txt")
documents = loader.load()

# in langchain "document" means a text fragment
# print(documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(texts, embeddings)

qa_model = pipeline(task="question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

def run_rag_system(query):
    # retrieve relevant docs
    docs = vector_store.similarity_search(query, k=3) # k= how many fragments we want to take
    context = "\n".join([doc.page_content for doc in docs])

    # generate answer 
    result = qa_model(question = query, context=context)
    answer = result["answer"]
    print(answer)

run_rag_system("What will I do later?")
#run_rag_system("When did I finish my studies?")
#run_rag_system("What did I do today?")