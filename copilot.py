import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load documents from a directory
def load_documents(directory_path):
    loader = DirectoryLoader(directory_path, loader_mapping={
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader
    })
    documents = loader.load()
    return documents

# Split documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

# Embed documents and create a vectorstore
def create_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Initialize the language model and the tokenizer
def initialize_model(model_name="facebook/opt-125m"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Create a conversational retrieval chain
def create_qa_chain(model, vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain

# Function to get a response from the QA chain
def get_response(qa_chain, query):
    result = qa_chain({"question": query})
    return result["answer"]

# Main chat loop function
def chat_loop(qa_chain):
    print("Company Knowledge Assistant Ready! (Type 'exit' to quit)")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            print("Thanks for using the Company Knowledge Assistant!")
            break
        response = get_response(qa_chain, query)
        print("\nAssistant:", response)

if __name__ == "__main__":
    # Load and process documents
    documents = load_documents("documents_dir")
    texts = split_documents(documents)
    vectorstore = create_vectorstore(texts)

    # Initialize the model and create the QA chain
    tokenizer, model = initialize_model("facebook/opt-125m")
    qa_chain = create_qa_chain(model, vectorstore)

    # Start the chat loop
    chat_loop(qa_chain)
