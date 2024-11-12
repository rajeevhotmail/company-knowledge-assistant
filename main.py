from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

# Load the text file with explicit UTF-8 encoding
loader = TextLoader("documents_dir/resume.txt", encoding='utf-8')

# Create vectorstore from documents
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Load the base model
#model_name = "mistralai/Mistral-7B-v0.1"
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

def get_response(query):
    result = qa_chain({"question": query})
    return result["answer"]

def chat_loop():
    print("Company Knowledge Assistant Ready! (Type 'exit' to quit)")
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            print("Thanks for using the Company Knowledge Assistant!")
            break
        response = get_response(query)
        print("\nAssistant:", response)

# Start the chat
chat_loop()
