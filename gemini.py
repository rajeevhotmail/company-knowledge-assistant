import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline

# 1. Load documents
if not os.path.exists('documents_dir'):
    raise FileNotFoundError("The 'documents_dir' directory does not exist.")

if not any(file.endswith('.txt') for file in os.listdir('documents_dir')):
    raise FileNotFoundError("No .txt files found in 'documents_dir'.")

loader = DirectoryLoader('documents_dir', glob="**/*.txt")
documents = loader.load()

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 3. Create embeddings and vectorstore
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# 4. Load the language model
model_name = "google/flan-t5-base"  # Or your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a HuggingFacePipeline
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

# 5. Set up the conversational chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# 6. Define functions for getting responses and the chat loop
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

# Start the chat loop
if __name__ == "__main__":
    chat_loop()