import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

# New function to retrieve context
def retrieve_context(retriever, question):
    relevant_docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in relevant_docs])
    print("the retrieved context is : ", context)
    return context

def get_conversational_chain(retriever, ques):
    #qa_model = pipeline("question-answering", model="distilbert-base-uncased")
    qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    context = retrieve_context(retriever, ques)  # Retrieve PDF context related to the question
    context = f"Based on the following text, answer the question accurately:\n\n{context}"
    response = qa_model(question=ques, context=context)
    print(response['answer'])
    st.write("Reply: ", response['answer'])

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    get_conversational_chain(retriever, user_question)

def main():
    st.set_page_config("Chat PDF")
    st.header("RAG-based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
