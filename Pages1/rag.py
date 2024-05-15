import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os
# Load PDF document
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])


if uploaded_file is not None:
    # Read PDF content
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
       file.write(uploaded_file.getvalue())
       file_name = uploaded_file.name
    try:
        # Load PDF content using PyPDFLoader
        current_dir = os.path.dirname(os.path.abspath(__file__))

# Form the full file path to the PDF file
        pdf_file_path = os.path.join(current_dir, "attention.pdf")
        print(pdf_file_path, type(file_name))
        loader = PyPDFLoader(pdf_file_path)
        text_documents = loader.load()

        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(text_documents)

        # Create vector store from document embeddings
        db=Chroma.from_documents(documents[:20], OllamaEmbeddings())

        # User input: ask a question
        question = st.text_input("Ask a question about the document:")

        if st.button("Submit"):
            if not question:
                st.warning("Please enter a question.")
            else:
                # Perform similarity search based on the question
                retrieved_results = db.similarity_search(question)
                
                if retrieved_results:
                    # Display the content of the most relevant document
                    st.write("Top matching document:")
                    st.write(retrieved_results[0].page_content)
                    st.success("RAG status: Green")  # Placeholder RAG status
                else:
                    st.warning("No relevant information found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
