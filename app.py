import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tiktoken  # For managing token counts

load_dotenv()

# Initialize LLM with OpenAI API key
llm = OpenAI(openai_api_key=os.getenv("OPEN_API_KEY"))

# Initialize OpenAI embeddings model
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))

def extract_text_from_pdf(pdf_file):
    # Extract text from the uploaded PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_text = pdf_reader.pages[page_num].extract_text()
        if page_text:
            text += page_text
    return text if text else ""

def create_retriever_from_document(document_text):
    # Split the text into much smaller chunks (token count controlled)
    text_splitter = CharacterTextSplitter(separator = "\n", chunk_size=1000, chunk_overlap=200, length_function = len)  # Much smaller chunks
    chunks = text_splitter.split_text(document_text)

    # Create Document objects for each chunk
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Use OpenAI embeddings to create a FAISS vector store
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    # Create a retriever object from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Limit to 1 chunk
    return retriever

def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)

def get_langchain_response(llm, retriever, question):
    # Create a Retrieval-based QA chain with strict token management
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    # Retrieve the most relevant chunk
    relevant_chunk = retriever.get_relevant_documents(question)[0].page_content

    # Strictly control the tokens by ensuring the question + chunk don't exceed the limit
    token_limit = 4097
    question_tokens = count_tokens(question)

    # Calculate the number of tokens the chunk can have
    max_chunk_tokens = token_limit - question_tokens - 256  # Reserve 256 for completion

    # Trim the relevant chunk if necessary to fit within the token limit
    relevant_chunk_tokens = count_tokens(relevant_chunk)

    if relevant_chunk_tokens > max_chunk_tokens:
        relevant_chunk = relevant_chunk[:max_chunk_tokens]

    # Rebuild the document and pass to the LLM for an answer
    result = qa_chain({"query":question, "input_documents": [Document(page_content = relevant_chunk)]})
    # result = qa_chain.run(question=question, input_documents=[Document(page_content=relevant_chunk)])
    return result["result"]

# Streamlit frontend
st.title("Conversational PDF Chatbot")

uploaded_pdf = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_pdf is not None:
    document_text = extract_text_from_pdf(uploaded_pdf)
    st.write("PDF Uploaded successfully. Please ask your questions.")
    
    # Create the retriever from the document text
    retriever = create_retriever_from_document(document_text)
    
    user_question = st.text_input("Ask a question")

    col1, col2 = st.columns([1,1])

    # with col1:
        
    if st.button("Generate"):
        if user_question:
            answer = get_langchain_response(llm, retriever, user_question)
            st.write(f"Answer: {answer}")
    
    # with col2:
    if st.button("Quit"):
        st.write("Thank you for using the app!")
