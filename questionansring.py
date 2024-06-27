### Using OpenAI chat model and Chroma vector store

#%pip install --upgrade --quiet langchain langchain-community langchainhub langchain-openai chromadb bs4

import streamlit as st
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
urls = ["https://www.gov.uk/government/news/cma-finds-fundamental-concerns-in-housebuilding-market"]
loader = WebBaseLoader(
    web_paths=(urls))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key="GghjqgqksYY5UvFkvI0WeNhjr6OKsH36J7MXd4fN"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key="AIzaSyCwkcTLWBJ90waMgC4juXdQjvwX_vkoKDs")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser())

if prompt:=st.chat_input("Enter question"):
    with st.spinner("Please wait."):
        st.write(rag_chain.invoke(prompt))