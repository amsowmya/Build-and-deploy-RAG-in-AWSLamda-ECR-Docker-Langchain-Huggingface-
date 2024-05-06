import os 
import sys 
import boto3 
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from langchain.vectorstores import FAISS 

from QASystem.ingestion import data_ingestion, get_vector_store
from QASystem.retrievalandgeneration import get_llama2_llm, get_response_llm

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def main():
    st.set_page_config("QA with doc")
    st.header("QA with Doc using langchain and AWSBedrock")

    user_question=st.text_input("Ask a question from pdf")

    with st.sidebar:
        st.title("Update or Create the vector store")
        if st.button("vectors update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

        if st.button("llama model"):
            with st.spinner("Processing..."):
                faiss_index=FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm=get_llama2_llm()

                st.write(get_response_llm(llm, faiss_index, user_question))
                st.success("Done")

if __name__=="__main__":
    main()