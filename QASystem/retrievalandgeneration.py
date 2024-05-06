from langchain.chains.retrieval_qa.base import RetrievalQA 
from langchain_community.vectorstores import FAISS 
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
from QASystem.ingestion import get_vector_store, data_ingestion
from langchain_community.embeddings import BedrockEmbeddings

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

prompt_template="""
HUMAN: Use the following piece context to provide a 
concise answer to the question at the end but use atlease summarixe with 
250 words with detailed explanation. If you don't know the answer just say
that you don't know, don't try to make up an answer. 
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT=PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def get_llama2_llm():
    llm=Bedrock(model_id="meta.llama2-13b-chat-v1", 
                client=bedrock
                # model_kwargs={"max_tokens":512}
                )
    return llm

def get_response_llm(llm, vectorstore_faiss, query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )
    answer=qa({"query":query})
    return answer["result"]

if __name__=="__main__":
    query="what is the population in Canada?"
    # docs=data_ingestion()
    # vectorstore_faiss=get_vector_store(docs)
    faiss_index=FAISS.load_local("faiss_index", embeddings=bedrock_embeddings,allow_dangerous_deserialization=True)
    llm=get_llama2_llm()
    print(get_response_llm(llm, faiss_index, query))
