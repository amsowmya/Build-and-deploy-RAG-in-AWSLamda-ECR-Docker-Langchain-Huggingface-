from setuptools import setup, find_packages

setup(
    name="QASystem",
    version="0.0.1",
    description="Gen ai application using Bedrock api",
    author="Sowmya AM",
    author_email="sowmya.anekonda@gmail.com",
    packages=find_packages(),
    install_requires=["langchainhub",
                      "bs4",
                      "tiktoken",
                      "openai",
                      "boto3==1.34.37",
                      "langchain_community",
                      "chromadb",
                      "awscli",
                      "streamlit",
                      "pypdf",
                      "faiss-cpu"]
)