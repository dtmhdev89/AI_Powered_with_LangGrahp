import os
from dotenv import load_dotenv
load_dotenv()

# tag::llm[]
# Create the LLM
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

# llm = ChatOpenAI(
#     openai_api_key=os.getenv('OPENAI_API_KEY'),
#     model_name="gpt-3.5-turbo"
# )
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
    base_url=os.getenv("OLLAMA_CONNECT_URL")
)

# end::llm[]

# tag::embedding[]
# Create the Embedding model
# from langchain_openai import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings(
#     openai_api_key=os.getenv('OPENAI_API_KEY')
# )

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# end::embedding[]
