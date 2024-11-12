import os
import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
openai_api_key=os.getenv["OPENAI_API_KEY"] 

# Load the documents
loader = CSVLoader(file_path='') #/content/propertydata.csv
docs = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Create embeddings
embeddings = OpenAIEmbeddings()
documents_content = [doc.page_content for doc in splits]
embeddings_list = embeddings.embed_documents(documents_content)

# Save embeddings and splits to disk
with open('embeddings.pkl', 'wb') as f:
    pickle.dump((embeddings_list, splits), f)
