import vertexai
# from loggerUtil import logger
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold, Image, FinishReason, Part
from vertexai import generative_models
from google.oauth2 import service_account
import json, time
from google.api_core.exceptions import ResourceExhausted
# from bs4 import BeautifulSoup
# from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import BSHTMLLoader
from langchain.schema import Document
from PyPDF2 import PdfReader
import os
import pandas as pd
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter, HTMLSectionSplitter, HTMLSemanticPreservingSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from base64 import b64decode
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from pymilvus import MilvusClient, DataType, utility, Collection,connections
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus as Milvus_hybridsearch, BM25BuiltInFunction
import os
import base64
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
# from get_llm_embedding import embeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv(".env")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "active-complex-405121-431aab878614_2.json"
vertexai.init(project='active-complex-405121', location='us-central1')
gemini_safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}
llm = None
llm_name = os.environ["LLM_NAME"]
llm = VertexAI(model_name=llm_name,
                    max_tokens=8192,                    
                    safety_settings=gemini_safety_settings
                    )
print(llm)

def create_collection(config_1):    
    URI=os.environ["URI"]
    TOKEN=os.environ["TOKEN"]
    USERNAME=os.environ["USERNAME"]
    PASSWORD=os.environ["PASSWORD"]    
    embeddings = AzureOpenAIEmbeddings(api_version="2023-05-15",
                                       model=config_1["embeddings"],
                                       api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                       azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_EMBEDD"])
    client = MilvusClient(
            uri=URI, # Cluster endpoint obtained from the console
            token=USERNAME+":"+PASSWORD # API key or a colon-separated cluster username and password
        )
    print(client.list_collections())
    collection_name = config_1["name"]
    if collection_name in client.list_collections():
        print(f"{collection_name} already exists and loaded.")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="my_vector", 
            index_type="AUTOINDEX",
            metric_type=config_1["similarityMetric"]
        )
        vectore_store = Milvus(collection_name=collection_name,
                        embedding_function=embeddings,
                        connection_args={"uri": URI,"user": USERNAME,"password": PASSWORD,"secure": True},                            
                        consistency_level="Strong",
                        auto_id=True,                        
                        index_params=index_params)
        print(f"{collection_name} collection create.")
        return vectore_store
    else:
        print(f"Creating {collection_name}.")        
        if config_1["storage"] == "External":
            URI = config_1["vectorStoreUrl"]
            USERNAME = config_1["vectorStoreUsername"]
            PASSWORD = config_1["vectorStorePassword"]       
        
        if config_1["searchType"] == "hybridSearch":
            index_params = client.prepare_index_params()
            # Add indexes
            index_params.add_index(
                field_name="dense",
                index_name="dense_index",
                index_type="AUTOINDEX",
                metric_type=config_1["similarityMetric"],
                params={"nlist": 128},
            )

            index_params.add_index(
                field_name="sparse",
                index_name="sparse_index",
                index_type="AUTOINDEX",  # Index type for sparse vectors
                metric_type="IP",  # Currently, only IP (Inner Product) is supported for sparse vectors
                params={"drop_ratio_build": 0.2},  # The ratio of small vector values to be dropped during indexing
            )
            vectore_store = Milvus_hybridsearch(embedding_function=embeddings,
                                                builtin_function=BM25BuiltInFunction(),
                                                collection_name=config_1["name"],                                                
                                                connection_args={"uri": URI,"user": USERNAME,"password": PASSWORD,"secure": True},
                                                consistency_level="Strong",
                                                enable_dynamic_field=True,
                                                auto_id=True,
                                                index_params=index_params,
                                            )      

        print(vectore_store)
        return vectore_store
    
def HTML_get_chunked_documents(file_path,chunk_size,chunk_overlap):
    #headers_to_split_on = [("h1", "Header 1"),("h2", "Header 2"),("h3", "Header 3"),("h4", "Header 4"), ]
    headers_to_split_on = []
    bs4_transformer = BeautifulSoupTransformer()
    docs_transformed = bs4_transformer.transform_documents(docs)

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    html_header_splits = html_splitter.split_text_from_file(file_path)    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(html_header_splits)
    return split_docs
    
def CSV_Xl_get_chunked_documents(df,chunk_by, chunk_size, chunk_overlap):
    if chunk_by == "RowWise":
        chunks = []
        for i in range(0, len(df), chunk_size - chunk_overlap):
            chunk_df = df.iloc[i:i + chunk_size]
            chunk_content = "\n".join(["\t".join(chunk_df.columns.tolist())] + ["\t".join(row) for row in chunk_df.astype(str).values.tolist()])
            chunks.append(chunk_content)

    elif chunk_by == "ColumnWise":
        chunks = []
        for i in range(0, len(df.columns), chunk_size - chunk_overlap):
            chunk_columns = df.iloc[:, i:i + chunk_size]
            chunk_content = "\n".join(["\t".join(chunk_columns.columns.tolist())] + ["\t".join(row) for row in chunk_columns.astype(str).values.tolist()])
            chunks.append(chunk_content)

    else:
        raise ValueError("chunk_by must be either 'row' or 'column'.")

    # Convert chunks to LangChain documents
    documents = [Document(page_content=str(chunk)) for chunk in chunks]
    return documents

def insert_data(config, documents):
    URI=os.environ["URI"]
    TOKEN=os.environ["TOKEN"]  
    USERNAME=os.environ["USERNAME"]
    PASSWORD=os.environ["PASSWORD"]    
    client = MilvusClient(
            uri=URI, # Cluster endpoint obtained from the console
            token=USERNAME+":"+PASSWORD # API key or a colon-separated cluster username and password
        )
    print(client.list_collections())
    collection_name = config["name"]
    embeddings = AzureOpenAIEmbeddings(api_version="2023-05-15",
                                       model=config["embeddings"],
                                       api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                       azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_EMBEDD"])
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="my_vector", 
        index_type="AUTOINDEX",
        metric_type=config["similarityMetric"])
    if collection_name in client.list_collections():        
        vectore_store = Milvus.from_documents(collection_name=collection_name,
                        embedding=embeddings,
                        documents=documents,
                        connection_args={"uri": URI,"user": USERNAME,"password": PASSWORD,"secure": True},                            
                        consistency_level="Strong",
                        #auto_id=True,                        
                        index_params=index_params)
    else:
        vectore_store = Milvus(collection_name=collection_name,
                        embedding_function=embeddings,                        
                        connection_args={"uri": URI,"user": USERNAME,"password": PASSWORD,"secure": True},                            
                        consistency_level="Strong",
                        auto_id=True,                        
                        index_params=index_params)
        vectore_store.add_documents(documents)
    print(f"Data ingested.")
    return f"{collection_name} created and documents inserted" 

        
    

def do_chunking(documents,chunk_method, chunk_size, chunk_overlap):  
    chunk_docs = None
    if chunk_method == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        chunk_docs = text_splitter.split_documents(documents)
    elif chunk_method == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)        
        chunk_docs = text_splitter.split_documents(documents)
        print(chunk_docs)
    elif chunk_method == "TokenTextSplitter":
        text_splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        chunk_docs = text_splitter.split_documents(documents)
    else:
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
        chunk_docs = text_splitter.split_documents(documents)    
    return chunk_docs
def convert_to_documents(job_id, fileName, chunkingMethod, chunk_size,chunk_overlap):
    """Convert a file to a LangChain Document."""   
    filename, ext = os.path.splitext(os.path.basename(fileName))  
    chunked_docs=None
    Proj_path = os.environ["PROJ_PATH"]
    dir_path = os.path.join(Proj_path, job_id)
    file_path = dir_path+"\\"+fileName
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()        
        chunked_docs = do_chunking(docs,chunkingMethod,chunk_size,chunk_overlap)
    elif ext == '.doc':
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        chunked_docs = do_chunking(docs,chunkingMethod,chunk_size,chunk_overlap)
    elif ext == '.docx':
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        chunked_docs = do_chunking(docs,chunkingMethod,chunk_size,chunk_overlap)
    elif ext == '.txt':
        loader = TextLoader(file_path)
        docs = loader.load()
        chunked_docs = do_chunking(docs,chunkingMethod,chunk_size,chunk_overlap)
    elif ext == '.html':
        loader = UnstructuredHTMLLoader(file_path)
        docs = loader.load()
        chunked_docs = do_chunking(docs,chunkingMethod,chunk_size,chunk_overlap)
        # chunked_docs = HTML_get_chunked_documents(file_path,chunk_size,chunk_overlap)        
    elif ext == '.xml':
        loader = UnstructuredXMLLoader(file_path)
        docs = loader.load()
        chunked_docs = do_chunking(docs, chunkingMethod,chunk_size,chunk_overlap)
    elif ext == ".csv":
        loader = CSVLoader(file_path=file_path)
        docs = loader.load()
        chunked_docs = do_chunking(docs,chunkingMethod,chunk_size,chunk_overlap)
    elif ext in ['.xlsx', '.xls']:       
        df = pd.read_excel(file_path)
        chunked_docs = CSV_Xl_get_chunked_documents(df,chunkingMethod, chunk_size, chunk_overlap)    
    elif ext == ".json":
        loader = JSONLoader(file_path=file_path,jq_schema=".",text_content=False,is_content_key_jq_parsable=True)
        docs = loader.load()
        chunked_docs = do_chunking(docs,chunkingMethod,chunk_size,chunk_overlap)
    else:
        raise ValueError(f"Unsupported file type: {ext}")    
    return chunked_docs

def base64_to_file(base64_string, filename, job_id):
    Proj_path = os.environ["PROJ_PATH"]  
    file_path = os.path.join(Proj_path, str(job_id))
    os.makedirs(file_path, exist_ok=True)
    fp = file_path+"\\"+filename
    file_data = b64decode(base64_string, validate=True)    
    # Write the bytes to a file in binary mode
    with open(fp, "wb") as f:
        f.write(file_data)
        f.close()
    return None


def get_retriever(config): 
    URI=os.environ["URI"]
    TOKEN=os.environ["TOKEN"]  
    USERNAME=os.environ["USERNAME"]
    PASSWORD=os.environ["PASSWORD"]  
    
    client = MilvusClient(
            uri=URI, # Cluster endpoint obtained from the console
            token=USERNAME+":"+PASSWORD # API key or a colon-separated cluster username and password
        )
    print(client.list_collections())
    collection_name = config["collectionName"]
    if config["collectionName"] not in client.list_collections():        
        return f"{collection_name} does not exists. Please create a collection first."
    else:
        retriever = None
        retrieverType = config["retrieverType"]        
        searchType="mmr",
        search_kwargs = {'k': 5,}
        embeddings = AzureOpenAIEmbeddings(api_version="2023-05-15",
                                       model=config["embeddings"],
                                       api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                       azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_EMBEDD"])   
        vectore_store = Milvus(collection_name=collection_name,
                               embedding_function=embeddings,
                               connection_args={"uri": URI,"token":TOKEN,"secure": True})

        if retrieverType == "MultiQueryRetriever":        
            retriever = MultiQueryRetriever.from_llm(retriever=vectore_store.as_retriever(), llm=llm)
        elif retrieverType == "BM25Retriever":
            retriever=vectore_store.as_retriever()              
        elif retrieverType == "EnsembleRetriever": 
            retriever_base=vectore_store.as_retriever()       
            retriever_MQ = MultiQueryRetriever.from_llm(retriever=vectore_store.as_retriever(), llm=llm)             
            retriever = EnsembleRetriever(retrievers=[retriever_base, retriever_MQ], weights=[0.5, 0.5])
        elif retrieverType == "BasicRetriever":
            retriever=vectore_store.as_retriever()        
        return retriever

def delete_document(document_id, collection_name):
    print(f'Deleting {document_id} from collection {collection_name}')
    URI=os.environ["URI"]
    TOKEN=os.environ["TOKEN"]
    USERNAME=os.environ["USERNAME"]
    PASSWORD=os.environ["PASSWORD"]
    client = MilvusClient(
		uri=URI, # Cluster endpoint obtained from the console
		token=USERNAME+":"+PASSWORD # API key or a colon-separated cluster username and password
	)
    print(client.list_collections())
    if collection_name in client.list_collections():               
        res = client.delete(
                collection_name=collection_name,
                filter="doc_id like '%{}%'".format(document_id)
            )               
        print(f"Deleted {document_id} from collection {collection_name}")
        return res
    else:
        print(f'{collection_name} does not exists!')
def delete_collection(collection_name):
    URI=os.environ["URI"]
    TOKEN=os.environ["TOKEN"]
    USERNAME=os.environ["USERNAME"]
    PASSWORD=os.environ["PASSWORD"]
    client = MilvusClient(
		uri=URI, # Cluster endpoint obtained from the console
		token=USERNAME+":"+PASSWORD # API key or a colon-separated cluster username and password
	)
    status = client.drop_collection(collection_name=collection_name)
    return status

def extract_web_data(config):
    fc_api_key = os.environ["FC_API_KEY"]
    loader = FireCrawlLoader(api_key=fc_api_key,                        
                            url=config["url"],
                            mode=config["mode"],                         
                            params={"maxDepth":config["maxDepth"],})
    docs = loader.load()
    for i in docs:
        i.metadata = {'sourceUrl':i.metadata['sourceURL']}
    return docs
