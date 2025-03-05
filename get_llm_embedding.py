import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
import vertexai
# from loggerUtil import logger
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold, Image, FinishReason, Part
from vertexai import generative_models
from google.oauth2 import service_account
import json, time
from google.api_core.exceptions import ResourceExhausted
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv(".env")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "active-complex-405121-431aab878614_2.json"
vertexai.init(project='active-complex-405121', location='us-central1')
gemini_safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(api_version="2024-08-01-preview",
                      api_key=os.environ["AZURE_OPENAI_API_KEY"],
                      azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"])
embeddings = AzureOpenAIEmbeddings(api_version="2023-05-15",
                                   api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_EMBEDD"])
# print(llm)
# print(embeddings)
