from duckduckgo_search import DDGS 
from google.generativeai import generative_models
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings , HuggingFaceEndpoint
from typing import *
search = DDGS()
LLM = generative_models.GenerativeAIModel("gemini-1.5-flash")

class NewsRequest:
    input_str: str
def embed(text: str):
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2").embed_query(text)
def search(query: str):
    results = search.news(query)
    return results
def check_relevance(query: str, result: str):
    cosine_similarity = cosine_similarity([embed(query)], [embed(result)])
    if cosine_similarity > 0.6 and cosine_similarity < 1:
        return True
    else:
        return False
