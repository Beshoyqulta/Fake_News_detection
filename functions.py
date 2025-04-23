from duckduckgo_search import DDGS 
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings , HuggingFaceEndpoint
from typing import *
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydantic import BaseModel
import os
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("/mnt/c/Users/SG/Desktop/myenv/for univ/results/checkpoint-500")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

search = DDGS()
genai.configure(api_key="AIzaSyBVusfvWlCSmOKbw6KUQPBl9IYqvEDZnOk")
llm = genai.GenerativeModel(model_name="gemini-2.0-flash",system_instruction="you are Rashed you are a fake news detector and you are here to help the user to detect if the news is fake or not").start_chat()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
class NewsRequest(BaseModel):
    input_str: str
def check_relevance(query: str):
    news_list = search.news(query)
    relevant_news = [] 
    for news in news_list:
        cosine_similarity_score = cosine_similarity([embedding_model.embed_query(query)], [embedding_model.embed_query(news["title"])])
        if cosine_similarity_score > 0.5 and cosine_similarity_score < 1:
            relevant_news.append(news)
            return True, relevant_news
        else:
            return False


def check_trusted(news: dict):

    inputs = tokenizer(news['source'], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        is_trusted = bool(prediction)

    return is_trusted

def check_fakeness(query: str):
    if check_relevance(query) != True:
        return llm.send_message("All you have to do here is to introduce yourself then you will tell the user that his input is not relevant").text
    elif check_relevance(query)[0] == True:
        for news in check_relevance(query)[1]:
            if check_trusted(news) == True:
                trusted_news =f"""
                                You Will Tell The User That:/n
                                This Source is trusted and the news is not fake,/n
                                and you will return to the user the following information:/n
                                the title is {news['title']},/n
                                the link is {news['url']},/n
                                the date is {news['date']},/n
                                the source is {news['source']}/n 
                                """
                
            else:
                not_trusted_news = f"""
                                You Will Tell The User That:/n
                                This Source is not trusted and the news is fake,/n
                                and you will return to the user the following information:/n
                                the title is {news['title']},/n
                                the link is {news['url']},/n
                                the date is {news['date']},/n
                                the source is {news['source']}/n
                                """
            return trusted_news + not_trusted_news 
def generate_content(query: str):
    response = llm.send_message(check_fakeness(query)).text
    return response