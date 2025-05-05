from duckduckgo_search import DDGS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings , HuggingFaceEndpoint
from typing import *
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pydantic import BaseModel
import os
tokenizer = BertTokenizer.from_pretrained("mohamedzabady/bert-fake-news")
bert_model = BertForSequenceClassification.from_pretrained("mohamedzabady/bert-fake-news")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

search = DDGS()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key="AIzaSyBVusfvWlCSmOKbw6KUQPBl9IYqvEDZnOk",
    temperature=0.5,
)

system="""Role: You are Rashed, an AI assistant specialized in detecting and analyzing fake news. Your goal is to help users verify news credibility with evidence-based analysis.

Instructions:

Source Evaluation:

Check if the news comes from a trusted source (e.g., BBC, Reuters). If yes, return structured details (title, source, date, summary, link).

If untrusted, analyze linguistic patterns (sensationalism, lack of sources, emotional manipulation).
Fact-Checking:
Cross-reference claims with reputable fact-checking platforms (Snopes, Politifact).

Flag inconsistencies, outdated data, or logical fallacies.

User Interaction:

For verified news:

✅ **Trusted News**: "[Headline]" ([Source], [Date]).  
📌 **Key Points**: [Neutral summary].  
🔍 [Read more](link).  
For potential fake news:

⚠️ **Suspicious Content Detected**: "[Headline]"  
🚩 **Red Flags**: [List reasons, e.g., "Unverified sources", "Clickbait language"].  
📢 **Suggestions**: Check [Trusted Source] for updates.  
For irrelevant queries:

❓ **Off-Topic**: Your input isn’t news-related. Try: "Is [claim] true?" or "News about [topic]."
"""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system,
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
class NewsRequest(BaseModel):
    input_str: str
def check_relevance(query: str):
    news_list = search.news(query)
    relevant_news = [] 
    for news in news_list:
        cosine_similarity_score = cosine_similarity([embedding_model.embed_query(query)], [embedding_model.embed_query(news["title"])])
        if cosine_similarity_score > 0.7:
            relevant_news.append(news)
    return relevant_news


def check_trusted(news: dict):

    inputs = tokenizer(news['source'], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        is_trusted = bool(prediction)

    return is_trusted

# def check_fakeness(query: str):
#     trusted_template = """"""
#     untrusted_template = """"""
#     results = check_relevance(query)
#     if not results:
#         messages = f"""
#                 No relevant news was found for: '{query}'. Possible reasons:
#                 - The topic is too new or niche.
#                 - The query may not be news-related.
#                 - Limited data coverage in our system.
                
#                 Suggestions:
#                 1. Rephrase your search (e.g., use keywords like 'COVID updates' instead of 'Is virus bad?').
#                 2. Check real-time sources like BBC/Reuters for breaking news."""

#         return llm.invoke(messages).content
#     for news in results:
#         if check_trusted(news):
#             trusted_template+= f"""
#         **✅ Trusted News Report**  
#         **Title:** {news['title']}  
#         **Source:** {news['source']} (Trusted)  
#         **Date:** {news['date']}  
#         **Summary:** {news['body']}  
#         **Read More:** [Full Article]({news['url']})\n\n

#             """
#         else:
#             untrusted_template+= f"""
#         ⚠️ Source not trusted. These news might be fake
#         **Source:** {news['source']} (Untrusted)\n
#             """
#     return trusted_template + untrusted_template

def generate_content(query: str):
    trusted_template = """"""
    untrusted_template = """"""
    results = check_relevance(query)
    if not results:
        messages = f"""
                No relevant news was found for: '{query}'. Possible reasons:
                - The topic is too new or niche.
                - The query may not be news-related.
                - Limited data coverage in our system.
                
                Suggestions:
                1. Rephrase your search (e.g., use keywords like 'COVID updates' instead of 'Is virus bad?').
                2. Check real-time sources like BBC/Reuters for breaking news."""

        return llm.invoke(messages).content
    for news in results:
        if check_trusted(news):
            trusted_template+= f"""
        **✅ Trusted News Report**  
        **Title:** {news['title']}  
        **Source:** {news['source']} (Trusted)  
        **Date:** {news['date']}  
        **Summary:** {news['body']}
        **Read More:** [Full Article]({news['url']})\n\n

            """
        else:
            untrusted_template+= f"""
        ⚠️ Source not trusted. These news might be fake
        **Source:** {news['source']} (Untrusted)\n
            """
    return llm.invoke(trusted_template + untrusted_template).content

