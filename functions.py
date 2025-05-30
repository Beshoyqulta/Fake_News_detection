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
tokenizer = BertTokenizer.from_pretrained("mohamedzabady/bert-fake-news",token="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")
bert_model = BertForSequenceClassification.from_pretrained("mohamedzabady/bert-fake-news",token="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")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

search = DDGS()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key="AIzaSyBVusfvWlCSmOKbw6KUQPBl9IYqvEDZnOk",
    temperature=0.0,
)

system = """
You are a concise fact-checking assistant. Analyze news reports and determine if claims are REAL or FAKE.

## Output Format (MANDATORY):
Start with: **REAL (X%)** or **FAKE (X%)** where X is your confidence percentage.

## Response Structure:
**REAL/FAKE (X%)**

**Reasoning:** Brief explanation based on trusted sources.

**Sources:** Number of trusted vs untrusted sources.

## Guidelines:
- **REAL**: Supported by trusted sources
- **FAKE**: Contradicted by trusted sources OR only untrusted sources support it
- **Confidence**: 60-75% = Some evidence, 76-89% = Strong evidence, 90%+ = Very strong evidence
- **Keep responses short** - 2-3 sentences maximum for reasoning
- **Always use markdown formatting**
- **Prioritize trusted sources heavily**

## Example:
**FAKE (85%)**

**Reasoning:** Multiple trusted news sources contradict this claim with verified information.

**Sources:** 3 trusted sources against, 1 untrusted source supporting.
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
        if cosine_similarity_score > 0.3:
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
                1. Rephrase your search.
                2. Check real-time sources """

        return chain.invoke(messages).content
    for news in results:
        if check_trusted(news):
            trusted_template+= f"""
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
            
    return chain.invoke(trusted_template + untrusted_template + f"\n\n based on Reports, the query \"{query}\" are TRUE or FALSE AND WHY \n\n MARKDOWN Answer :").content

