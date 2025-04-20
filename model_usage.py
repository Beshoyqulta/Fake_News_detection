import tldextract
from duckduckgo_search import DDGS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

ddg=DDGS()
print("Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained("results/checkpoint-500")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



def retrieve_domains(query: str):
    print(f"Retrieving domains for query: {query}.........")
    results = ddg.text(query)
    domains = []
    for result in results:
        extract = tldextract.extract(result['href'])
        domain = f"{extract.domain}.{extract.suffix}"
        domains.append(domain)
    return domains
def classify_domains(domains: list):
    results = []
    for domain in domains:
        inputs = tokenizer(domain, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            is_trusted = bool(prediction)

        results.append((domain, is_trusted))
    return results

print(classify_domains(retrieve_domains('cristiano ronaldo')))