# test_bert.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load fine-tuned model
MODEL_PATH = "./models/bert_classifier"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Category labels
CATEGORIES = ["Conversational", "RAG", "API_Call"]


def classify(query: str):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confidence, pred = torch.max(probs, dim=-1)

    return CATEGORIES[pred.item()], confidence.item()


# Test cases
test_queries = [
    # Conversational
    "Hello!",
    "Thanks for your help",
    "Can you explain that?",

    # RAG
    "What are the best places to visit in Paris?",
    "Plan a trip to Tokyo",
    "Best food in Thailand",

    # API Call
    "Flight price from NYC to London",
    "Weather in Dubai today",
    "Hotel prices in Rome",
]

print("=" * 60)
print("BERT Query Classifier Test")
print("=" * 60)

for query in test_queries:
    category, confidence = classify(query)
    print(f"\nQuery: \"{query}\"")
    print(f"  → {category} (confidence: {confidence:.2%})")
