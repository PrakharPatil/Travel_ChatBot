# modules/orchestration.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline


class QueryOrchestrator:
    def __init__(self, bert_model_path='./models/bert_classifier', llm_model='mistralai/Mistral-7B-Instruct-v0.2'):
        # Load fine-tuned BERT
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
        self.categories = ["Conversational", "RAG", "API_Call"]

        # Backup LLM for uncertain cases
        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            dtype=torch.float16,
            device_map="auto"
        )

    def classify_with_bert(self, query: str):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence, pred = torch.max(probs, dim=-1)

        return self.categories[pred.item()], confidence.item()

    def classify_with_llm(self, query: str):
        prompt = f"""<s>[INST] You are a query classifier for a travel chatbot. Classify the following query into one of these categories:
- Conversational: Greetings, clarifications, casual chat
- RAG: Queries requiring information from travel database (destinations, attractions, travel plans)
- API_Call: Queries requiring real-time data (flight prices, hotel availability, weather)

Query: "{query}"

Respond with ONLY the category name. [/INST]"""

        result = self.llm(prompt, max_new_tokens=10, do_sample=False)[0]['generated_text']
        category = result.split('[/INST]')[-1].strip()

        for cat in self.categories:
            if cat.lower() in category.lower():
                return cat
        return "Conversational"  # Default

    def classify(self, query: str):
        # Primary classification with BERT
        category, confidence = self.classify_with_bert(query)

        # If confidence is low, use LLM as fallback
        if confidence < 0.7:
            category = self.classify_with_llm(query)

        return category
