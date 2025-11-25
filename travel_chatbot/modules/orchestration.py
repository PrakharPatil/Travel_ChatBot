# modules/orchestration.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline


class QueryOrchestrator:
    def __init__(self, bert_model_path, llm_model='google/flan-t5-base'):
        # Keep BERT as is
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
        self.categories = ["Conversational", "RAG", "API_Call"]

        task = "text2text-generation" if "t5" in llm_model.lower() else "text-generation"
        self.llm = pipeline(
            task,
            model=llm_model,
            dtype=torch.float32,
            device_map="auto"
        )

    def classify_with_bert(self, query: str):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred = torch.max(probs, dim=-1)
        return self.categories[pred.item()], confidence.item()

    def classify_with_llm(self, query: str):
        prompt = f"Classify query into Conversational, RAG, or API_Call. Query: {query}"
        result = self.llm(prompt, max_new_tokens=10)[0]['generated_text']

        # Simple string matching for robustness
        res_lower = result.lower()
        if "api" in res_lower: return "API_Call"
        if "rag" in res_lower or "database" in res_lower: return "RAG"
        return "Conversational"

    def classify(self, query: str):
        cat, conf = self.classify_with_bert(query)
        if conf < 0.6:  # Lowered threshold
            return self.classify_with_llm(query)
        return cat