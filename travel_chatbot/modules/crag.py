# modules/crag.py
import requests
from transformers import pipeline
import torch


class CorrectiveRAG:
    def __init__(self, serpapi_key, llm_model='google/flan-t5-base'):
        self.serpapi_key = serpapi_key
        task = "text2text-generation" if "t5" in llm_model.lower() else "text-generation"
        self.llm = pipeline(
            task,
            model=llm_model,
            dtype=torch.float32,
            device_map="auto"
        )

    def _clean_llm_response(self, response):
        text = response[0]['generated_text']
        if "[/INST]" in text:
            return text.split('[/INST]')[-1].strip()
        return text.strip()

    def should_verify_with_llm(self, query: str, rag_context: dict):
        prompt = f"""Task: Decide if this needs web verification (prices, weather, current events).
Query: {query}
Context: {rag_context}
Respond 'VERIFY' or 'NO_VERIFY'."""

        result = self.llm(prompt, max_new_tokens=10, do_sample=False)
        decision = self._clean_llm_response(result).upper()
        return "VERIFY" in decision

    def search_web(self, query: str):
        if not self.serpapi_key: return []  # Handle missing key gracefully
        url = "https://serpapi.com/search.json"
        params = {"q": query, "api_key": self.serpapi_key, "num": 3}
        try:
            response = requests.get(url, params=params)
            results = response.json()
            web_context = []
            if 'organic_results' in results:
                for result in results['organic_results'][:3]:
                    web_context.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', '')
                    })
            return web_context
        except Exception:
            return []

    def correct_with_llm(self, query: str, rag_context: dict, web_context: list):
        prompt = f"""Task: Synthesize answer based on sources.
Query: {query}
Graph Data: {rag_context}
Web Data: {web_context}
Answer:"""

        result = self.llm(prompt, max_new_tokens=200, do_sample=True)
        return self._clean_llm_response(result)

    def process(self, query: str, rag_context: dict):
        if self.should_verify_with_llm(query, rag_context):
            web_context = self.search_web(query)
            corrected = self.correct_with_llm(query, rag_context, web_context)
            return {'verified': True, 'context': corrected, 'source': 'CRAG'}

        return {'verified': False, 'context': rag_context, 'source': 'RAG'}