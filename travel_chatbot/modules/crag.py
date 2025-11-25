# modules/crag.py
import requests
from transformers import pipeline
import torch


class CorrectiveRAG:
    def __init__(self, serpapi_key, llm_model='mistralai/Mistral-7B-Instruct-v0.2'):
        self.serpapi_key = serpapi_key
        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            dtype=torch.float16,
            device_map="auto"
        )

    def should_verify_with_llm(self, query: str, rag_context: dict):
        """LLM decides if verification is needed"""
        prompt = f"""<s>[INST] Determine if the following travel information needs real-time web verification.

Query: "{query}"
RAG Context: {rag_context}

Information that needs verification:
- Prices (flights, hotels, attractions)
- Operating hours
- Current availability
- Weather
- Recent changes or closures
- Seasonal events

Information that does NOT need verification:
- Historical facts
- General destination information
- Cultural information
- Geographic location

Respond with ONLY: "VERIFY" or "NO_VERIFY" [/INST]"""

        result = self.llm(prompt, max_new_tokens=20, do_sample=False)[0]['generated_text']
        decision = result.split('[/INST]')[-1].strip().upper()

        return "VERIFY" in decision

    def search_web(self, query: str):
        """Get web verification data"""
        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": 3
        }

        try:
            response = requests.get(url, params=params)
            results = response.json()

            web_context = []
            if 'organic_results' in results:
                for result in results['organic_results'][:3]:
                    web_context.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', '')
                    })

            return web_context
        except Exception as e:
            print(f"Web search error: {e}")
            return []

    def correct_with_llm(self, query: str, rag_context: dict, web_context: list):
        """LLM compares RAG output with web data and returns corrected version"""
        prompt = f"""<s>[INST] You are a fact-checker for travel information. Compare the RAG-retrieved information with web search results and provide the most accurate, up-to-date answer.

User Query: "{query}"

RAG Context: {rag_context}

Web Search Results: {web_context}

Task:
1. Compare the information from both sources
2. Identify any discrepancies
3. Provide the most accurate information, prioritizing web results for volatile data (prices, hours, availability)
4. Keep historical/static information from RAG if web doesn't contradict it

Provide a corrected, comprehensive answer. [/INST]"""

        result = self.llm(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]['generated_text']
        corrected_response = result.split('[/INST]')[-1].strip()

        return corrected_response

    def process(self, query: str, rag_context: dict):
        """Main CRAG pipeline"""
        # Phase 1: Decide if verification is needed
        needs_verification = self.should_verify_with_llm(query, rag_context)

        if not needs_verification:
            return {
                'verified': False,
                'context': rag_context,
                'source': 'RAG'
            }

        # Phase 2: Verify with web
        web_context = self.search_web(query)

        # Phase 3: Correct using LLM
        corrected_answer = self.correct_with_llm(query, rag_context, web_context)

        return {
            'verified': True,
            'context': corrected_answer,
            'source': 'CRAG (RAG + Web)',
            'web_sources': web_context
        }
