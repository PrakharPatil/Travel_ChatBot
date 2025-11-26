import json
import requests
import yaml
import re
from transformers import pipeline
import torch


class CorrectiveRAG:
    def __init__(self, serpapi_key, llm_model='google/flan-t5-base'):
        self.serpapi_key = serpapi_key

        # Dynamic task selection based on model architecture
        task = "text2text-generation" if "t5" in llm_model.lower() else "text-generation"
        print(f"CRAG Module: Loading {llm_model} with task '{task}'")

        self.llm = pipeline(
            task,
            model=llm_model,
            dtype=torch.float32,
            device_map="auto"
        )

    def _clean_llm_response(self, response):
        """Helper to clean and normalize LLM text response."""
        if isinstance(response, list):
            text = response[0]['generated_text']
        else:
            text = str(response)

        if "[/INST]" in text:
            text = text.split('[/INST]')[-1]

        return text.strip()

    def _filter_web_content(self, results):
        """Clean web snippets to remove ads, navigation text, or short noise."""
        cleaned = []
        print(f"DEBUG: Filtering {len(results)} raw web results...")

        for res in results:
            snippet = res.get('snippet', '')
            # Relaxed filter: Allow shorter snippets (15 chars)
            if len(snippet) > 15:
                # Remove common garbage characters
                snippet = re.sub(r'\s+', ' ', snippet).strip()
                cleaned.append(f"Title: {res.get('title')}\nInfo: {snippet}")

        print(f"DEBUG: {len(cleaned)} results passed filtering.")
        return cleaned[:3]  # Return top 3 cleanest results

    def should_verify_with_llm(self, query: str, rag_context: list):
        """
        Determines if the retrieved Graph context is sufficient or if we need external web verification.
        Uses Few-Shot prompting for higher accuracy.
        """
        context_str = str(rag_context)[:800]

        prompt = f"""Task: Check if Context answers Query.
Respond 'VERIFY' if info is missing.
Respond 'NO_VERIFY' if sufficient.

Q: "Price of Eiffel Tower tickets"
C: [{{'name': 'Eiffel Tower'}}]
Decision: VERIFY

Q: "Where is the Louvre?"
C: [{{'city': 'Paris'}}]
Decision: NO_VERIFY

Q: "{query}"
C: {context_str}
Decision:"""

        try:
            result = self.llm(prompt, max_new_tokens=10, do_sample=False)
            text = self._clean_llm_response(result)
            print(f"DEBUG: Decision Raw Output: '{text}'")

            # Check for explicit decision or fallback to keyword
            if "VERIFY" in text.upper() and "NO_VERIFY" not in text.upper():
                return True
            if "NO_VERIFY" in text.upper():
                return False

            # Fallback: aggressive verification if unsure
            return "VERIFY" in text.upper()

        except Exception as e:
            print(f"CRAG Decision Error: {e}")
            return True  # Fail safe: verify if error

    def search_web(self, query: str):
        """Performs a Google Search using SerpAPI."""
        if not self.serpapi_key:
            print("CRAG Warning: No SerpAPI key provided.")
            return []

        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": 5,  # Fetch more to filter later
            "engine": "google"
        }

        try:
            print(f"CRAG: Searching web for '{query}'...")
            response = requests.get(url, params=params, timeout=10)
            results = response.json()

            if 'organic_results' in results:
                return self._filter_web_content(results['organic_results'])

            return []
        except Exception as e:
            print(f"CRAG Search Error: {e}")
            return []

    def correct_with_llm(self, query: str, rag_context: list, web_context: list):
        """
        Synthesizes a final answer using both Graph RAG (if valid) and Web Context.
        """
        web_text = "\n\n".join(web_context)
        graph_text = str(rag_context)

        # Debug print to ensure LLM actually sees data
        if not web_text:
            print("DEBUG: Web context is empty! LLM will likely fail.")
        else:
            print(f"DEBUG: Sending {len(web_text)} chars of web data to LLM.")

        prompt = f"""Task: Answer the query using the Web Data.

Query: {query}
Web Data: {web_text}

Answer:"""

        result = self.llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.6)
        return self._clean_llm_response(result)

    def process(self, query: str, rag_context: list):
        """
        Main pipeline: Evaluate -> (Search) -> Synthesize

        Returns dict with BOTH 'context' (for app.py) and 'final_answer' (for display)
        """
        try:
            # 1. Decision Step
            needs_verification = self.should_verify_with_llm(query, rag_context)

            if needs_verification:
                print("CRAG: Verification triggered (Context insufficient or dynamic data needed).")

                # 2. Action Step
                web_context = self.search_web(query)

                if not web_context:
                    print("CRAG: Web search returned no results. Falling back to original context.")
                    return {
                        'context': rag_context if rag_context else [{"info": "No context available"}],  # For app.py
                        'verified': False,
                        'final_answer': "I tried to check the web but couldn't connect. Here is what I know: " + str(
                            rag_context),
                        'source': 'RAG (Fallback)'
                    }

                # 3. Correction Step - Generate synthesized answer
                corrected_answer = self.correct_with_llm(query, rag_context, web_context)

                # Create structured context for MainLLM
                fused_context = []
                if rag_context:
                    fused_context.extend(rag_context if isinstance(rag_context, list) else [rag_context])
                fused_context.extend([{"source": "web", "info": item} for item in web_context])

                return {
                    'context': fused_context,  # CRITICAL: For app.py → MainLLM
                    'verified': True,
                    'final_answer': corrected_answer,  # For standalone CRAG testing/display
                    'source': 'CRAG (Web + Graph)',
                    'web_context': web_context
                }

            # 4. Passive Step (No action needed)
            print("CRAG: Context is sufficient. No verification needed.")
            return {
                'context': rag_context if rag_context else [{"info": "No context available"}],  # For app.py
                'verified': False,
                'final_answer': None,  # Signals orchestrator to use base RAG answer
                'source': 'RAG (Graph Only)'
            }

        except Exception as e:
            # Error handling: Always return valid structure
            print(f"CRAG Error: {e}")
            import traceback
            traceback.print_exc()

            return {
                'context': rag_context if rag_context else [{"error": str(e)}],
                'verified': False,
                'final_answer': f"Error during verification: {str(e)}",
                'source': 'ERROR',
                'error': str(e)
            }


# --- TESTING FUNCTIONALITIES ---
if __name__ == "__main__":
    import os

    # 1. SETUP: Load Config from File Only
    SERPAPI_KEY = None

    try:
        config_path = '../configs/config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Look for the key in the root of the YAML as provided
                if 'serpapi_key' in config:
                    SERPAPI_KEY = config['serpapi_key']
                else:
                    print(f"ERROR: 'serpapi_key' not found in {config_path}")
        else:
            print(f"ERROR: {config_path} does not exist. Please create it.")
    except Exception as e:
        print(f"Config Loading Error: {e}")

    # 2. INITIALIZE
    if SERPAPI_KEY:
        print("\n--- Initializing CorrectiveRAG Module ---")
        crag = CorrectiveRAG(serpapi_key=SERPAPI_KEY, llm_model='google/flan-t5-base')

        # 3. TEST CASES

        # Case A: Static Info (Graph is sufficient)
        query_a = "Where is the Eiffel Tower?"
        context_a = [{'name': 'Eiffel Tower', 'city': 'Paris', 'country': 'France'}]
        print(f"\n--- Test A: '{query_a}' (Expect NO_VERIFY) ---")
        result_a = crag.process(query_a, context_a)
        print(f"Source: {result_a['source']}")
        print(f"✓ Has 'context' key: {'context' in result_a}")
        if result_a.get('final_answer'):
            print(f"Final Answer: {result_a['final_answer']}")

        # Case B: Dynamic Info (Graph is insufficient)
        query_b = "What is the ticket price for Eiffel Tower in 2025?"
        context_b = [{'name': 'Eiffel Tower', 'city': 'Paris'}]  # Missing price
        print(f"\n--- Test B: '{query_b}' (Expect VERIFY) ---")
        result_b = crag.process(query_b, context_b)
        print(f"Source: {result_b['source']}")
        print(f"✓ Has 'context' key: {'context' in result_b}")
        print(f"Result: {result_b.get('final_answer')}")

        # Case C: Empty Context (Graph failed completely)
        query_c = "Who won the World Cup in 2022?"
        context_c = []
        print(f"\n--- Test C: '{query_c}' (Expect VERIFY) ---")
        result_c = crag.process(query_c, context_c)
        print(f"Source: {result_c['source']}")
        print(f"✓ Has 'context' key: {'context' in result_c}")
        print(f"Result: {result_c.get('final_answer')}")

        # Validation
        print(f"\n{'=' * 70}")
        print(f"Compatibility Validation")
        print(f"{'=' * 70}")
        print(f"✓ All results have 'context' key: {all('context' in r for r in [result_a, result_b, result_c])}")
        print(
            f"✓ All results have 'final_answer' key: {all('final_answer' in r for r in [result_a, result_b, result_c])}")

    else:
        print("CRAG Test Aborted: Missing valid configuration.")
