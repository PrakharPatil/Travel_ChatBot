from transformers import pipeline
import torch


class MemoryManager:
    def __init__(self, llm_model='google/flan-t5-base', max_stm_turns=10):
        self.stm = []
        self.ltm = {
            'preferences': {},
            'dislikes': {},
            'facts': {}
        }
        self.max_stm_turns = max_stm_turns

        # Dynamic task selection
        task = "text2text-generation" if "t5" in llm_model.lower() else "text-generation"
        print(f"MemoryManager: Loading {llm_model} with task '{task}'")

        self.llm = pipeline(
            task,
            model=llm_model,
            dtype=torch.float32,
            device_map="auto"
        )

    def _clean_llm_response(self, response):
        """Helper to safely clean response"""
        text = response[0]['generated_text']
        if "[/INST]" in text:
            return text.split('[/INST]')[-1].strip()
        return text.strip()

    def add_to_stm(self, query: str, response: str):
        self.stm.append({'query': query, 'response': response})
        if len(self.stm) > self.max_stm_turns:
            self.stm.pop(0)

    def generate_stm_summary_with_llm(self):
        if not self.stm:
            return "No conversation history yet."

        # Limit context to last 3 interactions to keep prompt small
        recent_stm = self.stm[-3:]
        conversation = " ".join([f"User: {t['query']} Bot: {t['response']}" for t in recent_stm])

        # FIX: Few-Shot Summary
        prompt = f"""Task: Summarize the travel conversation in 1 sentence.

Example 1:
Context: User: Hi. Bot: Hello! User: Book a flight to Paris. Bot: Done.
Summary: The user booked a flight to Paris.

Example 2:
Context: User: I like sushi. Bot: Noted. User: Where is a good place? Bot: Tokyo.
Summary: The user likes sushi and asked for restaurant recommendations in Tokyo.

Task:
Context: {conversation}
Summary:"""

        result = self.llm(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
        return self._clean_llm_response(result)

    def extract_user_preferences_with_llm(self, query: str, response: str):
        # FIX: Chain of Thought (CoT) + Parsing Strategy
        # We need to distinguish between a *question* about a topic and a *preference* for it.

        prompt = f"""Task: Extract permanent User Data (Preferences, Dislikes, Facts).
Ignore temporary questions.

Example 1:
Input: "I love hiking but I hate rain."
Thought: 'Love' implies positive preference. 'Hate' implies dislike.
Output: Preference: hiking | Dislike: rain

Example 2:
Input: "Is it raining in London?"
Thought: This is a question about weather, not a dislike of rain.
Output: None

Example 3:
Input: "I am a vegan and I need a hotel in Berlin."
Thought: 'Vegan' is a fact/dietary preference. 'Hotel in Berlin' is a current need (Fact).
Output: Fact: Vegan | Fact: Location Berlin

Task:
Input: "{query}"
Thought:"""

        # Increased tokens for "Thought" generation
        result = self.llm(prompt, max_new_tokens=60, do_sample=False)
        response_text = self._clean_llm_response(result)

        extracted = {
            'preferences': [],
            'dislikes': [],
            'facts': []
        }

        # Robust Parsing: Look for delimiters (| or newline)
        parts = response_text.replace('\n', '|').split('|')

        for part in parts:
            part = part.strip()
            # We look for the keyword after the colon to avoid parsing the "Thought" section
            if "Preference:" in part:
                val = part.split("Preference:")[-1].strip()
                if val and val.lower() != 'none': extracted['preferences'].append(val)
            elif "Dislike:" in part:
                val = part.split("Dislike:")[-1].strip()
                if val and val.lower() != 'none': extracted['dislikes'].append(val)
            elif "Fact:" in part:
                val = part.split("Fact:")[-1].strip()
                if val and val.lower() != 'none': extracted['facts'].append(val)

        return extracted

    def update_ltm(self, query: str, response: str):
        extracted = self.extract_user_preferences_with_llm(query, response)

        # Update Logic
        for pref in extracted['preferences']:
            self.ltm['preferences'][pref] = self.ltm['preferences'].get(pref, 0) + 1
            print(f"Memory: Learned Preference -> {pref}")

        for dislike in extracted['dislikes']:
            self.ltm['dislikes'][dislike] = self.ltm['dislikes'].get(dislike, 0) + 1
            print(f"Memory: Learned Dislike -> {dislike}")

        for fact in extracted['facts']:
            clean_fact = fact.replace('Fact:', '').strip()
            if clean_fact:
                # Simple deduplication using the fact as the key
                self.ltm['facts'][clean_fact] = "Active"
                print(f"Memory: Learned Fact -> {clean_fact}")

    def get_memory_context(self):
        stm_summary = self.generate_stm_summary_with_llm()

        ltm_summary = []
        if self.ltm['preferences']:
            # Get top 3 most mentioned preferences
            top_prefs = sorted(self.ltm['preferences'].items(), key=lambda x: x[1], reverse=True)[:3]
            ltm_summary.append(f"Likes: {', '.join([p[0] for p in top_prefs])}")

        if self.ltm['dislikes']:
            top_dislikes = sorted(self.ltm['dislikes'].items(), key=lambda x: x[1], reverse=True)[:3]
            ltm_summary.append(f"Dislikes: {', '.join([d[0] for d in top_dislikes])}")

        if self.ltm['facts']:
            # Just take the last 3 facts
            facts_list = list(self.ltm['facts'].keys())[-3:]
            ltm_summary.append(f"User Info: {', '.join(facts_list)}")

        return {
            'stm_summary': stm_summary,
            'ltm_summary': ' | '.join(ltm_summary) if ltm_summary else 'No prior data',
            'recent_turns': self.stm[-3:]
        }