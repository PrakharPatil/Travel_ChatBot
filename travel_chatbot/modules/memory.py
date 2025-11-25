# modules/memory.py
from transformers import pipeline
import torch


class MemoryManager:
    def __init__(self, llm_model='mistralai/Mistral-7B-Instruct-v0.2', max_stm_turns=10):
        self.stm = []  # Short-term memory: list of conversation turns
        self.ltm = {  # Long-term memory: user preferences, facts
            'preferences': {},
            'dislikes': {},
            'facts': {}
        }
        self.max_stm_turns = max_stm_turns

        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            dtype=torch.float16,
            device_map="auto"
        )

    def add_to_stm(self, query: str, response: str):
        """Add conversation turn to short-term memory"""
        self.stm.append({
            'query': query,
            'response': response
        })

        # Keep only recent turns
        if len(self.stm) > self.max_stm_turns:
            self.stm.pop(0)

    def generate_stm_summary_with_llm(self):
        """LLM generates summary of conversation"""
        if not self.stm:
            return "No conversation history yet."

        conversation = "\n".join([
            f"User: {turn['query']}\nBot: {turn['response']}"
            for turn in self.stm
        ])

        prompt = f"""<s>[INST] Summarize this conversation in 2-3 sentences, focusing on key information and user intent:

{conversation}

Summary: [/INST]"""

        result = self.llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']
        summary = result.split('[/INST]')[-1].strip()

        return summary

    def extract_user_preferences_with_llm(self, query: str, response: str):
        """LLM extracts user preferences from conversation"""
        prompt = f"""<s>[INST] Extract user preferences, dislikes, and personal facts from this conversation turn:

User: {query}
Bot: {response}

Extract:
1. Preferences (things user likes, wants, prefers)
2. Dislikes (things user doesn't like, wants to avoid)
3. Personal facts (user's location, travel dates, budget, etc.)

Respond in this format:
Preferences: <comma-separated list or "none">
Dislikes: <comma-separated list or "none">
Facts: <comma-separated list or "none">
[/INST]"""

        result = self.llm(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
        response_text = result.split('[/INST]')[-1].strip()

        # Parse response
        extracted = {
            'preferences': [],
            'dislikes': [],
            'facts': []
        }

        for line in response_text.split('\n'):
            if 'Preferences:' in line:
                prefs = line.split('Preferences:')[-1].strip()
                if prefs.lower() != 'none':
                    extracted['preferences'] = [p.strip() for p in prefs.split(',')]
            elif 'Dislikes:' in line:
                dislikes = line.split('Dislikes:')[-1].strip()
                if dislikes.lower() != 'none':
                    extracted['dislikes'] = [d.strip() for d in dislikes.split(',')]
            elif 'Facts:' in line:
                facts = line.split('Facts:')[-1].strip()
                if facts.lower() != 'none':
                    extracted['facts'] = [f.strip() for f in facts.split(',')]

        return extracted

    def update_ltm(self, query: str, response: str):
        """Update long-term memory with extracted information"""
        extracted = self.extract_user_preferences_with_llm(query, response)

        # Update LTM
        for pref in extracted['preferences']:
            if pref:
                self.ltm['preferences'][pref] = self.ltm['preferences'].get(pref, 0) + 1

        for dislike in extracted['dislikes']:
            if dislike:
                self.ltm['dislikes'][dislike] = self.ltm['dislikes'].get(dislike, 0) + 1

        for fact in extracted['facts']:
            if fact:
                key = fact.split(':')[0].strip() if ':' in fact else fact
                value = fact.split(':')[-1].strip() if ':' in fact else fact
                self.ltm['facts'][key] = value

    def get_memory_context(self):
        """Get formatted memory context for main LLM"""
        stm_summary = self.generate_stm_summary_with_llm()

        ltm_summary = []
        if self.ltm['preferences']:
            top_prefs = sorted(self.ltm['preferences'].items(), key=lambda x: x[1], reverse=True)[:3]
            ltm_summary.append(f"User prefers: {', '.join([p[0] for p in top_prefs])}")

        if self.ltm['dislikes']:
            top_dislikes = sorted(self.ltm['dislikes'].items(), key=lambda x: x[1], reverse=True)[:3]
            ltm_summary.append(f"User dislikes: {', '.join([d[0] for d in top_dislikes])}")

        if self.ltm['facts']:
            ltm_summary.append(
                f"Known facts: {', '.join([f'{k}: {v}' for k, v in list(self.ltm['facts'].items())[:5]])}")

        return {
            'stm_summary': stm_summary,
            'ltm_summary': ' | '.join(ltm_summary) if ltm_summary else 'No long-term preferences yet',
            'recent_turns': self.stm[-3:] if len(self.stm) > 0 else []
        }
