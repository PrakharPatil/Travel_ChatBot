import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re


class MemoryManager:
    def __init__(self, llm_model='google/flan-t5-base', max_stm_turns=10):
        self.stm = []
        self.ltm = {
            'preferences': {},
            'dislikes': {},
            'facts': {}
        }
        self.max_stm_turns = max_stm_turns

        # Use direct model loading instead of pipeline
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"MemoryManager: Loading {llm_model} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            llm_model,
            dtype=torch.float16 if self.device.type == "mps" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _generate(self, prompt: str, max_length: int = 50):
        """Helper method for T5 generation"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def add_to_stm(self, query: str, response: str):
        self.stm.append({'query': query, 'response': response})
        if len(self.stm) > self.max_stm_turns:
            self.stm.pop(0)

    def generate_stm_summary_with_llm(self):
        if not self.stm:
            return "No conversation history yet."

        # Limit context to last 3 interactions
        recent_stm = self.stm[-3:]
        conversation = "\n".join([
            f"User: {t['query']}\nBot: {t['response']}"
            for t in recent_stm
        ])

        prompt = f"""Summarize this conversation in one sentence:

{conversation}

Summary:"""

        return self._generate(prompt, max_length=50)

    def _rule_based_extraction(self, query: str):
        """Primary extraction method using rules (more reliable than LLM)"""
        extracted = {
            'preferences': [],
            'dislikes': [],
            'facts': []
        }

        query_lower = query.lower()

        # === PREFERENCES ===
        # Pattern 1: "I love/like/prefer/enjoy X" (including "also", "really")
        love_patterns = [
            r"i\s+(?:also\s+)?(?:really\s+)?(?:love|like|prefer|enjoy|adore)\s+(.+?)(?:\.|,|but|and|$)",
            r"i'?m\s+(?:really\s+)?(?:into|fond of)\s+(.+?)(?:\.|,|but|and|$)",
        ]

        for pattern in love_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                pref = match.strip()
                # Clean up common words
                pref = re.sub(r'\b(the|a|an|to|very|really|so|much|also)\b', '', pref).strip()
                if pref and len(pref) > 2 and pref not in ['it', 'that', 'this', 'there']:
                    extracted['preferences'].append(pref)

        # === DISLIKES ===
        # Pattern 2: "I hate/dislike/can't stand X"
        hate_patterns = [
            r"i\s+(?:really\s+)?(?:hate|dislike|can'?t stand|avoid|don'?t like)\s+(.+?)(?:\.|,|but|and|$)",
            r"i'?m\s+(?:not (?:a\s+)?fan of|against)\s+(.+?)(?:\.|,|but|and|$)"
        ]

        for pattern in hate_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                dislike = match.strip()
                dislike = re.sub(r'\b(the|a|an|to|very|really|so|much)\b', '', dislike).strip()
                if dislike and len(dislike) > 2 and dislike not in ['it', 'that', 'this', 'there']:
                    extracted['dislikes'].append(dislike)

        # === FACTS ===
        # Pattern 3: Destination/Travel intent (use set to avoid duplicates)
        destinations = set()

        # Match "go to Paris", "visit Paris", "trip to Paris"
        destination_patterns = [
            r"(?:go|going|travel|visit|trip)\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:in|to|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        ]

        for pattern in destination_patterns:
            matches = re.findall(pattern, query)  # Use original case
            for match in matches:
                city = match.strip()
                if city and len(city) > 2 and city.lower() not in ['i', 'a', 'it', 'and', 'but']:
                    destinations.add(city)

        # Add unique destinations to facts
        for city in destinations:
            extracted['facts'].append(f"interested in {city}")

        # Pattern 4: Planning/Trip context
        if re.search(r'plan(?:ning)?\s+(?:a\s+)?trip', query_lower):
            extracted['facts'].append("planning a trip")

        # Pattern 5: Budget - IMPROVED
        budget_patterns = [
            r'\$\s*(\d+[,\d]*)',  # $2000 or $2,000
            r'(\d+[,\d]*)\s*(?:dollars?|usd)',  # 2000 dollars
            r'budget\s+(?:is|of)?\s*\$?(\d+[,\d]*)',  # budget is 2000
        ]

        for pattern in budget_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                amount = match.group(1).replace(',', '')
                extracted['facts'].append(f"budget ${amount}")
                break  # Only extract once

        # Pattern 6: Dietary restrictions
        dietary_keywords = ['vegan', 'vegetarian', 'gluten-free', 'kosher', 'halal', 'allergic']
        for keyword in dietary_keywords:
            if keyword in query_lower:
                extracted['facts'].append(f"dietary: {keyword}")

        # Pattern 7: Seating preference
        if 'window seat' in query_lower or 'aisle seat' in query_lower:
            if 'window' in query_lower:
                extracted['preferences'].append('window seats')
            elif 'aisle' in query_lower:
                extracted['preferences'].append('aisle seats')

        return extracted

    def extract_user_preferences_with_llm(self, query: str, response: str):
        """Hybrid extraction: Rule-based (primary) + LLM (fallback)"""

        # PRIMARY: Rule-based extraction
        extracted = self._rule_based_extraction(query)

        # FALLBACK: Only use LLM if rule-based found nothing
        if not any(extracted.values()):
            # Try LLM with very simple prompt
            prompt = f"""What does the user like or dislike?

User said: "{query}"

Answer (one word or short phrase):"""

            result = self._generate(prompt, max_length=20)

            print(f"DEBUG: LLM Raw Extraction: '{result}'")

            # Parse LLM result if it's not "None" or empty
            if result and result.lower() not in ['none', 'nothing', 'n/a', 'no']:
                # Try to categorize the LLM output
                if any(word in query.lower() for word in ['love', 'like', 'prefer', 'enjoy']):
                    extracted['preferences'].append(result)
                elif any(word in query.lower() for word in ['hate', 'dislike', 'avoid']):
                    extracted['dislikes'].append(result)
        else:
            print(f"DEBUG: Rule-based extraction found: {extracted}")

        return extracted

    def update_ltm(self, query: str, response: str):
        extracted = self.extract_user_preferences_with_llm(query, response)

        # Update preferences (deduplicate)
        for pref in extracted['preferences']:
            clean_pref = pref.strip().lower()
            if clean_pref and len(clean_pref) > 2:
                self.ltm['preferences'][clean_pref] = self.ltm['preferences'].get(clean_pref, 0) + 1
                print(f"Memory [LTM]: ✅ Learned Preference -> {clean_pref}")

        # Update dislikes (deduplicate)
        for dislike in extracted['dislikes']:
            clean_dislike = dislike.strip().lower()
            if clean_dislike and len(clean_dislike) > 2:
                self.ltm['dislikes'][clean_dislike] = self.ltm['dislikes'].get(clean_dislike, 0) + 1
                print(f"Memory [LTM]: ❌ Learned Dislike -> {clean_dislike}")

        # Update facts (deduplicate - don't add same fact twice)
        for fact in extracted['facts']:
            clean_fact = fact.replace('Fact:', '').strip()
            if clean_fact and len(clean_fact) > 2:
                # Only add if not already present
                if clean_fact not in self.ltm['facts']:
                    self.ltm['facts'][clean_fact] = "Active"
                    print(f"Memory [LTM]: 📝 Learned Fact -> {clean_fact}")

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
            # Take last 3 facts
            facts_list = list(self.ltm['facts'].keys())[-3:]
            ltm_summary.append(f"User Info: {', '.join(facts_list)}")

        return {
            'stm_summary': stm_summary,
            'ltm_summary': ' | '.join(ltm_summary) if ltm_summary else 'No prior data',
            'recent_turns': self.stm[-3:]
        }

    def reset(self):
        """Reset all memory"""
        self.stm = []
        self.ltm = {
            'preferences': {},
            'dislikes': {},
            'facts': {}
        }
        print("Memory [RESET]: All memory cleared")


# --- TESTING FUNCTIONALITIES ---
if __name__ == "__main__":
    print("\n--- Initializing Memory Manager ---")
    memory = MemoryManager(llm_model='google/flan-t5-base')

    # Simulation of a conversation
    test_interactions = [
        ("Hi, I am planning a trip.", "Where would you like to go?"),
        ("I want to go to Paris, but I hate flying.", "We can look at trains."),
        ("I also love Italian food.", "Paris has great Italian options."),
        ("Is it raining there?", "It might be."),
        ("I prefer window seats and I'm vegan.", "Noted! I'll keep that in mind."),
        ("My budget is $2000.", "That's a good budget for Paris.")
    ]

    print("\n--- Starting Conversation Simulation ---")
    for i, (user_q, bot_a) in enumerate(test_interactions):
        print(f"\n{'=' * 60}")
        print(f"Turn {i + 1}: User='{user_q}'")

        # 1. Update Short Term Memory
        memory.add_to_stm(user_q, bot_a)

        # 2. Update Long Term Memory (Extract Preferences)
        memory.update_ltm(user_q, bot_a)

    print("\n" + "=" * 60)
    print("--- Final Memory State ---")
    print("=" * 60)
    context = memory.get_memory_context()
    print(f"\n📋 STM Summary:\n   {context['stm_summary']}")
    print(f"\n👤 LTM Summary:\n   {context['ltm_summary']}")
    print(f"\n🔍 Raw LTM:\n{json.dumps(memory.ltm, indent=2)}")
