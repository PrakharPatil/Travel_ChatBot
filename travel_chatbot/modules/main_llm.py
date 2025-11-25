from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class MainLLM:
    def __init__(self, model_name='google/flan-t5-base'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Fallback for CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        print(f"MainLLM: Device set to use {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _format_context_string(self, context: dict):
        """Helper to flatten the context dictionary into a readable string"""
        parts = []

        # 1. Memory (Who is the user?)
        if context.get('memory'):
            mem = context['memory']
            # Check both summary fields
            if mem.get('ltm_summary') and mem['ltm_summary'] != 'No prior data':
                parts.append(f"User Info: {mem['ltm_summary']}")
            elif mem.get('stm_summary'):
                parts.append(f"User Info: {mem['stm_summary']}")

        # 2. API (Live Data)
        if context.get('api_result'):
            api_data = context['api_result']
            if isinstance(api_data, dict):
                data_str = ", ".join([f"{k}: {v}" for k, v in api_data.items() if k != 'raw'])
                parts.append(f"Live Status: {data_str}")
            else:
                parts.append(f"Live Status: {api_data}")

        # 3. RAG (Database Knowledge)
        if context.get('rag_context'):
            rag = context['rag_context']
            if isinstance(rag, list):
                rag_str = " | ".join([str(item) for item in rag[:3]])
                parts.append(f"Database: {rag_str}")
            else:
                parts.append(f"Database: {rag}")

        if not parts:
            return "No specific context available."

        return "\n".join(parts)

    def generate_response(self, query: str, context: dict):
        """Generate response using Few-Shot CoT"""

        # 1. Prepare context
        context_str = self._format_context_string(context)

        # 2. Comprehensive Prompt with 5 Distinct Examples
        prompt = f"""Task: You are a helpful AI Travel Assistant. Follow the format: Thought (internal reasoning), then Answer (final response to user).

Example 1 (Conversational):
Context: No specific context available.
User Query: "Hello! How are you?"
Thought: The user is greeting me. I should greet them back warmly.
Answer: I'm doing great! I'm ready to help you plan your next adventure. Where would you like to go?

Example 2 (Using Database/RAG):
Context: Database: {{"city": "Tokyo", "attraction": "Senso-ji Temple"}} | {{"city": "Tokyo", "food": "Sushi"}}
User Query: "What is there to do in Tokyo?"
Thought: The database mentions Senso-ji Temple and Sushi in Tokyo.
Answer: In Tokyo, you must visit the historic Senso-ji Temple. Afterward, I highly recommend trying some authentic Sushi!

Example 3 (Using Live API):
Context: Live Status: Weather: Rain, Temp: 12C, Location: London
User Query: "Should I pack shorts for London?"
Thought: Live status says it is raining and 12C in London. Shorts are a bad idea.
Answer: It is currently 12°C and raining in London, so I would suggest packing a raincoat and warm trousers instead of shorts.

Example 4 (Using User Info/Memory):
Context: User Info: The user loves hiking and nature.
User Query: "Suggest a place to visit in the US."
Thought: User loves hiking. I should suggest a National Park like Yosemite.
Answer: Since you love hiking and nature, I highly recommend visiting Yosemite National Park. The trails there are incredible.

Example 5 (Context Conflict/Correction):
Context: User Info: Budget traveler. Database: {{"hotel": "Ritz Carlton (Luxury)"}}, {{"hotel": "Backpacker Hostel (Cheap)"}}
User Query: "Where should I stay?"
Thought: The user is on a budget. I should ignore the Luxury hotel and suggest the Hostel.
Answer: For a budget-friendly option, the Backpacker Hostel is a great choice.

Actual Task:
Context:
{context_str}
User Query: "{query}"
Thought:"""

        # 3. Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        # 4. Generate
        # increased max_new_tokens to allow for the thought process + answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,  # Lower temperature for more consistent formatting
                do_sample=True,
                repetition_penalty=1.1
            )

        # 5. Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 6. PARSING LOGIC (The Fix)
        # The model generates: "The user is... Answer: The response..."
        # We split by 'Answer:' and take the last part.

        if "Answer:" in full_output:
            # We take the text AFTER "Answer:"
            final_response = full_output.split("Answer:", 1)[-1].strip()
        else:
            # Fallback: If the model forgot to say "Answer:", we try to salvage the response.
            # Usually, if it fails format, the whole thing might be the answer.
            final_response = full_output.strip()

        # Debug print to see what the model actually thought (Optional)
        # print(f"\n--- DEBUG ---\nFull Generation: {full_output}\nParsed: {final_response}\n-------------")

        return final_response


# --- Quick Test ---
if __name__ == "__main__":
    llm = MainLLM()

    # Test Case 7: RAG Query 3
    test_context = {
        "memory": {"ltm_summary": "User likes spicy food."},
        "rag_context": [{"city": "London", "food": "Chicken Tikka Masala"},
                        {"city": "London", "food": "Fish and Chips"}]
    }

    response = llm.generate_response("What food should I try in London?", test_context)
    print(f"Bot Response: {response}")