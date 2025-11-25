# modules/main_llm.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class MainLLM:
    def __init__(self, model_name='google/flan-t5-base'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device set to use {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_response(self, query: str, context: dict):
        """Generate response using context from RAG/API/Memory"""

        # Build prompt
        prompt = f"You are a helpful travel assistant. Answer the travel question naturally.\n\n"

        # Add memory context
        if 'memory' in context:
            memory = context['memory']
            if memory.get('stm_summary'):
                prompt += f"Recent conversation: {memory['stm_summary']}\n"
            if memory.get('ltm_summary'):
                prompt += f"User preferences: {memory['ltm_summary']}\n"

        # Add RAG context
        if context.get('rag_context'):
            prompt += f"\nKnowledge: {context['rag_context']}\n"

        # Add API results
        if context.get('api_result'):
            prompt += f"\nLive data: {context['api_result']}\n"

        prompt += f"\nQuestion: {query}\nAnswer:"

        # Generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
