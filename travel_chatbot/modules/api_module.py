import json
import requests
import yaml
from transformers import pipeline
import torch


class AskToActAPI:
    def __init__(self, serpapi_key, llm_model='google/flan-t5-base'):
        self.serpapi_key = serpapi_key

        # Dynamic task selection
        task = "text2text-generation" if "t5" in llm_model.lower() else "text-generation"
        print(f"API Module: Loading {llm_model} with task '{task}'")

        self.llm = pipeline(
            task,
            model=llm_model,
            dtype=torch.float32,
            device_map="auto"
        )

        self.api_schemas = {
            "flight_search": {
                "required": ["departure_id", "arrival_id", "outbound_date"],
                "optional": ["return_date"]
            },
            "weather": {
                "required": ["location"],
                "optional": []
            },
            "hotel_search": {
                "required": ["location", "check_in_date"],
                "optional": ["currency"]
            }
        }

    def _clean_llm_response(self, response):
        """Helper to clean response"""
        text = response[0]['generated_text']
        if "[/INST]" in text:
            return text.split('[/INST]')[-1].strip()
        return text.strip()

    def detect_api_type_with_llm(self, query: str):
        # STRATEGY: Chain of Thought (CoT)
        # We ask the model to "think" about the intent before assigning a label.

        prompt = f"""Task: Identify the API Intent (flight_search, weather, hotel_search).

Example 1:
Query: "Is it raining in Tokyo today?"
Thought: The user is asking about rain and conditions. This is a weather request.
Intent: weather

Example 2:
Query: "I need to fly from NY to London next week."
Thought: The user wants to travel by air between cities. This is a flight request.
Intent: flight_search

Example 3:
Query: "Find a cheap room in Paris for 2 nights."
Thought: 'Room' implies accommodation/hotel.
Intent: hotel_search

Task:
Query: "{query}"
Thought:"""

        # Increased max tokens to allow for the "Thought" generation
        result = self.llm(prompt, max_new_tokens=40, do_sample=False)
        response_text = self._clean_llm_response(result).lower()

        # Robust Parsing: Look for the intent keyword
        if "weather" in response_text: return "weather"
        if "hotel" in response_text: return "hotel_search"
        if "flight" in response_text: return "flight_search"
        if "fly" in response_text: return "flight_search"

        return "flight_search"  # Default fallback

    def extract_parameters_with_llm(self, query: str, api_type: str, conversation_history: list = None):
        # STRATEGY: Text-to-Key-Value (No JSON)
        # T5 is bad at JSON brackets. We ask for "Key: Value | Key: Value" format instead.

        context_str = ""
        if conversation_history:
            # We include history so it can resolve "there" or "tomorrow" based on context
            context_str = " ".join([t['query'] for t in conversation_history[-2:]])

        full_input = f"{context_str} {query}".strip()

        prompt = ""
        if api_type == "weather":
            prompt = f"""Task: Extract Location for weather.
Example 1:
Query: "Weather in Tokyo"
Thought: The location is Tokyo.
Output: Location: Tokyo

Example 2:
Query: "How is it in New York?"
Thought: The location is New York.
Output: Location: New York

Task:
Query: "{full_input}"
Thought:"""

        elif api_type == "flight_search":
            prompt = f"""Task: Extract Departure, Arrival, and Date.
Example 1:
Query: "Fly from NY to London on 2025-05-01"
Thought: From NY (Departure) to London (Arrival) on 2025-05-01 (Date).
Output: Departure: NY | Arrival: London | Date: 2025-05-01

Example 2:
Query: "Book a flight to Paris from Berlin tomorrow"
Thought: From Berlin (Departure) to Paris (Arrival) date is tomorrow.
Output: Departure: Berlin | Arrival: Paris | Date: tomorrow

Task:
Query: "{full_input}"
Thought:"""

        elif api_type == "hotel_search":
            prompt = f"""Task: Extract Location and Check-in Date.
Example 1:
Query: "Hotel in Dubai for tomorrow"
Thought: City is Dubai, time is tomorrow.
Output: Location: Dubai | Date: tomorrow

Task:
Query: "{full_input}"
Thought:"""

        result = self.llm(prompt, max_new_tokens=80, do_sample=False)
        text = self._clean_llm_response(result)

        # --- Python Parsing Logic (Replaces fragile JSON parsing) ---
        parameters = {}

        # We split by '|' or newlines, then look for "Key:"
        parts = text.replace('|', '\n').split('\n')

        for part in parts:
            part = part.strip()
            if "Location:" in part:
                parameters['location'] = part.split("Location:")[-1].strip()
            elif "Departure:" in part:
                parameters['departure_id'] = part.split("Departure:")[-1].strip()
            elif "Arrival:" in part:
                parameters['arrival_id'] = part.split("Arrival:")[-1].strip()
            elif "Date:" in part:
                # Map 'Date' to the specific field needed based on API type
                val = part.split("Date:")[-1].strip()
                if api_type == "flight_search":
                    parameters['outbound_date'] = val
                else:
                    parameters['check_in_date'] = val

        return parameters

    def identify_missing_parameters(self, parameters: dict, api_type: str):
        schema = self.api_schemas[api_type]
        missing = []
        for param in schema['required']:
            # Check if key exists and is not empty
            if param not in parameters or not parameters[param] or parameters[param].lower() == "none":
                missing.append(param)
        return missing

    def generate_clarification_with_llm(self, missing_params: list, api_type: str):
        # STRATEGY: Conversational Refinement
        prompt = f"""Task: Politely ask the user for missing travel information.

Example 1:
Missing: location
Output: "Where would you like to check the weather for?"

Example 2:
Missing: departure_id, outbound_date
Output: "Where are you flying from, and when do you want to leave?"

Example 3:
Missing: arrival_id
Output: "What is your destination city?"

Task:
Missing: {', '.join(missing_params)}
Output:"""

        result = self.llm(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
        return self._clean_llm_response(result)

    def call_flight_api(self, params: dict):
        #
        if not self.serpapi_key:
            return {"flights": [{"airline": "Mock Airline", "price": "$500", "duration": "10h"}]}

        url = "https://serpapi.com/search.json"
        api_params = {
            "engine": "google_flights",
            "departure_id": params.get('departure_id'),
            "arrival_id": params.get('arrival_id'),
            "outbound_date": params.get('outbound_date'),
            "api_key": self.serpapi_key,
            "currency": "USD"
        }

        try:
            print(f"Calling Flight API with: {api_params}")
            response = requests.get(url, params=api_params, timeout=10)
            data = response.json()

            if 'best_flights' in data:
                return data['best_flights'][:2]
            return {"status": "No flights found", "raw": str(data)[:100]}

        except Exception as e:
            return {"error": f"Flight API Error: {str(e)}"}

    def call_weather_api(self, params: dict):
        location = params.get('location', 'London').replace(' ', '+')
        url = f"https://wttr.in/{location}?format=3"

        try:
            response = requests.get(url, timeout=5)
            return {"weather_report": response.text.strip()}
        except Exception as e:
            return {"error": f"Weather API Error: {str(e)}"}

    def process(self, query: str, conversation_history: list = None):
        # 1. Detect
        api_type = self.detect_api_type_with_llm(query)
        print(f"API Detected: {api_type}")

        # 2. Extract
        parameters = self.extract_parameters_with_llm(query, api_type, conversation_history)
        print(f"Extracted Params: {parameters}")

        # 3. Check Missing
        missing = self.identify_missing_parameters(parameters, api_type)

        if missing:
            clarification = self.generate_clarification_with_llm(missing, api_type)
            return {
                'status': 'missing_params',
                'missing': missing,
                'clarification': clarification,
                'partial_params': parameters
            }

        # 4. Execute
        if api_type == "flight_search":
            api_result = self.call_flight_api(parameters)
        elif api_type == "weather":
            api_result = self.call_weather_api(parameters)
        else:
            api_result = {"error": "API not implemented"}

        return {
            'status': 'success',
            'api_type': api_type,
            'parameters': parameters,
            'result': api_result
        }

# --- TESTING FUNCTIONALITIES ---
if __name__ == "__main__":
    import os

    # 1. SETUP: Load Config
    SERPAPI_KEY = None
    try:
        with open('../configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            if 'apis' in config and 'serpapi_key' in config['apis']:
                SERPAPI_KEY = config['apis']['serpapi_key']
    except Exception as e:
        print(f"Config Warning: {e}. Using mock data for flights.")

    # 2. INITIALIZE
    print("\n--- Initializing API Module ---")
    agent = AskToActAPI(serpapi_key=SERPAPI_KEY, llm_model='google/flan-t5-base')

    # 3. TEST CASES
    test_queries = [
        "What is the weather in Tokyo?",
        "I want to book a flight to Paris.",  # Should trigger missing params
        "Fly from New York to London on 2025-05-01" # Should trigger success (or mock)
    ]

    for q in test_queries:
        print(f"\n--- Testing Query: '{q}' ---")
        result = agent.process(q)
        print(f"Result:\n{json.dumps(result, indent=2)}")