# modules/api_module.py
import json
import requests
from transformers import pipeline
import torch


class AskToActAPI:
    def __init__(self, serpapi_key, llm_model='mistralai/Mistral-7B-Instruct-v0.2'):
        self.serpapi_key = serpapi_key
        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            dtype=torch.float16,
            device_map="auto"
        )

        # API schemas
        self.api_schemas = {
            "flight_search": {
                "required": ["departure_id", "arrival_id", "outbound_date"],
                "optional": ["return_date", "adults", "currency"]
            },
            "weather": {
                "required": ["location"],
                "optional": ["date", "units"]
            },
            "hotel_search": {
                "required": ["location", "check_in_date", "check_out_date"],
                "optional": ["adults", "currency", "rating"]
            }
        }

    def detect_api_type_with_llm(self, query: str):
        """LLM determines which API to call"""
        prompt = f"""<s>[INST] Determine which API should be called for this travel query:

Query: "{query}"

Available APIs:
- flight_search: For flight information, prices, availability
- weather: For weather information
- hotel_search: For hotel information, prices, availability

Respond with ONLY the API name. [/INST]"""

        result = self.llm(prompt, max_new_tokens=20, do_sample=False)[0]['generated_text']
        api_type = result.split('[/INST]')[-1].strip().lower()

        for api in self.api_schemas.keys():
            if api in api_type:
                return api

        return "flight_search"  # Default

    def extract_parameters_with_llm(self, query: str, api_type: str, conversation_history: list = None):
        """LLM extracts API parameters from query"""
        schema = self.api_schemas[api_type]

        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation:\n" + "\n".join([
                f"User: {turn['query']}\nBot: {turn['response']}"
                for turn in conversation_history[-3:]
            ])

        prompt = f"""<s>[INST] Extract API parameters from the user query. Consider the conversation history for context.

API: {api_type}
Required parameters: {schema['required']}
Optional parameters: {schema['optional']}

Query: "{query}"{history_context}

Extract the parameters and respond in JSON format. For missing required parameters, set value to null.

Example format:
{{
  "departure_id": "JFK",
  "arrival_id": "LAX",
  "outbound_date": "2025-12-01",
  "adults": 1
}}

Respond with ONLY valid JSON. [/INST]"""

        result = self.llm(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
        json_str = result.split('[/INST]')[-1].strip()

        # Extract JSON from response
        try:
            # Find JSON object in response
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = json_str[start:end]

            parameters = json.loads(json_str)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            parameters = {}

        return parameters

    def identify_missing_parameters(self, parameters: dict, api_type: str):
        """Check for missing required parameters"""
        schema = self.api_schemas[api_type]
        missing = []

        for param in schema['required']:
            if param not in parameters or parameters[param] is None or parameters[param] == "":
                missing.append(param)

        return missing

    def generate_clarification_with_llm(self, missing_params: list, api_type: str):
        """LLM generates natural clarification question"""
        prompt = f"""<s>[INST] Generate a natural, friendly question to ask the user for missing information.

API Type: {api_type}
Missing Parameters: {missing_params}

Generate ONE concise question that asks for all missing information naturally.

Examples:
- "Where are you flying from and to? Also, what date?"
- "Which city would you like to check the weather for?"
- "When do you want to check in and check out?"

Generate ONLY the question. [/INST]"""

        result = self.llm(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]['generated_text']
        question = result.split('[/INST]')[-1].strip()

        return question

    def call_flight_api(self, params: dict):
        """Call Google Flights via SerpAPI"""
        url = "https://serpapi.com/search.json"

        api_params = {
            "engine": "google_flights",
            "departure_id": params.get('departure_id'),
            "arrival_id": params.get('arrival_id'),
            "outbound_date": params.get('outbound_date'),
            "api_key": self.serpapi_key
        }

        if params.get('return_date'):
            api_params['return_date'] = params['return_date']

        try:
            response = requests.get(url, params=api_params)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def call_weather_api(self, params: dict):
        """Call weather API"""
        # Using wttr.in as free alternative
        location = params.get('location', '').replace(' ', '+')
        url = f"https://wttr.in/{location}?format=j1"

        try:
            response = requests.get(url)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def process(self, query: str, conversation_history: list = None):
        """Main AskToAct pipeline"""
        # Step 1: Detect API type
        api_type = self.detect_api_type_with_llm(query)

        # Step 2: Extract parameters
        parameters = self.extract_parameters_with_llm(query, api_type, conversation_history)

        # Step 3: Check for missing parameters
        missing = self.identify_missing_parameters(parameters, api_type)

        if missing:
            # Generate clarification question
            clarification = self.generate_clarification_with_llm(missing, api_type)
            return {
                'status': 'missing_params',
                'missing': missing,
                'clarification': clarification,
                'partial_params': parameters
            }

        # Step 4: Call API
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
