import json
import requests
import re


class AskToActAPI:
    def __init__(self, serpapi_key, llm_pipeline):
        self.serpapi_key = serpapi_key
        self.llm = llm_pipeline  # Use shared pipeline

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
        prompt = f"""<s>[INST] Determine which API should be called for this travel query:
Query: "{query}"

Available APIs:
- flight_search: For flight information, prices, availability
- weather: For weather information
- hotel_search: For hotel information, prices, availability

Respond with ONLY the API name. [/INST]"""

        # Reduced max_new_tokens for speed
        result = self.llm(prompt, max_new_tokens=15, do_sample=False)[0]['generated_text']
        api_type = result.split('[/INST]')[-1].strip().lower()

        for api in self.api_schemas.keys():
            if api in api_type:
                return api
        return "flight_search"

    def extract_parameters_with_llm(self, query: str, api_type: str, conversation_history: list = None):
        schema = self.api_schemas[api_type]

        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation:\n" + "\n".join([
                f"User: {turn['query']}\nBot: {turn['response']}"
                for turn in conversation_history[-3:]
            ])

        prompt = f"""<s>[INST] Extract API parameters from the user query.
API: {api_type}
Required parameters: {schema['required']}
Optional parameters: {schema['optional']}

Query: "{query}"{history_context}

Extract the parameters and respond in valid JSON format.
Example:
{{
  "departure_id": "JFK",
  "arrival_id": "LAX",
  "outbound_date": "2025-12-01"
}}
Respond with ONLY valid JSON. [/INST]"""

        result = self.llm(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
        text_output = result.split('[/INST]')[-1].strip()

        # Robust JSON extraction
        try:
            match = re.search(r'\{.*\}', text_output, re.DOTALL)
            if match:
                json_str = match.group()
                parameters = json.loads(json_str)
            else:
                parameters = {}
        except Exception as e:
            print(f"JSON parsing error: {e}")
            parameters = {}

        return parameters

    def identify_missing_parameters(self, parameters: dict, api_type: str):
        schema = self.api_schemas[api_type]
        missing = []
        for param in schema['required']:
            if param not in parameters or parameters[param] is None or parameters[param] == "":
                missing.append(param)
        return missing

    def generate_clarification_with_llm(self, missing_params: list, api_type: str):
        prompt = f"""<s>[INST] Generate a natural question to ask for missing information: {missing_params} for {api_type}.
Generate ONLY the question. [/INST]"""
        result = self.llm(prompt, max_new_tokens=60, do_sample=True, temperature=0.7)[0]['generated_text']
        return result.split('[/INST]')[-1].strip()

    def call_flight_api(self, params: dict):
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
            response = requests.get(url, params=api_params, timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def call_weather_api(self, params: dict):
        location = params.get('location', '').replace(' ', '+')
        url = f"https://wttr.in/{location}?format=j1"
        try:
            response = requests.get(url, timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def process(self, query: str, conversation_history: list = None):
        api_type = self.detect_api_type_with_llm(query)
        parameters = self.extract_parameters_with_llm(query, api_type, conversation_history)
        missing = self.identify_missing_parameters(parameters, api_type)

        if missing:
            clarification = self.generate_clarification_with_llm(missing, api_type)
            return {
                'status': 'missing_params',
                'missing': missing,
                'clarification': clarification,
                'partial_params': parameters
            }

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