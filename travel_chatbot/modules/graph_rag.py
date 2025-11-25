# modules/graph_rag.py
import json
from neo4j import GraphDatabase
from transformers import pipeline
import torch


class GraphRAG:
    def __init__(self, neo4j_config, llm_model='google/flan-t5-base'):
        self.driver = GraphDatabase.driver(
            neo4j_config['uri'],
            auth=(neo4j_config['user'], neo4j_config['password'])
        )
        task = "text2text-generation" if "t5" in llm_model.lower() else "text-generation"
        print(f"GraphRAG: Loading {llm_model} with task '{task}'")
        self.llm = pipeline(task, model=llm_model, dtype=torch.float32, device_map="auto")

    def _clean_llm_response(self, response):
        text = response[0]['generated_text']
        if "[/INST]" in text: return text.split('[/INST]')[-1].strip()
        return text.strip()

    def create_graph_from_data(self, data_path='data/travel_data.json'):
        # ... (Same as your original code) ...
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            with self.driver.session() as session:
                for city in data.get('cities', []):
                    session.run(
                        "MERGE (c:City {name: $name}) SET c.country = $country, c.best_time = $best_time, c.budget = $budget",
                        name=city['name'], country=city['country'], best_time=city['best_time'], budget=city['budget']
                    )
                    for attraction in city['attractions']:
                        session.run(
                            "MERGE (a:Attraction {name: $att}) MERGE (c:City {name: $city}) MERGE (c)-[:HAS_ATTRACTION]->(a)",
                            att=attraction, city=city['name']
                        )
            print("Graph creation/update complete.")
        except Exception as e:
            print(f"Data loading error: {e}")

    def extract_entities_with_llm(self, query: str):
        # FIX: CoT + 3 Examples
        prompt = f"""Task: Extract the main City from the user query.

Example 1:
Query: "What is the food like in Tokyo?"
Thought: The user is asking about 'Tokyo'.
City: Tokyo

Example 2:
Query: "I want to visit Paris museums."
Thought: The specific location mentioned is 'Paris'.
City: Paris

Example 3:
Query: "How much does it cost to go to New York?"
Thought: The destination is 'New York'.
City: New York

Task:
Query: "{query}"
Thought:"""

        # Increased tokens to allow for "Thought" generation
        result = self.llm(prompt, max_new_tokens=40, do_sample=False)
        response = self._clean_llm_response(result)

        # Parse logic: Look for "City:"
        if "City:" in response:
            city = response.split("City:")[-1].strip()
        else:
            # Fallback: take the last word if format fails
            city = response.split()[-1].strip()

        return {'city': city, 'info_type': 'general', 'preferences': ''}

    def generate_cypher_with_llm(self, entities):
        city = entities.get('city', 'Paris')

        # FIX: CoT + Schema Awareness + 3 Examples
        prompt = f"""Task: Write a Neo4j Cypher query.
Schema: (City {{name: str}})-[:HAS_ATTRACTION]->(Attraction {{name: str}})

Example 1:
Goal: Find attractions in Paris.
Thought: I need to match the City node 'Paris' and find connected Attraction nodes.
Cypher: MATCH (c:City {{name: 'Paris'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name

Example 2:
Goal: Find Tokyo attractions.
Thought: Target city is 'Tokyo'. I will match the City pattern.
Cypher: MATCH (c:City {{name: 'Tokyo'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name

Example 3:
Goal: Show me what to do in London.
Thought: 'What to do' implies attractions. Target is 'London'.
Cypher: MATCH (c:City {{name: 'London'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name

Task:
Goal: Find {city} attractions.
Thought:"""

        try:
            result = self.llm(prompt, max_new_tokens=120, do_sample=False)
            raw_text = self._clean_llm_response(result)

            # Extract Cypher after the "Cypher:" keyword if present
            if "Cypher:" in raw_text:
                cypher = raw_text.split("Cypher:")[-1].strip()
            else:
                cypher = raw_text

            # Clean markdown
            cypher = cypher.replace('```cypher', '').replace('```', '').strip()

            # Robustness Check
            if "MATCH" not in cypher or "RETURN" not in cypher:
                raise ValueError("Invalid Cypher structure")
            if "name:" in cypher and "{" not in cypher:
                raise ValueError("Syntax error (missing brackets)")

            return cypher

        except Exception as e:
            print(f"LLM Cypher Fallback triggered: {e}")
            return f"MATCH (c:City {{name: '{city}'}})-[:HAS_ATTRACTION]->(a:Attraction) RETURN c.name as city, collect(a.name) as attractions"

    def format_context(self, records):
        return [dict(record) for record in records] if records else []

    def retrieve_context(self, query: str):
        entities = self.extract_entities_with_llm(query)
        print(f"GraphRAG Entity: {entities['city']}")

        cypher = self.generate_cypher_with_llm(entities)
        print(f"GraphRAG Cypher: {cypher}")

        with self.driver.session() as session:
            try:
                result = session.run(cypher)
                records = list(result)
                if records: return self.format_context(records)

                # Fuzzy Fallback
                print("No exact match, trying fuzzy search...")
                fuzzy = f"MATCH (c:City) WHERE toLower(c.name) CONTAINS toLower('{entities['city']}') RETURN c.name, c.country"
                return self.format_context(list(session.run(fuzzy)))
            except Exception as e:
                return [{"error": str(e)}]

    def close(self):
        self.driver.close()