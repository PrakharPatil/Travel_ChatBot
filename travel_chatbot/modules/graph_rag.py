import json
import re
import yaml  # Added yaml import
from neo4j import GraphDatabase
from transformers import pipeline
import torch


class GraphRAG:
    def __init__(self, neo4j_config, llm_model='google/flan-t5-base'):
        self.driver = GraphDatabase.driver(
            neo4j_config['uri'],
            auth=(neo4j_config['user'], neo4j_config['password'])
        )
        # T5 uses text2text-generation
        task = "text2text-generation" if "t5" in llm_model.lower() else "text-generation"
        print(f"GraphRAG: Loading {llm_model} with task '{task}'")

        # Load model
        self.llm = pipeline(task, model=llm_model, dtype=torch.float32, device_map="auto")

    def _clean_llm_response(self, response):
        """Cleans the raw text from the LLM"""
        if isinstance(response, list) and 'generated_text' in response[0]:
            text = response[0]['generated_text']
        else:
            text = str(response)

        # Remove common markdown and 'Thought' debris
        text = text.replace('```cypher', '').replace('```', '').strip()
        if "Cypher:" in text:
            text = text.split("Cypher:")[-1].strip()
        return text

    def create_graph_from_data(self, data_path='data/travel_data.json'):
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            with self.driver.session() as session:
                # Create constraint (optional, speeds up lookups)
                try:
                    session.run("CREATE CONSTRAINT FOR (c:City) REQUIRE c.name IS UNIQUE")
                except:
                    pass  # Constraint might already exist

                for city in data.get('cities', []):
                    # Create City
                    session.run(
                        """
                        MERGE (c:City {name: $name}) 
                        SET c.country = $country, c.best_time = $best_time, c.budget = $budget
                        """,
                        name=city['name'], country=city['country'], best_time=city['best_time'], budget=city['budget']
                    )
                    # Create Attractions and Relationships
                    for attraction in city['attractions']:
                        session.run(
                            """
                            MERGE (a:Attraction {name: $att}) 
                            MERGE (c:City {name: $city}) 
                            MERGE (c)-[:HAS_ATTRACTION]->(a)
                            """,
                            att=attraction, city=city['name']
                        )
            print("Graph creation/update complete.")
        except Exception as e:
            print(f"Data loading error: {e}")

    def extract_entities_with_llm(self, query: str):
        # We simplify this for T5-base. It prefers direct questions over "Thoughts".
        prompt = f"""Extract the city name from the sentence.
Query: What to do in Tokyo?
City: Tokyo

Query: I want to visit Paris.
City: Paris

Query: Is London expensive?
City: London

Query: {query}
City:"""

        result = self.llm(prompt, max_new_tokens=20, do_sample=False)
        raw_text = self._clean_llm_response(result)

        # Cleanup extra words if the model babbles
        city = raw_text.split('\n')[0].strip()

        # Fallback if empty
        if not city:
            city = "Paris"  # Default fallback

        return {'city': city}

    def generate_cypher_with_llm(self, entities):
        city = entities.get('city', 'Paris')

        # STRATEGY: 5 Strict Examples (Few-Shot Prompting)
        # We explicitly teach the schema: (City {name})-[:HAS_ATTRACTION]->(Attraction {name})
        prompt = f"""Translate the question into a Neo4j Cypher query.
Schema: (City {{name}})-[:HAS_ATTRACTION]->(Attraction {{name}})

Q: Find attractions in Paris.
A: MATCH (c:City {{name: 'Paris'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name

Q: What can I do in Tokyo?
A: MATCH (c:City {{name: 'Tokyo'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name

Q: Show me places to visit in London.
A: MATCH (c:City {{name: 'London'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name

Q: List the tourist spots for Berlin.
A: MATCH (c:City {{name: 'Berlin'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name

Q: What are the sights in New York?
A: MATCH (c:City {{name: 'New York'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name

Q: Find attractions in {city}.
A:"""

        try:
            # Generate
            result = self.llm(prompt, max_new_tokens=100, do_sample=False)
            cypher = self._clean_llm_response(result)

            print(f"Generated Cypher: {cypher}")

            # VALIDATION: Common T5-base errors fixing
            # T5 sometimes misses the closing brace '}' or single quotes
            if "name:" in cypher and "{" not in cypher:
                # Force inject the braces if missing
                cypher = f"MATCH (c:City {{name: '{city}'}})-[:HAS_ATTRACTION]->(a) RETURN c.name, a.name"

            # Ensure proper quote closure
            if cypher.count("'") % 2 != 0:
                cypher = cypher + "'"

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

        with self.driver.session() as session:
            try:
                result = session.run(cypher)
                records = list(result)
                if records:
                    return self.format_context(records)

                # Fuzzy Fallback if exact match fails
                print("No exact match, trying fuzzy search...")
                fuzzy = f"MATCH (c:City) WHERE toLower(c.name) CONTAINS toLower('{entities['city']}') RETURN c.name, c.country"
                return self.format_context(list(session.run(fuzzy)))
            except Exception as e:
                return [{"error": str(e)}]

    def close(self):
        self.driver.close()


# --- TESTING BLOCK ---
if __name__ == "__main__":
    import os

    # 1. SETUP: Load configuration from YAML
    NEO4J_CONFIG = {}
    try:
        with open('../configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            if 'neo4j' in config:
                NEO4J_CONFIG = config['neo4j']
            else:
                raise ValueError("Key 'neo4j' not found in config.yaml")
    except FileNotFoundError:
        print("Config file not found. Please ensure 'configs/config.yaml' exists.")
        exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        exit(1)

    # 2. SETUP: Create dummy data
    test_data_path = 'data/test_travel_data.json'
    if not os.path.exists('data'):
        os.makedirs('data')

    dummy_data = {
        "cities": [
            {
                "name": "Paris",
                "country": "France",
                "best_time": "Spring",
                "budget": "High",
                "attractions": ["Eiffel Tower", "Louvre Museum"]
            },
            {
                "name": "Tokyo",
                "country": "Japan",
                "best_time": "Spring",
                "budget": "High",
                "attractions": ["Senso-ji", "Tokyo Tower"]
            }
        ]
    }

    with open(test_data_path, 'w') as f:
        json.dump(dummy_data, f)

    # 3. EXECUTION
    rag = None
    try:
        print("\n--- Initializing GraphRAG ---")
        rag = GraphRAG(NEO4J_CONFIG, llm_model='google/flan-t5-base')

        print("\n--- Loading Data ---")
        rag.create_graph_from_data(test_data_path)

        test_query = "What famous attractions are there in Paris?"
        print(f"\n--- Testing Query: '{test_query}' ---")
        results = rag.retrieve_context(test_query)

        print("\n--- Final Results ---")
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Tip: Check your Neo4j credentials in configs/config.yaml")

    finally:
        if rag:
            rag.close()
            print("\n--- Connection Closed ---")