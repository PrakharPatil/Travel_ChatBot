# modules/graph_rag.py
import json
from neo4j import GraphDatabase
from transformers import pipeline
import torch


class GraphRAG:
    def __init__(self, neo4j_config, llm_model='mistralai/Mistral-7B-Instruct-v0.2'):
        self.driver = GraphDatabase.driver(
            neo4j_config['uri'],
            auth=(neo4j_config['user'], neo4j_config['password'])
        )

        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            dtype=torch.float16,
            device_map="auto"
        )

    def create_graph_from_data(self, data_path='data/travel_data.json'):
        """Create Neo4j graph from travel data using LLM for entity extraction"""
        with open(data_path, 'r') as f:
            data = json.load(f)

        with self.driver.session() as session:
            for city in data['cities']:
                # Create city node
                session.run(
                    """
                    MERGE (c:City {name: $name})
                    SET c.country = $country,
                        c.best_time = $best_time,
                        c.budget = $budget
                    """,
                    name=city['name'],
                    country=city['country'],
                    best_time=city['best_time'],
                    budget=city['budget']
                )

                # Create attraction nodes and relationships
                for attraction in city['attractions']:
                    session.run(
                        """
                        MERGE (a:Attraction {name: $attraction})
                        MERGE (c:City {name: $city})
                        MERGE (c)-[:HAS_ATTRACTION]->(a)
                        """,
                        attraction=attraction,
                        city=city['name']
                    )

                # Create cuisine nodes
                for cuisine in city['cuisine']:
                    session.run(
                        """
                        MERGE (cu:Cuisine {name: $cuisine})
                        MERGE (c:City {name: $city})
                        MERGE (c)-[:OFFERS_CUISINE]->(cu)
                        """,
                        cuisine=cuisine,
                        city=city['name']
                    )

    def extract_entities_with_llm(self, query: str):
        """Use LLM to extract entities and intent from query"""
        prompt = f"""<s>[INST] Extract the following information from the travel query:
- City/Destination (if mentioned)
- Type of information sought (attractions, cuisine, budget, best time, etc.)
- Any specific preferences

Query: "{query}"

Respond in this format:
City: <city name or "any">
Info Type: <type>
Preferences: <any specific preferences>
[/INST]"""

        result = self.llm(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
        response = result.split('[/INST]')[-1].strip()

        # Parse response
        entities = {
            'city': 'any',
            'info_type': 'general',
            'preferences': []
        }

        for line in response.split('\n'):
            if 'City:' in line:
                entities['city'] = line.split('City:')[-1].strip()
            elif 'Info Type:' in line:
                entities['info_type'] = line.split('Info Type:')[-1].strip()
            elif 'Preferences:' in line:
                entities['preferences'] = line.split('Preferences:')[-1].strip()

        return entities

    def generate_cypher_with_llm(self, entities: dict):
        """Generate Cypher query using LLM"""
        prompt = f"""<s>[INST] Generate a Neo4j Cypher query based on these extracted entities:
City: {entities['city']}
Info Type: {entities['info_type']}
Preferences: {entities['preferences']}

The graph has these node types:
- City (properties: name, country, best_time, budget)
- Attraction (properties: name)
- Cuisine (properties: name)

Relationships:
- (City)-[:HAS_ATTRACTION]->(Attraction)
- (City)-[:OFFERS_CUISINE]->(Cuisine)

Generate ONLY the Cypher query, nothing else. [/INST]"""

        result = self.llm(prompt, max_new_tokens=200, do_sample=False)[0]['generated_text']
        cypher = result.split('[/INST]')[-1].strip()

        return cypher

    # def retrieve_context(self, query: str):
    #     """Main retrieval method"""
    #     # Extract entities using LLM
    #     entities = self.extract_entities_with_llm(query)
    #
    #     # Generate Cypher query using LLM
    #     cypher = self.generate_cypher_with_llm(entities)
    #
    #     # Execute query
    #     try:
    #         with self.driver.session() as session:
    #             result = session.run(cypher)
    #             context = [dict(record) for record in result]
    #     except Exception as e:
    #         print(f"Cypher execution error: {e}")
    #         # Fallback: Simple query
    #         with self.driver.session() as session:
    #             result = session.run(
    #                 """
    #                 MATCH (c:City)-[:HAS_ATTRACTION]->(a:Attraction)
    #                 WHERE c.name CONTAINS $query OR a.name CONTAINS $query
    #                 RETURN c.name as city, collect(a.name) as attractions, c.country as country
    #                 LIMIT 5
    #                 """,
    #                 query=entities['city']
    #             )
    #             context = [dict(record) for record in result]
    #
    #     return context
    # modules/graph_rag.py - Fix retrieve_context method

    def retrieve_context(self, query: str):
        """Main method to retrieve context from knowledge graph"""

        # Step 1: Extract entities
        entities = self.extract_entities_with_llm(query)
        print(f"Extracted entities: {entities}")

        # Step 2: Generate Cypher query
        cypher = self.generate_cypher_with_llm(entities)
        print(f"Generated Cypher: {cypher}")

        # Step 3: Execute query
        with self.driver.session() as session:
            try:
                # Check if cypher is empty or whitespace
                if not cypher or not cypher.strip():
                    print("Empty Cypher query, using fallback")
                    # Fallback query
                    result = session.run(
                        """
                        MATCH (c:City)
                        WHERE toLower(c.name) CONTAINS toLower($city_name)
                        OPTIONAL MATCH (c)-[:HAS_ATTRACTION]->(a:Attraction)
                        RETURN c.name AS city, collect(a.name)[0..5] AS attractions
                        LIMIT 1
                        """,
                        city_name=entities.get('city', 'Paris')  # Use parameter correctly
                    )
                else:
                    result = session.run(cypher)

                records = list(result)

                if records:
                    context = self.format_context(records)
                    return context
                else:
                    return f"No information found about {entities.get('city', 'the requested destination')}."

            except Exception as e:
                print(f"Cypher execution error: {e}")
                # Ultimate fallback
                return f"Information about {entities.get('city', 'travel destinations')} from our knowledge base."

    def close(self):
        self.driver.close()
