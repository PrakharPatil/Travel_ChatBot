# app.py
from flask import Flask, request, jsonify
import yaml
import json
import os
from modules.orchestration import QueryOrchestrator
from modules.graph_rag import GraphRAG
from modules.crag import CorrectiveRAG
from modules.api_module import AskToActAPI
from modules.memory import MemoryManager
from modules.main_llm import MainLLM

app = Flask(__name__)

# Load config
try:
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("Config file not found. Please ensure 'configs/config.yaml' exists.")
    exit(1)

# Initialize all modules
print("Initializing modules...")

# FIX: Robust Orchestrator loading
# If the local BERT path doesn't exist, fallback to HuggingFace default to prevent crash
bert_path = './models/bert_classifier'
if not os.path.exists(bert_path):
    print(f"Warning: Local BERT path '{bert_path}' not found. Using 'distilbert-base-uncased'.")
    bert_path = 'distilbert-base-uncased'

orchestrator = QueryOrchestrator(
    bert_model_path=bert_path,
    llm_model=config['models']['llm_small']
)

graph_rag = GraphRAG(
    neo4j_config=config['neo4j'],
    llm_model=config['models']['llm_small']
)

crag = CorrectiveRAG(
    serpapi_key=config['serpapi_key'],
    llm_model=config['models']['llm_small']
)

api_module = AskToActAPI(
    serpapi_key=config['serpapi_key'],
    llm_model=config['models']['llm_small']
)

# Initialize Memory
memory = MemoryManager(
    llm_model=config['models']['llm_small'],
    max_stm_turns=config['memory']['stm_max_turns']
)

main_llm = MainLLM(model_name=config['models']['llm_main'])

print("All modules initialized!")

# Initialize graph data (run once)
try:
    # Check if file exists before trying to load
    if os.path.exists('data/travel_data.json'):
        graph_rag.create_graph_from_data('data/travel_data.json')
        print("Graph data loaded!")
    else:
        print("Warning: 'data/travel_data.json' not found. Skipping graph data load.")
except Exception as e:
    print(f"Graph initialization warning: {e}")


@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Step 1: Classify query using BERT + LLM
        query_type = orchestrator.classify(user_query)
        print(f"Query classified as: {query_type}")

        # Step 2: Get memory context
        memory_context = memory.get_memory_context()

        # Step 3: Process based on query type
        context = {
            'memory': memory_context,
            'query_type': query_type
        }

        if query_type == "Conversational":
            # Simple conversational response
            response = main_llm.generate_response(user_query, context)

        elif query_type == "RAG":
            # Graph RAG pipeline
            rag_context = graph_rag.retrieve_context(user_query)

            # Corrective RAG (now returns dict with 'context' key)
            crag_result = crag.process(user_query, rag_context)

            # Use the 'context' key from CRAG result
            # The fixed CRAG module always returns {'context': ...}
            context['rag_context'] = crag_result.get('context', rag_context)

            response = main_llm.generate_response(user_query, context)

        elif query_type == "API_Call":
            # API module (AskToAct)
            api_result = api_module.process(user_query, memory_context.get('recent_turns', []))

            if api_result['status'] == 'missing_params':
                # Need clarification
                response = api_result['clarification']
            else:
                # Got API result
                context['api_result'] = api_result['result']
                response = main_llm.generate_response(user_query, context)

        else:
            response = "I'm not sure how to help with that. Can you rephrase?"

        # Step 4: Update memory
        memory.add_to_stm(user_query, response)
        memory.update_ltm(user_query, response)

        return jsonify({
            'response': response,
            'query_type': query_type,
            'memory_summary': memory_context['stm_summary']
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/reset_memory', methods=['POST'])
def reset_memory():
    """Reset conversation memory"""
    global memory
    try:
        memory = MemoryManager(
            llm_model=config['models']['llm_small'],
            max_stm_turns=config['memory']['stm_max_turns']
        )
        print("Memory reset successfully.")
        return jsonify({'message': 'Memory reset successfully'})
    except Exception as e:
        print(f"Error resetting memory: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089, debug=False)
