# app.py
from flask import Flask, request, jsonify
import yaml
import json
from modules.orchestration import QueryOrchestrator
from modules.graph_rag import GraphRAG
from modules.crag import CorrectiveRAG
from modules.api_module import AskToActAPI
from modules.memory import MemoryManager
from modules.main_llm import MainLLM

app = Flask(__name__)

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize all modules
print("Initializing modules...")
orchestrator = QueryOrchestrator(
    bert_model_path='./models/bert_classifier',
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

memory = MemoryManager(
    llm_model=config['models']['llm_small'],
    max_stm_turns=config['memory']['stm_max_turns']
)

main_llm = MainLLM(model_name=config['models']['llm_main'])

print("All modules initialized!")

# Initialize graph data (run once)
try:
    graph_rag.create_graph_from_data('data/travel_data.json')
    print("Graph data loaded!")
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

            # Corrective RAG
            crag_result = crag.process(user_query, rag_context)

            context['rag_context'] = crag_result['context']
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
    memory = MemoryManager(
        llm_model=config['models']['llm_small'],
        max_stm_turns=config['memory']['stm_max_turns']
    )
    return jsonify({'message': 'Memory reset successfully'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089, debug=False)
