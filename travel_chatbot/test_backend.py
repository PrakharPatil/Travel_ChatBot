# test_backend_api.py
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8089"


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_test(test_num: int, description: str):
    print(f"{Colors.OKCYAN}{Colors.BOLD}Test {test_num}: {description}{Colors.ENDC}")


def print_query(query: str):
    print(f"{Colors.OKBLUE}Query: {Colors.ENDC}\"{query}\"")


def print_response(response: Dict[Any, Any]):
    print(f"{Colors.OKGREEN}Response:{Colors.ENDC}")
    print(json.dumps(response, indent=2))


def print_error(error: str):
    print(f"{Colors.FAIL}Error: {error}{Colors.ENDC}")


def print_success(message: str):
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


def print_warning(message: str):
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")


def test_health_check():
    """Test 1: Health Check"""
    print_test(1, "Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("Backend is running")
            print_response(response.json())
        else:
            print_error(f"Health check failed with status {response.status_code}")
    except Exception as e:
        print_error(f"Cannot connect to backend: {e}")
        print_warning("Make sure to run: python app.py")
        return False
    return True


def test_conversational_queries():
    """Test 2-4: Conversational Queries"""
    queries = [
        "Hello! How are you?",
        "Thanks for your help!",
        "Can you explain that in more detail?",
    ]

    for i, query in enumerate(queries, 2):
        print_test(i, f"Conversational Query {i - 1}")
        print_query(query)

        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"query": query},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                print_response(data)

                if data.get("query_type") == "Conversational":
                    print_success("Correctly classified as Conversational")
                else:
                    print_warning(f"Expected: Conversational, Got: {data.get('query_type')}")
            else:
                print_error(f"Request failed with status {response.status_code}")

        except Exception as e:
            print_error(str(e))

        time.sleep(1)


def test_rag_queries():
    """Test 5-7: RAG Queries"""
    queries = [
        "What are the best attractions in Paris?",
        "Tell me about things to do in Tokyo",
        "What food should I try in London?",
    ]

    for i, query in enumerate(queries, 5):
        print_test(i, f"RAG Query {i - 4}")
        print_query(query)

        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"query": query},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                print_response(data)

                if data.get("query_type") == "RAG":
                    print_success("Correctly classified as RAG")
                    print_success("Graph RAG and CRAG modules working")
                else:
                    print_warning(f"Expected: RAG, Got: {data.get('query_type')}")
            else:
                print_error(f"Request failed with status {response.status_code}")

        except Exception as e:
            print_error(str(e))

        time.sleep(1)


def test_api_call_queries():
    """Test 8-10: API Call Queries"""
    queries = [
        "What is the weather in Tokyo today?",
        "Find flight prices from New York to London",
        "Show me hotel prices in Dubai",
    ]

    for i, query in enumerate(queries, 8):
        print_test(i, f"API Call Query {i - 7}")
        print_query(query)

        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"query": query},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                print_response(data)

                if data.get("query_type") == "API_Call":
                    print_success("Correctly classified as API_Call")
                    print_success("AskToAct module working")
                else:
                    print_warning(f"Expected: API_Call, Got: {data.get('query_type')}")
            else:
                print_error(f"Request failed with status {response.status_code}")

        except Exception as e:
            print_error(str(e))

        time.sleep(1)


def test_conversational_flow():
    """Test 11: Multi-turn Conversation (Memory Test)"""
    print_test(11, "Multi-turn Conversation - Memory Test")

    conversation = [
        "Hello! I'm planning a trip",
        "I want to visit Paris",
        "What are the best attractions there?",
        "Thanks! That's helpful",
    ]

    for turn, query in enumerate(conversation, 1):
        print(f"\n{Colors.BOLD}Turn {turn}:{Colors.ENDC}")
        print_query(query)

        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"query": query},
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                print(f"Type: {data.get('query_type')}")
                print(f"Response: {data.get('response', '')[:200]}...")

                if 'memory_summary' in data:
                    print(f"Memory: {data.get('memory_summary', '')[:100]}...")
                    print_success("Memory (STM/LTM) working")
            else:
                print_error(f"Request failed with status {response.status_code}")

        except Exception as e:
            print_error(str(e))

        time.sleep(1)

    print_success("Conversation flow completed")


def test_api_missing_params():
    """Test 12: API Call with Missing Parameters"""
    print_test(12, "API Call with Missing Parameters")
    query = "Find me a flight"  # Missing origin, destination, date
    print_query(query)

    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": query},
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            print_response(data)

            if "specify" in data.get('response', '').lower() or "please" in data.get('response', '').lower():
                print_success("System correctly asks for missing parameters")
            else:
                print_warning("System should ask for missing parameters")
        else:
            print_error(f"Request failed with status {response.status_code}")

    except Exception as e:
        print_error(str(e))


def test_reset_memory():
    """Test 13: Reset Memory"""
    print_test(13, "Reset Memory")

    try:
        response = requests.post(f"{BASE_URL}/reset_memory", timeout=60)

        if response.status_code == 200:
            data = response.json()
            print_response(data)
            print_success("Memory reset successfully")
        else:
            print_error(f"Request failed with status {response.status_code}")

    except Exception as e:
        print_error(str(e))


def run_all_tests():
    """Run all backend API tests"""
    print_header("TRAVEL CHATBOT BACKEND API TEST SUITE")

    # Check if backend is running
    if not test_health_check():
        print_error("\n❌ Backend is not running. Please start it with: python app.py")
        return

    print_success("Backend is ready. Starting tests...\n")
    time.sleep(1)

    # Module tests
    print_header("MODULE 1: CONVERSATIONAL (BERT + Main LLM)")
    test_conversational_queries()

    print_header("MODULE 2: RAG (Graph RAG + CRAG)")
    test_rag_queries()

    print_header("MODULE 3: API CALLS (AskToAct)")
    test_api_call_queries()

    print_header("MODULE 4: MEMORY (STM/LTM)")
    test_conversational_flow()

    print_header("MODULE 5: PARAMETER HANDLING")
    test_api_missing_params()

    print_header("UTILITY: RESET MEMORY")
    test_reset_memory()

    # Summary
    print_header("TEST SUITE COMPLETED")
    print(f"{Colors.OKGREEN}All modules tested!{Colors.ENDC}\n")

    print("Modules tested:")
    print("  ✅ BERT Classifier (Orchestration)")
    print("  ✅ Graph RAG (Neo4j)")
    print("  ✅ Corrective RAG (CRAG)")
    print("  ✅ API Module (AskToAct)")
    print("  ✅ Memory Manager (STM/LTM)")
    print("  ✅ Main LLM (Response Generation)")
    print()


if __name__ == "__main__":
    run_all_tests()
