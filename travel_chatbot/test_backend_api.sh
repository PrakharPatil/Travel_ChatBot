#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8089"

# Print functions
print_header() {
    echo -e "\n${CYAN}${BOLD}========================================${NC}"
    echo -e "${CYAN}${BOLD}$1${NC}"
    echo -e "${CYAN}${BOLD}========================================${NC}\n"
}

print_test() {
    echo -e "${BLUE}${BOLD}Test $1: $2${NC}"
}

print_query() {
    echo -e "${YELLOW}Query:${NC} \"$1\""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Test 1: Health Check
print_header "TEST 1: HEALTH CHECK"
print_test "1" "Checking if backend is running"
response=$(curl -s -w "\n%{http_code}" ${BASE_URL}/health)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" == "200" ]; then
    print_success "Backend is running"
    echo "$body" | jq '.'
else
    print_error "Backend not running. Start with: python app.py"
    exit 1
fi

sleep 1

# Test 2-4: Conversational Queries
print_header "MODULE 1: CONVERSATIONAL (BERT + Main LLM)"

print_test "2" "Conversational Query 1"
print_query "Hello! How are you?"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello! How are you?"}' | jq '.'
sleep 2

print_test "3" "Conversational Query 2"
print_query "Thanks for your help!"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Thanks for your help!"}' | jq '.'
sleep 2

print_test "4" "Conversational Query 3"
print_query "Can you explain that in more detail?"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Can you explain that in more detail?"}' | jq '.'
sleep 2

# Test 5-7: RAG Queries
print_header "MODULE 2: RAG (Graph RAG + CRAG)"

print_test "5" "RAG Query 1"
print_query "What are the best attractions in Paris?"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best attractions in Paris?"}' | jq '.'
sleep 2

print_test "6" "RAG Query 2"
print_query "Tell me about things to do in Tokyo"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about things to do in Tokyo"}' | jq '.'
sleep 2

print_test "7" "RAG Query 3"
print_query "What food should I try in London?"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What food should I try in London?"}' | jq '.'
sleep 2

# Test 8-10: API Call Queries
print_header "MODULE 3: API CALLS (AskToAct)"

print_test "8" "API Call Query 1"
print_query "What is the weather in Tokyo today?"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Tokyo today?"}' | jq '.'
sleep 2

print_test "9" "API Call Query 2"
print_query "Find flight prices from New York to London"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find flight prices from New York to London"}' | jq '.'
sleep 2

print_test "10" "API Call Query 3"
print_query "Show me hotel prices in Dubai"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me hotel prices in Dubai"}' | jq '.'
sleep 2

# Test 11: Multi-turn Conversation
print_header "MODULE 4: MEMORY (STM/LTM)"

print_test "11" "Multi-turn Conversation Test"

echo -e "${YELLOW}Turn 1:${NC} Hello! I'm planning a trip"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello! I am planning a trip"}' | jq '.'
sleep 2

echo -e "${YELLOW}Turn 2:${NC} I want to visit Paris"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to visit Paris"}' | jq '.'
sleep 2

echo -e "${YELLOW}Turn 3:${NC} What are the best attractions there?"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best attractions there?"}' | jq '.'
sleep 2

echo -e "${YELLOW}Turn 4:${NC} Thanks! That's helpful"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Thanks! That is helpful"}' | jq '.'
sleep 2

# Test 12: Missing Parameters
print_header "MODULE 5: PARAMETER HANDLING"

print_test "12" "API Call with Missing Parameters"
print_query "Find me a flight"
curl -s -X POST ${BASE_URL}/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find me a flight"}' | jq '.'
sleep 2

# Test 13: Reset Memory
print_header "UTILITY: RESET MEMORY"

print_test "13" "Reset Memory"
curl -s -X POST ${BASE_URL}/reset_memory \
  -H "Content-Type: application/json" | jq '.'
sleep 1

# Summary
print_header "TEST SUITE COMPLETED"
print_success "All modules tested!"
echo ""
echo "Modules tested:"
echo "  ✅ BERT Classifier (Orchestration)"
echo "  ✅ Graph RAG (Neo4j)"
echo "  ✅ Corrective RAG (CRAG)"
echo "  ✅ API Module (AskToAct)"
echo "  ✅ Memory Manager (STM/LTM)"
echo "  ✅ Main LLM (Response Generation)"
echo ""
