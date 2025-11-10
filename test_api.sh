#!/bin/bash

# Test script for the QA API
# Usage: ./test_api.sh [base_url]
# Example: ./test_api.sh http://localhost:8000

BASE_URL=${1:-http://localhost:8000}

echo "Testing QA API at $BASE_URL"
echo "================================"
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""
echo ""

# Test question 1
echo "2. Testing question: 'When is Layla planning her trip to London?'"
curl -s -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}' | python3 -m json.tool
echo ""
echo ""

# Test question 2
echo "3. Testing question: 'How many cars does Vikram Desai have?'"
curl -s -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many cars does Vikram Desai have?"}' | python3 -m json.tool
echo ""
echo ""

# Test question 3
echo "4. Testing question: 'What are Amira'\''s favorite restaurants?'"
curl -s -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What are Amira's favorite restaurants?\"}" | python3 -m json.tool
echo ""
echo ""

# Test insights endpoint
echo "5. Testing insights endpoint..."
curl -s "$BASE_URL/insights" | python3 -m json.tool
echo ""

