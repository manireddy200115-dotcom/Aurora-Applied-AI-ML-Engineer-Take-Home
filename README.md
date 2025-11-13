# Member Data Question-Answering System

A simple question-answering system that answers natural-language questions about member data from a public API.

## Features

- **Natural Language Processing**: Answers questions like:
  - "When is Layla planning her trip to London?"
  - "How many cars does Vikram Desai have?"
  - "What are Amira's favorite restaurants?"
- **RESTful API**: Simple `/ask` endpoint that accepts questions and returns answers
- **Data Caching**: Efficient caching of API responses to reduce load
- **Data Insights**: Analysis endpoint for identifying data quality issues and anomalies

## API Endpoints

### POST `/ask`
Answer a natural-language question about member data.

**Request:**
```json
{
  "question": "When is Layla planning her trip to London?"
}
```

**Response:**
```json
{
  "answer": "Layla is planning a trip to London on 2024-06-15."
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET `/insights`
Get data insights and anomaly analysis.

**Response:**
```json
{
  "total_messages": 150,
  "anomalies": [...],
  "statistics": {...},
  "data_quality_issues": [...]
}
```

## Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
uvicorn app.main:app --reload
```

3. **Test the API:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

### Docker

1. **Build the image:**
```bash
docker build -t member-qa-service .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 member-qa-service
```

### Deployment

The service can be deployed to Render, Railway, Fly.io, or any platform that supports Docker.

**Render:**
- Connect your GitHub repository
- Render will automatically detect `render.yaml` and deploy the service

**Railway:**
- Connect your GitHub repository
- Railway will detect the Dockerfile and deploy automatically

**Fly.io:**
```bash
fly launch
fly deploy
```

## Architecture

The system is built with a modular architecture:

- **`main.py`**: FastAPI application with endpoints
- **`extractor.py`**: Fetches and caches data from the external API
- **`rag_qa.py`**: RAG-based QA system using semantic embeddings (ML) with keyword fallback
- **`insights.py`**: Data analysis and anomaly detection

## Design Notes: Alternative Approaches

### Approach 1: Semantic Search with Embeddings (Current Implementation)
**Implementation:** Using `sentence-transformers` with `all-MiniLM-L6-v2` model

**Pros:**
- True ML-based semantic understanding
- Handles synonyms and related concepts
- No per-request API costs
- Can find relevant information even with different wording
- Falls back to keyword matching if embeddings unavailable

**Cons:**
- Requires model download (~80MB) on first run
- Slightly slower than pure keyword matching (but still fast)
- Requires PyTorch dependency

**How it works:**
1. Converts questions and messages into vector embeddings using a neural network
2. Calculates cosine similarity between question and all messages
3. Returns the most semantically similar message
4. Falls back to keyword matching if similarity is too low

### Approach 2: Large Language Model (LLM) Integration
**Considered:** Using OpenAI GPT, Anthropic Claude, or open-source models like Llama 2

**Pros:**
- Handles complex, nuanced questions
- Better understanding of context
- Can answer questions not explicitly in the data
- More natural language understanding

**Cons:**
- Requires API keys and costs per request
- Slower response times
- Less predictable outputs
- Privacy concerns with external APIs

**Implementation would involve:**
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant answering questions about member data."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
)
```

### Approach 3: Semantic Search with Embeddings (IMPLEMENTED)
**Status:** This is now the primary approach used in the system

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
question_embedding = model.encode(question, convert_to_numpy=True)
message_embeddings = model.encode(messages, convert_to_numpy=True)

# Calculate cosine similarity to find most relevant message
similarities = np.dot(message_embeddings, question_embedding) / (
    np.linalg.norm(message_embeddings, axis=1) * np.linalg.norm(question_embedding)
)
best_match = messages[np.argmax(similarities)]
```

**Benefits:**
- True semantic understanding (handles "trip" vs "journey", "car" vs "vehicle")
- Better accuracy for nuanced questions
- Cached embeddings for performance
- Automatic fallback to keyword matching if needed

### Approach 4: Named Entity Recognition (NER) + Structured Queries
**Considered:** Using spaCy or similar for entity extraction, then querying structured data

**Pros:**
- Better entity extraction (names, dates, locations)
- Can build structured queries
- More accurate for specific information

**Cons:**
- Requires training or pre-trained models
- More setup complexity
- Still needs query logic

### Approach 5: Hybrid Approach (Recommended for Production)
**Best of both worlds:**
- Use rule-based patterns for common queries (fast, free)
- Fall back to LLM for complex questions
- Use semantic search for finding relevant context
- Cache frequent queries

**Implementation:**
```python
def answer(question):
    # Try rule-based first
    answer = rule_based_qa(question)
    if answer and confidence > threshold:
        return answer
    
    # Find relevant context with semantic search
    context = semantic_search(question, messages)
    
    # Use LLM for complex reasoning
    answer = llm_answer(question, context)
    return answer
```

### Why We Chose the Current Approach

We implemented **Approach 3 (Semantic Search with Embeddings)** because:
1. **True ML/AI**: Uses neural network embeddings for semantic understanding
2. **Better Accuracy**: Handles synonyms, related concepts, and different phrasings
3. **No API Costs**: Runs locally with no per-request charges
4. **Fast**: Embeddings are cached, making subsequent queries very fast
5. **Robust Fallback**: Automatically falls back to keyword matching if needed
6. **Production-Ready**: Lightweight model (all-MiniLM-L6-v2) balances speed and accuracy

The system now uses a **hybrid approach**:
- **Primary**: Semantic embeddings for ML-based similarity search
- **Fallback**: Keyword matching for edge cases or if embeddings unavailable
- **Smart Caching**: Message embeddings are cached to avoid recomputation

## Data Insights

### Analysis Methodology

The system analyzes member messages for:
- **Structure consistency**: Field presence and types
- **Format consistency**: Date formats, naming conventions
- **Data quality**: Missing fields, duplicates
- **Anomalies**: Unusual dates, inconsistent data
- **Pagination**: Data retrieval completeness

### Findings from Dataset Analysis

Based on comprehensive analysis of the member data API (3,349 total messages, analyzing first 100):

#### **Data Quality Strengths**

1. **Complete Field Coverage**
   - All messages have 100% field completeness
   - Required fields (`id`, `user_id`, `user_name`, `timestamp`, `message`) are always present
   - No missing or null values detected in the sample

2. **Consistent Data Structure**
   - All messages follow the same schema
   - Field types are consistent across all records
   - UUID format is standardized for IDs

3. **No Duplicate Messages**
   - No exact duplicate messages found in the sample
   - Each message has a unique `id` field

#### **Anomalies and Inconsistencies Identified**

1. **Future Date Anomaly** (High Priority)
   - **Issue**: Many messages have timestamps in the future (dates beyond current date)
   - **Example**: Messages dated 2025-11-04 when current date is earlier
   - **Impact**: Could indicate test data, timezone issues, or data entry errors
   - **Recommendation**: 
     - Validate timestamps against current date
     - Implement timezone normalization
     - Flag future-dated messages for review

2. **Pagination Limitation** (Medium Priority)
   - **Issue**: API returns only 100 messages per request despite having 3,349 total messages
   - **Current Behavior**: Only first 100 messages are accessible
   - **Impact**: 
     - 97% of data is not being analyzed (3,249 messages missing)
     - Answers may be incomplete or inaccurate for users not in first 100 messages
   - **Recommendation**:
     - Implement pagination support in the extractor
     - Use `?page=` or `?offset=` parameters if available
     - Fetch all messages in batches

3. **Uneven User Distribution** (Medium Priority)
   - **Issue**: Message distribution across users is uneven
   - **Finding**: Top user (Sophia Al-Farsi) has 16 messages, while some users have fewer
   - **Impact**: Answers may be biased toward more active users
   - **Recommendation**: Consider weighting or normalization for user representation

4. **Date Range Span** (Medium Priority)
   - **Issue**: Date range spans from 2024-11-14 to 2025-11-04 (nearly 1 year span)
   - **Finding**: 91 unique dates in first 100 messages
   - **Impact**: Temporal queries may need date filtering
   - **Recommendation**: Implement date range filtering for time-sensitive queries

5. **Message Length Variation** (Low Priority)
   - **Finding**: Message lengths range from 47 to 84 characters (average: 63.6)
   - **Status**: Within normal range, no anomalies detected
   - **Note**: Very consistent message length suggests structured or templated messages

#### **Statistical Summary**

- **Total Messages**: 3,349 (API total)
- **Messages Analyzed**: 100 (sample)
- **Unique Users**: 10 (in first 100 messages)
- **Average Messages per User**: 10.0
- **Field Completeness**: 100% for all fields
- **Date Range**: 2024-11-14 to 2025-11-04
- **Unique Dates**: 91 (in sample)
- **Duplicate Messages**: 0

#### **Recommended Improvements**

1. **Implement Pagination**: Fetch all messages, not just first 100
2. **Date Validation**: Add checks for future dates and timezone handling
3. **Data Sampling**: For large datasets, implement intelligent sampling
4. **Caching Strategy**: Cache paginated results separately
5. **Monitoring**: Track data quality metrics over time

*Note: These insights are generated dynamically. Run the `/insights` endpoint to get current analysis of the live data. The analysis currently processes the first 100 messages due to pagination limitations.*

## Testing

### Example Queries

```bash
# Trip planning question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'

# Count question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many cars does Vikram Desai have?"}'

# Preference question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Amira'\''s favorite restaurants?"}'
```

## Project Structure

```
assessment_aurora/
├── app/
│   ├── main.py          # FastAPI application
│   ├── qa.py            # Question-answering logic
│   ├── extractor.py     # Data fetching and caching
│   └── insights.py      # Data analysis
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── render.yaml         # Render deployment config
└── README.md           # This file
```

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server
- **Requests**: HTTP library for API calls
- **Pydantic**: Data validation
- **sentence-transformers**: ML library for semantic embeddings (core AI component)
- **PyTorch**: Deep learning framework (required by sentence-transformers)
- **NumPy**: Numerical computing for similarity calculations

## Performance Considerations

- **Caching**: API responses are cached for 5 minutes to reduce load
- **Async**: FastAPI's async support for better concurrency
- **Lightweight**: Minimal dependencies for fast startup

## Future Improvements

1. **Semantic Search**: Implemented with sentence-transformers
2. **LLM Integration**: Add optional LLM for answer generation (e.g., GPT-4, Claude)
3. **Confidence Scores**: Return similarity scores with answers
4. **Query Expansion**: Handle synonyms and related terms (partially done via embeddings)
5. **Multi-language Support**: Handle questions in multiple languages
6. **Vector Database**: Use a proper vector DB (e.g., Pinecone, Weaviate) for large datasets
7. **Rate Limiting**: Implement rate limiting for production use
8. **Authentication**: Add API key authentication
9. **Monitoring**: Add logging and metrics collection
10. **Fine-tuning**: Fine-tune embeddings on domain-specific data

## License

This project is created for assessment purposes.

## Contact

For questions or issues, please open an issue in the repository.

