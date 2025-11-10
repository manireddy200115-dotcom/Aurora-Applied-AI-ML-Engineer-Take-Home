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
- **`qa.py`**: Core question-answering logic with multiple strategies
- **`insights.py`**: Data analysis and anomaly detection

## Design Notes: Alternative Approaches

### Approach 1: Rule-Based Pattern Matching (Current Implementation)
**Pros:**
- Fast and lightweight
- No external dependencies or API costs
- Predictable and debuggable
- Works well for structured queries

**Cons:**
- Limited to predefined patterns
- Requires manual updates for new question types
- May miss nuanced questions

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

### Approach 3: Semantic Search with Embeddings
**Considered:** Using sentence transformers (e.g., `sentence-transformers`) to create embeddings

**Pros:**
- Better semantic understanding than keyword matching
- Can find relevant information even with different wording
- No per-request costs
- Can be run locally

**Cons:**
- Requires model download and setup
- More complex implementation
- Still needs answer extraction logic

**Implementation would involve:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embedding = model.encode(question)
message_embeddings = model.encode(messages)
# Find most similar messages using cosine similarity
```

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

For this assessment, we chose **Approach 1 (Rule-Based)** because:
1. **Simplicity**: Easy to understand and maintain
2. **No Dependencies**: No external APIs or large models
3. **Fast**: Sub-millisecond response times
4. **Cost-Effective**: No per-request costs
5. **Reliable**: Predictable behavior for the given examples

For a production system with more complex requirements, we would recommend **Approach 5 (Hybrid)**.

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

#### ✅ **Data Quality Strengths**

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

#### ⚠️ **Anomalies and Inconsistencies Identified**

1. **Future Date Anomaly** 🔴
   - **Issue**: Many messages have timestamps in the future (dates beyond current date)
   - **Example**: Messages dated 2025-11-04 when current date is earlier
   - **Impact**: Could indicate test data, timezone issues, or data entry errors
   - **Recommendation**: 
     - Validate timestamps against current date
     - Implement timezone normalization
     - Flag future-dated messages for review

2. **Pagination Limitation** 🟡
   - **Issue**: API returns only 100 messages per request despite having 3,349 total messages
   - **Current Behavior**: Only first 100 messages are accessible
   - **Impact**: 
     - 97% of data is not being analyzed (3,249 messages missing)
     - Answers may be incomplete or inaccurate for users not in first 100 messages
   - **Recommendation**:
     - Implement pagination support in the extractor
     - Use `?page=` or `?offset=` parameters if available
     - Fetch all messages in batches

3. **Uneven User Distribution** 🟡
   - **Issue**: Message distribution across users is uneven
   - **Finding**: Top user (Sophia Al-Farsi) has 16 messages, while some users have fewer
   - **Impact**: Answers may be biased toward more active users
   - **Recommendation**: Consider weighting or normalization for user representation

4. **Date Range Span** 🟡
   - **Issue**: Date range spans from 2024-11-14 to 2025-11-04 (nearly 1 year span)
   - **Finding**: 91 unique dates in first 100 messages
   - **Impact**: Temporal queries may need date filtering
   - **Recommendation**: Implement date range filtering for time-sensitive queries

5. **Message Length Variation** 🟢
   - **Finding**: Message lengths range from 47 to 84 characters (average: 63.6)
   - **Status**: Within normal range, no anomalies detected
   - **Note**: Very consistent message length suggests structured or templated messages

#### 📊 **Statistical Summary**

- **Total Messages**: 3,349 (API total)
- **Messages Analyzed**: 100 (sample)
- **Unique Users**: 10 (in first 100 messages)
- **Average Messages per User**: 10.0
- **Field Completeness**: 100% for all fields
- **Date Range**: 2024-11-14 to 2025-11-04
- **Unique Dates**: 91 (in sample)
- **Duplicate Messages**: 0

#### 🔧 **Recommended Improvements**

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

## Performance Considerations

- **Caching**: API responses are cached for 5 minutes to reduce load
- **Async**: FastAPI's async support for better concurrency
- **Lightweight**: Minimal dependencies for fast startup

## Future Improvements

1. **Enhanced NLP**: Integrate transformer models for better understanding
2. **Query Expansion**: Handle synonyms and related terms
3. **Confidence Scores**: Return confidence levels with answers
4. **Query History**: Track and learn from previous queries
5. **Multi-language Support**: Handle questions in multiple languages
6. **Rate Limiting**: Implement rate limiting for production use
7. **Authentication**: Add API key authentication
8. **Monitoring**: Add logging and metrics collection

## License

This project is created for assessment purposes.

## Contact

For questions or issues, please open an issue in the repository.

