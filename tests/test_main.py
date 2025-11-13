"""
Unit tests for main API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_ask_endpoint_empty_question(self, client):
        """Test /ask endpoint with empty question."""
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 400
    
    def test_ask_endpoint_missing_question(self, client):
        """Test /ask endpoint with missing question field."""
        response = client.post("/ask", json={})
        assert response.status_code == 422  # Validation error
    
    def test_ask_endpoint_valid_question(self, client):
        """Test /ask endpoint with valid question."""
        # Mock the QA system to return a predictable answer
        from unittest.mock import patch
        with patch('app.main.qa_system') as mock_qa:
            mock_qa.answer.return_value = ("Test answer", 0.85)
            
            response = client.post("/ask", json={"question": "Test question?"})
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "confidence" in data
            assert data["answer"] == "Test answer"
            assert data["confidence"] == 0.85
    
    def test_insights_endpoint(self, client):
        """Test /insights endpoint."""
        # Mock the insights analyzer
        from unittest.mock import patch
        with patch('app.main.insights_analyzer') as mock_insights:
            mock_insights.analyze.return_value = {
                "total_messages": 100,
                "anomalies": [],
                "statistics": {},
                "data_quality_issues": []
            }
            
            response = client.get("/insights")
            
            assert response.status_code == 200
            data = response.json()
            assert "total_messages" in data
            assert "anomalies" in data

