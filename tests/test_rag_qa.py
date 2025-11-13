"""
Unit tests for RAGQASystem module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.rag_qa import RAGQASystem
from app.extractor import DataExtractor


class TestRAGQASystem:
    """Test cases for RAGQASystem."""
    
    @pytest.fixture
    def mock_extractor(self):
        """Create a mock extractor."""
        extractor = Mock(spec=DataExtractor)
        extractor.fetch_messages.return_value = [
            {
                'id': '1',
                'user_name': 'Layla Johnson',
                'message': 'I am planning a trip to London on 2024-06-15'
            },
            {
                'id': '2',
                'user_name': 'Vikram Desai',
                'message': 'I have 3 cars in my garage'
            },
            {
                'id': '3',
                'user_name': 'Amira Hassan',
                'message': 'My favorite restaurants are Le Bernardin and Nobu'
            }
        ]
        return extractor
    
    @pytest.fixture
    def qa_system(self, mock_extractor):
        """Create a QA system instance without embeddings/SLM for faster tests."""
        with patch('app.rag_qa.SentenceTransformer'), \
             patch('app.rag_qa.AutoTokenizer'), \
             patch('app.rag_qa.AutoModelForSeq2SeqLM'), \
             patch('app.rag_qa.pipeline'):
            system = RAGQASystem(mock_extractor, use_embeddings=False, use_slm=False)
            return system
    
    def test_init(self, mock_extractor):
        """Test RAGQASystem initialization."""
        with patch('app.rag_qa.SentenceTransformer'), \
             patch('app.rag_qa.AutoTokenizer'), \
             patch('app.rag_qa.AutoModelForSeq2SeqLM'), \
             patch('app.rag_qa.pipeline'):
            system = RAGQASystem(mock_extractor, use_embeddings=False, use_slm=False)
            assert system.extractor == mock_extractor
            assert system.use_embeddings is False
            assert system.use_slm is False
            assert system.embedding_model is None
            assert system.slm_pipeline is None
    
    def test_answer_empty_question(self, qa_system):
        """Test that empty questions raise ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            qa_system.answer("")
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            qa_system.answer("   ")
    
    def test_answer_no_messages(self, qa_system):
        """Test answer when no messages are available."""
        qa_system.extractor.fetch_messages.return_value = []
        qa_system.messages = []
        
        answer, confidence = qa_system.answer("What is Layla doing?")
        
        assert "couldn't find" in answer.lower()
        assert confidence == 0.0
    
    def test_format_message(self, qa_system):
        """Test message formatting."""
        msg = {
            'user_name': 'Layla Johnson',
            'message': 'Planning trip to London',
            'timestamp': '2024-06-15'
        }
        formatted = qa_system._format_message(msg)
        
        assert 'Layla Johnson' in formatted
        assert 'Planning trip to London' in formatted
    
    def test_retrieve_top_k_keyword_fallback(self, qa_system):
        """Test keyword-based retrieval when embeddings are disabled."""
        qa_system.messages = qa_system.extractor.fetch_messages()
        
        retrieved = qa_system._retrieve_top_k("Layla London trip")
        
        assert len(retrieved) > 0
        assert 'message' in retrieved[0]
        assert 'similarity' in retrieved[0]
        # Should find Layla's message about London
        assert 'Layla' in qa_system._format_message(retrieved[0]['message'])
    
    def test_generate_template_answer_when_question(self, qa_system):
        """Test template answer generation for 'when' questions."""
        context_messages = [{
            'message': {
                'user_name': 'Layla Johnson',
                'message': 'I am planning a trip to London on 2024-06-15'
            },
            'similarity': 0.9
        }]
        
        answer = qa_system._generate_template_answer("When is Layla going to London?", context_messages)
        
        assert "2024-06-15" in answer or "Layla" in answer
    
    def test_generate_template_answer_how_many(self, qa_system):
        """Test template answer generation for 'how many' questions."""
        context_messages = [{
            'message': {
                'user_name': 'Vikram Desai',
                'message': 'I have 3 cars in my garage'
            },
            'similarity': 0.9
        }]
        
        answer = qa_system._generate_template_answer("How many cars does Vikram have?", context_messages)
        
        assert "3" in answer or "Vikram" in answer
    
    def test_answer_integration(self, qa_system):
        """Test full answer generation flow."""
        qa_system.messages = qa_system.extractor.fetch_messages()
        
        answer, confidence = qa_system.answer("When is Layla planning her trip to London?")
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert 0.0 <= confidence <= 1.0
