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
        """Create a QA system instance."""
        with patch('app.rag_qa.EMBEDDINGS_AVAILABLE', False):
            system = RAGQASystem(mock_extractor, use_embeddings=False)
            return system
    
    def test_init(self, mock_extractor):
        """Test RAGQASystem initialization."""
        with patch('app.rag_qa.EMBEDDINGS_AVAILABLE', False):
            system = RAGQASystem(mock_extractor, use_embeddings=False)
            assert system.extractor == mock_extractor
            assert system.use_embeddings is False
            assert system.embedding_model is None
    
    def test_answer_empty_question(self, qa_system):
        """Test that empty questions raise ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            qa_system.answer("")
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            qa_system.answer("   ")
    
    def test_answer_no_messages(self, qa_system):
        """Test answer when no messages are available."""
        qa_system.extractor.fetch_messages.return_value = []
        
        answer, confidence = qa_system.answer("What is Layla doing?")
        
        assert "couldn't find any member data" in answer.lower()
        assert confidence == 0.0
    
    def test_extract_person_name_dynamic(self, qa_system):
        """Test dynamic person name extraction."""
        question = "When is Layla planning her trip?"
        person = qa_system._extract_person_name(question, qa_system.extractor.fetch_messages())
        
        assert person == "Layla Johnson"
    
    def test_extract_person_name_full_name(self, qa_system):
        """Test extraction of full names."""
        question = "What is Vikram Desai doing?"
        person = qa_system._extract_person_name(question, qa_system.extractor.fetch_messages())
        
        assert person == "Vikram Desai"
    
    def test_filter_by_person(self, qa_system):
        """Test filtering messages by person."""
        messages = qa_system.extractor.fetch_messages()
        filtered = qa_system._filter_by_person(messages, "Layla")
        
        assert len(filtered) == 1
        assert filtered[0]['user_name'] == 'Layla Johnson'
    
    def test_extract_answer_when_question(self, qa_system):
        """Test answer extraction for 'when' questions."""
        messages = [{
            'user_name': 'Layla Johnson',
            'message': 'I am planning a trip to London on 2024-06-15'
        }]
        
        answer = qa_system._extract_answer("When is Layla going to London?", messages)
        
        assert "2024-06-15" in answer
        assert "Layla Johnson" in answer
        assert "London" in answer
    
    def test_extract_answer_how_many(self, qa_system):
        """Test answer extraction for 'how many' questions."""
        messages = [{
            'user_name': 'Vikram Desai',
            'message': 'I have 3 cars in my garage'
        }]
        
        answer = qa_system._extract_answer("How many cars does Vikram have?", messages)
        
        assert "3" in answer
        assert "cars" in answer.lower()
        assert "Vikram Desai" in answer
    
    def test_extract_answer_favorite(self, qa_system):
        """Test answer extraction for 'favorite' questions."""
        messages = [{
            'user_name': 'Amira Hassan',
            'message': 'My favorite restaurants are Le Bernardin and Nobu'
        }]
        
        answer = qa_system._extract_answer("What are Amira's favorite restaurants?", messages)
        
        assert "Amira Hassan" in answer
        assert "restaurants" in answer.lower()
        assert "Le Bernardin" in answer or "Nobu" in answer
    
    def test_classify_question(self, qa_system):
        """Test question classification."""
        assert qa_system._classify_question("How many cars?") == 'count'
        assert qa_system._classify_question("When is the trip?") == 'temporal'
        assert qa_system._classify_question("Where is she going?") == 'location'
        assert qa_system._classify_question("What are favorite restaurants?") == 'list'
        assert qa_system._classify_question("What is she doing?") == 'action'
        assert qa_system._classify_question("Tell me about this") == 'generic'
    
    def test_suggest_alternatives_person_not_found(self, qa_system):
        """Test alternative suggestions when person not found."""
        messages = qa_system.extractor.fetch_messages()
        answer = qa_system._suggest_alternatives("What is Unknown Person doing?", "Unknown Person", messages)
        
        assert "couldn't find" in answer.lower()
        assert "available members" in answer.lower()
    
    def test_suggest_alternatives_no_answer(self, qa_system):
        """Test alternative suggestions when no answer found."""
        messages = qa_system.extractor.fetch_messages()
        answer = qa_system._suggest_alternatives("Random question?", None, messages)
        
        assert "couldn't find" in answer.lower()
        assert "members" in answer.lower()

