"""
Unit tests for DataExtractor module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from app.extractor import DataExtractor


class TestDataExtractor:
    """Test cases for DataExtractor."""
    
    def test_init(self):
        """Test DataExtractor initialization."""
        extractor = DataExtractor()
        assert extractor._cache is None
        assert extractor._cache_timestamp is None
        assert extractor._cache_ttl == 300
    
    @patch('app.extractor.requests.get')
    def test_fetch_messages_success(self, mock_get):
        """Test successful message fetching."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'total': 2,
            'items': [
                {'id': '1', 'user_name': 'John Doe', 'message': 'Test message 1'},
                {'id': '2', 'user_name': 'Jane Smith', 'message': 'Test message 2'}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        extractor = DataExtractor()
        messages = extractor.fetch_messages(force_refresh=True)
        
        assert len(messages) == 2
        assert messages[0]['id'] == '1'
        assert messages[1]['id'] == '2'
    
    @patch('app.extractor.requests.get')
    def test_fetch_messages_pagination(self, mock_get):
        """Test pagination fetches all messages."""
        # Mock first page
        first_page = Mock()
        first_page.json.return_value = {
            'total': 250,
            'items': [{'id': f'{i}', 'user_name': 'User', 'message': f'Message {i}'} 
                     for i in range(100)]
        }
        first_page.raise_for_status = Mock()
        
        # Mock second page
        second_page = Mock()
        second_page.json.return_value = {
            'total': 250,
            'items': [{'id': f'{i}', 'user_name': 'User', 'message': f'Message {i}'} 
                     for i in range(100, 200)]
        }
        second_page.raise_for_status = Mock()
        
        # Mock third page
        third_page = Mock()
        third_page.json.return_value = {
            'total': 250,
            'items': [{'id': f'{i}', 'user_name': 'User', 'message': f'Message {i}'} 
                     for i in range(200, 250)]
        }
        third_page.raise_for_status = Mock()
        
        mock_get.side_effect = [first_page, second_page, third_page]
        
        extractor = DataExtractor()
        messages = extractor.fetch_messages(force_refresh=True)
        
        # Should fetch all 250 messages across 3 pages
        assert len(messages) == 250
        assert mock_get.call_count == 3
    
    def test_cache_usage(self):
        """Test that cache is used when valid."""
        extractor = DataExtractor()
        extractor._cache = [{'id': '1', 'message': 'Cached message'}]
        extractor._cache_timestamp = datetime.now()
        
        messages = extractor.fetch_messages(force_refresh=False)
        
        assert messages == extractor._cache
        assert len(messages) == 1
    
    def test_cache_expiry(self):
        """Test that cache is refreshed when expired."""
        extractor = DataExtractor()
        extractor._cache = [{'id': '1', 'message': 'Old message'}]
        extractor._cache_timestamp = datetime.now() - timedelta(seconds=400)  # Expired
        
        with patch('app.extractor.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'total': 1,
                'items': [{'id': '2', 'message': 'New message'}]
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            messages = extractor.fetch_messages(force_refresh=False)
            
            assert mock_get.called
            assert len(messages) == 1
    
    def test_get_member_data_filtering(self):
        """Test member data filtering by name."""
        extractor = DataExtractor()
        extractor._cache = [
            {'id': '1', 'user_name': 'John Doe', 'message': 'Message 1'},
            {'id': '2', 'user_name': 'Jane Smith', 'message': 'Message 2'},
            {'id': '3', 'user_name': 'John Smith', 'message': 'Message 3'}
        ]
        extractor._cache_timestamp = datetime.now()
        
        john_messages = extractor.get_member_data('John')
        
        assert len(john_messages) == 2
        assert all('john' in msg['user_name'].lower() for msg in john_messages)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        extractor = DataExtractor()
        extractor._cache = [{'id': '1'}]
        extractor._cache_timestamp = datetime.now()
        
        extractor.clear_cache()
        
        assert extractor._cache is None
        assert extractor._cache_timestamp is None

