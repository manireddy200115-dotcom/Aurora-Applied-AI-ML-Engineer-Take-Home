"""
Data extraction module for fetching and parsing member messages from the API.
"""
import requests
import logging
from typing import List, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

API_BASE_URL = "https://november7-730026606190.europe-west1.run.app"


class DataExtractor:
    """Handles fetching and caching of member messages from the API."""
    
    def __init__(self):
        self.api_url = f"{API_BASE_URL}/messages"
        self._cache = None
        self._cache_timestamp = None
        self._cache_ttl = 300  # Cache for 5 minutes
    
    def fetch_messages(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch messages from the API.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of message dictionaries
        """
        # Check cache validity
        if not force_refresh and self._cache is not None:
            if self._cache_timestamp:
                age = (datetime.now() - self._cache_timestamp).total_seconds()
                if age < self._cache_ttl:
                    logger.info("Returning cached messages")
                    return self._cache
        
        try:
            logger.info(f"Fetching messages from {self.api_url}")
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, dict):
                # If response is a dict, try common keys
                if "items" in data:
                    messages = data["items"]
                elif "messages" in data:
                    messages = data["messages"]
                elif "data" in data:
                    messages = data["data"]
                else:
                    # If it's a single message dict, wrap it
                    messages = [data]
            elif isinstance(data, list):
                messages = data
            else:
                messages = []
            
            # Update cache
            self._cache = messages
            self._cache_timestamp = datetime.now()
            
            logger.info(f"Fetched {len(messages)} messages")
            return messages
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching messages: {str(e)}")
            # Return cached data if available, even if stale
            if self._cache is not None:
                logger.warning("Using stale cache due to API error")
                return self._cache
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            if self._cache is not None:
                return self._cache
            return []
    
    def get_member_data(self, member_name: str = None) -> List[Dict[str, Any]]:
        """
        Get messages filtered by member name (case-insensitive partial match).
        
        Args:
            member_name: Optional member name to filter by
            
        Returns:
            List of messages for the member
        """
        messages = self.fetch_messages()
        
        if not member_name:
            return messages
        
        # Normalize member name for matching
        member_name_lower = member_name.lower().strip()
        # Extract first name if full name provided
        first_name = member_name_lower.split()[0] if member_name_lower.split() else member_name_lower
        
        filtered = []
        for msg in messages:
            if isinstance(msg, dict):
                # Check user_name field specifically
                user_name = msg.get('user_name', '').lower()
                if member_name_lower in user_name or first_name in user_name:
                    filtered.append(msg)
                # Also check message content
                elif member_name_lower in json.dumps(msg).lower():
                    filtered.append(msg)
        
        return filtered
    
    def clear_cache(self):
        """Clear the cached messages."""
        self._cache = None
        self._cache_timestamp = None

