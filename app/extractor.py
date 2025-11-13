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
        Fetch all messages from the API using pagination.
        Ensures all messages are retrieved regardless of total count.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of message dictionaries (all messages from API)
        """
        # Check cache validity
        if not force_refresh and self._cache is not None:
            if self._cache_timestamp:
                age = (datetime.now() - self._cache_timestamp).total_seconds()
                if age < self._cache_ttl:
                    logger.info(f"Returning cached messages ({len(self._cache)} messages)")
                    return self._cache
        
        try:
            # Use pagination as primary method to ensure we get all messages
            logger.info(f"Fetching all messages from {self.api_url} using pagination")
            messages = self._fetch_messages_paginated()
            
            if messages:
                # Update cache
                self._cache = messages
                self._cache_timestamp = datetime.now()
                logger.info(f"Successfully fetched {len(messages)} messages")
                return messages
            
            logger.warning("No messages fetched from API")
            return []
            
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
    
    
    def _fetch_messages_paginated(self) -> List[Dict[str, Any]]:
        """
        Fetch all messages using pagination (offset-based).
        Continues fetching until all messages are retrieved.
        
        Returns:
            List of all message dictionaries
        """
        all_messages = []
        page_size = 100
        offset = 0
        max_pages = 200  # Increased safety limit for large datasets
        
        try:
            # Get first page to determine total
            response = requests.get(
                self.api_url, 
                params={'limit': page_size, 'offset': offset}, 
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict) or 'items' not in data:
                logger.error("Unexpected API response format for pagination")
                return []
            
            total = data.get('total', 0)
            first_page_items = data.get('items', [])
            all_messages.extend(first_page_items)
            
            if total == 0:
                logger.warning("API reports 0 total messages")
                return all_messages
            
            logger.info(f"API reports {total} total messages. Fetched {len(first_page_items)} in first page.")
            
            # Fetch remaining pages
            pages_fetched = 1
            while len(all_messages) < total and pages_fetched < max_pages:
                offset += page_size
                
                # Safety check: if offset exceeds total, we should have all messages
                if offset >= total:
                    logger.info(f"Offset {offset} >= total {total}, should have all messages")
                    break
                
                try:
                    response = requests.get(
                        self.api_url, 
                        params={'limit': page_size, 'offset': offset}, 
                        timeout=10
                    )
                    response.raise_for_status()
                    page_data = response.json()
                    
                    if isinstance(page_data, dict) and 'items' in page_data:
                        page_messages = page_data['items']
                        if not page_messages:  # Empty page, we're done
                            logger.info(f"Empty page at offset {offset}, stopping pagination")
                            break
                        all_messages.extend(page_messages)
                        pages_fetched += 1
                        
                        # Log progress every 10 pages
                        if pages_fetched % 10 == 0:
                            logger.info(f"Progress: {len(all_messages)}/{total} messages fetched ({pages_fetched} pages)")
                    else:
                        logger.warning(f"Unexpected response format on page {pages_fetched}")
                        break
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error fetching page at offset {offset}: {str(e)}")
                    # Continue with what we have rather than failing completely
                    break
            
            # Verify we got all messages
            if len(all_messages) < total:
                logger.warning(
                    f"Pagination incomplete: fetched {len(all_messages)}/{total} messages "
                    f"across {pages_fetched} pages"
                )
            else:
                logger.info(
                    f"Pagination complete: fetched all {len(all_messages)} messages "
                    f"across {pages_fetched} pages"
                )
            
            return all_messages
            
        except Exception as e:
            logger.error(f"Error in pagination: {str(e)}", exc_info=True)
            # Return what we have so far rather than failing completely
            if all_messages:
                logger.warning(f"Returning partial results: {len(all_messages)} messages")
            return all_messages
    
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

