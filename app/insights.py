"""
Data insights module for analyzing member messages and identifying anomalies.
"""
import json
import re
import logging
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime
from app.extractor import DataExtractor

logger = logging.getLogger(__name__)


class DataInsights:
    """Analyzes member data for insights and anomalies."""
    
    def __init__(self, extractor: DataExtractor):
        self.extractor = extractor
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the member data and return insights.
        
        Returns:
            Dictionary containing various insights and anomalies
        """
        messages = self.extractor.fetch_messages(force_refresh=True)
        
        insights = {
            "total_messages": len(messages),
            "anomalies": [],
            "statistics": {},
            "data_quality_issues": []
        }
        
        if not messages:
            insights["anomalies"].append("No messages found in the API response")
            return insights
        
        # Analyze message structure
        insights["statistics"]["message_structure"] = self._analyze_structure(messages)
        
        # Check for missing fields
        insights["data_quality_issues"].extend(self._check_missing_fields(messages))
        
        # Check for inconsistent data formats
        insights["data_quality_issues"].extend(self._check_format_consistency(messages))
        
        # Check for duplicate messages
        duplicates = self._find_duplicates(messages)
        if duplicates:
            insights["anomalies"].append(f"Found {len(duplicates)} potential duplicate messages")
        
        # Check for date inconsistencies
        date_issues = self._check_date_consistency(messages)
        insights["data_quality_issues"].extend(date_issues)
        
        # Check for member name inconsistencies
        name_issues = self._check_name_consistency(messages)
        insights["anomalies"].extend(name_issues)
        
        # Analyze content patterns
        insights["statistics"]["content_patterns"] = self._analyze_content_patterns(messages)
        
        return insights
    
    def _analyze_structure(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze the structure of messages."""
        if not messages:
            return {}
        
        field_counts = Counter()
        for msg in messages:
            if isinstance(msg, dict):
                field_counts.update(msg.keys())
        
        return {
            "unique_fields": list(field_counts.keys()),
            "field_frequency": dict(field_counts),
            "most_common_fields": [field for field, _ in field_counts.most_common(5)]
        }
    
    def _check_missing_fields(self, messages: List[Dict]) -> List[str]:
        """Check for messages with missing expected fields."""
        issues = []
        
        if not messages:
            return issues
        
        # Identify common fields across messages
        all_fields = set()
        for msg in messages:
            if isinstance(msg, dict):
                all_fields.update(msg.keys())
        
        # Check if any messages are missing common fields
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                missing = all_fields - set(msg.keys())
                if missing and len(msg) < len(all_fields) * 0.5:  # Missing more than 50% of fields
                    issues.append(f"Message {i} is missing many common fields: {missing}")
        
        return issues
    
    def _check_format_consistency(self, messages: List[Dict]) -> List[str]:
        """Check for inconsistent data formats."""
        issues = []
        
        # Check date formats
        date_formats = []
        for msg in messages:
            msg_str = json.dumps(msg)
            # Look for dates
            dates = re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}', msg_str)
            date_formats.extend(dates)
        
        if date_formats:
            # Check if dates are in consistent format
            formats = set()
            for date in date_formats:
                if '-' in date and len(date.split('-')[0]) == 4:
                    formats.add('ISO')
                elif '/' in date:
                    formats.add('US')
                elif '-' in date:
                    formats.add('EU')
            
            if len(formats) > 1:
                issues.append(f"Inconsistent date formats found: {formats}")
        
        # Check for inconsistent field types
        field_types = {}
        for msg in messages:
            if isinstance(msg, dict):
                for key, value in msg.items():
                    if key not in field_types:
                        field_types[key] = type(value).__name__
                    elif field_types[key] != type(value).__name__:
                        issues.append(f"Inconsistent type for field '{key}': expected {field_types[key]}, found {type(value).__name__}")
        
        return issues
    
    def _find_duplicates(self, messages: List[Dict]) -> List[int]:
        """Find potential duplicate messages."""
        seen = {}
        duplicates = []
        
        for i, msg in enumerate(messages):
            msg_str = json.dumps(msg, sort_keys=True)
            msg_hash = hash(msg_str)
            if msg_hash in seen:
                duplicates.append(i)
            else:
                seen[msg_hash] = i
        
        return duplicates
    
    def _check_date_consistency(self, messages: List[Dict]) -> List[str]:
        """Check for date-related inconsistencies."""
        issues = []
        
        dates = []
        for msg in messages:
            msg_str = json.dumps(msg)
            found_dates = re.findall(r'\d{4}-\d{2}-\d{2}', msg_str)
            dates.extend(found_dates)
        
        if dates:
            # Check for future dates that seem unreasonable
            try:
                current_year = datetime.now().year
                for date_str in dates:
                    year = int(date_str.split('-')[0])
                    if year > current_year + 10:  # More than 10 years in future
                        issues.append(f"Unusual future date found: {date_str}")
                    elif year < 2000:  # Very old dates
                        issues.append(f"Very old date found: {date_str}")
            except:
                pass
        
        return issues
    
    def _check_name_consistency(self, messages: List[Dict]) -> List[str]:
        """Check for name inconsistencies (e.g., different capitalizations)."""
        issues = []
        
        names = set()
        for msg in messages:
            msg_str = json.dumps(msg)
            # Look for capitalized words that might be names
            potential_names = re.findall(r'\b[A-Z][a-z]+\b', msg_str)
            names.update(potential_names)
        
        # Check for similar names with different capitalizations
        name_lower_map = {}
        for name in names:
            name_lower = name.lower()
            if name_lower in name_lower_map and name_lower_map[name_lower] != name:
                issues.append(f"Inconsistent name capitalization: '{name_lower_map[name_lower]}' vs '{name}'")
            else:
                name_lower_map[name_lower] = name
        
        return issues
    
    def _analyze_content_patterns(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in message content."""
        patterns = {
            "contains_dates": 0,
            "contains_locations": 0,
            "contains_numbers": 0,
            "average_length": 0
        }
        
        total_length = 0
        
        for msg in messages:
            msg_str = json.dumps(msg).lower()
            total_length += len(msg_str)
            
            # Check for dates
            if re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}', msg_str):
                patterns["contains_dates"] += 1
            
            # Check for locations dynamically (capitalized words after location prepositions)
            # Pattern: "to/in/at/from [Capitalized Word]" suggests a location
            location_pattern = r'\b(to|in|at|from|going to)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
            if re.search(location_pattern, json.dumps(msg), re.IGNORECASE):
                patterns["contains_locations"] += 1
            
            # Check for numbers
            if re.search(r'\d+', msg_str):
                patterns["contains_numbers"] += 1
        
        if messages:
            patterns["average_length"] = total_length / len(messages)
        
        return patterns

