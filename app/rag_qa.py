"""
RAG-based Question Answering System.

Retrieves top-k relevant messages using semantic embeddings,
then uses a Small Language Model (SLM) to generate answers.
"""
import logging
import numpy as np
import os
import pickle
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)

# Cache directory for embeddings
CACHE_DIR = ".cache"
EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "embeddings_cache.pkl")

# Try to use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")


class RAGQASystem:
    """
    RAG-based QA system that:
    1. Retrieves top-k messages using semantic embeddings
    2. Generates answers using a Small Language Model (SLM)
    """
    
    def __init__(
        self,
        extractor,
        use_embeddings: bool = True,
        use_slm: bool = True,
        top_k: int = 5,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        slm_model_name: str = "google/flan-t5-small"
    ):
        """
        Initialize the RAG QA system.
        
        Args:
            extractor: DataExtractor instance for fetching messages
            use_embeddings: Whether to use semantic embeddings for retrieval
            use_slm: Whether to use SLM for answer generation
            top_k: Number of top messages to retrieve
            embedding_model_name: Name of the embedding model
            slm_model_name: Name of the SLM for answer generation
        """
        self.extractor = extractor
        self.use_embeddings = use_embeddings
        self.use_slm = use_slm
        self.top_k = top_k
        
        # Initialize embedding model for retrieval
        self.embedding_model = None
        self.message_embeddings = None
        self.messages = None
        self._closest_name_suggestion = None  # Store closest name suggestion for mismatches
        
        if use_embeddings:
            try:
                logger.info(f"Loading embedding model: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Falling back to keyword search.")
                self.use_embeddings = False
        
        # Initialize SLM for answer generation - LAZY LOADING
        # Don't load at startup to save memory, load on first use
        self.slm_pipeline = None
        self.slm_model_name = slm_model_name if use_slm else None
        self.use_slm = use_slm
        self._slm_loaded = False  # Track if SLM has been loaded
        
        # Don't load all messages upfront - load on-demand based on questions
        # This is more efficient and accurate
        self.messages = None
        self.message_embeddings = None
        logger.info("System initialized. Messages will be loaded on-demand based on questions.")
    
    def _get_messages_hash(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a hash of messages to detect changes."""
        if not messages:
            return ""
        # Create a hash from message IDs and content
        message_ids = [str(msg.get('id', '')) for msg in messages[:100]]  # Sample first 100
        content = "".join(message_ids)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_embeddings_cache(self) -> Optional[Tuple[np.ndarray, str]]:
        """Load cached embeddings from disk if available."""
        if not os.path.exists(EMBEDDINGS_CACHE_FILE):
            return None
        
        try:
            with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                embeddings = cache_data.get('embeddings')
                messages_hash = cache_data.get('messages_hash')
                logger.info(f"Loaded cached embeddings (hash: {messages_hash[:8]}...)")
                return embeddings, messages_hash
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")
            return None
    
    def _save_embeddings_cache(self, embeddings: np.ndarray, messages_hash: str):
        """Save embeddings to disk cache."""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_data = {
                'embeddings': embeddings,
                'messages_hash': messages_hash
            }
            with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Saved embeddings cache (hash: {messages_hash[:8]}...)")
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")
    
    def _load_messages(self, force_refresh: bool = False):
        """
        Load all messages from API and compute embeddings.
        Uses cache when possible to avoid recomputation.
        
        Args:
            force_refresh: If True, force refresh messages and recompute embeddings
        """
        try:
            logger.info("Loading messages from API...")
            # Only force refresh if explicitly requested
            self.messages = self.extractor.fetch_messages(force_refresh=force_refresh)
            logger.info(f"Loaded {len(self.messages)} messages")
            
            if not self.messages:
                self.message_embeddings = None
                return
            
            # Check if we can use cached embeddings
            current_hash = self._get_messages_hash(self.messages)
            cached_data = None if force_refresh else self._load_embeddings_cache()
            
            # Validate cached embeddings match current messages
            if cached_data and cached_data[1] == current_hash:
                cached_embeddings = cached_data[0]
                # CRITICAL: Ensure embeddings count matches messages count
                if cached_embeddings.shape[0] == len(self.messages):
                    # Use cached embeddings
                    self.message_embeddings = cached_embeddings
                    logger.info(f"Using cached embeddings (messages unchanged, {len(self.messages)} messages)")
                else:
                    logger.warning(
                        f"Cache mismatch: {cached_embeddings.shape[0]} embeddings vs {len(self.messages)} messages. "
                        "Recomputing embeddings."
                    )
                    cached_data = None  # Force recomputation
            
            if not cached_data and self.use_embeddings and self.embedding_model:
                # Compute new embeddings
                logger.info(f"Computing embeddings for {len(self.messages)} messages (this may take a minute)...")
                message_texts = [self._format_message(msg) for msg in self.messages]
                self.message_embeddings = self.embedding_model.encode(
                    message_texts,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    batch_size=32  # Process in batches for efficiency
                )
                # Save to cache
                self._save_embeddings_cache(self.message_embeddings, current_hash)
                logger.info(f"✓ Message embeddings computed and cached ({self.message_embeddings.shape[0]} embeddings)")
            else:
                self.message_embeddings = None
                
        except Exception as e:
            logger.error(f"Error loading messages: {e}")
            self.messages = []
            self.message_embeddings = None
    
    def _format_message(self, msg: Dict[str, Any]) -> str:
        """Format a message dictionary into a searchable text string."""
        parts = []
        if isinstance(msg, dict):
            if 'user_name' in msg:
                parts.append(f"User: {msg['user_name']}")
            if 'message' in msg:
                parts.append(msg['message'])
            # Include other relevant fields
            for key in ['timestamp', 'content']:
                if key in msg:
                    parts.append(f"{key}: {msg[key]}")
        return " | ".join(parts)
    
    def _extract_person_name(self, question: str) -> Optional[str]:
        """Extract person name from question."""
        import re
        # Look for capitalized names (first name or full name)
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', question)
        # Filter out common words
        common_words = {'When', 'What', 'How', 'Where', 'Who', 'Why', 'The', 'Is', 'Are', 'Does', 'Have', 'Has', 'Many', 'Favorite', 'Restaurants', 'Planning', 'Trip', 'London', 'Paris', 'Cars', 'Car'}
        for name in names:
            if name not in common_words:
                return name
        return None
    
    def _find_closest_name(self, query_name: str, all_messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find the closest matching name in the dataset using fuzzy matching.
        Returns the closest name if similarity is above threshold, None otherwise.
        """
        if not query_name or not all_messages:
            return None
        
        # Get all unique user names from messages
        unique_names = set()
        for msg in all_messages:
            user_name = msg.get('user_name', '')
            if user_name:
                unique_names.add(user_name)
        
        if not unique_names:
            return None
        
        query_lower = query_name.lower().strip()
        query_first_name = query_lower.split()[0] if query_lower.split() else query_lower
        
        # First, try exact first name match
        for name in unique_names:
            name_lower = name.lower()
            name_first = name_lower.split()[0] if name_lower.split() else name_lower
            if query_first_name == name_first:
                return name
        
        # If no exact match, try fuzzy matching using simple string similarity
        from difflib import SequenceMatcher
        
        best_match = None
        best_score = 0.0
        threshold = 0.6  # Minimum similarity threshold
        
        for name in unique_names:
            name_lower = name.lower()
            name_first = name_lower.split()[0] if name_lower.split() else name_lower
            
            # Calculate similarity with first name
            first_name_similarity = SequenceMatcher(None, query_first_name, name_first).ratio()
            
            # Calculate similarity with full name
            full_name_similarity = SequenceMatcher(None, query_lower, name_lower).ratio()
            
            # Take the maximum similarity
            similarity = max(first_name_similarity, full_name_similarity)
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = name
        
        return best_match
    
    def _load_relevant_messages(self, question: str) -> List[Dict[str, Any]]:
        """
        Load only messages relevant to the question.
        Filters by person name AND keywords for better accuracy.
        """
        # Reset closest name suggestion for each new question
        self._closest_name_suggestion = None
        
        # Extract person name from question
        person_name = self._extract_person_name(question)
        
        # Extract keywords from question
        question_lower = question.lower()
        keywords = []
        if 'london' in question_lower:
            keywords.append('london')
        if 'trip' in question_lower or 'planning' in question_lower or 'going' in question_lower:
            keywords.append('trip')
            keywords.append('planning')
            keywords.append('going')
        if 'car' in question_lower or 'vehicle' in question_lower or 'cars' in question_lower:
            keywords.append('car')
            keywords.append('vehicle')
            keywords.append('cars')
        if 'restaurant' in question_lower or 'favorite' in question_lower or 'favourite' in question_lower:
            keywords.append('restaurant')
            keywords.append('favorite')
            keywords.append('favourite')
        
        # Get all messages (from cache if available)
        all_messages = self.extractor.fetch_messages(force_refresh=False)
        
        if not all_messages:
            logger.warning("No messages available")
            return []
        
        import json
        relevant = []
        
        # Filter by person name AND keywords (both must match for better accuracy)
        if person_name and keywords:
            logger.info(f"Filtering messages for person: {person_name} AND keywords: {keywords}")
            person_lower = person_name.lower()
            for msg in all_messages:
                user_name = msg.get('user_name', '').lower()
                msg_text = json.dumps(msg).lower()
                
                # Check person match
                person_match = person_lower in user_name or person_lower in msg_text
                # Check keyword match (at least one keyword must be present)
                keyword_match = any(kw in msg_text for kw in keywords)
                
                if person_match and keyword_match:
                    relevant.append(msg)
            
            if relevant:
                # Remove duplicates based on message content
                seen = set()
                unique = []
                for msg in relevant:
                    msg_id = msg.get('id', '')
                    msg_content = msg.get('message', '')
                    key = (msg_id, msg_content)
                    if key not in seen:
                        seen.add(key)
                        unique.append(msg)
                
                logger.info(f"Found {len(unique)} unique messages matching person and keywords")
                return unique[:200]  # Limit for efficiency
        
        # Fallback: filter by person name only
        if person_name:
            logger.info(f"Filtering messages for person: {person_name} (no keyword match)")
            filtered = self.extractor.get_member_data(person_name)
            if filtered:
                # Remove duplicates
                seen = set()
                unique = []
                for msg in filtered:
                    msg_id = msg.get('id', '')
                    msg_content = msg.get('message', '')
                    key = (msg_id, msg_content)
                    if key not in seen:
                        seen.add(key)
                        unique.append(msg)
                logger.info(f"Found {len(unique)} unique messages for {person_name}")
                return unique[:200]
            else:
                # No exact match found - try to find closest name
                closest_name = self._find_closest_name(person_name, all_messages)
                if closest_name:
                    logger.info(f"No exact match for '{person_name}', found closest match: '{closest_name}'")
                    # Store closest name for later use in answer
                    self._closest_name_suggestion = closest_name
                    # Try filtering with closest name
                    filtered = self.extractor.get_member_data(closest_name)
                    if filtered:
                        seen = set()
                        unique = []
                        for msg in filtered:
                            msg_id = msg.get('id', '')
                            msg_content = msg.get('message', '')
                            key = (msg_id, msg_content)
                            if key not in seen:
                                seen.add(key)
                                unique.append(msg)
                        logger.info(f"Found {len(unique)} unique messages for closest match '{closest_name}'")
                        return unique[:200]
        
        # Fallback: filter by keywords only
        if keywords:
            logger.info(f"Filtering messages by keywords: {keywords}")
            for msg in all_messages:
                msg_text = json.dumps(msg).lower()
                if any(kw in msg_text for kw in keywords):
                    relevant.append(msg)
            if relevant:
                # Remove duplicates
                seen = set()
                unique = []
                for msg in relevant:
                    msg_id = msg.get('id', '')
                    msg_content = msg.get('message', '')
                    key = (msg_id, msg_content)
                    if key not in seen:
                        seen.add(key)
                        unique.append(msg)
                logger.info(f"Found {len(unique)} unique messages matching keywords")
                return unique[:200]
        
        # Last resort: return a sample
        logger.info(f"No specific filter matched, using first 200 messages")
        return all_messages[:200]
    
    def _retrieve_top_k(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant messages for the question.
        Loads only relevant messages on-demand, then uses embeddings.
        
        Args:
            question: The question to answer
            
        Returns:
            List of top-k most relevant messages
        """
        # Load only relevant messages for this question
        relevant_messages = self._load_relevant_messages(question)
        
        if not relevant_messages:
            return []
        
        # Compute embeddings only for relevant messages (on-demand)
        if self.use_embeddings and self.embedding_model:
            try:
                # Compute embeddings for relevant messages only
                logger.debug(f"Computing embeddings for {len(relevant_messages)} relevant messages")
                message_texts = [self._format_message(msg) for msg in relevant_messages]
                message_embeddings = self.embedding_model.encode(
                    message_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Compute question embedding
                question_embedding = self.embedding_model.encode(
                    question,
                    convert_to_numpy=True
                )
                
                # Calculate cosine similarities
                similarities = np.dot(
                    message_embeddings,
                    question_embedding
                ) / (
                    np.linalg.norm(message_embeddings, axis=1) *
                    np.linalg.norm(question_embedding)
                )
                
                # Get top-k indices
                k = min(self.top_k, len(similarities))
                top_indices = np.argsort(similarities)[::-1][:k]
                
                # Return top-k messages with their similarity scores
                retrieved = []
                for idx in top_indices:
                    if 0 <= idx < len(relevant_messages):
                        retrieved.append({
                            'message': relevant_messages[idx],
                            'similarity': float(similarities[idx])
                        })
                
                if retrieved:
                    similarities_str = ', '.join([f'{r["similarity"]:.3f}' for r in retrieved])
                    logger.info(f"Retrieved {len(retrieved)} messages with similarities: {similarities_str}")
                    return retrieved
                
            except Exception as e:
                logger.warning(f"Error in semantic search: {e}. Falling back to keyword search.")
        
        # Fallback: keyword-based search on relevant messages
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        scored_messages = []
        for msg in relevant_messages:
            msg_text = self._format_message(msg).lower()
            msg_words = set(msg_text.split())
            
            # Simple word overlap score
            overlap = len(question_words & msg_words)
            if overlap > 0:
                scored_messages.append({
                    'message': msg,
                    'similarity': overlap / len(question_words)
                })
        
        # Sort by score and return top-k
        scored_messages.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_messages[:self.top_k]
    
    def _validate_context_relevance(self, question: str, context_messages: List[Dict[str, Any]]) -> bool:
        """
        Validate that the context messages are actually relevant to the question.
        Returns True if context is relevant, False otherwise.
        """
        if not context_messages:
            return False
        
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        # Extract key entities from question
        person_name = self._extract_person_name(question)
        has_person = person_name is not None
        
        # Check keywords
        has_location = any(loc in question_lower for loc in ['london', 'paris', 'tokyo', 'trip', 'going'])
        has_count = 'how many' in question_lower or 'count' in question_lower
        has_restaurant = 'restaurant' in question_lower or 'favorite' in question_lower
        
        # Check if any message actually contains relevant information
        relevant_count = 0
        for item in context_messages[:5]:  # Check top 5
            msg = item['message']
            msg_text = self._format_message(msg).lower()
            msg_words = set(msg_text.split())
            
            # Check person match
            person_match = True
            if has_person:
                person_lower = person_name.lower()
                user_name = msg.get('user_name', '').lower()
                person_match = person_lower in user_name or person_lower in msg_text
            
            # Check keyword overlap (at least 2-3 keywords should match)
            keyword_overlap = len(question_words & msg_words)
            keyword_match = keyword_overlap >= 2
            
            # Check specific requirements
            specific_match = True
            if has_location:
                specific_match = any(loc in msg_text for loc in ['london', 'paris', 'tokyo', 'trip', 'going', 'planning'])
            elif has_count:
                import re
                specific_match = bool(re.search(r'\d+', msg_text))  # Has numbers
            elif has_restaurant:
                specific_match = 'restaurant' in msg_text or 'favorite' in msg_text
            
            # Message is relevant if person matches AND (keywords match OR specific requirement matches)
            if person_match and (keyword_match or specific_match):
                relevant_count += 1
        
        # Need at least 1-2 relevant messages
        return relevant_count >= 1
    
    def _load_slm_if_needed(self):
        """Lazy load SLM only when needed to save memory."""
        if not self.use_slm or not self.slm_model_name:
            return False
        
        if self._slm_loaded and self.slm_pipeline is not None:
            return True
        
        try:
            logger.info(f"Loading SLM: {self.slm_model_name} (lazy loading)")
            tokenizer = AutoTokenizer.from_pretrained(self.slm_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.slm_model_name)
            self.slm_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if DEVICE == "cuda" else -1,
                max_length=512
            )
            self._slm_loaded = True
            logger.info("SLM loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load SLM: {e}. Falling back to template-based answers.")
            self.use_slm = False
            return False
    
    def _generate_answer_with_slm(
        self,
        question: str,
        context_messages: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using SLM from retrieved context.
        Only generates answer if context is actually relevant.
        SLM is loaded lazily on first use to save memory.
        
        Args:
            question: The question to answer
            context_messages: List of retrieved messages with similarity scores
            
        Returns:
            Generated answer string
        """
        if not context_messages:
            return "I couldn't find any relevant information in the member data to answer your question."
        
        # Validate context relevance BEFORE using SLM
        if not self._validate_context_relevance(question, context_messages):
            logger.warning("Context messages are not relevant to the question, not using SLM")
            return "I couldn't find any relevant information in the member data to answer your question."
        
        # Format context from retrieved messages
        context_texts = []
        for item in context_messages:
            msg = item['message']
            context_texts.append(self._format_message(msg))
        
        context = "\n".join(context_texts[:3])  # Use top 3 for context
        
        # Lazy load SLM if needed
        if self.use_slm and self.slm_model_name:
            if not self._load_slm_if_needed():
                # SLM failed to load, use template fallback
                return self._generate_template_answer(question, context_messages)
        
        if self.use_slm and self.slm_pipeline:
            try:
                # Create prompt optimized for flan-t5-small
                # Flan-T5 works better with instruction-style prompts
                prompt = f"""Given the following information, answer the question.

Information:
{context}

Question: {question}

Answer:"""
                
                logger.debug(f"SLM prompt: {prompt[:200]}...")
                
                # Generate answer with flan-t5-small
                # Use greedy decoding (do_sample=False) to avoid repetitive outputs
                result = self.slm_pipeline(
                    prompt,
                    max_length=80,
                    min_length=5,
                    num_return_sequences=1,
                    do_sample=False  # Greedy decoding for more consistent, less repetitive results
                )
                
                answer = result[0]['generated_text'].strip()
                logger.debug(f"SLM generated answer: {answer}")
                
                # Clean up the answer - remove common prefixes
                for prefix in ['answer:', 'Answer:', 'answer', 'Answer']:
                    if answer.lower().startswith(prefix.lower()):
                        answer = answer[len(prefix):].strip()
                        if answer.startswith(':'):
                            answer = answer[1:].strip()
                
                # Remove repetitive patterns (common issue with small models)
                # Split by common separators and take unique parts
                import re
                # Remove excessive repetition (same word/phrase repeated 3+ times)
                words = answer.split()
                if len(words) > 0:
                    # Simple deduplication: remove consecutive identical words
                    deduplicated = []
                    prev_word = None
                    for word in words:
                        if word != prev_word:
                            deduplicated.append(word)
                            prev_word = word
                        elif len(deduplicated) == 0 or deduplicated[-1] != word:
                            deduplicated.append(word)
                    answer = ' '.join(deduplicated)
                
                # Validate answer is meaningful and not just repeating the question
                if not answer or len(answer) < 5:
                    logger.warning("SLM generated empty/short answer")
                    return "I couldn't find specific information to answer your question in the member data."
                
                # Check if answer is just repeating question words (hallucination indicator)
                answer_lower = answer.lower()
                question_lower = question.lower()
                question_words = set(question_lower.split())
                answer_words = set(answer_lower.split())
                
                # If answer is mostly question words, it's likely hallucinated
                overlap_ratio = len(answer_words & question_words) / len(answer_words) if answer_words else 0
                if overlap_ratio > 0.7 and len(answer_words) < 10:
                    logger.warning(f"Answer appears to repeat question (overlap: {overlap_ratio:.2f}), likely hallucinated")
                    return "I couldn't find specific information to answer your question in the member data."
                
                # Check if answer contains actual information (dates, numbers, names, etc.)
                import re
                has_date = bool(re.search(r'\d{4}-\d{2}-\d{2}', answer))
                has_number = bool(re.search(r'\b\d+\b', answer))
                has_capitalized = bool(re.search(r'\b[A-Z][a-z]+\b', answer))
                
                # Check for suspicious patterns (hallucination indicators)
                # Long strings of repeated digits (like 3000000...)
                if re.search(r'\d{20,}', answer):
                    logger.warning("Answer contains suspicious long number pattern, likely hallucinated")
                    return "I couldn't find specific information to answer your question in the member data."
                
                # Check if answer is just repeating the same word/character
                words = answer.split()
                if len(words) > 0:
                    unique_words = len(set(words))
                    if unique_words < 3 and len(words) > 5:
                        logger.warning("Answer is mostly repetitive, likely hallucinated")
                        return "I couldn't find specific information to answer your question in the member data."
                
                # If answer is too generic and has no specific info, it's likely wrong
                if not (has_date or has_number or has_capitalized) and len(answer_words) < 8:
                    logger.warning("Answer lacks specific information, likely not based on data")
                    return "I couldn't find specific information to answer your question in the member data."
                
                # Validate numbers are reasonable (not astronomical)
                numbers = re.findall(r'\b\d+\b', answer)
                for num_str in numbers:
                    try:
                        num = int(num_str)
                        if num > 1000000:  # Unreasonably large number
                            logger.warning(f"Answer contains unreasonably large number ({num}), likely hallucinated")
                            return "I couldn't find specific information to answer your question in the member data."
                    except ValueError:
                        pass
                
                if len(answer.split()) <= 30:  # Reasonable length
                    return answer
                else:
                    logger.warning(f"SLM generated answer too long ({len(answer.split())} words)")
                    return answer[:200]  # Truncate if too long
                
            except Exception as e:
                logger.warning(f"Error generating answer with SLM: {e}. Returning no-data message.")
                return "I couldn't find specific information to answer your question in the member data."
        else:
            # No SLM, use template but validate first
            if self._validate_context_relevance(question, context_messages):
                return self._generate_template_answer(question, context_messages)
            else:
                return "I couldn't find any relevant information in the member data to answer your question."
    
    def _generate_template_answer(
        self,
        question: str,
        context_messages: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a simple template-based answer from context.
        Fallback when SLM is not available.
        """
        if not context_messages:
            return "I couldn't find any relevant information to answer your question."
        
        # Use the most relevant message
        top_message = context_messages[0]['message']
        msg_text = self._format_message(top_message)
        
        # Extract key information
        user_name = top_message.get('user_name', 'Unknown')
        
        # Simple answer extraction
        if 'when' in question.lower() or 'date' in question.lower():
            # Try to extract date
            import re
            dates = re.findall(r'\d{4}-\d{2}-\d{2}', msg_text)
            if dates:
                return f"{user_name} is planning for {dates[0]}."
        
        if 'how many' in question.lower():
            # Try to extract numbers
            import re
            numbers = re.findall(r'\b(\d+)\b', msg_text)
            if numbers:
                return f"{user_name} has {numbers[0]}."
        
        # Default: return formatted message
        return f"Based on the data: {msg_text}"
    
    def answer(self, question: str) -> Tuple[str, float]:
        """
        Answer a question about member data.
        
        Args:
            question: The question to answer
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        question = question.strip()
        logger.info(f"Answering question: {question}")
        
        # Retrieve top-k relevant messages
        retrieved = self._retrieve_top_k(question)
        
        if not retrieved:
            # Check if we have a closest name suggestion
            person_name = self._extract_person_name(question)
            if person_name and self._closest_name_suggestion:
                suggestion_msg = f" I couldn't find '{person_name}' in the dataset. Did you mean '{self._closest_name_suggestion}'?"
                self._closest_name_suggestion = None  # Reset
                return (
                    "I couldn't find any relevant information in the member data to answer your question." + suggestion_msg,
                    0.0
                )
            return (
                "I couldn't find any relevant information in the member data to answer your question.",
                0.0
            )
        
        # Check if retrieved messages actually contain relevant information
        top_similarity = retrieved[0]['similarity']
        
        # STRICT validation: If similarity is low, messages are likely not relevant
        if top_similarity < 0.4:
            logger.warning(f"Low similarity score ({top_similarity:.3f}), likely no relevant data")
            # Check if we have a closest name suggestion
            if self._closest_name_suggestion:
                suggestion_msg = f" I couldn't find '{self._extract_person_name(question)}' in the dataset. Did you mean '{self._closest_name_suggestion}'?"
                self._closest_name_suggestion = None  # Reset
                return (
                    "I couldn't find any relevant information in the member data to answer your question." + suggestion_msg,
                    0.0
                )
            return (
                "I couldn't find any relevant information in the member data to answer your question.",
                0.0
            )
        
        # Validate context relevance BEFORE generating answer
        if not self._validate_context_relevance(question, retrieved):
            logger.warning("Retrieved messages are not relevant to the question")
            # Check if we have a closest name suggestion
            if self._closest_name_suggestion:
                suggestion_msg = f" I couldn't find '{self._extract_person_name(question)}' in the dataset. Did you mean '{self._closest_name_suggestion}'?"
                self._closest_name_suggestion = None  # Reset
                return (
                    "I couldn't find any relevant information in the member data to answer your question." + suggestion_msg,
                    0.0
                )
            return (
                "I couldn't find any relevant information in the member data to answer your question.",
                0.0
            )
        
        confidence = min(top_similarity, 1.0)
        
        # Generate answer using SLM (which will also validate internally)
        answer = self._generate_answer_with_slm(question, retrieved)
        
        # Final validation: Check if answer indicates no data found
        if "couldn't find" in answer.lower() or "no relevant" in answer.lower():
            confidence = 0.0
        
        logger.info(f"Generated answer with confidence: {confidence:.3f}")
        return answer, confidence

