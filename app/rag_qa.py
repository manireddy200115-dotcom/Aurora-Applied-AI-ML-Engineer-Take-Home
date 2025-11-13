"""
RAG-based question-answering system for member data.

Simple, unified approach:
- Semantic embeddings for retrieval (finding relevant messages)
- Small Language Model (SLM) for generation (creating natural answers)
- Handles both related and unrelated questions gracefully
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from app.extractor import DataExtractor

# Try to import sentence-transformers, fall back to keyword matching if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not available, falling back to keyword matching")

# Try to import transformers for SLM (Small Language Model) answer generation
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    SLM_AVAILABLE = True
except ImportError:
    SLM_AVAILABLE = False
    logging.warning("transformers not available, SLM answer generation disabled")

logger = logging.getLogger(__name__)


class RAGQASystem:
    """
    RAG-based QA system using semantic search + SLM for answer generation.
    
    Simple, unified approach:
    1. Semantic embeddings to find relevant messages (Retrieval)
    2. SLM to generate natural answers from context (Generation)
    3. Handles both related and unrelated questions
    """
    
    def __init__(
        self, 
        extractor: DataExtractor, 
        use_embeddings: bool = True,
        use_slm: bool = True,
        slm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Smaller, more compatible model
    ):
        """
        Initialize the QA system.
        
        Args:
            extractor: DataExtractor instance for fetching messages
            use_embeddings: Whether to use semantic embeddings (requires sentence-transformers)
            use_slm: Whether to use SLM for answer generation (requires transformers)
            slm_model_name: Name of the SLM model to use
        """
        self.extractor = extractor
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        self.use_slm = use_slm and SLM_AVAILABLE
        self.embedding_model = None
        self.slm_model = None
        self.slm_tokenizer = None
        self.message_embeddings_cache = None
        self.messages_cache = None
        self._known_names_cache = None
        self.slm_model_name = slm_model_name
        
        if self.use_embeddings:
            try:
                # Use a lightweight, fast model for semantic search
                logger.info("Loading sentence transformer model for semantic search...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Falling back to keyword matching.")
                self.use_embeddings = False
        
        if self.use_slm:
            try:
                logger.info(f"Loading SLM model: {slm_model_name}...")
                logger.info("Note: First-time model download may take a few minutes (~2-3GB)")
                
                # Use CPU by default, can be changed to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")
                
                # Load tokenizer and model
                self.slm_tokenizer = AutoTokenizer.from_pretrained(
                    slm_model_name,
                    trust_remote_code=True
                )
                
                # Set pad token if not set
                if self.slm_tokenizer.pad_token is None:
                    self.slm_tokenizer.pad_token = self.slm_tokenizer.eos_token
                
                # Load model with appropriate settings
                model_kwargs = {
                    "trust_remote_code": True,
                }
                
                if device == "cuda":
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto"
                    })
                else:
                    model_kwargs.update({
                        "torch_dtype": torch.float32,
                    })
                
                self.slm_model = AutoModelForCausalLM.from_pretrained(
                    slm_model_name,
                    **model_kwargs
                )
                
                if device == "cpu":
                    self.slm_model = self.slm_model.to(device)
                
                # Set to evaluation mode
                self.slm_model.eval()
                
                logger.info(f"SLM model ({slm_model_name}) loaded successfully on {device}")
            except Exception as e:
                logger.warning(f"Failed to load SLM model: {e}")
                logger.warning("Falling back to rule-based answer extraction.")
                logger.warning("To disable SLM, set use_slm=False when initializing RAGQASystem.")
                self.use_slm = False
                self.slm_model = None
                self.slm_tokenizer = None
    
    def answer(self, question: str) -> Tuple[str, float]:
        """
        Answer a question using semantic search and SLM-based answer generation.
        
        Process:
        1. Get all messages from the API
        2. Extract person name from question (if mentioned)
        3. Filter messages by person
        4. Use semantic embeddings to find top relevant messages
        5. Generate natural answer using SLM from retrieved context
        
        Args:
            question: Natural language question
            
        Returns:
            Tuple of (answer string, confidence score 0.0-1.0)
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        question = question.strip()
        logger.info(f"Processing question: {question} (embeddings: {self.use_embeddings})")
        
        # Get all messages
        messages = self.extractor.fetch_messages()
        if not messages:
            return ("I couldn't find any member data to answer your question.", 0.0)
        
        # Extract person name from question
        person = self._extract_person_name(question, messages)
        
        # Filter by person if mentioned
        if person:
            messages = self._filter_by_person(messages, person)
            if not messages:
                answer = self._suggest_alternatives(question, person, messages)
                return (answer, 0.0)
        
        # Find top relevant messages using semantic search
        if self.use_embeddings:
            relevant_messages, confidence = self._find_top_relevant_semantic(question, messages, top_k=5)
        else:
            relevant_message, confidence = self._find_most_relevant(question, messages)
            relevant_messages = [relevant_message] if relevant_message else []
        
        # Always use SLM for answer generation (simplified system)
        if not self.use_slm or not self.slm_model:
            raise RuntimeError("SLM is required but not available. Please ensure transformers is installed and model can be loaded.")
        
        # Generate answer using SLM
        answer = self._generate_slm_answer(question, relevant_messages, person)
        
        # Adjust confidence based on whether we found relevant messages
        if not relevant_messages:
            confidence = 0.0
        
        return (answer, confidence)
    
    def _extract_person_name(self, question: str, messages: List[Dict]) -> Optional[str]:
        """
        Extract person name from question dynamically.
        
        Args:
            question: The question string
            messages: All messages to extract known names from
            
        Returns:
            Person name if found, None otherwise
        """
        # Cache known names to avoid repeated extraction
        if self._known_names_cache is None:
            self._known_names_cache = set()
            for msg in messages:
                if isinstance(msg, dict):
                    user_name = msg.get('user_name', '').strip()
                    if user_name:
                        self._known_names_cache.add(user_name.lower())
        
        question_lower = question.lower()
        words = question.split()
        
        # Check against known names (case-insensitive)
        for name in self._known_names_cache:
            # Check for full name match
            if name in question_lower:
                # Find the actual name with proper casing
                for msg in messages:
                    if isinstance(msg, dict):
                        user_name = msg.get('user_name', '').strip()
                        if user_name.lower() == name:
                            return user_name
        
        # Check for capitalized words (likely names)
        for word in words:
            clean_word = word.rstrip("'s").rstrip("'").rstrip(',').rstrip('?')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                # Skip common question words
                if clean_word.lower() not in {'what', 'when', 'where', 'who', 'how', 'why', 'this', 'that'}:
                    # Check if it's part of a full name
                    word_idx = words.index(word)
                    if word_idx + 1 < len(words):
                        next_word = words[word_idx + 1].rstrip("'s").rstrip("'").rstrip(',').rstrip('?')
                        if next_word and next_word[0].isupper():
                            full_name = f"{clean_word} {next_word}"
                            # Verify it's a known name
                            if full_name.lower() in self._known_names_cache:
                                for msg in messages:
                                    if isinstance(msg, dict):
                                        user_name = msg.get('user_name', '').strip()
                                        if user_name.lower() == full_name.lower():
                                            return user_name
                    else:
                        # Single capitalized word - check if it matches a known first name
                        for msg in messages:
                            if isinstance(msg, dict):
                                user_name = msg.get('user_name', '').strip()
                                if user_name.lower().startswith(clean_word.lower()):
                                    return user_name
        
        return None
    
    def _filter_by_person(self, messages: List[Dict], person_name: str) -> List[Dict]:
        """Filter messages by person name (case-insensitive partial match)."""
        person_lower = person_name.lower()
        first_name = person_lower.split()[0] if person_lower.split() else person_lower
        
        filtered = []
        for msg in messages:
            if isinstance(msg, dict):
                user_name = msg.get('user_name', '').lower()
                if person_lower in user_name or first_name in user_name:
                    filtered.append(msg)
        
        return filtered
    
    def _find_top_relevant_semantic(
        self, question: str, messages: List[Dict], top_k: int = 3
    ) -> Tuple[List[Dict], float]:
        """
        Find top K most relevant messages using semantic embeddings (ML-based).
        
        This is the core ML component - uses neural network embeddings for semantic similarity.
        
        Args:
            question: The question string
            messages: List of messages to search
            top_k: Number of top messages to return
            
        Returns:
            Tuple of (list of top K messages sorted by relevance, confidence score)
        """
        if not self.embedding_model:
            best, confidence = self._find_most_relevant(question, messages)
            return ([best] if best else [], confidence)
        
        try:
            # Create embedding for the question
            question_embedding = self.embedding_model.encode(question, convert_to_numpy=True)
            
            # Check if we need to update embeddings cache
            if self.messages_cache != messages or self.message_embeddings_cache is None:
                # Create embeddings for all messages
                message_texts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        # Combine user name and message for better context
                        user_name = msg.get('user_name', '')
                        message_text = msg.get('message', '')
                        combined_text = f"{user_name}: {message_text}"
                        message_texts.append(combined_text)
                    else:
                        message_texts.append(str(msg))
                
                logger.debug(f"Creating embeddings for {len(message_texts)} messages...")
                self.message_embeddings_cache = self.embedding_model.encode(
                    message_texts, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                self.messages_cache = messages
                logger.debug("Embeddings created and cached")
            
            # Calculate cosine similarity between question and all messages
            # This is the ML-based semantic similarity calculation
            similarities = np.dot(self.message_embeddings_cache, question_embedding) / (
                np.linalg.norm(self.message_embeddings_cache, axis=1) * np.linalg.norm(question_embedding)
            )
            
            # Get top K indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort descending, take top K
            top_similarities = similarities[top_indices]
            
            # Get confidence from top similarity score (normalize to 0-1 range)
            # Cosine similarity is already in -1 to 1, but for embeddings it's typically 0-1
            # Normalize to ensure it's in 0-1 range
            max_similarity = float(top_similarities[0]) if len(top_similarities) > 0 else 0.0
            confidence = max(0.0, min(1.0, max_similarity))  # Clamp to [0, 1]
            
            logger.debug(f"Top {top_k} semantic similarity scores: {top_similarities}, confidence: {confidence:.3f}")
            
            # Filter by threshold and return top messages
            threshold = 0.3
            top_messages = []
            for idx, sim in zip(top_indices, top_similarities):
                if sim > threshold:
                    top_messages.append(messages[idx])
            
            if top_messages:
                return (top_messages, confidence)
            else:
                logger.warning(f"All similarities below threshold ({threshold}), trying fallback")
                best, confidence = self._find_most_relevant(question, messages)
                return ([best] if best else [], confidence)
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}. Falling back to keyword matching.")
            best, confidence = self._find_most_relevant(question, messages)
            return ([best] if best else [], confidence)
    
    def _find_most_relevant(self, question: str, messages: List[Dict]) -> Tuple[Optional[Dict], float]:
        """
        Find the most relevant message using keyword matching (fallback method).
        
        Args:
            question: The question string
            messages: List of messages to search
            
        Returns:
            Tuple of (most relevant message or None, confidence score 0.0-1.0)
        """
        question_lower = question.lower()
        
        # Extract meaningful keywords (remove stop words)
        stop_words = {
            'what', 'when', 'where', 'who', 'how', 'why', 'is', 'are', 'was', 'were',
            'does', 'do', 'did', 'has', 'have', 'had', 'the', 'a', 'an', 'to', 'for',
            'of', 'in', 'on', 'at', 'by', 'with', 'and', 'or', 'but', 'this', 'that',
            'these', 'those', 'can', 'could', 'should', 'would', 'may', 'might', 'many',
            'much', 'some', 'any', 'all', 'each', 'every'
        }
        
        question_words = set([
            w.lower() for w in question.split() 
            if w.lower() not in stop_words and len(w) > 2
        ])
        
        best_message = None
        best_score = 0
        max_possible_score = len(question_words) * 2 + 10  # Rough estimate
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            message = msg.get('message', '').lower()
            user_name = msg.get('user_name', '').lower()
            
            # Score based on keyword overlap
            message_words = set([w for w in message.split() if len(w) > 2])
            overlap = len(question_words & message_words)
            
            # Boost for exact phrase matches
            phrase_score = 0
            for phrase in question_words:
                if phrase in message:
                    phrase_score += 2
            
            # Boost for question-specific patterns
            if 'doing' in question_lower or 'planning' in question_lower:
                action_words = ['book', 'reserve', 'arrange', 'organize', 'secure', 'plan', 'going']
                if any(action in message for action in action_words):
                    phrase_score += 3
            
            # Temporal matching
            if 'this week' in question_lower:
                if 'this week' in message or 'this weekend' in message:
                    phrase_score += 5
            elif 'when' in question_lower or 'time' in question_lower or 'date' in question_lower:
                if re.search(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|(this|next|last|today|tomorrow)', message):
                    phrase_score += 3
            
            if 'where' in question_lower or 'going' in question_lower:
                if re.search(r'\b(to|in|at|from)\s+[A-Z][a-z]+', msg.get('message', '')):
                    phrase_score += 3
            
            if 'how many' in question_lower or 'how much' in question_lower:
                # Look for numbers in message
                numbers = re.findall(r'\b\d+\b', message)
                if numbers:
                    phrase_score += 2
            
            total_score = overlap * 2 + phrase_score
            
            if total_score > best_score:
                best_score = total_score
                best_message = msg
        
        # Normalize confidence score to 0-1 range
        if max_possible_score > 0:
            confidence = min(1.0, best_score / max_possible_score)
        else:
            confidence = 0.0
        
        return (best_message, confidence)
    
    def _generate_slm_answer(
        self, 
        question: str, 
        relevant_messages: List[Dict], 
        person: Optional[str] = None
    ) -> str:
        """
        Generate answer using Small Language Model (SLM).
        
        This method uses an instruction-tuned SLM to generate natural, well-phrased
        answers from retrieved context. It handles both related and unrelated questions.
        
        Args:
            question: The question string
            relevant_messages: List of relevant messages (can be empty for unrelated questions)
            person: Person name if mentioned
            
        Returns:
            Generated answer string
        """
        if not self.use_slm or not self.slm_model or not self.slm_tokenizer:
            # Simple fallback if SLM not available
            if relevant_messages:
                msg = relevant_messages[0]
                user_name = msg.get('user_name', 'Unknown')
                message_text = msg.get('message', '')
                return f"{user_name}: {message_text}"
            else:
                return "I couldn't find relevant information to answer your question."
        
        try:
            # Build context from relevant messages
            context_parts = []
            for msg in relevant_messages[:5]:  # Use top 5 messages
                if isinstance(msg, dict):
                    user_name = msg.get('user_name', 'Unknown')
                    message_text = msg.get('message', '')
                    context_parts.append(f"{user_name}: {message_text}")
            
            context = "\n".join(context_parts) if context_parts else "No relevant information found."
            
            # Create prompt for instruction-tuned model
            if relevant_messages:
                # Related question - answer based on context
                prompt = f"""Based ONLY on the member messages below, answer the question. If the answer is not in the messages, say "I don't have that information in the member data."

Member messages:
{context}

Question: {question}

Answer (be brief, 1 sentence if possible):"""
            else:
                # Unrelated question - provide helpful response
                prompt = f"""You are a member data assistant. The user asked: "{question}"

This question is NOT about member data. You can ONLY answer questions about members' travel plans, preferences, and activities.

Respond with exactly: "I can only answer questions about member data, such as travel plans, preferences, and activities."

Response:"""
            
            # Tokenize and generate
            inputs = self.slm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            # Move to same device as model
            device = next(self.slm_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate answer
            with torch.no_grad():
                # Fix for Phi-3 compatibility - don't pass past_key_values if not needed
                generation_kwargs = {
                    "max_new_tokens": 80,  # Reduced for more concise answers
                    "temperature": 0.5,  # Lower temperature for more focused responses
                    "top_p": 0.9,
                    "do_sample": True,
                    "pad_token_id": self.slm_tokenizer.pad_token_id or self.slm_tokenizer.eos_token_id,
                    "eos_token_id": self.slm_tokenizer.eos_token_id,
                    "repetition_penalty": 1.2,  # Reduce repetition
                }
                
                # Remove past_key_values from inputs if present (causes issues with Phi-3)
                clean_inputs = {k: v for k, v in inputs.items() if k != "past_key_values"}
                
                outputs = self.slm_model.generate(
                    **clean_inputs,
                    **generation_kwargs,
                )
            
            # Decode response
            generated_text = self.slm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part (after "Answer:" or "Response:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            elif "Response:" in generated_text:
                answer = generated_text.split("Response:")[-1].strip()
            else:
                # Take everything after the prompt
                answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer - remove unwanted formatting
            answer = self._clean_slm_answer(answer)
            
            if not answer:
                # If SLM failed, provide a simple fallback
                if relevant_messages:
                    # Try to extract basic info from first message
                    msg = relevant_messages[0]
                    user_name = msg.get('user_name', 'Unknown')
                    message_text = msg.get('message', '')
                    return f"{user_name}: {message_text}"
                else:
                    return "I can only answer questions about member data, such as travel plans, preferences, and activities."
            
            # For unrelated questions, ensure we give the right response
            if not relevant_messages:
                answer_lower = answer.lower()
                # Check if answer acknowledges it's unrelated
                if not any(phrase in answer_lower for phrase in [
                    "member data", "travel plans", "preferences", "activities",
                    "can only answer", "only answer questions"
                ]):
                    # Answer doesn't acknowledge limitation, use standard response
                    return "I can only answer questions about member data, such as travel plans, preferences, and activities."
            
            logger.debug(f"SLM generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with SLM: {e}", exc_info=True)
            # Simple fallback on error
            if relevant_messages:
                msg = relevant_messages[0]
                user_name = msg.get('user_name', 'Unknown')
                message_text = msg.get('message', '')
                return f"{user_name}: {message_text}"
            else:
                return "I couldn't find relevant information to answer your question."
    
    def _clean_slm_answer(self, answer: str) -> str:
        """
        Clean up SLM-generated answer to remove unwanted formatting.
        
        Args:
            answer: Raw answer from SLM
            
        Returns:
            Cleaned answer string
        """
        if not answer:
            return answer
        
        # Remove hashtags and everything after them
        if '#' in answer:
            answer = answer.split('#')[0].strip()
        
        # Remove emojis (common unicode ranges)
        import re
        # Remove emoji patterns
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+",
            flags=re.UNICODE
        )
        answer = emoji_pattern.sub('', answer)
        
        # Remove multiple spaces
        answer = re.sub(r'\s+', ' ', answer)
        
        # Take only first sentence for maximum conciseness
        sentences = re.split(r'[.!?]+', answer)
        if len(sentences) > 1:
            # Take first complete sentence
            answer = sentences[0].strip()
            if not answer.endswith(('.', '!', '?')):
                answer += '.'
        
        # Remove trailing punctuation issues
        answer = answer.strip()
        
        # Remove any remaining special formatting
        answer = re.sub(r'[#@]\w+', '', answer)  # Remove hashtags and mentions
        answer = re.sub(r'\s+', ' ', answer).strip()  # Clean up spaces
        
        return answer
    
    def _suggest_alternatives(
        self, question: str, person: Optional[str], all_messages: List[Dict]
    ) -> str:
        """
        Provide helpful suggestions when answer not found.
        
        Args:
            question: The original question
            person: Person name if mentioned (can be None)
            all_messages: All available messages
            
        Returns:
            Helpful error message with suggestions
        """
        # Get available names
        names = set()
        for msg in all_messages:
            if isinstance(msg, dict):
                user_name = msg.get('user_name', '').strip()
                if user_name:
                    names.add(user_name)
        
        names_list = sorted(list(names))
        
        if person:
            # Person not found or no messages
            if names_list:
                sample = ", ".join(names_list[:5])
                more = f" and {len(names_list) - 5} more" if len(names_list) > 5 else ""
                return (
                    f"I couldn't find '{person}' in the member data. "
                    f"Available members include: {sample}{more}. "
                    f"You can ask questions about any of these {len(names_list)} members."
                )
            return f"I couldn't find '{person}' in the member data."
        
        # General fallback
        if names_list:
            sample = ", ".join(names_list[:3])
            return (
                f"I couldn't find the answer to your question. "
                f"The database contains {len(names_list)} members (e.g., {sample}). "
                f"Try asking about a specific member, or rephrase your question."
            )
        
        return "I couldn't find the answer to your question in the available member data."
