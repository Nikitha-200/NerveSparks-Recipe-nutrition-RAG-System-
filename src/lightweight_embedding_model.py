"""
Lightweight Embedding Model
Uses TF-IDF and cosine similarity for semantic search without heavy dependencies.
"""

from typing import Union, List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class LightweightEmbeddingModel:
    """Lightweight embedding model using TF-IDF for semantic search."""
    
    def __init__(self, max_features: int = 1000):
        """
        Initialize the lightweight embedding model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        self.is_fitted = False
        self.embedding_dimension = max_features
        
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better TF-IDF performance.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text into TF-IDF embeddings.
        
        Args:
            text: Text or list of texts to encode
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(t) for t in text]
        
        if not self.is_fitted:
            # Fit the vectorizer on the first batch
            embeddings = self.vectorizer.fit_transform(processed_texts).toarray()
            self.is_fitted = True
        else:
            # Transform using fitted vectorizer
            embeddings = self.vectorizer.transform(processed_texts).toarray()
        
        if normalize:
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms
            
        return embeddings
    
    def encode_recipe_text(self, recipe_text: str) -> np.ndarray:
        """
        Encode recipe text specifically.
        
        Args:
            recipe_text: Recipe text to encode
            
        Returns:
            Recipe embedding
        """
        return self.encode_text(recipe_text)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode search query.
        
        Args:
            query: Search query to encode
            
        Returns:
            Query embedding
        """
        return self.encode_text(query)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Ensure embeddings are 2D
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
            
        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0, 0]
        return float(similarity)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts in batch.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Batch of embeddings
        """
        return self.encode_text(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': 'tfidf_lightweight',
            'embedding_dimension': self.embedding_dimension,
            'max_features': self.max_features,
            'is_fitted': self.is_fitted,
            'model_type': 'tfidf_lightweight'
        }
    
    def encode_with_metadata(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Encode texts with metadata.
        
        Args:
            texts: List of texts to encode
            metadata: Optional metadata for each text
            
        Returns:
            Dictionary with embeddings and metadata
        """
        embeddings = self.batch_encode(texts)
        
        result = {
            'embeddings': embeddings,
            'texts': texts
        }
        
        if metadata:
            result['metadata'] = metadata
            
        return result 