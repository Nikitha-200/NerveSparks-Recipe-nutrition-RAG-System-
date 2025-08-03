"""
Real Embedding Model using Sentence Transformers
Provides high-quality text embeddings for semantic search.
"""

from typing import Union, List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:
    """Real embedding model using Sentence Transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text into embeddings.
        
        Args:
            text: Text or list of texts to encode
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(text, convert_to_numpy=True)
        
        if normalize:
            # Normalize embeddings to unit length
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
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
        # Ensure embeddings are 1D
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()
            
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
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
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'device': str(self.model.device),
            'max_seq_length': self.model.max_seq_length,
            'model_type': 'sentence_transformer'
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