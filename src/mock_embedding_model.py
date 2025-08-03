"""
Mock Embedding Model Module
Simulates text embeddings for the Recipe RAG system without heavy dependencies.
"""

from typing import List, Optional, Union
import numpy as np
import hashlib


class MockEmbeddingModel:
    """Mock embedding model for the Recipe RAG system."""
    
    def __init__(self, model_name: str = "mock-embedding-model"):
        """
        Initialize the mock embedding model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.embedding_dimension = 384
        
    def encode_text(self, text: Union[str, List[str]], 
                   normalize: bool = True) -> np.ndarray:
        """
        Encode text into mock embeddings.
        
        Args:
            text: Text or list of texts to encode
            normalize: Whether to normalize the embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(text, str):
            text = [text]
        
        embeddings = []
        for t in text:
            # Create deterministic embedding based on text hash
            hash_obj = hashlib.md5(t.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert hash to embedding vector
            embedding = np.frombuffer(hash_bytes, dtype=np.float32)
            
            # Extend to required dimension
            while len(embedding) < self.embedding_dimension:
                embedding = np.concatenate([embedding, embedding])
            embedding = embedding[:self.embedding_dimension]
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def encode_recipe_text(self, recipe_text: str) -> np.ndarray:
        """
        Encode recipe text into embedding.
        
        Args:
            recipe_text: Recipe text to encode
            
        Returns:
            Recipe embedding as numpy array
        """
        return self.encode_text(recipe_text)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode search query into embedding.
        
        Args:
            query: Search query to encode
            
        Returns:
            Query embedding as numpy array
        """
        return self.encode_text(query)
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
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
        similarity = np.dot(embedding1, embedding2.T) / (
            np.linalg.norm(embedding1, axis=1, keepdims=True) * 
            np.linalg.norm(embedding2, axis=1, keepdims=True)
        )
        return float(similarity[0, 0])
    
    def batch_encode(self, texts: List[str], 
                    batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of texts efficiently.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode_text(batch, normalize=True)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': 'cpu',
            'max_seq_length': 512,
            'embedding_dimension': self.embedding_dimension
        }
    
    def encode_with_metadata(self, texts: List[str], 
                           metadata: Optional[List[dict]] = None) -> dict:
        """
        Encode texts with optional metadata.
        
        Args:
            texts: List of texts to encode
            metadata: Optional list of metadata dictionaries
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        embeddings = self.batch_encode(texts)
        
        result = {
            'embeddings': embeddings,
            'texts': texts
        }
        
        if metadata:
            result['metadata'] = metadata
            
        return result 