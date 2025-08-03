"""
Lightweight Vector Store
Uses in-memory storage with efficient similarity search without heavy dependencies.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from pathlib import Path
import pickle
import json


class LightweightVectorStore:
    """Lightweight vector store using in-memory storage and cosine similarity."""
    
    def __init__(self, persist_directory: str = "lightweight_vector_db"):
        """
        Initialize the lightweight vector store.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # In-memory storage
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []
        
        # Load existing data if available
        self._load_data()
        
    def _load_data(self):
        """Load existing data from disk."""
        try:
            data_file = self.persist_directory / "vector_data.pkl"
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
                    self.embeddings = data.get('embeddings', [])
                    self.ids = data.get('ids', [])
        except Exception:
            pass
    
    def _save_data(self):
        """Save data to disk."""
        try:
            data = {
                'documents': self.documents,
                'metadatas': self.metadatas,
                'embeddings': self.embeddings,
                'ids': self.ids
            }
            data_file = self.persist_directory / "vector_data.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass
    
    def add_recipes(self, recipe_texts: List[str], 
                   recipe_metadata: Optional[List[Dict[str, Any]]] = None,
                   embeddings: Optional[np.ndarray] = None) -> List[str]:
        """
        Add recipes to the vector store.
        
        Args:
            recipe_texts: List of recipe text chunks
            recipe_metadata: Optional metadata for each recipe
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of document IDs
        """
        # Generate unique IDs
        doc_ids = [str(uuid.uuid4()) for _ in recipe_texts]
        
        # Prepare metadata
        if recipe_metadata is None:
            recipe_metadata = [{"text": text} for text in recipe_texts]
        
        # Add to storage
        self.documents.extend(recipe_texts)
        self.metadatas.extend(recipe_metadata)
        self.ids.extend(doc_ids)
        
        if embeddings is not None:
            self.embeddings.extend(embeddings.tolist())
        else:
            # Create random embeddings for demo
            mock_embeddings = np.random.rand(len(recipe_texts), 100)
            self.embeddings.extend(mock_embeddings.tolist())
        
        # Save to disk
        self._save_data()
        
        return doc_ids
    
    def search_recipes(self, query: str, n_results: int = 5,
                      filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for recipes using cosine similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional filter criteria
            
        Returns:
            Dictionary with search results
        """
        if not self.documents:
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}
        
        # Convert query to embedding (simplified)
        query_embedding = self._text_to_embedding(query)
        
        # Get filtered indices
        filtered_indices = self._apply_filters(filter_dict) if filter_dict else list(range(len(self.documents)))
        
        if not filtered_indices:
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}
        
        # Calculate similarities
        similarities = []
        for idx in filtered_indices:
            doc_embedding = np.array(self.embeddings[idx])
            # Ensure both embeddings have the same dimension
            query_emb = self._ensure_embedding_dimension(query_embedding)
            doc_emb = self._ensure_embedding_dimension(doc_embedding)
            similarity = cosine_similarity([query_emb], [doc_emb])[0, 0]
            similarities.append((idx, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        n_results = min(n_results, len(similarities))
        top_indices = [idx for idx, _ in similarities[:n_results]]
        
        # Format results
        return {
            'documents': [self.documents[i] for i in top_indices],
            'metadatas': [self.metadatas[i] for i in top_indices],
            'distances': [1 - similarities[i][1] for i in range(n_results)],
            'ids': [self.ids[i] for i in top_indices]
        }
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to embedding (simplified TF-IDF approach).
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        # Simple bag-of-words approach
        words = text.lower().split()
        embedding = np.zeros(100)  # Fixed size for simplicity
        
        for word in words:
            # Simple hash-based feature
            hash_val = hash(word) % 100
            embedding[hash_val] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _ensure_embedding_dimension(self, embedding: np.ndarray, target_dim: int = 100) -> np.ndarray:
        """
        Ensure embedding has the correct dimension.
        
        Args:
            embedding: Input embedding
            target_dim: Target dimension
            
        Returns:
            Resized embedding
        """
        if embedding.shape[0] != target_dim:
            # Resize to target dimension
            if embedding.shape[0] > target_dim:
                embedding = embedding[:target_dim]
            else:
                # Pad with zeros
                padding = np.zeros(target_dim - embedding.shape[0])
                embedding = np.concatenate([embedding, padding])
        
        return embedding
    
    def _apply_filters(self, filter_dict: Dict[str, Any]) -> List[int]:
        """
        Apply filters to get matching document indices.
        
        Args:
            filter_dict: Filter criteria
            
        Returns:
            List of matching document indices
        """
        if not filter_dict:
            return list(range(len(self.documents)))
        
        matching_indices = []
        
        for idx, metadata in enumerate(self.metadatas):
            matches = True
            
            for key, value in filter_dict.items():
                if key in metadata:
                    if isinstance(value, dict) and "$in" in value:
                        # Handle $in operator
                        if metadata[key] not in value["$in"]:
                            matches = False
                            break
                    elif isinstance(value, dict) and "$contains" in value:
                        # Handle $contains operator
                        if value["$contains"] not in str(metadata[key]):
                            matches = False
                            break
                    else:
                        # Direct equality
                        if metadata[key] != value:
                            matches = False
                            break
                else:
                    matches = False
                    break
            
            if matches:
                matching_indices.append(idx)
        
        return matching_indices
    
    def filter_by_dietary_restriction(self, restriction: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Filter recipes by dietary restriction.
        
        Args:
            restriction: Dietary restriction to filter by
            n_results: Number of results to return
            
        Returns:
            Dictionary with filtered results
        """
        where_filter = {
            "dietary_tags": {"$contains": restriction}
        }
        
        return self.search_recipes("", n_results, where_filter)
    
    def filter_by_health_condition(self, condition: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Filter recipes by health condition.
        
        Args:
            condition: Health condition to filter by
            n_results: Number of results to return
            
        Returns:
            Dictionary with filtered results
        """
        where_filter = {
            "health_benefits": {"$contains": condition}
        }
        
        return self.search_recipes("", n_results, where_filter)
    
    def get_recipe_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get recipe by document ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Recipe data or None
        """
        try:
            idx = self.ids.index(doc_id)
            return {
                'document': self.documents[idx],
                'metadata': self.metadatas[idx],
                'id': doc_id
            }
        except ValueError:
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        return {
            'total_documents': len(self.documents),
            'total_recipes': len(self.documents),  # Add this key for compatibility
            'collection_name': 'lightweight_vector_store',
            'persist_directory': str(self.persist_directory)
        }
    
    def delete_collection(self):
        """Delete the collection."""
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []
        self._save_data()
    
    def reset_collection(self):
        """Reset the collection."""
        self.delete_collection() 