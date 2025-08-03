"""
Mock Vector Store Module
Simulates ChromaDB operations for storing and retrieving recipe embeddings.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid
from pathlib import Path
import json


class MockVectorStore:
    """Mock vector store for the Recipe RAG system."""
    
    def __init__(self, persist_directory: str = "mock_chroma_db"):
        """
        Initialize the mock vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # In-memory storage
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []
        
    def add_recipes(self, recipe_texts: List[str], 
                   recipe_metadata: Optional[List[Dict[str, Any]]] = None,
                   embeddings: Optional[np.ndarray] = None) -> List[str]:
        """
        Add recipes to the vector store.
        
        Args:
            recipe_texts: List of recipe text strings
            recipe_metadata: Optional list of recipe metadata dictionaries
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of document IDs
        """
        # Generate document IDs
        doc_ids = [str(uuid.uuid4()) for _ in recipe_texts]
        
        # Prepare metadata
        if recipe_metadata is None:
            recipe_metadata = [{"text": text} for text in recipe_texts]
        
        # Add documents to storage
        self.documents.extend(recipe_texts)
        self.metadatas.extend(recipe_metadata)
        self.ids.extend(doc_ids)
        
        if embeddings is not None:
            self.embeddings.extend(embeddings.tolist())
        else:
            # Create mock embeddings
            mock_embeddings = np.random.rand(len(recipe_texts), 384)
            self.embeddings.extend(mock_embeddings.tolist())
        
        return doc_ids
    
    def search_recipes(self, query: str, 
                      n_results: int = 5,
                      filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for recipes using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional filter dictionary for metadata
            
        Returns:
            Dictionary containing search results
        """
        # Simple mock search - return first n_results
        if not self.documents:
            return {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': []
            }
        
        # Filter by metadata if provided
        filtered_indices = list(range(len(self.documents)))
        if filter_dict:
            filtered_indices = self._apply_filters(filter_dict)
        
        # Return top results
        n_results = min(n_results, len(filtered_indices))
        selected_indices = filtered_indices[:n_results]
        
        return {
            'documents': [self.documents[i] for i in selected_indices],
            'metadatas': [self.metadatas[i] for i in selected_indices],
            'distances': [0.1 + 0.9 * (i / len(selected_indices)) for i in range(len(selected_indices))],
            'ids': [self.ids[i] for i in selected_indices]
        }
    
    def search_by_embedding(self, query_embedding: np.ndarray,
                           n_results: int = 5,
                           filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for recipes using pre-computed query embedding.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filter_dict: Optional filter dictionary for metadata
            
        Returns:
            Dictionary containing search results
        """
        return self.search_recipes("mock_query", n_results, filter_dict)
    
    def filter_by_dietary_restriction(self, restriction: str) -> List[Dict[str, Any]]:
        """
        Filter recipes by dietary restriction.
        
        Args:
            restriction: Dietary restriction to filter by
            
        Returns:
            List of filtered recipes
        """
        filtered_recipes = []
        for i, metadata in enumerate(self.metadatas):
            if 'dietary_tags' in metadata and restriction in metadata['dietary_tags']:
                filtered_recipes.append({
                    'id': self.ids[i],
                    'text': self.documents[i],
                    'metadata': metadata
                })
        
        return filtered_recipes
    
    def filter_by_health_condition(self, condition: str) -> List[Dict[str, Any]]:
        """
        Filter recipes by health condition.
        
        Args:
            condition: Health condition to filter by
            
        Returns:
            List of filtered recipes
        """
        filtered_recipes = []
        for i, metadata in enumerate(self.metadatas):
            if 'health_benefits' in metadata and condition in metadata['health_benefits']:
                filtered_recipes.append({
                    'id': self.ids[i],
                    'text': self.documents[i],
                    'metadata': metadata
                })
        
        return filtered_recipes
    
    def get_recipe_by_id(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific recipe by ID.
        
        Args:
            recipe_id: Recipe ID to retrieve
            
        Returns:
            Recipe dictionary or None if not found
        """
        try:
            index = self.ids.index(recipe_id)
            return {
                'id': self.ids[index],
                'text': self.documents[index],
                'metadata': self.metadatas[index]
            }
        except ValueError:
            return None
    
    def update_recipe(self, recipe_id: str, 
                     new_text: str,
                     new_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a recipe in the vector store.
        
        Args:
            recipe_id: Recipe ID to update
            new_text: New recipe text
            new_metadata: New recipe metadata
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            index = self.ids.index(recipe_id)
            self.documents[index] = new_text
            if new_metadata:
                self.metadatas[index] = new_metadata
            return True
        except ValueError:
            return False
    
    def delete_recipe(self, recipe_id: str) -> bool:
        """
        Delete a recipe from the vector store.
        
        Args:
            recipe_id: Recipe ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            index = self.ids.index(recipe_id)
            del self.documents[index]
            del self.metadatas[index]
            del self.embeddings[index]
            del self.ids[index]
            return True
        except ValueError:
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        return {
            'total_recipes': len(self.documents),
            'collection_name': 'mock_recipe_collection',
            'persist_directory': str(self.persist_directory)
        }
    
    def clear_collection(self) -> bool:
        """
        Clear all recipes from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        self.documents.clear()
        self.metadatas.clear()
        self.embeddings.clear()
        self.ids.clear()
        return True
    
    def get_similar_recipes(self, recipe_id: str, 
                           n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find recipes similar to a given recipe.
        
        Args:
            recipe_id: ID of the reference recipe
            n_results: Number of similar recipes to return
            
        Returns:
            List of similar recipes
        """
        # Get the reference recipe
        reference_recipe = self.get_recipe_by_id(recipe_id)
        if not reference_recipe:
            return []
        
        # Return other recipes as similar
        similar_recipes = []
        for i, doc_id in enumerate(self.ids):
            if doc_id != recipe_id:
                similar_recipes.append({
                    'id': doc_id,
                    'text': self.documents[i],
                    'metadata': self.metadatas[i],
                    'similarity': 0.8 - (len(similar_recipes) * 0.1)  # Decreasing similarity
                })
                if len(similar_recipes) >= n_results:
                    break
        
        return similar_recipes
    
    def _apply_filters(self, filter_dict: Dict[str, Any]) -> List[int]:
        """
        Apply filters to get matching indices.
        
        Args:
            filter_dict: Filter dictionary
            
        Returns:
            List of matching indices
        """
        matching_indices = []
        
        for i, metadata in enumerate(self.metadatas):
            matches = True
            
            for key, value in filter_dict.items():
                if key in metadata:
                    if isinstance(value, dict) and '$in' in value:
                        # Check if any value in the list matches
                        if not any(v in metadata[key] for v in value['$in']):
                            matches = False
                            break
                    elif isinstance(value, dict) and '$contains' in value:
                        # Check if the value contains the specified item
                        if value['$contains'] not in metadata[key]:
                            matches = False
                            break
                    else:
                        # Direct comparison
                        if metadata[key] != value:
                            matches = False
                            break
                else:
                    matches = False
                    break
            
            if matches:
                matching_indices.append(i)
        
        return matching_indices 