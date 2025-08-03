"""
Real Vector Store using ChromaDB
Provides efficient vector storage and retrieval for recipe embeddings.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
import chromadb
from chromadb.config import Settings
import uuid
from pathlib import Path


class VectorStore:
    """Real vector store using ChromaDB for recipe embeddings."""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection_name = "recipe_embeddings"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Recipe embeddings for semantic search"}
        )
        
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
        
        # Add to collection
        self.collection.add(
            documents=recipe_texts,
            metadatas=recipe_metadata,
            ids=doc_ids,
            embeddings=embeddings.tolist() if embeddings is not None else None
        )
        
        return doc_ids
    
    def search_recipes(self, query: str, n_results: int = 5,
                      filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for recipes using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Optional filter criteria
            
        Returns:
            Dictionary with search results
        """
        # Convert filter dict to ChromaDB format
        where_filter = self._convert_filter_to_chroma_format(filter_dict) if filter_dict else None
        
        # Search the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }
    
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
            results = self.collection.get(ids=[doc_id])
            if results['documents']:
                return {
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0],
                    'id': doc_id
                }
        except Exception:
            pass
        return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            return {
                'error': str(e),
                'collection_name': self.collection_name
            }
    
    def _convert_filter_to_chroma_format(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert filter dictionary to ChromaDB format.
        
        Args:
            filter_dict: Filter dictionary
            
        Returns:
            ChromaDB-compatible filter
        """
        chroma_filter = {}
        
        for key, value in filter_dict.items():
            if isinstance(value, dict) and "$in" in value:
                # Handle $in operator
                chroma_filter[key] = {"$in": value["$in"]}
            elif isinstance(value, dict) and "$contains" in value:
                # Handle $contains operator
                chroma_filter[key] = {"$contains": value["$contains"]}
            else:
                # Direct equality
                chroma_filter[key] = value
                
        return chroma_filter
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
    
    def reset_collection(self):
        """Reset the collection by deleting and recreating it."""
        self.delete_collection()
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Recipe embeddings for semantic search"}
        ) 