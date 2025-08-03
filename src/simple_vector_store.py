from typing import Dict, List, Any, Optional, Union
import numpy as np
import uuid
from pathlib import Path
import pickle
import json


class SimpleVectorStore:
    
    def __init__(self, persist_directory: str = "simple_vector_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []
        
        self._load_data()
        
    def _load_data(self):
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
        doc_ids = [str(uuid.uuid4()) for _ in recipe_texts]
        
        if recipe_metadata is None:
            recipe_metadata = [{"text": text} for text in recipe_texts]
        
        self.documents.extend(recipe_texts)
        self.metadatas.extend(recipe_metadata)
        self.ids.extend(doc_ids)
        
        if embeddings is not None:
            self.embeddings.extend(embeddings.tolist())
        else:
            simple_embeddings = []
            for text in recipe_texts:
                embedding = self._text_to_simple_embedding(text)
                simple_embeddings.append(embedding.tolist())
            self.embeddings.extend(simple_embeddings)
        
        self._save_data()
        return doc_ids
    
    def search_recipes(self, query: str, n_results: int = 5,
                      filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query_embedding = self._text_to_simple_embedding(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            doc_embedding = np.array(doc_embedding)
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if filter_dict:
            filtered_indices = self._apply_filters(filter_dict)
            similarities = [(i, sim) for i, sim in similarities if i in filtered_indices]
        
        top_indices = [idx for idx, _ in similarities[:n_results]]
        
        return {
            'ids': [self.ids[i] for i in top_indices],
            'documents': [self.documents[i] for i in top_indices],
            'metadatas': [self.metadatas[i] for i in top_indices],
            'distances': [1 - sim for _, sim in similarities[:n_results]]
        }
    
    def _text_to_simple_embedding(self, text: str) -> np.ndarray:
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
        
        embedding_dim = 100
        embedding = np.zeros(embedding_dim)
        
        for i, (word, freq) in enumerate(list(word_freq.items())[:embedding_dim]):
            embedding[i] = freq
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1.ndim > 1:
            vec1 = vec1.flatten()
        if vec2.ndim > 1:
            vec2 = vec2.flatten()
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _apply_filters(self, filter_dict: Dict[str, Any]) -> List[int]:
        matching_indices = []
        
        for i, metadata in enumerate(self.metadatas):
            matches = True
            
            for key, value in filter_dict.items():
                if key not in metadata:
                    matches = False
                    break
                
                metadata_value = metadata[key]
                
                if isinstance(value, dict) and "$in" in value:
                    if isinstance(metadata_value, list):
                        if not any(filter_val in metadata_value for filter_val in value["$in"]):
                            matches = False
                            break
                    else:
                        if metadata_value not in value["$in"]:
                            matches = False
                            break
                
                elif isinstance(value, dict) and "$contains" in value:
                    if isinstance(metadata_value, list):
                        if not any(filter_val in metadata_value for filter_val in value["$contains"]):
                            matches = False
                            break
                    else:
                        if value["$contains"] not in metadata_value:
                            matches = False
                            break
                
                elif isinstance(value, dict) and "$not_contains" in value:
                    if isinstance(metadata_value, list):
                        if any(filter_val in metadata_value for filter_val in value["$not_contains"]):
                            matches = False
                            break
                    else:
                        if value["$not_contains"] in metadata_value:
                            matches = False
                            break
                
                else:
                    if metadata_value != value:
                        matches = False
                        break
            
            if matches:
                matching_indices.append(i)
        
        return matching_indices
    
    def filter_by_dietary_restriction(self, restriction: str, n_results: int = 5) -> Dict[str, Any]:
        filter_dict = {'dietary_tags': {"$in": [restriction]}}
        return self.search_recipes("", n_results, filter_dict)
    
    def filter_by_health_condition(self, condition: str, n_results: int = 5) -> Dict[str, Any]:
        filter_dict = {'health_benefits': {"$in": [condition]}}
        return self.search_recipes("", n_results, filter_dict)
    
    def get_recipe_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            index = self.ids.index(doc_id)
            return {
                'id': self.ids[index],
                'document': self.documents[index],
                'metadata': self.metadatas[index]
            }
        except ValueError:
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            'total_documents': len(self.documents),
            'total_recipes': len(self.documents),
            'collection_name': 'simple_vector_store',
            'persist_directory': str(self.persist_directory)
        }
    
    def delete_collection(self):
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []
        self._save_data()
    
    def reset_collection(self):
        self.delete_collection() 