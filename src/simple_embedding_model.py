from typing import Union, List, Dict, Any
import numpy as np
import re
import hashlib


class SimpleEmbeddingModel:
    
    def __init__(self, embedding_dimension: int = 100):
        self.embedding_dimension = embedding_dimension
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
        
    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    def _build_vocabulary(self, texts: List[str]):
        word_freq = {}
        
        for text in texts:
            processed_text = self._preprocess_text(text)
            words = processed_text.split()
            
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for i, (word, _) in enumerate(sorted_words[:self.embedding_dimension]):
            self.word_to_index[word] = i
            self.index_to_word[i] = word
        
        self.vocab_size = len(self.word_to_index)
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        embedding = np.zeros(self.embedding_dimension)
        processed_text = self._preprocess_text(text)
        words = processed_text.split()
        
        for word in words:
            if word in self.word_to_index:
                idx = self.word_to_index[word]
                embedding[idx] += 1
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        
        if self.vocab_size == 0:
            self._build_vocabulary(text)
        
        embeddings = []
        for t in text:
            embedding = self._text_to_embedding(t)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def encode_recipe_text(self, recipe_text: str) -> np.ndarray:
        return self.encode_text(recipe_text)
    
    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_text(query)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()
            
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        return self.encode_text(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': 'simple_embedding_model',
            'embedding_dimension': self.embedding_dimension,
            'vocab_size': self.vocab_size,
            'model_type': 'simple_bow'
        }
    
    def encode_with_metadata(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        embeddings = self.batch_encode(texts)
        
        result = {
            'embeddings': embeddings,
            'texts': texts
        }
        
        if metadata:
            result['metadata'] = metadata
            
        return result 