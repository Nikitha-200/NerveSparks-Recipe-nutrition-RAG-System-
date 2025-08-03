import unittest
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = RAGPipeline()
    
    def test_initialization(self):
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.data_processor)
        self.assertIsNotNone(self.pipeline.embedding_model)
        self.assertIsNotNone(self.pipeline.vector_store)
        self.assertIsNotNone(self.pipeline.dietary_analyzer)
        self.assertIsNotNone(self.pipeline.substitution_engine)
    
    def test_search_recipes(self):
        query = "vegetarian high protein"
        results = self.pipeline.search_recipes(
            query=query,
            dietary_restrictions=["vegetarian"],
            n_results=3
        )
        
        self.assertIn('results', results)
        self.assertIn('query', results)
        self.assertIn('total_found', results)
        self.assertIn('filters_applied', results)
    
    def test_get_ingredient_substitutions(self):
        substitutions = self.pipeline.get_ingredient_substitutions(
            ingredient="milk",
            dietary_restrictions=["dairy-free"]
        )
        
        self.assertIn('original_ingredient', substitutions)
        self.assertIn('substitutions', substitutions)
        self.assertIn('total_options', substitutions)
    
    def test_analyze_recipe_compatibility(self):
        sample_recipe = self.pipeline.data_processor.recipes[0]
        
        compatibility = self.pipeline.analyze_recipe_compatibility(
            recipe=sample_recipe,
            dietary_restrictions=["vegetarian"],
            allergies=["peanut"]
        )
        
        self.assertIn('compatibility_analysis', compatibility)
        self.assertIn('substitution_suggestions', compatibility)
    
    def test_get_system_stats(self):
        stats = self.pipeline.get_system_stats()
        
        self.assertIn('vector_store', stats)
        self.assertIn('embedding_model', stats)
        self.assertIn('total_recipes', stats)
        self.assertIn('dietary_restrictions', stats)
        self.assertIn('health_conditions', stats)
        self.assertIn('allergies', stats)


if __name__ == '__main__':
    unittest.main() 