import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall
)
from datasets import Dataset


class RAGASEvaluator:
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.evaluation_data = []
    
    def create_evaluation_dataset(self, queries: List[str], 
                                expected_answers: List[str] = None) -> Dataset:
        evaluation_data = []
        
        for i, query in enumerate(queries):
            results = self.rag_pipeline.search_recipes(
                query=query,
                n_results=3
            )
            
            context = self._create_context_from_results(results['results'])
            
            answer = self._generate_answer(query, context)
            
            expected_answer = expected_answers[i] if expected_answers else self._generate_expected_answer(query)
            
            evaluation_data.append({
                'question': query,
                'contexts': [context],
                'answer': answer,
                'ground_truth': expected_answer
            })
        
        return Dataset.from_list(evaluation_data)
    
    def _create_context_from_results(self, results: List[Dict[str, Any]]) -> str:
        context_parts = []
        
        for result in results:
            recipe = result['recipe']
            context_parts.append(f"Recipe: {recipe['title']}")
            context_parts.append(f"Description: {recipe['description']}")
            context_parts.append(f"Ingredients: {', '.join([ing['name'] for ing in recipe['ingredients']])}")
            context_parts.append(f"Nutrition: {recipe['nutritional_info']['calories']} calories, {recipe['nutritional_info']['protein']}g protein")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str) -> str:
        query_lower = query.lower()
        
        if 'vegetarian' in query_lower:
            return "Here are vegetarian recipes that match your search criteria."
        elif 'high protein' in query_lower:
            return "Here are high-protein recipes suitable for your needs."
        elif 'gluten-free' in query_lower:
            return "Here are gluten-free recipes that meet your requirements."
        elif 'diabetes' in query_lower:
            return "Here are diabetes-friendly recipes with controlled carbohydrates."
        else:
            return "Here are recipes that match your search criteria."
    
    def _generate_expected_answer(self, query: str) -> str:
        query_lower = query.lower()
        
        if 'vegetarian' in query_lower:
            return "Vegetarian recipes with plant-based ingredients."
        elif 'high protein' in query_lower:
            return "High-protein recipes with lean meats, eggs, or legumes."
        elif 'gluten-free' in query_lower:
            return "Gluten-free recipes without wheat, barley, or rye."
        elif 'diabetes' in query_lower:
            return "Diabetes-friendly recipes with low glycemic index."
        else:
            return "Recipes matching the search criteria."
    
    def evaluate_system(self, test_queries: List[str], 
                       expected_answers: List[str] = None) -> Dict[str, float]:
        try:
            dataset = self.create_evaluation_dataset(test_queries, expected_answers)
            
            results = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_relevancy,
                    context_recall
                ]
            )
            
            return {
                'faithfulness': results['faithfulness'],
                'answer_relevancy': results['answer_relevancy'],
                'context_relevancy': results['context_relevancy'],
                'context_recall': results['context_recall']
            }
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            return {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_relevancy': 0.0,
                'context_recall': 0.0
            }
    
    def evaluate_search_accuracy(self, test_queries: List[str]) -> Dict[str, float]:
        accuracies = []
        relevances = []
        
        for query in test_queries:
            results = self.rag_pipeline.search_recipes(query=query, n_results=5)
            
            accuracy = self._calculate_search_accuracy(query, results['results'])
            relevance = self._calculate_search_relevance(query, results['results'])
            
            accuracies.append(accuracy)
            relevances.append(relevance)
        
        return {
            'search_accuracy': np.mean(accuracies),
            'search_relevance': np.mean(relevances),
            'total_queries': len(test_queries)
        }
    
    def _calculate_search_accuracy(self, query: str, results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        
        query_lower = query.lower()
        relevant_count = 0
        
        for result in results:
            recipe = result['recipe']
            recipe_text = f"{recipe['title']} {recipe['description']}".lower()
            
            if 'vegetarian' in query_lower and 'vegetarian' in recipe_text:
                relevant_count += 1
            elif 'high protein' in query_lower and 'protein' in recipe_text:
                relevant_count += 1
            elif 'gluten-free' in query_lower and 'gluten' in recipe_text:
                relevant_count += 1
            else:
                relevant_count += 0.5
        
        return relevant_count / len(results)
    
    def _calculate_search_relevance(self, query: str, results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        
        query_words = set(query.lower().split())
        total_relevance = 0
        
        for result in results:
            recipe = result['recipe']
            recipe_text = f"{recipe['title']} {recipe['description']}".lower()
            recipe_words = set(recipe_text.split())
            
            overlap = len(query_words.intersection(recipe_words))
            relevance = overlap / len(query_words) if query_words else 0
            total_relevance += relevance
        
        return total_relevance / len(results)
    
    def evaluate_compatibility_analysis(self, test_recipes: List[Dict[str, Any]],
                                    test_profiles: List[Dict[str, Any]]) -> Dict[str, float]:
        accuracies = []
        
        for recipe in test_recipes:
            for profile in test_profiles:
                compatibility = self.rag_pipeline.analyze_recipe_compatibility(
                    recipe=recipe,
                    dietary_restrictions=profile.get('dietary_restrictions', []),
                    allergies=profile.get('allergies', []),
                    health_conditions=profile.get('health_conditions', [])
                )
                
                expected_compatible = self._calculate_expected_compatibility(recipe, profile)
                actual_compatible = compatibility['overall_compatible']
                
                accuracy = 1.0 if expected_compatible == actual_compatible else 0.0
                accuracies.append(accuracy)
        
        return {
            'compatibility_accuracy': np.mean(accuracies),
            'total_analyses': len(accuracies)
        }
    
    def _calculate_expected_compatibility(self, recipe: Dict[str, Any], 
                                      profile: Dict[str, Any]) -> bool:
        recipe_tags = recipe.get('dietary_tags', [])
        recipe_ingredients = [ing['name'].lower() for ing in recipe.get('ingredients', [])]
        
        dietary_restrictions = profile.get('dietary_restrictions', [])
        allergies = profile.get('allergies', [])
        
        for restriction in dietary_restrictions:
            if restriction not in recipe_tags:
                return False
        
        for allergy in allergies:
            if allergy in recipe_ingredients:
                return False
        
        return True
    
    def generate_evaluation_report(self, test_queries: List[str],
                                test_recipes: List[Dict[str, Any]] = None,
                                test_profiles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        ragas_results = self.evaluate_system(test_queries)
        search_results = self.evaluate_search_accuracy(test_queries)
        
        report = {
            'ragas_metrics': ragas_results,
            'search_metrics': search_results,
            'total_queries': len(test_queries)
        }
        
        if test_recipes and test_profiles:
            compatibility_results = self.evaluate_compatibility_analysis(test_recipes, test_profiles)
            report['compatibility_metrics'] = compatibility_results
        
        return report


def create_test_queries() -> List[str]:
    return [
        "vegetarian high protein recipes",
        "gluten-free breakfast options",
        "diabetes-friendly dinner recipes",
        "low-carb lunch ideas",
        "vegan dessert recipes"
    ]


def create_test_profiles() -> List[Dict[str, Any]]:
    return [
        {
            'dietary_restrictions': ['vegetarian'],
            'allergies': ['peanut'],
            'health_conditions': []
        },
        {
            'dietary_restrictions': ['gluten-free'],
            'allergies': [],
            'health_conditions': ['celiac_disease']
        },
        {
            'dietary_restrictions': ['diabetes_friendly'],
            'allergies': [],
            'health_conditions': ['diabetes']
        }
    ] 