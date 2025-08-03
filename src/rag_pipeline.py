from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .data_processor import DataProcessor
from .simple_embedding_model import SimpleEmbeddingModel as EmbeddingModel
from .simple_vector_store import SimpleVectorStore as VectorStore
from .dietary_analyzer import DietaryAnalyzer
from .substitution_engine import SubstitutionEngine
from .recipe_integration import RecipeIntegrator


class RAGPipeline:
    
    def __init__(self, data_dir: str = "data"):
        self.data_processor = DataProcessor(data_dir)
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.dietary_analyzer = None
        self.substitution_engine = None
        self.recipe_integrator = RecipeIntegrator()
        
        self._load_data()
        
    def _load_data(self):
        data = self.data_processor.load_all_data()
        
        self.dietary_analyzer = DietaryAnalyzer(data['dietary_guidelines'])
        self.substitution_engine = SubstitutionEngine(
            data['dietary_guidelines'], data['nutritional_data']
        )
        
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        recipe_chunks = self.data_processor.create_recipe_chunks()
        recipe_metadata = self.data_processor.get_recipe_metadata()
        
        embeddings = self.embedding_model.batch_encode(recipe_chunks)
        
        self.vector_store.add_recipes(
            recipe_texts=recipe_chunks,
            recipe_metadata=recipe_metadata,
            embeddings=embeddings
        )
    
    def search_recipes(self, query: str,
                      dietary_restrictions: List[str] = None,
                      allergies: List[str] = None,
                      health_conditions: List[str] = None,
                      n_results: int = 5,
                      include_dynamic: bool = True) -> Dict[str, Any]:
        filter_dict = self._build_filter_dict(dietary_restrictions, allergies, health_conditions)
        
        search_results = self.vector_store.search_recipes(
            query=query,
            n_results=n_results * 2,
            filter_dict=filter_dict
        )
        
        analyzed_results = []
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            search_results['ids'], search_results['documents'],
            search_results['metadatas'], search_results['distances']
        )):
            recipe = self._find_recipe_by_text(document)
            if recipe:
                compatibility = self.dietary_analyzer.analyze_recipe_compatibility(
                    recipe, dietary_restrictions, allergies, health_conditions
                )
                
                search_score = 1.0 - (distance / max(search_results['distances']))
                overall_score = self._calculate_overall_score(search_score, compatibility['overall_score'])
                
                analyzed_results.append({
                    'recipe': recipe,
                    'compatibility': compatibility,
                    'search_score': search_score,
                    'overall_score': overall_score,
                    'distance': distance,
                    'metadata': metadata
                })
        
        if include_dynamic:
            dynamic_recipes = self._get_dynamic_recipes(
                query, dietary_restrictions, allergies, health_conditions, n_results
            )
            
            for dynamic_recipe in dynamic_recipes:
                compatibility = self.dietary_analyzer.analyze_recipe_compatibility(
                    dynamic_recipe, dietary_restrictions, allergies, health_conditions
                )
                
                analyzed_results.append({
                    'recipe': dynamic_recipe,
                    'compatibility': compatibility,
                    'search_score': 0.8,
                    'overall_score': compatibility['overall_score'] * 0.8,
                    'distance': 0.2,
                    'metadata': {
                        'title': dynamic_recipe['title'],
                        'cuisine_type': dynamic_recipe.get('cuisine_type', 'Dynamic'),
                        'dietary_tags': dynamic_recipe.get('dietary_tags', []),
                        'health_benefits': dynamic_recipe.get('health_benefits', []),
                        'calories': dynamic_recipe.get('nutritional_info', {}).get('calories', 0),
                        'protein': dynamic_recipe.get('nutritional_info', {}).get('protein', 0),
                        'carbohydrates': dynamic_recipe.get('nutritional_info', {}).get('carbohydrates', 0),
                        'fat': dynamic_recipe.get('nutritional_info', {}).get('fat', 0),
                        'fiber': dynamic_recipe.get('nutritional_info', {}).get('fiber', 0)
                    }
                })
        
        analyzed_results.sort(key=lambda x: x['overall_score'], reverse=True)
        unique_results = []
        seen_titles = set()
        for result in analyzed_results:
            title = result['recipe']['title']
            if title not in seen_titles:
                unique_results.append(result)
                seen_titles.add(title)
        
        return {
            'results': unique_results[:n_results],
            'total_found': len(unique_results),
            'query': query,
            'filters_applied': {
                'dietary_restrictions': dietary_restrictions,
                'allergies': allergies,
                'health_conditions': health_conditions
            },
            'dynamic_recipes_included': include_dynamic
        }
    
    def _get_dynamic_recipes(self, query: str, dietary_restrictions: List[str] = None,
                            allergies: List[str] = None, health_conditions: List[str] = None,
                            n_results: int = 5) -> List[Dict[str, Any]]:
        dynamic_recipes = self.recipe_integrator.generate_recipes(
            query=query,
            dietary_restrictions=dietary_restrictions,
            allergies=allergies,
            health_conditions=health_conditions,
            n_recipes=n_results
        )
        
        return dynamic_recipes[:n_results]
    
    def get_personalized_recommendations(self,
                                       user_profile: Dict[str, Any],
                                       n_recommendations: int = 5,
                                       include_dynamic: bool = True) -> Dict[str, Any]:
        dietary_restrictions = user_profile.get('dietary_restrictions', [])
        allergies = user_profile.get('allergies', [])
        health_conditions = user_profile.get('health_conditions', [])
        preferences = user_profile.get('preferences', [])
        nutritional_goals = user_profile.get('nutritional_goals', {})
        
        search_query = self._build_search_query(preferences, nutritional_goals)
        
        search_results = self.search_recipes(
            query=search_query,
            dietary_restrictions=dietary_restrictions,
            allergies=allergies,
            health_conditions=health_conditions,
            n_results=n_recommendations * 2,
            include_dynamic=include_dynamic
        )
        
        optimized_results = []
        for result in search_results['results']:
            recipe = result['recipe']
            
            if nutritional_goals:
                optimization = self.substitution_engine.optimize_recipe_nutrition(
                    recipe, nutritional_goals, dietary_restrictions, allergies
                )
                result['nutrition_optimization'] = optimization
            
            optimized_results.append(result)
        
        return {
            'recommendations': optimized_results[:n_recommendations],
            'user_profile': user_profile,
            'search_query': search_query,
            'dynamic_recipes_included': include_dynamic
        }
    
    def get_ingredient_substitutions(self, ingredient: str,
                                   dietary_restrictions: List[str] = None,
                                   allergies: List[str] = None) -> Dict[str, Any]:
        substitutions = self.substitution_engine.find_substitutions(
            ingredient, dietary_restrictions, allergies
        )
        
        return {
            'original_ingredient': ingredient,
            'substitutions': substitutions,
            'total_options': len(substitutions),
            'filters_applied': {
                'dietary_restrictions': dietary_restrictions,
                'allergies': allergies
            }
        }
    
    def analyze_recipe_compatibility(self, recipe: Dict[str, Any],
                                   dietary_restrictions: List[str] = None,
                                   allergies: List[str] = None,
                                   health_conditions: List[str] = None) -> Dict[str, Any]:
        return self.dietary_analyzer.analyze_recipe_compatibility(
            recipe, dietary_restrictions, allergies, health_conditions
        )
    
    def _build_filter_dict(self, dietary_restrictions: List[str] = None,
                          allergies: List[str] = None,
                          health_conditions: List[str] = None) -> Optional[Dict[str, Any]]:
        filters = {}
        
        if dietary_restrictions:
            filters['dietary_tags'] = {"$in": dietary_restrictions}
        
        if allergies:
            incompatible_ingredients = []
            for allergy in allergies:
                allergy_info = self.data_processor.get_allergy_info(allergy)
                if allergy_info and 'incompatible_ingredients' in allergy_info:
                    incompatible_ingredients.extend(allergy_info['incompatible_ingredients'])
            
            if incompatible_ingredients:
                filters['ingredients'] = {"$not_contains": incompatible_ingredients}
        
        if health_conditions:
            health_benefits_mapping = {
                'diabetes': ['diabetes_friendly', 'blood_sugar_control'],
                'heart_disease': ['heart_healthy', 'cholesterol_lowering'],
                'hypertension': ['blood_pressure_control', 'heart_healthy'],
                'celiac_disease': ['celiac_safe', 'gluten_free'],
                'lactose_intolerance': ['lactose_intolerance_safe', 'dairy_free'],
                'obesity': ['weight_management', 'low_carb']
            }
            relevant_benefits = []
            for condition in health_conditions:
                if condition in health_benefits_mapping:
                    relevant_benefits.extend(health_benefits_mapping[condition])
            if relevant_benefits:
                filters['health_benefits'] = {"$in": relevant_benefits}
        
        return filters if filters else None
    
    def _build_search_query(self, preferences: List[str], nutritional_goals: Dict[str, Any]) -> str:
        query_parts = []
        
        if preferences:
            query_parts.extend(preferences)
        
        if nutritional_goals:
            if nutritional_goals.get('high_protein'):
                query_parts.append("high protein")
            if nutritional_goals.get('low_carb'):
                query_parts.append("low carb")
            if nutritional_goals.get('low_fat'):
                query_parts.append("low fat")
            if nutritional_goals.get('high_fiber'):
                query_parts.append("high fiber")
        
        return " ".join(query_parts) if query_parts else "healthy recipe"
    
    def _find_recipe_by_text(self, text: str) -> Optional[Dict[str, Any]]:
        if " | " not in text:
            return None
        
        title_part = text.split(" | ")[0]
        if title_part.startswith("Title: "):
            title = title_part[7:]
            
            for recipe in self.data_processor.recipes:
                if recipe.get('title') == title:
                    return recipe
        
        return None
    
    def _calculate_overall_score(self, search_score: float, compatibility_score: float) -> float:
        return (search_score * 0.6) + (compatibility_score * 0.4)
    
    def _extract_ingredients_from_issue(self, issue: str) -> List[str]:
        ingredients = []
        if "contains" in issue.lower():
            parts = issue.split("contains")
            if len(parts) > 1:
                ingredient_part = parts[1].strip()
                ingredients = [ing.strip() for ing in ingredient_part.split(",")]
        return ingredients
    
    def get_system_stats(self) -> Dict[str, Any]:
        vector_stats = self.vector_store.get_collection_stats()
        model_info = self.embedding_model.get_model_info()
        
        recipes = self.data_processor.recipes
        all_dietary_tags = []
        all_health_benefits = []
        all_cuisine_types = []
        all_ingredients = set()
        
        for recipe in recipes:
            all_dietary_tags.extend(recipe.get('dietary_tags', []))
            all_health_benefits.extend(recipe.get('health_benefits', []))
            all_cuisine_types.append(recipe.get('cuisine_type', 'Unknown'))
            
            for ingredient in recipe.get('ingredients', []):
                all_ingredients.add(ingredient['name'].lower())
        
        nutrition_stats = {
            'calories': {'min': 0, 'max': 0, 'avg': 0},
            'protein': {'min': 0, 'max': 0, 'avg': 0},
            'carbohydrates': {'min': 0, 'max': 0, 'avg': 0},
            'fat': {'min': 0, 'max': 0, 'avg': 0},
            'fiber': {'min': 0, 'max': 0, 'avg': 0}
        }
        
        if recipes:
            calories = [r.get('nutritional_info', {}).get('calories', 0) for r in recipes]
            protein = [r.get('nutritional_info', {}).get('protein', 0) for r in recipes]
            carbs = [r.get('nutritional_info', {}).get('carbohydrates', 0) for r in recipes]
            fat = [r.get('nutritional_info', {}).get('fat', 0) for r in recipes]
            fiber = [r.get('nutritional_info', {}).get('fiber', 0) for r in recipes]
            
            nutrition_stats['calories'] = {
                'min': min(calories), 'max': max(calories), 'avg': sum(calories) / len(calories)
            }
            nutrition_stats['protein'] = {
                'min': min(protein), 'max': max(protein), 'avg': sum(protein) / len(protein)
            }
            nutrition_stats['carbohydrates'] = {
                'min': min(carbs), 'max': max(carbs), 'avg': sum(carbs) / len(carbs)
            }
            nutrition_stats['fat'] = {
                'min': min(fat), 'max': max(fat), 'avg': sum(fat) / len(fat)
            }
            nutrition_stats['fiber'] = {
                'min': min(fiber), 'max': max(fiber), 'avg': sum(fiber) / len(fiber)
            }
        
        dietary_coverage = {}
        for restriction in self.dietary_analyzer.restrictions:
            compatible_count = sum(1 for recipe in recipes if restriction in recipe.get('dietary_tags', []))
            dietary_coverage[restriction] = {
                'total_recipes': len(recipes),
                'compatible_recipes': compatible_count,
                'coverage_percentage': (compatible_count / len(recipes) * 100) if recipes else 0
            }
        
        health_coverage = {}
        for condition in self.dietary_analyzer.health_conditions:
            condition_mapping = {
                'diabetes': ['diabetes_friendly', 'blood_sugar_control'],
                'heart_disease': ['heart_healthy', 'cholesterol_lowering'],
                'hypertension': ['blood_pressure_control', 'heart_healthy'],
                'celiac_disease': ['celiac_safe', 'gluten_free'],
                'lactose_intolerance': ['lactose_intolerance_safe', 'dairy_free'],
                'obesity': ['weight_management', 'low_carb']
            }
            
            relevant_benefits = condition_mapping.get(condition, [])
            compatible_count = sum(1 for recipe in recipes 
                                if any(benefit in recipe.get('health_benefits', []) 
                                      for benefit in relevant_benefits))
            
            health_coverage[condition] = {
                'total_recipes': len(recipes),
                'compatible_recipes': compatible_count,
                'coverage_percentage': (compatible_count / len(recipes) * 100) if recipes else 0
            }
        
        return {
            'vector_store': vector_stats,
            'embedding_model': model_info,
            'total_recipes': len(recipes),
            'dietary_restrictions': len(self.dietary_analyzer.restrictions),
            'health_conditions': len(self.dietary_analyzer.health_conditions),
            'allergies': len(self.dietary_analyzer.allergies),
            'unique_ingredients': len(all_ingredients),
            'cuisine_types': len(set(all_cuisine_types)),
            'dietary_tags_available': len(set(all_dietary_tags)),
            'health_benefits_available': len(set(all_health_benefits)),
            'nutrition_stats': nutrition_stats,
            'dietary_coverage': dietary_coverage,
            'health_coverage': health_coverage,
            'embedding_dimension': model_info.get('embedding_dimension', 0),
            'vocabulary_size': model_info.get('vocab_size', 0),
            'vector_store_size': vector_stats.get('total_documents', 0),
            'dynamic_integration': {
                'enabled': True,
                'available_sources': self.recipe_integrator.get_available_sources(),
                'source_stats': self.recipe_integrator.get_source_stats(),
                'dynamic_recipes_generated': 'Unlimited',
                'integration_capabilities': [
                    'Spoonacular API (requires key)',
                    'Edamam API (requires key)',
                    'Mock Dynamic Generation (active)',
                    'Web Scraping (configurable)',
                    'Database Integration (configurable)'
                ]
            },
            'last_updated': 'Real-time',
            'data_source': 'JSON files + Dynamic Integration'
        } 