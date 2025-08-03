import requests
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random


@dataclass
class RecipeSource:
    name: str
    api_url: str
    api_key: Optional[str] = None
    rate_limit: int = 100
    enabled: bool = True


class RecipeIntegrator:
    
    def __init__(self):
        self.sources = {
            'spoonacular': RecipeSource(
                name='Spoonacular',
                api_url='https://api.spoonacular.com/recipes',
                api_key=None,
                rate_limit=150,
                enabled=False
            ),
            'edamam': RecipeSource(
                name='Edamam',
                api_url='https://api.edamam.com/api/recipes/v2',
                api_key=None,
                rate_limit=10,
                enabled=False
            ),
            'mock_dynamic': RecipeSource(
                name='Mock Dynamic Recipes',
                api_url='mock://dynamic',
                rate_limit=1000,
                enabled=True
            )
        }
        
        self.categories = {
            'breakfast': ['pancakes', 'oatmeal', 'smoothie', 'eggs', 'toast'],
            'lunch': ['salad', 'sandwich', 'soup', 'pasta', 'rice'],
            'dinner': ['chicken', 'fish', 'beef', 'vegetarian', 'vegan'],
            'dessert': ['cake', 'cookies', 'ice_cream', 'pudding', 'fruit'],
            'snack': ['nuts', 'fruit', 'yogurt', 'chips', 'smoothie']
        }
        
        self.cuisines = [
            'mediterranean', 'asian', 'indian', 'american', 'italian', 
            'mexican', 'french', 'thai', 'japanese', 'chinese'
        ]
        
        self.dietary_options = [
            'vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'keto',
            'low_sodium', 'diabetes_friendly', 'heart_healthy'
        ]
    
    def fetch_recipes_from_api(self, source_name: str, query: str = None, 
                              dietary_restrictions: List[str] = None,
                              max_recipes: int = 10) -> List[Dict[str, Any]]:
        source = self.sources.get(source_name)
        if not source or not source.enabled:
            return []
        
        try:
            if source_name == 'mock_dynamic':
                return self._generate_mock_recipes(query, dietary_restrictions, max_recipes)
            
            return self._generate_mock_recipes(query, dietary_restrictions, max_recipes)
            
        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            return []
    
    def generate_recipes(self, query: str = None, 
                        dietary_restrictions: List[str] = None,
                        allergies: List[str] = None,
                        health_conditions: List[str] = None,
                        n_recipes: int = 5) -> List[Dict[str, Any]]:
        return self._generate_mock_recipes(query, dietary_restrictions, n_recipes)
    
    def _generate_mock_recipes(self, query: str = None, 
                              dietary_restrictions: List[str] = None,
                              max_recipes: int = 10) -> List[Dict[str, Any]]:
        recipes = []
        
        if query:
            keywords = query.lower().split()
        else:
            keywords = random.choice(list(self.categories.values()))
        
        for i in range(max_recipes):
            recipe_type = random.choice(list(self.categories.keys()))
            recipe_keywords = keywords[:3] if keywords else random.choice(list(self.categories.values()))
            
            recipe = self._create_mock_recipe(
                f"dynamic_recipe_{i+1}",
                recipe_type,
                recipe_keywords,
                dietary_restrictions or []
            )
            recipes.append(recipe)
        
        return recipes
    
    def _create_mock_recipe(self, recipe_id: str, recipe_type: str,
                           keywords: List[str], dietary_restrictions: List[str]) -> Dict[str, Any]:
        cuisine = random.choice(self.cuisines)
        
        available_dietary = [opt for opt in self.dietary_options if opt not in dietary_restrictions]
        dietary_tags = random.sample(available_dietary, min(2, len(available_dietary)))
        
        if dietary_restrictions:
            dietary_tags.extend(dietary_restrictions)
        
        health_benefits = []
        if 'diabetes_friendly' in dietary_tags:
            health_benefits.extend(['diabetes_friendly', 'blood_sugar_control'])
        if 'heart_healthy' in dietary_tags:
            health_benefits.extend(['heart_healthy', 'cholesterol_lowering'])
        if 'vegetarian' in dietary_tags:
            health_benefits.append('plant_based')
        if 'vegan' in dietary_tags:
            health_benefits.extend(['plant_based', 'dairy_free'])
        
        ingredients = self._generate_ingredients(recipe_type, dietary_restrictions)
        nutrition_info = self._generate_nutrition_info(recipe_type, dietary_restrictions)
        
        title_parts = [recipe_type.title()]
        if keywords:
            title_parts.append(keywords[0].title())
        if cuisine:
            title_parts.append(f"{cuisine.title()} Style")
        
        title = " ".join(title_parts)
        
        return {
            'id': recipe_id,
            'title': title,
            'description': f"A delicious {recipe_type} recipe with {cuisine} influences",
            'cuisine_type': cuisine,
            'dietary_tags': dietary_tags,
            'health_benefits': health_benefits,
            'ingredients': ingredients,
            'instructions': self._generate_instructions(ingredients),
            'nutritional_info': nutrition_info,
            'prep_time': random.randint(10, 45),
            'cook_time': random.randint(15, 60),
            'servings': random.randint(2, 6),
            'difficulty': random.choice(['easy', 'medium', 'hard']),
            'source': 'dynamic_generation'
        }
    
    def _generate_ingredients(self, recipe_type: str, dietary_restrictions: List[str]) -> List[Dict[str, Any]]:
        base_ingredients = {
            'breakfast': ['eggs', 'milk', 'flour', 'butter', 'sugar', 'vanilla'],
            'lunch': ['chicken', 'rice', 'vegetables', 'olive oil', 'garlic', 'onion'],
            'dinner': ['beef', 'pasta', 'tomatoes', 'cheese', 'herbs', 'wine'],
            'dessert': ['flour', 'sugar', 'eggs', 'butter', 'vanilla', 'chocolate'],
            'snack': ['nuts', 'fruits', 'yogurt', 'honey', 'cinnamon', 'seeds']
        }
        
        ingredients = base_ingredients.get(recipe_type, ['ingredient1', 'ingredient2', 'ingredient3'])
        
        if 'vegetarian' in dietary_restrictions:
            ingredients = [ing for ing in ingredients if ing not in ['beef', 'chicken']]
        if 'vegan' in dietary_restrictions:
            ingredients = [ing for ing in ingredients if ing not in ['milk', 'eggs', 'cheese', 'butter']]
        if 'gluten-free' in dietary_restrictions:
            ingredients = [ing for ing in ingredients if ing not in ['flour', 'pasta']]
        
        recipe_ingredients = []
        for i, ingredient in enumerate(ingredients[:6]):
            amount = random.randint(1, 4)
            unit = random.choice(['cup', 'tbsp', 'tsp', 'oz', 'piece'])
            recipe_ingredients.append({
                'name': ingredient,
                'amount': amount,
                'unit': unit,
                'notes': random.choice(['', 'fresh', 'organic', 'diced', 'chopped'])
            })
        
        return recipe_ingredients
    
    def _generate_nutrition_info(self, recipe_type: str, dietary_restrictions: List[str]) -> Dict[str, Any]:
        base_calories = {
            'breakfast': (200, 400),
            'lunch': (300, 600),
            'dinner': (400, 800),
            'dessert': (150, 350),
            'snack': (100, 250)
        }
        
        cal_range = base_calories.get(recipe_type, (200, 500))
        calories = random.randint(cal_range[0], cal_range[1])
        
        protein_ratio = 0.15 if 'vegetarian' not in dietary_restrictions else 0.10
        carb_ratio = 0.55 if 'keto' not in dietary_restrictions else 0.20
        fat_ratio = 0.30 if 'keto' not in dietary_restrictions else 0.70
        
        protein = int(calories * protein_ratio / 4)
        carbs = int(calories * carb_ratio / 4)
        fat = int(calories * fat_ratio / 9)
        fiber = random.randint(2, 8)
        
        return {
            'calories': calories,
            'protein': protein,
            'carbohydrates': carbs,
            'fat': fat,
            'fiber': fiber,
            'sodium': random.randint(200, 800),
            'sugar': random.randint(5, 25)
        }
    
    def _generate_instructions(self, ingredients: List[Dict[str, Any]]) -> List[str]:
        instructions = [
            "Prepare all ingredients as specified",
            "Heat cooking surface to medium temperature",
            "Combine ingredients in the specified order",
            "Cook until desired consistency is reached",
            "Let rest for a few minutes before serving",
            "Garnish and serve immediately"
        ]
        
        return instructions[:len(ingredients) + 1]
    
    def get_available_sources(self) -> List[str]:
        return [name for name, source in self.sources.items() if source.enabled]
    
    def get_source_stats(self) -> Dict[str, Any]:
        enabled_sources = self.get_available_sources()
        return {
            'total_sources': len(self.sources),
            'enabled_sources': len(enabled_sources),
            'available_sources': enabled_sources,
            'total_rate_limit': sum(self.sources[name].rate_limit for name in enabled_sources)
        } 