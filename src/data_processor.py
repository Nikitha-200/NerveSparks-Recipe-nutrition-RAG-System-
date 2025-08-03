

import json
import os
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path


class DataProcessor:
    
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.recipes = []
        self.nutritional_data = {}
        self.dietary_guidelines = {}
        
    def load_recipes(self) -> List[Dict[str, Any]]:
        recipe_file = self.data_dir / "recipes.json"
        if recipe_file.exists():
            with open(recipe_file, 'r', encoding='utf-8') as f:
                self.recipes = json.load(f)
        return self.recipes
    
    def load_nutritional_data(self) -> Dict[str, Any]:
        nutrition_file = self.data_dir / "nutritional_data.json"
        if nutrition_file.exists():
            with open(nutrition_file, 'r', encoding='utf-8') as f:
                self.nutritional_data = json.load(f)
        return self.nutritional_data
    
    def load_dietary_guidelines(self) -> Dict[str, Any]:
        guidelines_file = self.data_dir / "dietary_guidelines.json"
        if guidelines_file.exists():
            with open(guidelines_file, 'r', encoding='utf-8') as f:
                self.dietary_guidelines = json.load(f)
        return self.dietary_guidelines
    
    def load_all_data(self) -> Dict[str, Any]:
        return {
            'recipes': self.load_recipes(),
            'nutritional_data': self.load_nutritional_data(),
            'dietary_guidelines': self.load_dietary_guidelines()
        }
    
    def process_recipe_text(self, recipe: Dict[str, Any]) -> str:
        text_parts = [
            f"Title: {recipe.get('title', '')}",
            f"Description: {recipe.get('description', '')}",
            f"Cuisine Type: {recipe.get('cuisine_type', '')}",
            f"Dietary Tags: {', '.join(recipe.get('dietary_tags', []))}",
            f"Health Benefits: {', '.join(recipe.get('health_benefits', []))}",
            f"Ingredients: {', '.join([ing['name'] for ing in recipe.get('ingredients', [])])}",
            f"Instructions: {' '.join(recipe.get('instructions', []))}",
            f"Nutritional Info: Calories {recipe.get('nutritional_info', {}).get('calories', 0)}, "
            f"Protein {recipe.get('nutritional_info', {}).get('protein', 0)}g, "
            f"Carbs {recipe.get('nutritional_info', {}).get('carbohydrates', 0)}g, "
            f"Fat {recipe.get('nutritional_info', {}).get('fat', 0)}g"
        ]
        return " | ".join(text_parts)
    
    def get_recipes_by_dietary_restriction(self, restriction: str) -> List[Dict[str, Any]]:
        return [recipe for recipe in self.recipes 
                if restriction in recipe.get('dietary_tags', [])]
    
    def get_recipes_by_health_condition(self, condition: str) -> List[Dict[str, Any]]:
        return [recipe for recipe in self.recipes 
                if condition in recipe.get('health_benefits', [])]
    
    def get_recipes_by_ingredient(self, ingredient: str) -> List[Dict[str, Any]]:
        matching_recipes = []
        for recipe in self.recipes:
            recipe_ingredients = [ing['name'].lower() for ing in recipe.get('ingredients', [])]
            if ingredient.lower() in recipe_ingredients:
                matching_recipes.append(recipe)
        return matching_recipes
    
    def get_nutritional_info(self, ingredient: str) -> Optional[Dict[str, Any]]:
        return self.nutritional_data.get('ingredients', {}).get(ingredient)
    
    def get_dietary_restriction_info(self, restriction: str) -> Optional[Dict[str, Any]]:
        return self.dietary_guidelines.get('dietary_restrictions', {}).get(restriction)
    
    def get_health_condition_info(self, condition: str) -> Optional[Dict[str, Any]]:
        return self.dietary_guidelines.get('health_conditions', {}).get(condition)
    
    def get_allergy_info(self, allergy: str) -> Optional[Dict[str, Any]]:
        return self.dietary_guidelines.get('allergies', {}).get(allergy)
    
    def get_ingredient_substitutions(self, ingredient: str) -> Optional[Dict[str, Any]]:
        return self.dietary_guidelines.get('ingredient_substitutions', {}).get(ingredient)
    
    def create_recipe_chunks(self, chunk_size: int = 1000) -> List[str]:
        chunks = []
        for recipe in self.recipes:
            recipe_text = self.process_recipe_text(recipe)
            
            if len(recipe_text) > chunk_size:
                sections = recipe_text.split(" | ")
                current_chunk = ""
                
                for section in sections:
                    if len(current_chunk + section) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = section
                    else:
                        current_chunk += " | " + section if current_chunk else section
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(recipe_text)
        
        return chunks
    
    def get_recipe_metadata(self) -> List[Dict[str, Any]]:
        metadata = []
        for recipe in self.recipes:
            metadata.append({
                'id': recipe.get('id'),
                'title': recipe.get('title'),
                'cuisine_type': recipe.get('cuisine_type'),
                'dietary_tags': recipe.get('dietary_tags', []),
                'health_benefits': recipe.get('health_benefits', []),
                'calories': recipe.get('nutritional_info', {}).get('calories', 0),
                'protein': recipe.get('nutritional_info', {}).get('protein', 0),
                'carbohydrates': recipe.get('nutritional_info', {}).get('carbohydrates', 0),
                'fat': recipe.get('nutritional_info', {}).get('fat', 0),
                'fiber': recipe.get('nutritional_info', {}).get('fiber', 0)
            })
        return metadata 