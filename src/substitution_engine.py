from typing import Dict, List, Any, Optional, Tuple
import re


class SubstitutionEngine:
    
    def __init__(self, dietary_guidelines: Dict[str, Any], nutritional_data: Dict[str, Any]):
        self.dietary_guidelines = dietary_guidelines
        self.nutritional_data = nutritional_data
        self.substitutions = dietary_guidelines.get('ingredient_substitutions', {})
        self.ingredients = nutritional_data.get('ingredients', {})
        
    def find_substitutions(self, ingredient: str, 
                          dietary_restrictions: List[str] = None,
                          allergies: List[str] = None) -> List[Dict[str, Any]]:
        normalized_ingredient = self._normalize_ingredient_name(ingredient)
        
        direct_substitutions = self.substitutions.get(normalized_ingredient, {})
        substitution_options = []
        
        if direct_substitutions:
            for substitute in direct_substitutions.get('substitutes', []):
                option = self._create_substitution_option(
                    ingredient, substitute, dietary_restrictions, allergies
                )
                if option:
                    substitution_options.append(option)
        
        ingredient_based_substitutions = self._find_ingredient_based_substitutions(
            normalized_ingredient, dietary_restrictions, allergies
        )
        substitution_options.extend(ingredient_based_substitutions)
        
        substitution_options.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return substitution_options
    
    def _normalize_ingredient_name(self, ingredient: str) -> str:
        modifiers = ['fresh', 'dried', 'frozen', 'canned', 'organic', 'raw', 'cooked']
        normalized = ingredient.lower().strip()
        
        for modifier in modifiers:
            normalized = normalized.replace(modifier, '').strip()
        
        units = ['cup', 'tbsp', 'tsp', 'oz', 'lb', 'g', 'kg', 'ml', 'l']
        for unit in units:
            normalized = re.sub(rf'\d+\s*{unit}', '', normalized).strip()
        
        return normalized
    
    def _create_substitution_option(self, original_ingredient: str,
                                  substitute_info: Dict[str, Any],
                                  dietary_restrictions: List[str] = None,
                                  allergies: List[str] = None) -> Optional[Dict[str, Any]]:
        substitute_name = substitute_info.get('name', '')
        ratio = substitute_info.get('ratio', '1:1')
        notes = substitute_info.get('notes', '')
        
        compatibility_score = self._calculate_substitution_compatibility(
            substitute_name, dietary_restrictions, allergies
        )
        
        nutritional_similarity = self._calculate_nutritional_similarity(
            self.ingredients.get(original_ingredient, {}),
            self.ingredients.get(substitute_name, {})
        )
        
        return {
            'original_ingredient': original_ingredient,
            'substitute_name': substitute_name,
            'ratio': ratio,
            'notes': notes,
            'compatibility_score': compatibility_score,
            'nutritional_similarity': nutritional_similarity,
            'overall_score': (compatibility_score * 0.7) + (nutritional_similarity * 0.3)
        }
    
    def _find_ingredient_based_substitutions(self, ingredient: str,
                                          dietary_restrictions: List[str] = None,
                                          allergies: List[str] = None) -> List[Dict[str, Any]]:
        substitution_options = []
        
        for substitute_name, substitute_data in self.ingredients.items():
            if substitute_name != ingredient:
                compatibility_score = self._calculate_substitution_compatibility(
                    substitute_name, dietary_restrictions, allergies
                )
                
                if compatibility_score > 0.5:
                    nutritional_similarity = self._calculate_nutritional_similarity(
                        self.ingredients.get(ingredient, {}),
                        substitute_data
                    )
                    
                    if nutritional_similarity > 0.3:
                        substitution_options.append({
                            'original_ingredient': ingredient,
                            'substitute_name': substitute_name,
                            'ratio': '1:1',
                            'notes': 'Nutritionally similar alternative',
                            'compatibility_score': compatibility_score,
                            'nutritional_similarity': nutritional_similarity,
                            'overall_score': (compatibility_score * 0.7) + (nutritional_similarity * 0.3)
                        })
        
        return substitution_options
    
    def _calculate_substitution_compatibility(self, substitute: str,
                                          dietary_restrictions: List[str] = None,
                                          allergies: List[str] = None) -> float:
        compatibility_score = 1.0
        
        if not dietary_restrictions and not allergies:
            return compatibility_score
        
        substitute_info = self.ingredients.get(substitute, {})
        substitute_tags = substitute_info.get('dietary_tags', [])
        
        for restriction in dietary_restrictions or []:
            restriction_info = self.dietary_guidelines.get('dietary_restrictions', {}).get(restriction, {})
            excluded_ingredients = restriction_info.get('excluded_ingredients', [])
            
            if substitute.lower() in [ing.lower() for ing in excluded_ingredients]:
                compatibility_score -= 0.5
            elif restriction in substitute_tags:
                compatibility_score += 0.2
        
        for allergy in allergies or []:
            allergy_info = self.dietary_guidelines.get('allergies', {}).get(allergy, {})
            incompatible_ingredients = allergy_info.get('incompatible_ingredients', [])
            
            if substitute.lower() in [ing.lower() for ing in incompatible_ingredients]:
                compatibility_score = 0.0
                break
        
        return max(0.0, min(1.0, compatibility_score))
    
    def _calculate_nutritional_similarity(self, nutrition1: Dict[str, Any],
                                       nutrition2: Dict[str, Any]) -> float:
        if not nutrition1 or not nutrition2:
            return 0.0
        
        nutrients = ['calories', 'protein', 'carbohydrates', 'fat', 'fiber']
        similarities = []
        
        for nutrient in nutrients:
            val1 = nutrition1.get(nutrient, 0)
            val2 = nutrition2.get(nutrient, 0)
            
            if val1 > 0 and val2 > 0:
                similarity = 1 - abs(val1 - val2) / max(val1, val2)
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def optimize_recipe_nutrition(self, recipe: Dict[str, Any],
                               target_nutrients: Dict[str, float],
                               dietary_restrictions: List[str] = None,
                               allergies: List[str] = None) -> Dict[str, Any]:
        current_nutrition = recipe.get('nutritional_info', {})
        
        optimization_suggestions = []
        nutrient_analysis = {}
        
        for nutrient, target_value in target_nutrients.items():
            current_value = current_nutrition.get(nutrient, 0)
            difference = target_value - current_value
            
            if abs(difference) > 0.1:
                if difference > 0:
                    suggestions = self._find_nutrient_boost_suggestions(
                        recipe, nutrient, difference, dietary_restrictions, allergies
                    )
                else:
                    suggestions = self._find_nutrient_reduction_suggestions(
                        recipe, nutrient, abs(difference), dietary_restrictions, allergies
                    )
                
                if suggestions:
                    optimization_suggestions.extend(suggestions)
                
                nutrient_analysis[nutrient] = {
                    'current': current_value,
                    'target': target_value,
                    'difference': difference,
                    'suggestions_count': len(suggestions)
                }
        
        optimization_score = self._calculate_optimization_score(current_nutrition, target_nutrients)
        
        return {
            'optimization_score': optimization_score,
            'suggestions': optimization_suggestions,
            'nutrient_analysis': nutrient_analysis,
            'current_nutrition': current_nutrition,
            'target_nutrition': target_nutrients
        }
    
    def _find_nutrient_boost_suggestions(self, recipe: Dict[str, Any],
                                      nutrient: str,
                                      deficit: float,
                                      dietary_restrictions: List[str] = None,
                                      allergies: List[str] = None) -> List[Dict[str, Any]]:
        suggestions = []
        
        for ingredient_name, ingredient_data in self.ingredients.items():
            ingredient_nutrition = ingredient_data.get('nutrition', {})
            nutrient_value = ingredient_nutrition.get(nutrient, 0)
            
            if nutrient_value > 0:
                compatibility_score = self._calculate_substitution_compatibility(
                    ingredient_name, dietary_restrictions, allergies
                )
                
                if compatibility_score > 0.7:
                    suggestions.append({
                        'type': 'add_ingredient',
                        'ingredient': ingredient_name,
                        'nutrient': nutrient,
                        'nutrient_value': nutrient_value,
                        'compatibility_score': compatibility_score,
                        'suggestion': f"Add {ingredient_name} to boost {nutrient}"
                    })
        
        suggestions.sort(key=lambda x: x['nutrient_value'], reverse=True)
        return suggestions[:3]
    
    def _find_nutrient_reduction_suggestions(self, recipe: Dict[str, Any],
                                          nutrient: str,
                                          excess: float,
                                          dietary_restrictions: List[str] = None,
                                          allergies: List[str] = None) -> List[Dict[str, Any]]:
        suggestions = []
        
        recipe_ingredients = recipe.get('ingredients', [])
        
        for ingredient in recipe_ingredients:
            ingredient_name = ingredient['name']
            ingredient_data = self.ingredients.get(ingredient_name, {})
            ingredient_nutrition = ingredient_data.get('nutrition', {})
            nutrient_value = ingredient_nutrition.get(nutrient, 0)
            
            if nutrient_value > 0:
                substitutions = self.find_substitutions(
                    ingredient_name, dietary_restrictions, allergies
                )
                
                for substitution in substitutions[:2]:
                    substitute_nutrition = self.ingredients.get(substitution['substitute_name'], {}).get('nutrition', {})
                    substitute_nutrient_value = substitute_nutrition.get(nutrient, 0)
                    
                    if substitute_nutrient_value < nutrient_value:
                        suggestions.append({
                            'type': 'substitute_ingredient',
                            'original_ingredient': ingredient_name,
                            'substitute_ingredient': substitution['substitute_name'],
                            'nutrient': nutrient,
                            'reduction': nutrient_value - substitute_nutrient_value,
                            'compatibility_score': substitution['compatibility_score'],
                            'suggestion': f"Substitute {ingredient_name} with {substitution['substitute_name']} to reduce {nutrient}"
                        })
        
        suggestions.sort(key=lambda x: x['reduction'], reverse=True)
        return suggestions[:3]
    
    def _calculate_optimization_score(self, current_nutrition: Dict[str, Any],
                                   target_nutrition: Dict[str, float]) -> float:
        if not target_nutrition:
            return 1.0
        
        scores = []
        
        for nutrient, target_value in target_nutrition.items():
            current_value = current_nutrition.get(nutrient, 0)
            
            if target_value > 0:
                if current_value >= target_value * 0.8:
                    scores.append(1.0)
                elif current_value >= target_value * 0.5:
                    scores.append(0.7)
                else:
                    scores.append(0.3)
        
        return sum(scores) / len(scores) if scores else 1.0 