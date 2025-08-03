from typing import Dict, List, Any, Optional, Set, Tuple
import re


class DietaryAnalyzer:
    
    def __init__(self, dietary_guidelines: Dict[str, Any]):
        self.dietary_guidelines = dietary_guidelines
        self.restrictions = dietary_guidelines.get('dietary_restrictions', {})
        self.health_conditions = dietary_guidelines.get('health_conditions', {})
        self.allergies = dietary_guidelines.get('allergies', {})
        
    def analyze_recipe_compatibility(self, recipe: Dict[str, Any],
                                   user_restrictions: List[str],
                                   user_allergies: List[str],
                                   user_health_conditions: List[str]) -> Dict[str, Any]:
        restriction_compatibility = self._check_dietary_restrictions(
            recipe, user_restrictions or []
        )
        
        allergy_compatibility = self._check_allergies(
            recipe, user_allergies or []
        )
        
        health_compatibility = self._check_health_conditions(
            recipe, user_health_conditions or []
        )
        
        overall_score = self._calculate_compatibility_score(
            restriction_compatibility, allergy_compatibility, health_compatibility
        )
        
        return {
            'overall_compatible': overall_score >= 0.7,
            'overall_score': overall_score,
            'restriction_compatibility': restriction_compatibility,
            'allergy_compatibility': allergy_compatibility,
            'health_compatibility': health_compatibility,
            'issues': self._identify_issues(
                restriction_compatibility, allergy_compatibility, health_compatibility
            ),
            'suggestions': self._generate_suggestions(
                recipe, restriction_compatibility, allergy_compatibility, health_compatibility
            )
        }
    
    def _check_dietary_restrictions(self, recipe: Dict[str, Any], 
                                   user_restrictions: List[str]) -> Dict[str, Any]:
        recipe_ingredients = [ing['name'].lower() for ing in recipe.get('ingredients', [])]
        recipe_tags = recipe.get('dietary_tags', [])
        
        compatibility_results = {}
        
        if not user_restrictions:
            return {
                'overall_compatible': True,
                'compatibility_score': 1.0,
                'restriction_results': {}
            }
        
        for restriction in user_restrictions:
            restriction_info = self.restrictions.get(restriction, {})
            excluded_ingredients = restriction_info.get('excluded_ingredients', [])
            
            has_restriction_tag = restriction in recipe_tags
            
            conflicting_ingredients = []
            for ingredient in recipe_ingredients:
                for excluded in excluded_ingredients:
                    if self._ingredient_matches(excluded.lower(), ingredient):
                        conflicting_ingredients.append(ingredient)
            
            is_compatible = has_restriction_tag and len(conflicting_ingredients) == 0
            compatibility_score = 1.0 if is_compatible else 0.0
            
            if conflicting_ingredients:
                compatibility_score = 0.0
            elif not has_restriction_tag:
                compatibility_score = 0.5
            
            compatibility_results[restriction] = {
                'compatible': is_compatible,
                'score': compatibility_score,
                'has_restriction_tag': has_restriction_tag,
                'conflicting_ingredients': conflicting_ingredients,
                'excluded_ingredients': excluded_ingredients
            }
        
        overall_restriction_score = sum(result['score'] for result in compatibility_results.values()) / len(compatibility_results) if compatibility_results else 1.0
        
        return {
            'overall_compatible': overall_restriction_score >= 0.7,
            'compatibility_score': overall_restriction_score,
            'restriction_results': compatibility_results
        }
    
    def _check_allergies(self, recipe: Dict[str, Any], 
                        user_allergies: List[str]) -> Dict[str, Any]:
        recipe_ingredients = [ing['name'].lower() for ing in recipe.get('ingredients', [])]
        
        allergy_results = {}
        
        if not user_allergies:
            return {
                'overall_compatible': True,
                'compatibility_score': 1.0,
                'allergy_results': {}
            }
        
        for allergy in user_allergies:
            allergy_info = self.allergies.get(allergy, {})
            incompatible_ingredients = allergy_info.get('incompatible_ingredients', [])
            
            conflicting_ingredients = []
            for ingredient in recipe_ingredients:
                for incompatible in incompatible_ingredients:
                    if self._ingredient_matches(incompatible.lower(), ingredient):
                        conflicting_ingredients.append(ingredient)
            
            is_compatible = len(conflicting_ingredients) == 0
            compatibility_score = 1.0 if is_compatible else 0.0
            
            allergy_results[allergy] = {
                'compatible': is_compatible,
                'score': compatibility_score,
                'conflicting_ingredients': conflicting_ingredients,
                'incompatible_ingredients': incompatible_ingredients
            }
        
        overall_allergy_score = sum(result['score'] for result in allergy_results.values()) / len(allergy_results) if allergy_results else 1.0
        
        return {
            'overall_compatible': overall_allergy_score >= 0.7,
            'compatibility_score': overall_allergy_score,
            'allergy_results': allergy_results
        }
    
    def _check_health_conditions(self, recipe: Dict[str, Any], 
                                user_health_conditions: List[str]) -> Dict[str, Any]:
        recipe_health_benefits = recipe.get('health_benefits', [])
        nutritional_info = recipe.get('nutritional_info', {})
        
        health_results = {}
        
        if not user_health_conditions:
            return {
                'overall_compatible': True,
                'compatibility_score': 1.0,
                'health_results': {}
            }
        
        for condition in user_health_conditions:
            condition_info = self.health_conditions.get(condition, {})
            recommended_benefits = condition_info.get('recommended_benefits', [])
            avoid_nutrients = condition_info.get('avoid_nutrients', [])
            recommended_nutrients = condition_info.get('recommended_nutrients', [])
            
            has_recommended_benefits = any(benefit in recipe_health_benefits for benefit in recommended_benefits)
            
            nutritional_score = self._analyze_nutritional_content(
                nutritional_info, recommended_nutrients, avoid_nutrients
            )
            
            overall_condition_score = (has_recommended_benefits * 0.6) + (nutritional_score * 0.4)
            
            health_results[condition] = {
                'compatible': overall_condition_score >= 0.6,
                'score': overall_condition_score,
                'has_recommended_benefits': has_recommended_benefits,
                'nutritional_score': nutritional_score,
                'recommended_benefits': recommended_benefits,
                'avoid_nutrients': avoid_nutrients,
                'recommended_nutrients': recommended_nutrients
            }
        
        overall_health_score = sum(result['score'] for result in health_results.values()) / len(health_results) if health_results else 1.0
        
        return {
            'overall_compatible': overall_health_score >= 0.6,
            'compatibility_score': overall_health_score,
            'health_results': health_results
        }
    
    def _analyze_nutritional_content(self, nutritional_info: Dict[str, Any],
                                   recommended_nutrients: List[str],
                                   avoid_nutrients: List[str]) -> float:
        score = 1.0
        
        for nutrient in avoid_nutrients:
            if nutrient in nutritional_info:
                value = nutritional_info[nutrient]
                if value > 0:
                    score -= 0.2
        
        for nutrient in recommended_nutrients:
            if nutrient in nutritional_info:
                value = nutritional_info[nutrient]
                if value > 0:
                    score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _ingredient_matches(self, pattern: str, ingredient: str) -> bool:
        if pattern == ingredient:
            return True
        
        if pattern in ingredient:
            return True
        
        if ingredient in pattern:
            return True
        
        pattern_words = pattern.split()
        ingredient_words = ingredient.split()
        
        for pattern_word in pattern_words:
            for ingredient_word in ingredient_words:
                if pattern_word in ingredient_word or ingredient_word in pattern_word:
                    return True
        
        return False
    
    def _calculate_compatibility_score(self, restriction_compatibility: Dict[str, Any],
                                   allergy_compatibility: Dict[str, Any],
                                   health_compatibility: Dict[str, Any]) -> float:
        restriction_score = restriction_compatibility.get('compatibility_score', 1.0)
        allergy_score = allergy_compatibility.get('compatibility_score', 1.0)
        health_score = health_compatibility.get('compatibility_score', 1.0)
        
        if allergy_score == 0.0:
            return 0.0
        
        if restriction_score == 0.0:
            return 0.0
        
        overall_score = (restriction_score * 0.4) + (allergy_score * 0.4) + (health_score * 0.2)
        
        return overall_score
    
    def _identify_issues(self, restriction_compatibility: Dict[str, Any],
                       allergy_compatibility: Dict[str, Any],
                       health_compatibility: Dict[str, Any]) -> List[str]:
        issues = []
        
        restriction_results = restriction_compatibility.get('restriction_results', {})
        for restriction, result in restriction_results.items():
            if not result['compatible']:
                if result['conflicting_ingredients']:
                    issues.append(f"Contains ingredients incompatible with {restriction}: {', '.join(result['conflicting_ingredients'])}")
                elif not result['has_restriction_tag']:
                    issues.append(f"Not tagged as {restriction}")
        
        allergy_results = allergy_compatibility.get('allergy_results', {})
        for allergy, result in allergy_results.items():
            if not result['compatible']:
                if result['conflicting_ingredients']:
                    issues.append(f"Contains ingredients that may cause {allergy} reaction: {', '.join(result['conflicting_ingredients'])}")
        
        health_results = health_compatibility.get('health_results', {})
        for condition, result in health_results.items():
            if not result['compatible']:
                if not result['has_recommended_benefits']:
                    issues.append(f"Not optimized for {condition}")
                if result['nutritional_score'] < 0.5:
                    issues.append(f"Nutritional content not ideal for {condition}")
        
        return issues
    
    def _generate_suggestions(self, recipe: Dict[str, Any],
                           restriction_compatibility: Dict[str, Any],
                           allergy_compatibility: Dict[str, Any],
                           health_compatibility: Dict[str, Any]) -> List[str]:
        suggestions = []
        
        restriction_results = restriction_compatibility.get('restriction_results', {})
        for restriction, result in restriction_results.items():
            if not result['compatible']:
                if result['conflicting_ingredients']:
                    suggestions.append(f"Consider substituting {', '.join(result['conflicting_ingredients'])} for {restriction}-friendly alternatives")
                elif not result['has_restriction_tag']:
                    suggestions.append(f"Recipe may be compatible with {restriction} but not explicitly tagged")
        
        allergy_results = allergy_compatibility.get('allergy_results', {})
        for allergy, result in allergy_results.items():
            if not result['compatible']:
                if result['conflicting_ingredients']:
                    suggestions.append(f"Substitute {', '.join(result['conflicting_ingredients'])} to avoid {allergy} triggers")
        
        health_results = health_compatibility.get('health_results', {})
        for condition, result in health_results.items():
            if not result['compatible']:
                if not result['has_recommended_benefits']:
                    suggestions.append(f"Consider adding ingredients beneficial for {condition}")
                if result['nutritional_score'] < 0.5:
                    suggestions.append(f"Adjust portion size or ingredients for better {condition} management")
        
        return suggestions
    
    def get_compatible_recipes(self, recipes: List[Dict[str, Any]],
                           user_restrictions: List[str],
                           user_allergies: List[str],
                           user_health_conditions: List[str],
                           min_score: float = 0.7) -> List[Dict[str, Any]]:
        compatible_recipes = []
        
        for recipe in recipes:
            compatibility = self.analyze_recipe_compatibility(
                recipe, user_restrictions, user_allergies, user_health_conditions
            )
            
            if compatibility['overall_score'] >= min_score:
                compatible_recipes.append({
                    'recipe': recipe,
                    'compatibility': compatibility
                })
        
        compatible_recipes.sort(key=lambda x: x['compatibility']['overall_score'], reverse=True)
        
        return compatible_recipes 