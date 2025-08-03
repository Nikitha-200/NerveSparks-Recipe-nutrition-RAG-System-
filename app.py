import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any
import time

from src.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="Recipe & Nutrition RAG System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .recipe-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .compatibility-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E8B57;
    }
    .warning {
        color: #FF6B6B;
        font-weight: bold;
    }
    .success {
        color: #4CAF50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_pipeline():
    return RAGPipeline()


def main():
    with st.spinner("Loading Recipe RAG System..."):
        rag_pipeline = initialize_rag_pipeline()
    
    st.markdown('<h1 class="main-header">🍽️ Recipe & Nutrition RAG System</h1>', unsafe_allow_html=True)
    st.markdown("### Personalized Meal Recommendations with Dietary Restrictions")
    
    with st.sidebar:
        st.markdown("## 👤 User Profile")
        
        st.subheader("Dietary Restrictions")
        dietary_options = ["vegetarian", "vegan", "gluten-free", "dairy-free", "keto", "low_sodium", "diabetes_friendly"]
        selected_dietary = st.multiselect(
            "Select your dietary restrictions:",
            dietary_options,
            help="Choose all that apply"
        )
        
        st.subheader("Allergies")
        allergy_options = ["peanut", "tree_nut", "shellfish", "egg", "soy", "wheat"]
        selected_allergies = st.multiselect(
            "Select your allergies:",
            allergy_options,
            help="Choose all that apply"
        )
        
        st.subheader("Health Conditions")
        health_options = ["diabetes", "heart_disease", "hypertension", "celiac_disease", "lactose_intolerance", "obesity"]
        selected_health = st.multiselect(
            "Select your health conditions:",
            health_options,
            help="Choose all that apply"
        )
        
        st.subheader("Nutritional Goals")
        calorie_goal = st.number_input("Daily Calorie Goal:", min_value=1000, max_value=5000, value=2000, step=100)
        protein_goal = st.number_input("Daily Protein Goal (g):", min_value=20, max_value=200, value=50, step=5)
        fiber_goal = st.number_input("Daily Fiber Goal (g):", min_value=10, max_value=100, value=25, step=5)
        
        st.subheader("Preferences")
        cuisine_preferences = st.multiselect(
            "Preferred Cuisines:",
            ["mediterranean", "asian", "indian", "american", "italian", "mexican"]
        )
        
        user_profile = {
            'dietary_restrictions': selected_dietary,
            'allergies': selected_allergies,
            'health_conditions': selected_health,
            'preferences': cuisine_preferences,
            'nutritional_goals': {
                'calories': calorie_goal,
                'protein': protein_goal,
                'fiber': fiber_goal
            }
        }
        
        st.markdown("---")
        st.markdown("### 📊 System Stats")
        stats = rag_pipeline.get_system_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Recipes", stats.get('total_recipes', 0))
            st.metric("Unique Ingredients", stats.get('unique_ingredients', 0))
            st.metric("Cuisine Types", stats.get('cuisine_types', 0))
        with col2:
            st.metric("Dietary Restrictions", stats.get('dietary_restrictions', 0))
            st.metric("Health Conditions", stats.get('health_conditions', 0))
            st.metric("Allergies", stats.get('allergies', 0))
        
        nutrition_stats = stats.get('nutrition_stats', {})
        if nutrition_stats.get('calories', {}).get('avg', 0) > 0:
            st.markdown("#### 🍽️ Nutrition Ranges")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Calories", f"{nutrition_stats.get('calories', {}).get('avg', 0):.0f}")
                st.metric("Avg Protein", f"{nutrition_stats.get('protein', {}).get('avg', 0):.1f}g")
            with col2:
                st.metric("Avg Carbs", f"{nutrition_stats.get('carbohydrates', {}).get('avg', 0):.1f}g")
                st.metric("Avg Fat", f"{nutrition_stats.get('fat', {}).get('avg', 0):.1f}g")
            with col3:
                st.metric("Avg Fiber", f"{nutrition_stats.get('fiber', {}).get('avg', 0):.1f}g")
        
        st.markdown("#### 🎯 Coverage Analysis")
        
        dietary_coverage = stats.get('dietary_coverage', {})
        if dietary_coverage:
            st.write("**Dietary Coverage:**")
            for restriction, coverage in dietary_coverage.items():
                if coverage.get('compatible_recipes', 0) > 0:
                    st.write(f"• {restriction}: {coverage.get('compatible_recipes', 0)}/{coverage.get('total_recipes', 0)} ({coverage.get('coverage_percentage', 0):.1f}%)")
        
        health_coverage = stats.get('health_coverage', {})
        if health_coverage:
            st.write("**Health Coverage:**")
            for condition, coverage in health_coverage.items():
                if coverage.get('compatible_recipes', 0) > 0:
                    st.write(f"• {condition}: {coverage.get('compatible_recipes', 0)}/{coverage.get('total_recipes', 0)} ({coverage.get('coverage_percentage', 0):.1f}%)")
        
        st.markdown("#### ⚙️ System Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vector Store Size", stats.get('vector_store_size', 0))
            st.metric("Embedding Dimension", stats.get('embedding_dimension', 0))
        with col2:
            st.metric("Vocabulary Size", stats.get('vocabulary_size', 0))
            st.metric("Data Source", stats.get('data_source', 'Unknown'))
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Recipe Search",
        "🎯 Personalized Recommendations", 
        "🔄 Ingredient Substitutions",
        "📊 Nutrition Analysis",
        "🌐 Dynamic Integration",
        "📈 System Analytics"
    ])

    with tab1:
        recipe_search_tab(rag_pipeline, user_profile)
    
    with tab2:
        personalized_recommendations_tab(rag_pipeline, user_profile)
    
    with tab3:
        ingredient_substitutions_tab(rag_pipeline, user_profile)
    
    with tab4:
        nutrition_analysis_tab(rag_pipeline, user_profile)
    
    with tab5:
        dynamic_integration_tab(rag_pipeline, user_profile)
    
    with tab6:
        system_analytics_tab(rag_pipeline)


def recipe_search_tab(rag_pipeline, user_profile):
    """Recipe search functionality."""
    st.markdown('<h2 class="sub-header">🔍 Recipe Search</h2>', unsafe_allow_html=True)
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for recipes:",
            placeholder="e.g., high protein vegetarian meals, gluten-free breakfast, etc."
        )
    
    with col2:
        n_results = st.selectbox("Number of results:", [3, 5, 10, 15])
    
    if st.button("🔍 Search Recipes", type="primary"):
        if search_query:
            with st.spinner("Searching for recipes..."):
                results = rag_pipeline.search_recipes(
                    query=search_query,
                    dietary_restrictions=user_profile['dietary_restrictions'],
                    allergies=user_profile['allergies'],
                    health_conditions=user_profile['health_conditions'],
                    n_results=n_results
                )
                
                display_search_results(results)
        else:
            st.warning("Please enter a search query.")


def personalized_recommendations_tab(rag_pipeline, user_profile):
    """Personalized recommendations functionality."""
    st.markdown('<h2 class="sub-header">🎯 Personalized Recommendations</h2>', unsafe_allow_html=True)
    
    if st.button("🎯 Get Personalized Recommendations", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            recommendations = rag_pipeline.get_personalized_recommendations(
                user_profile=user_profile,
                n_recommendations=5
            )
            
            display_recommendations(recommendations)


def ingredient_substitutions_tab(rag_pipeline, user_profile):
    """Ingredient substitutions functionality."""
    st.markdown('<h2 class="sub-header">🔄 Ingredient Substitutions</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ingredient = st.text_input(
            "Enter ingredient to substitute:",
            placeholder="e.g., milk, eggs, wheat flour, etc."
        )
    
    with col2:
        st.markdown("### Current Filters")
        st.write(f"**Dietary:** {', '.join(user_profile['dietary_restrictions']) if user_profile['dietary_restrictions'] else 'None'}")
        st.write(f"**Allergies:** {', '.join(user_profile['allergies']) if user_profile['allergies'] else 'None'}")
    
    if st.button("🔄 Find Substitutions", type="primary"):
        if ingredient:
            with st.spinner("Finding substitution options..."):
                substitutions = rag_pipeline.get_ingredient_substitutions(
                    ingredient=ingredient,
                    dietary_restrictions=user_profile['dietary_restrictions'],
                    allergies=user_profile['allergies']
                )
                
                display_substitutions(substitutions)
        else:
            st.warning("Please enter an ingredient to substitute.")


def nutrition_analysis_tab(rag_pipeline, user_profile):
    """Nutrition analysis functionality."""
    st.markdown('<h2 class="sub-header">📊 Nutrition Analysis</h2>', unsafe_allow_html=True)
    
    # Get user profile details
    dietary_restrictions = user_profile.get('dietary_restrictions', [])
    allergies = user_profile.get('allergies', [])
    health_conditions = user_profile.get('health_conditions', [])
    nutritional_goals = user_profile.get('nutritional_goals', {})
    
    # Show user profile summary
    st.markdown("### 👤 Your Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Dietary:** {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}")
    with col2:
        st.write(f"**Allergies:** {', '.join(allergies) if allergies else 'None'}")
    with col3:
        st.write(f"**Health:** {', '.join(health_conditions) if health_conditions else 'None'}")
    
    if st.button("📊 Analyze Nutrition", type="primary"):
        with st.spinner("Analyzing nutrition data for your profile..."):
            # Get all recipes
            all_recipes = rag_pipeline.data_processor.recipes
            
            # Filter recipes based on user profile
            compatible_recipes = []
            incompatible_recipes = []
            
            for recipe in all_recipes:
                # Analyze compatibility
                compatibility = rag_pipeline.dietary_analyzer.analyze_recipe_compatibility(
                    recipe, dietary_restrictions, allergies, health_conditions
                )
                
                if compatibility['overall_score'] >= 0.7:  # Good compatibility
                    compatible_recipes.append(recipe)
                else:
                    incompatible_recipes.append(recipe)
            
            # Create nutrition analysis for compatible recipes
            nutrition_data = []
            for recipe in compatible_recipes:
                nutrition = recipe.get('nutritional_info', {})
                nutrition_data.append({
                    'title': recipe['title'],
                    'calories': nutrition.get('calories', 0),
                    'protein': nutrition.get('protein', 0),
                    'carbohydrates': nutrition.get('carbohydrates', 0),
                    'fat': nutrition.get('fat', 0),
                    'fiber': nutrition.get('fiber', 0),
                    'sodium': nutrition.get('sodium', 0),
                    'cuisine_type': recipe.get('cuisine_type', 'Unknown'),
                    'dietary_tags': ', '.join(recipe.get('dietary_tags', [])),
                    'compatibility_score': rag_pipeline.dietary_analyzer.analyze_recipe_compatibility(
                        recipe, dietary_restrictions, allergies, health_conditions
                    )['overall_score']
                })
            
            df = pd.DataFrame(nutrition_data)
            
            # Show compatibility summary
            st.markdown("### 📊 Compatibility Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Compatible Recipes", len(compatible_recipes))
            with col2:
                st.metric("Incompatible Recipes", len(incompatible_recipes))
            with col3:
                compatibility_percentage = len(compatible_recipes) / len(all_recipes) * 100 if all_recipes else 0
                st.metric("Compatibility Rate", f"{compatibility_percentage:.1f}%")
            
            if len(compatible_recipes) > 0:
                # Display nutrition charts for compatible recipes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Calories Distribution (Compatible Recipes)")
                    fig_calories = px.histogram(df, x='calories', nbins=20, 
                                             title="Recipe Calories Distribution")
                    st.plotly_chart(fig_calories, use_container_width=True)
                
                with col2:
                    st.subheader("Protein vs Fat (Compatible Recipes)")
                    fig_protein_fat = px.scatter(df, x='protein', y='fat', 
                                               color='cuisine_type',
                                               title="Protein vs Fat Content")
                    st.plotly_chart(fig_protein_fat, use_container_width=True)
                
                # Personalized nutrition summary
                st.markdown("### 🎯 Personalized Nutrition Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                avg_calories = df['calories'].mean()
                avg_protein = df['protein'].mean()
                avg_carbs = df['carbohydrates'].mean()
                avg_fiber = df['fiber'].mean()
                
                with col1:
                    st.metric("Avg Calories", f"{avg_calories:.0f}")
                    if nutritional_goals.get('calories'):
                        goal_calories = nutritional_goals['calories']
                        st.metric("Your Goal", f"{goal_calories}")
                        if avg_calories <= goal_calories:
                            st.success("✅ Within goal range")
                        else:
                            st.warning("⚠️ Above goal range")
                
                with col2:
                    st.metric("Avg Protein", f"{avg_protein:.1f}g")
                    if nutritional_goals.get('protein'):
                        goal_protein = nutritional_goals['protein']
                        st.metric("Your Goal", f"{goal_protein}g")
                        if avg_protein >= goal_protein:
                            st.success("✅ Meets protein goal")
                        else:
                            st.warning("⚠️ Below protein goal")
                
                with col3:
                    st.metric("Avg Carbs", f"{avg_carbs:.1f}g")
                
                with col4:
                    st.metric("Avg Fiber", f"{avg_fiber:.1f}g")
                    if nutritional_goals.get('fiber'):
                        goal_fiber = nutritional_goals['fiber']
                        st.metric("Your Goal", f"{goal_fiber}g")
                        if avg_fiber >= goal_fiber:
                            st.success("✅ Meets fiber goal")
                        else:
                            st.warning("⚠️ Below fiber goal")
                
                # Health condition specific analysis
                if health_conditions:
                    st.markdown("### 🏥 Health-Focused Analysis")
                    
                    # Diabetes-friendly analysis
                    if 'diabetes' in health_conditions:
                        diabetes_recipes = df[df['dietary_tags'].str.contains('diabetes_friendly', case=False, na=False)]
                        if len(diabetes_recipes) > 0:
                            st.write(f"**Diabetes-Friendly Recipes:** {len(diabetes_recipes)} found")
                            avg_glycemic = diabetes_recipes['carbohydrates'].mean()
                            st.metric("Avg Carbs (Diabetes-friendly)", f"{avg_glycemic:.1f}g")
                    
                    # Heart-healthy analysis
                    if 'heart_disease' in health_conditions or 'hypertension' in health_conditions:
                        heart_recipes = df[df['dietary_tags'].str.contains('heart_healthy', case=False, na=False)]
                        if len(heart_recipes) > 0:
                            st.write(f"**Heart-Healthy Recipes:** {len(heart_recipes)} found")
                            avg_sodium = heart_recipes['sodium'].mean()
                            st.metric("Avg Sodium (Heart-healthy)", f"{avg_sodium:.1f}mg")
                
                # Show top compatible recipes
                st.markdown("### 🏆 Top Compatible Recipes")
                top_recipes = df.nlargest(5, 'compatibility_score')[['title', 'calories', 'protein', 'compatibility_score']]
                st.dataframe(top_recipes, use_container_width=True)
                
            else:
                st.warning("❌ No compatible recipes found for your dietary profile.")
                st.info("💡 Try adjusting your dietary restrictions or allergies to see more options.")


def dynamic_integration_tab(rag_pipeline, user_profile):
    """Dynamic recipe integration functionality."""
    st.markdown('<h2 class="sub-header">🌐 Dynamic Recipe Integration</h2>', unsafe_allow_html=True)
    
    st.markdown("### 🚀 Unlimited Recipe Generation")
    st.write("This system can generate **unlimited recipes** dynamically based on your preferences and dietary needs!")
    
    # Show integration capabilities
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📡 Available Sources")
        sources = rag_pipeline.recipe_integrator.get_available_sources()
        for source in sources:
            st.write(f"✅ **{source.replace('_', ' ').title()}**")
        
        st.markdown("#### 🔧 Integration Capabilities")
        capabilities = [
            "Spoonacular API (requires key)",
            "Edamam API (requires key)", 
            "Mock Dynamic Generation (active)",
            "Web Scraping (configurable)",
            "Database Integration (configurable)"
        ]
        for capability in capabilities:
            st.write(f"🔗 {capability}")
    
    with col2:
        st.markdown("#### 📊 Source Statistics")
        source_stats = rag_pipeline.recipe_integrator.get_source_stats()
        st.metric("Total Sources", source_stats['total_sources'])
        st.metric("Enabled Sources", source_stats['enabled_sources'])
        st.metric("Rate Limit/Hour", source_stats['total_rate_limit'])
    
    # Dynamic recipe generation
    st.markdown("### 🎯 Generate Dynamic Recipes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dynamic_query = st.text_input(
            "What type of recipes would you like to generate?",
            placeholder="e.g., high protein breakfast, vegan dinner, gluten-free snacks"
        )
    
    with col2:
        num_recipes = st.selectbox("Number of recipes:", [5, 10, 15, 20])
        include_dynamic = st.checkbox("Include dynamic recipes", value=True)
    
    if st.button("🚀 Generate Dynamic Recipes", type="primary"):
        if dynamic_query:
            with st.spinner("Generating dynamic recipes..."):
                # Get dynamic recipes
                dynamic_recipes = rag_pipeline.recipe_integrator.fetch_recipes_from_api(
                    source_name='mock_dynamic',
                    query=dynamic_query,
                    dietary_restrictions=user_profile.get('dietary_restrictions', []),
                    max_recipes=num_recipes
                )
                
                if dynamic_recipes:
                    st.success(f"✅ Generated {len(dynamic_recipes)} dynamic recipes!")
                    
                    # Display dynamic recipes
                    for i, recipe in enumerate(dynamic_recipes, 1):
                        with st.expander(f"{i}. {recipe['title']} (Dynamic)"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Description:** {recipe['description']}")
                                st.markdown(f"**Cuisine:** {recipe['cuisine_type']}")
                                st.markdown(f"**Dietary Tags:** {', '.join(recipe['dietary_tags'])}")
                                st.markdown(f"**Health Benefits:** {', '.join(recipe['health_benefits'])}")
                                
                                # Ingredients
                                st.markdown("**Ingredients:**")
                                for ingredient in recipe['ingredients']:
                                    st.write(f"• {ingredient['amount']} {ingredient['unit']} {ingredient['name']}")
                            
                            with col2:
                                # Nutritional info
                                nutrition = recipe['nutritional_info']
                                st.metric("Calories", f"{nutrition['calories']}")
                                st.metric("Protein", f"{nutrition['protein']}g")
                                st.metric("Carbs", f"{nutrition['carbohydrates']}g")
                                st.metric("Fat", f"{nutrition['fat']}g")
                                
                                st.markdown("**📊 Recipe Info:**")
                                st.write(f"Prep Time: {recipe['prep_time']} min")
                                st.write(f"Cook Time: {recipe['cook_time']} min")
                                st.write(f"Servings: {recipe['servings']}")
                                st.write(f"Difficulty: {recipe['difficulty']}")
                                st.write(f"Source: {recipe['source']}")
                else:
                    st.warning("No dynamic recipes generated. Try a different query.")
        else:
            st.warning("Please enter a query to generate dynamic recipes.")
    
    # Show comparison
    st.markdown("### 📊 Static vs Dynamic Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Static Recipes (JSON)")
        st.write("• Fixed 8 recipes in database")
        st.write("• Pre-defined nutritional data")
        st.write("• Limited variety")
        st.write("• No real-time updates")
    
    with col2:
        st.markdown("#### 🌐 Dynamic Recipes (Generated)")
        st.write("• **Unlimited** recipe generation")
        st.write("• Real-time nutritional calculation")
        st.write("• Infinite variety based on query")
        st.write("• Always fresh and relevant")
    
    # Integration status
    st.markdown("### 🔗 Integration Status")
    
    integration_stats = rag_pipeline.get_system_stats().get('dynamic_integration', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dynamic Integration", "✅ Enabled" if integration_stats.get('enabled') else "❌ Disabled")
        st.metric("Available Sources", len(integration_stats.get('available_sources', [])))
    
    with col2:
        st.metric("Dynamic Recipes", integration_stats.get('dynamic_recipes_generated', 'N/A'))
        st.metric("Integration Capabilities", len(integration_stats.get('integration_capabilities', [])))
    
    with col3:
        st.metric("Data Source", "JSON + Dynamic")
        st.metric("Last Updated", "Real-time")


def system_analytics_tab(rag_pipeline):
    """System analytics functionality."""
    st.markdown('<h2 class="sub-header">📈 System Analytics</h2>', unsafe_allow_html=True)
    
    stats = rag_pipeline.get_system_stats()
    
    # System overview
    st.markdown("### 🏗️ System Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Recipes", stats.get('total_recipes', 0))
        st.metric("Vector Store Size", stats.get('vector_store', {}).get('total_recipes', 0))
        st.metric("Unique Ingredients", stats.get('unique_ingredients', 0))
    
    with col2:
        st.metric("Dietary Restrictions", stats.get('dietary_restrictions', 0))
        st.metric("Health Conditions", stats.get('health_conditions', 0))
        st.metric("Allergies", stats.get('allergies', 0))
    
    with col3:
        st.metric("Cuisine Types", stats.get('cuisine_types', 0))
        st.metric("Dietary Tags", stats.get('dietary_tags_available', 0))
        st.metric("Health Benefits", stats.get('health_benefits_available', 0))
    
    # Nutrition Analysis
    st.markdown("### 🍽️ Nutrition Analysis")
    nutrition_stats = stats.get('nutrition_stats', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Calories Distribution")
        calories_data = {
            'Min': nutrition_stats.get('calories', {}).get('min', 0),
            'Average': nutrition_stats.get('calories', {}).get('avg', 0),
            'Max': nutrition_stats.get('calories', {}).get('max', 0)
        }
        fig_calories = px.bar(x=list(calories_data.keys()), y=list(calories_data.values()),
                             title="Calories Range", color=list(calories_data.keys()))
        st.plotly_chart(fig_calories, use_container_width=True)
    
    with col2:
        st.subheader("Protein Distribution")
        protein_data = {
            'Min': nutrition_stats.get('protein', {}).get('min', 0),
            'Average': nutrition_stats.get('protein', {}).get('avg', 0),
            'Max': nutrition_stats.get('protein', {}).get('max', 0)
        }
        fig_protein = px.bar(x=list(protein_data.keys()), y=list(protein_data.values()),
                            title="Protein Range (g)", color=list(protein_data.keys()))
        st.plotly_chart(fig_protein, use_container_width=True)
    
    # Detailed nutrition table
    st.subheader("Nutrition Ranges")
    nutrition_df = pd.DataFrame({
        'Nutrient': ['Calories', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)'],
        'Min': [
            nutrition_stats.get('calories', {}).get('min', 0),
            nutrition_stats.get('protein', {}).get('min', 0),
            nutrition_stats.get('carbohydrates', {}).get('min', 0),
            nutrition_stats.get('fat', {}).get('min', 0),
            nutrition_stats.get('fiber', {}).get('min', 0)
        ],
        'Average': [
            nutrition_stats.get('calories', {}).get('avg', 0),
            nutrition_stats.get('protein', {}).get('avg', 0),
            nutrition_stats.get('carbohydrates', {}).get('avg', 0),
            nutrition_stats.get('fat', {}).get('avg', 0),
            nutrition_stats.get('fiber', {}).get('avg', 0)
        ],
        'Max': [
            nutrition_stats.get('calories', {}).get('max', 0),
            nutrition_stats.get('protein', {}).get('max', 0),
            nutrition_stats.get('carbohydrates', {}).get('max', 0),
            nutrition_stats.get('fat', {}).get('max', 0),
            nutrition_stats.get('fiber', {}).get('max', 0)
        ]
    })
    st.dataframe(nutrition_df, use_container_width=True)
    
    # Coverage Analysis
    st.markdown("### 🎯 Coverage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dietary Restrictions Coverage")
        dietary_coverage = stats.get('dietary_coverage', {})
        if dietary_coverage:
            dietary_data = []
            for restriction, coverage in dietary_coverage.items():
                if coverage.get('compatible_recipes', 0) > 0:
                    dietary_data.append({
                        'Restriction': restriction.replace('_', ' ').title(),
                        'Compatible': coverage.get('compatible_recipes', 0),
                        'Total': coverage.get('total_recipes', 0),
                        'Percentage': coverage.get('coverage_percentage', 0)
                    })
            
            if dietary_data:
                dietary_df = pd.DataFrame(dietary_data)
                fig_dietary = px.bar(dietary_df, x='Restriction', y='Percentage',
                                   title="Dietary Restrictions Coverage (%)")
                st.plotly_chart(fig_dietary, use_container_width=True)
    
    with col2:
        st.subheader("Health Conditions Coverage")
        if stats['health_coverage']:
            health_data = []
            for condition, coverage in stats['health_coverage'].items():
                if coverage['compatible_recipes'] > 0:
                    health_data.append({
                        'Condition': condition.replace('_', ' ').title(),
                        'Compatible': coverage['compatible_recipes'],
                        'Total': coverage['total_recipes'],
                        'Percentage': coverage['coverage_percentage']
                    })
            
            if health_data:
                health_df = pd.DataFrame(health_data)
                fig_health = px.bar(health_df, x='Condition', y='Percentage',
                                  title="Health Conditions Coverage (%)")
                st.plotly_chart(fig_health, use_container_width=True)
    
    # System Performance
    st.markdown("### ⚙️ System Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Embedding Dimension", stats['embedding_dimension'])
        st.metric("Vocabulary Size", stats['vocabulary_size'])
    
    with col2:
        st.metric("Vector Store Size", stats['vector_store_size'])
        st.metric("Data Source", stats['data_source'])
    
    with col3:
        st.metric("Last Updated", stats['last_updated'])
        st.metric("Model Type", stats['embedding_model']['model_name'])
    
    # Detailed system info
    st.markdown("### 📋 Detailed System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.json(stats['vector_store'])
    
    with col2:
        st.json(stats['embedding_model'])


def display_search_results(results):
    """Display search results."""
    st.subheader(f"Search Results for: '{results['query']}'")
    st.write(f"Found {results['total_found']} recipes")
    
    for i, result in enumerate(results['results'], 1):
        recipe = result['recipe']
        compatibility = result['compatibility']
        
        with st.expander(f"{i}. {recipe['title']} (Score: {result['overall_score']:.2f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {recipe['description']}")
                st.markdown(f"**Cuisine:** {recipe['cuisine_type']}")
                st.markdown(f"**Dietary Tags:** {', '.join(recipe['dietary_tags'])}")
                st.markdown(f"**Health Benefits:** {', '.join(recipe['health_benefits'])}")
                
                # Ingredients
                st.markdown("**Ingredients:**")
                for ingredient in recipe['ingredients']:
                    st.write(f"• {ingredient['amount']} {ingredient['unit']} {ingredient['name']}")
            
            with col2:
                # Compatibility score
                score = compatibility['overall_score']
                if score >= 0.8:
                    st.markdown(f'<p class="compatibility-score success">✅ Compatible ({score:.1%})</p>', unsafe_allow_html=True)
                elif score >= 0.6:
                    st.markdown(f'<p class="compatibility-score">⚠️ Partially Compatible ({score:.1%})</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="compatibility-score warning">❌ Not Compatible ({score:.1%})</p>', unsafe_allow_html=True)
                
                # Nutritional info
                nutrition = recipe['nutritional_info']
                st.metric("Calories", f"{nutrition['calories']}")
                st.metric("Protein", f"{nutrition['protein']}g")
                st.metric("Carbs", f"{nutrition['carbohydrates']}g")
                st.metric("Fat", f"{nutrition['fat']}g")
            
            # Issues and suggestions
            if compatibility['issues']:
                st.markdown("**⚠️ Issues:**")
                for issue in compatibility['issues']:
                    st.write(f"• {issue}")
            
            if compatibility['suggestions']:
                st.markdown("**💡 Suggestions:**")
                for suggestion in compatibility['suggestions']:
                    st.write(f"• {suggestion}")


def display_recommendations(recommendations):
    """Display personalized recommendations."""
    st.subheader("🎯 Personalized Recommendations")
    st.write(f"Based on your profile and preferences")
    
    for i, rec in enumerate(recommendations['recommendations'], 1):
        recipe = rec['recipe']
        compatibility = rec['compatibility']
        
        with st.expander(f"{i}. {recipe['title']} (Score: {rec['overall_score']:.2f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {recipe['description']}")
                st.markdown(f"**Cuisine:** {recipe['cuisine_type']}")
                
                # Instructions
                st.markdown("**Instructions:**")
                for j, instruction in enumerate(recipe['instructions'], 1):
                    st.write(f"{j}. {instruction}")
            
            with col2:
                # Compatibility
                score = compatibility['overall_score']
                if score >= 0.8:
                    st.markdown(f'<p class="compatibility-score success">✅ Perfect Match ({score:.1%})</p>', unsafe_allow_html=True)
                elif score >= 0.6:
                    st.markdown(f'<p class="compatibility-score">✅ Good Match ({score:.1%})</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="compatibility-score">⚠️ Partial Match ({score:.1%})</p>', unsafe_allow_html=True)
                
                # Nutrition optimization if available
                if 'nutrition_optimization' in rec:
                    optimization = rec['nutrition_optimization']
                    st.markdown("**📊 Nutrition Optimization:**")
                    if 'overall_score' in optimization:
                        st.write(f"Current Score: {optimization['overall_score']:.1%}")
                    
                    if 'optimization_suggestions' in optimization and optimization['optimization_suggestions']:
                        st.markdown("**💡 Suggestions:**")
                        for suggestion in optimization['optimization_suggestions'][:3]:
                            suggestion_text = suggestion.get('type', '')
                            if 'ingredient' in suggestion:
                                suggestion_text += f": {suggestion['ingredient']}"
                            elif 'substitute' in suggestion:
                                suggestion_text += f": {suggestion['substitute']}"
                            st.write(f"• {suggestion_text}")


def display_substitutions(substitutions):
    """Display ingredient substitutions."""
    st.subheader(f"🔄 Substitutions for: {substitutions['original_ingredient']}")
    st.write(f"Found {substitutions['total_options']} substitution options")
    
    if not substitutions['substitutions']:
        st.warning("No suitable substitutions found for this ingredient.")
        return
    
    for i, sub in enumerate(substitutions['substitutions'], 1):
        with st.expander(f"{i}. {sub['substitute_name']} (Score: {sub['compatibility_score']:.2f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Substitute:** {sub['substitute_name']}")
                st.markdown(f"**Ratio:** {sub['ratio']}")
                st.markdown(f"**Nutritional Difference:** {sub['nutritional_difference']}")
                
                if sub['health_benefits']:
                    st.markdown("**Health Benefits:**")
                    for benefit in sub['health_benefits']:
                        st.write(f"• {benefit}")
            
            with col2:
                # Compatibility score
                score = sub['compatibility_score']
                if score >= 0.8:
                    st.markdown(f'<p class="compatibility-score success">✅ Excellent ({score:.1%})</p>', unsafe_allow_html=True)
                elif score >= 0.6:
                    st.markdown(f'<p class="compatibility-score">✅ Good ({score:.1%})</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="compatibility-score">⚠️ Fair ({score:.1%})</p>', unsafe_allow_html=True)
                
                # Nutritional info if available
                if 'nutritional_info' in sub and sub['nutritional_info']:
                    nutrition = sub['nutritional_info']
                    st.metric("Calories/100g", f"{nutrition.get('calories_per_100g', 0)}")
                    st.metric("Protein", f"{nutrition.get('protein', 0)}g")
                    st.metric("Glycemic Index", f"{nutrition.get('glycemic_index', 0)}")


if __name__ == "__main__":
    main() 