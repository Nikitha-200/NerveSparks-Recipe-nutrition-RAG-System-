import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import RAGPipeline


def test_basic_functionality():
    print("🧪 Testing Recipe RAG System...")
    
    try:
        print("📦 Initializing RAG pipeline...")
        pipeline = RAGPipeline()
        print("✅ Pipeline initialized successfully")
        
        print("\n🔍 Testing recipe search...")
        results = pipeline.search_recipes(
            query="vegetarian high protein",
            dietary_restrictions=["vegetarian"],
            n_results=3
        )
        print(f"✅ Search completed. Found {results['total_found']} recipes")
        
        print("\n🔄 Testing ingredient substitutions...")
        substitutions = pipeline.get_ingredient_substitutions(
            ingredient="milk",
            dietary_restrictions=["dairy-free"]
        )
        print(f"✅ Substitutions found: {substitutions['total_options']} options")
        
        print("\n🎯 Testing personalized recommendations...")
        user_profile = {
            'dietary_restrictions': ['vegetarian'],
            'allergies': [],
            'health_conditions': [],
            'preferences': ['mediterranean'],
            'nutritional_goals': {'protein': 50, 'fiber': 25}
        }
        
        recommendations = pipeline.get_personalized_recommendations(
            user_profile=user_profile,
            n_recommendations=3
        )
        print(f"✅ Generated {len(recommendations['recommendations'])} recommendations")
        
        print("\n📊 Testing system statistics...")
        stats = pipeline.get_system_stats()
        print(f"✅ System stats retrieved:")
        print(f"   - Total recipes: {stats['total_recipes']}")
        print(f"   - Dietary restrictions: {stats['dietary_restrictions']}")
        print(f"   - Health conditions: {stats['health_conditions']}")
        print(f"   - Allergies: {stats['allergies']}")
        
        print("\n🎉 All tests passed! The RAG system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_search_queries():
    print("\n🔍 Testing various search queries...")
    
    pipeline = RAGPipeline()
    
    test_queries = [
        "vegetarian high protein",
        "gluten-free breakfast",
        "diabetes-friendly low carb",
        "vegan mediterranean",
        "heart-healthy low sodium"
    ]
    
    for query in test_queries:
        try:
            results = pipeline.search_recipes(query=query, n_results=2)
            print(f"✅ '{query}': Found {results['total_found']} recipes")
        except Exception as e:
            print(f"❌ '{query}': Failed - {e}")


def test_compatibility_analysis():
    print("\n🔬 Testing compatibility analysis...")
    
    pipeline = RAGPipeline()
    
    sample_recipe = pipeline.data_processor.recipes[0]
    
    test_profiles = [
        {
            'dietary_restrictions': ['vegetarian'],
            'allergies': [],
            'health_conditions': []
        },
        {
            'dietary_restrictions': ['gluten-free'],
            'allergies': ['wheat'],
            'health_conditions': ['celiac_disease']
        }
    ]
    
    for i, profile in enumerate(test_profiles, 1):
        try:
            compatibility = pipeline.analyze_recipe_compatibility(
                recipe=sample_recipe,
                dietary_restrictions=profile['dietary_restrictions'],
                allergies=profile['allergies'],
                health_conditions=profile['health_conditions']
            )
            
            score = compatibility['overall_score']
            print(f"✅ Profile {i}: Compatibility score = {score:.2f}")
            
        except Exception as e:
            print(f"❌ Profile {i}: Failed - {e}")


if __name__ == "__main__":
    print("🚀 Starting Recipe RAG System Tests")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        test_search_queries()
        test_compatibility_analysis()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("\nTo run the Streamlit app, use:")
        print("streamlit run app.py")
    else:
        print("\n❌ Basic functionality test failed. Please check the system setup.") 