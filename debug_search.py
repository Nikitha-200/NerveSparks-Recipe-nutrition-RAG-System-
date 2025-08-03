#!/usr/bin/env python3
"""
Debug script to test search functionality with different filter combinations.
"""

from src.rag_pipeline import RAGPipeline

def debug_search():
    print("🔍 Debugging Search Functionality")
    print("=" * 50)
    
    # Initialize pipeline
    rag_pipeline = RAGPipeline()
    
    # Test query
    query = "low carb vegetarian dinner"
    
    print(f"🔍 Testing query: '{query}'")
    print()
    
    # Test 1: No filters
    print("📋 Test 1: No filters")
    results1 = rag_pipeline.search_recipes(
        query=query,
        dietary_restrictions=[],
        allergies=[],
        health_conditions=[],
        n_results=5
    )
    print(f"   Results: {results1['total_found']} recipes")
    for i, result in enumerate(results1['results'][:3]):
        print(f"   {i+1}. {result['recipe']['title']}")
    print()
    
    # Test 2: Only vegetarian filter
    print("📋 Test 2: Vegetarian filter only")
    results2 = rag_pipeline.search_recipes(
        query=query,
        dietary_restrictions=['vegetarian'],
        allergies=[],
        health_conditions=[],
        n_results=5
    )
    print(f"   Results: {results2['total_found']} recipes")
    for i, result in enumerate(results2['results'][:3]):
        print(f"   {i+1}. {result['recipe']['title']}")
    print()
    
    # Test 3: Vegetarian + peanut allergy
    print("📋 Test 3: Vegetarian + peanut allergy")
    results3 = rag_pipeline.search_recipes(
        query=query,
        dietary_restrictions=['vegetarian'],
        allergies=['peanut'],
        health_conditions=[],
        n_results=5
    )
    print(f"   Results: {results3['total_found']} recipes")
    for i, result in enumerate(results3['results'][:3]):
        print(f"   {i+1}. {result['recipe']['title']}")
    print()
    
    # Test 4: Vegetarian + peanut allergy + diabetes
    print("📋 Test 4: Vegetarian + peanut allergy + diabetes")
    results4 = rag_pipeline.search_recipes(
        query=query,
        dietary_restrictions=['vegetarian'],
        allergies=['peanut'],
        health_conditions=['diabetes'],
        n_results=5
    )
    print(f"   Results: {results4['total_found']} recipes")
    for i, result in enumerate(results4['results'][:3]):
        print(f"   {i+1}. {result['recipe']['title']}")
    print()
    
    # Test 5: Just diabetes filter
    print("📋 Test 5: Diabetes filter only")
    results5 = rag_pipeline.search_recipes(
        query=query,
        dietary_restrictions=[],
        allergies=[],
        health_conditions=['diabetes'],
        n_results=5
    )
    print(f"   Results: {results5['total_found']} recipes")
    for i, result in enumerate(results5['results'][:3]):
        print(f"   {i+1}. {result['recipe']['title']}")
    print()
    
    # Test 6: Just peanut allergy
    print("📋 Test 6: Peanut allergy only")
    results6 = rag_pipeline.search_recipes(
        query=query,
        dietary_restrictions=[],
        allergies=['peanut'],
        health_conditions=[],
        n_results=5
    )
    print(f"   Results: {results6['total_found']} recipes")
    for i, result in enumerate(results6['results'][:3]):
        print(f"   {i+1}. {result['recipe']['title']}")
    print()

if __name__ == "__main__":
    debug_search() 