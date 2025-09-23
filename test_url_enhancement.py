"""Test script to demonstrate URL-enhanced keyword analysis."""
import pandas as pd
from cluster.semantic_analyzer import SemanticKeywordAnalyzer

def test_url_enhancement():
    """Test URL enhancement with sample Betway data."""
    
    # Sample data from Betway dataset
    test_cases = [
        ("daily lotto results for yesterday", "https://www.betway.co.za/luckynumbers"),
        ("uk teatime", "https://www.betway.co.za/luckynumbers"),
        ("sportingbet login", "https://www.betway.co.za/sport"),
        ("betway premiership fixtures", "https://www.betway.co.za/sport"),
        ("hollywood login password", "https://www.betway.co.za/account/loginform"),
        ("aviator betway", "https://www.betway.co.za/lobby/casino-games"),
        ("betway app download", "https://www.betway.co.za/Betway-App"),
        ("live soccer", "https://www.betway.co.za/Event/LiveSport")
    ]
    
    analyzer = SemanticKeywordAnalyzer()
    
    print("🧪 TESTING URL-ENHANCED KEYWORD ANALYSIS")
    print("=" * 60)
    
    # Test semantic-only analysis
    keywords_only = [case[0] for case in test_cases]
    semantic_results = analyzer.analyze_keywords(keywords_only)
    
    # Test URL-enhanced analysis
    enhanced_results = analyzer.analyze_keywords_with_urls(test_cases)
    
    print("📊 COMPARISON RESULTS:")
    print("-" * 60)
    
    improvements = 0
    for i, (keyword, url) in enumerate(test_cases):
        semantic_main = semantic_results.iloc[i]['Main']
        enhanced_main = enhanced_results.iloc[i]['Main']
        
        if semantic_main != enhanced_main:
            improvements += 1
            print(f"✅ IMPROVED: {keyword}")
            print(f"   Semantic Only: {semantic_main}")
            print(f"   URL Enhanced:  {enhanced_main}")
            print(f"   URL Pattern:   {url}")
            print()
        else:
            print(f"✓ Consistent: {keyword} → {semantic_main}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"Total keywords tested: {len(test_cases)}")
    print(f"Classifications improved: {improvements}")
    print(f"Improvement rate: {improvements/len(test_cases)*100:.1f}%")
    
    return {
        'semantic_results': semantic_results,
        'enhanced_results': enhanced_results,
        'improvements': improvements,
        'test_cases': test_cases
    }

if __name__ == "__main__":
    test_url_enhancement()