"""Debug URL analysis to see what's happening."""
from cluster.url_intent_analyzer import UrlIntentAnalyzer
from cluster.semantic_analyzer import SemanticKeywordAnalyzer

def debug_url_analysis():
    """Debug URL analysis step by step."""
    
    url_analyzer = UrlIntentAnalyzer()
    semantic_analyzer = SemanticKeywordAnalyzer()
    
    test_cases = [
        ("daily lotto results for yesterday", "https://www.betway.co.za/luckynumbers"),
        ("sportingbet login", "https://www.betway.co.za/account/loginform"),
        ("aviator game", "https://www.betway.co.za/lobby/casino-games"),
        ("live soccer betting", "https://www.betway.co.za/Event/LiveSport")
    ]
    
    print("🔍 DEBUG: URL ANALYSIS STEP-BY-STEP")
    print("=" * 60)
    
    for keyword, url in test_cases:
        print(f"\n🎯 Testing: '{keyword}'")
        print(f"📍 URL: {url}")
        
        # Step 1: URL Analysis
        url_analysis = url_analyzer.analyze_url(url)
        print(f"🔗 URL Analysis:")
        print(f"   Intent Signals: {url_analysis['intent_signals']}")
        print(f"   Confidence: {url_analysis['confidence']}")
        print(f"   Brand: {url_analysis.get('brand')}")
        
        # Step 2: Semantic Analysis
        semantic_result = semantic_analyzer.fuzzy_recognizer.analyze_intent(keyword)
        print(f"🧠 Semantic Analysis:")
        print(f"   Main: {semantic_result['main']}")
        print(f"   Sub: {semantic_result['sub']}")
        print(f"   Modifier: {semantic_result['modifier']}")
        
        # Step 3: Enhanced Analysis
        semantic_dict = {
            'main_topic': semantic_result['main'],
            'sub_topic': semantic_result['sub'],
            'modifier': semantic_result['modifier']
        }
        
        enhanced_result = url_analyzer.enhance_semantic_classification(
            keyword, url, semantic_dict
        )
        
        print(f"✨ Enhanced Result:")
        print(f"   Main: {enhanced_result.get('main_topic')}")
        print(f"   Sub: {enhanced_result.get('sub_topic')}")
        print(f"   Modifier: {enhanced_result.get('modifier')}")
        print(f"   URL Override: {enhanced_result.get('url_override', False)}")
        print(f"   URL Confidence: {enhanced_result.get('url_confidence', 0)}")
        
        print("-" * 40)

if __name__ == "__main__":
    debug_url_analysis()