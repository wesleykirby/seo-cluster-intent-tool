"""Test URL analysis correcting semantic misclassifications."""
from cluster.url_intent_analyzer import UrlIntentAnalyzer
from cluster.semantic_analyzer import SemanticKeywordAnalyzer

def test_misclassification_correction():
    """Test cases where URL should correct semantic misclassification."""
    
    url_analyzer = UrlIntentAnalyzer()
    semantic_analyzer = SemanticKeywordAnalyzer()
    
    # Test cases where semantic might be wrong but URL should correct
    test_cases = [
        # Cases where keyword alone might be ambiguous
        ("results", "https://www.betway.co.za/luckynumbers"),  # Should be Lottery not Generic
        ("login", "https://www.betway.co.za/account/loginform"),  # Should be Branded with Login modifier
        ("games", "https://www.betway.co.za/lobby/casino-games"), # Should be Casino not Generic
        ("sport", "https://www.betway.co.za/sport"),  # Should be Sports
        ("live", "https://www.betway.co.za/Event/LiveSport"),  # Should be Sports with Live modifier
        
        # Cross-brand competition cases from Betway data
        ("hollywoodbets", "https://www.betway.co.za/account/loginform"),  # Competitor brand ranking on Betway
        ("lottostar", "https://www.betway.co.za/luckynumbers"),  # Competitor lottery term on Betway lottery page
    ]
    
    print("🧪 TESTING MISCLASSIFICATION CORRECTION")
    print("=" * 60)
    
    improvements = 0
    
    for keyword, url in test_cases:
        print(f"\n🎯 Testing: '{keyword}'")
        print(f"📍 URL: {url}")
        
        # Get semantic classification
        semantic_result = semantic_analyzer.fuzzy_recognizer.analyze_intent(keyword)
        
        # Get URL analysis
        url_analysis = url_analyzer.analyze_url(url)
        
        # Try enhancement
        semantic_dict = {
            'main_topic': semantic_result['main'],
            'sub_topic': semantic_result['sub'],
            'modifier': semantic_result['modifier']
        }
        
        enhanced_result = url_analyzer.enhance_semantic_classification(
            keyword, url, semantic_dict
        )
        
        # Check for improvement
        semantic_main = semantic_result['main']
        enhanced_main = enhanced_result.get('main_topic', semantic_main)
        
        print(f"🧠 Semantic: {semantic_main}")
        print(f"🔗 URL Signals: {url_analysis['intent_signals']} (confidence: {url_analysis['confidence']})")
        print(f"✨ Enhanced: {enhanced_main}")
        
        if semantic_main != enhanced_main:
            improvements += 1
            print(f"✅ CORRECTED: {semantic_main} → {enhanced_main}")
        else:
            print(f"✓ Consistent classification")
    
    print(f"\n🎯 CORRECTION SUMMARY:")
    print(f"Test cases: {len(test_cases)}")
    print(f"Corrections made: {improvements}")
    print(f"Correction rate: {improvements/len(test_cases)*100:.1f}%")
    
    return improvements

def test_real_betway_examples():
    """Test with actual problematic examples from Betway data."""
    
    # Real examples from the Betway dataset where URL should help
    real_cases = [
        # High-volume lottery keywords that might be misclassified
        ("daily lotto results for yesterday", "https://www.betway.co.za/luckynumbers"),
        ("uk teatime", "https://www.betway.co.za/luckynumbers"),
        ("power balls results for yesterday", "https://www.betway.co.za/luckynumbers"),
        
        # Competitor terms ranking on Betway pages  
        ("sportingbet login", "https://www.betway.co.za/sport"),
        ("hollywood login password", "https://www.betway.co.za/account/loginform"),
        ("lottostar login password", "https://www.betway.co.za/account/loginform"),
        
        # Sports keywords
        ("south african premiership games", "https://www.betway.co.za/sport"),
        ("live soccer", "https://www.betway.co.za/Event/LiveSport"),
        
        # Casino/promotions
        ("free spins no deposit win real money", "https://www.betway.co.za/latestpromos"),
        ("25 free spins on registration no deposit south africa", "https://www.betway.co.za/lobby/casino-games/slots"),
    ]
    
    analyzer = SemanticKeywordAnalyzer()
    
    print("\n🎰 TESTING REAL BETWAY EXAMPLES")
    print("=" * 60)
    
    # Analyze with and without URL enhancement
    keywords_only = [case[0] for case in real_cases]
    semantic_results = analyzer.analyze_keywords(keywords_only)
    enhanced_results = analyzer.analyze_keywords_with_urls(real_cases)
    
    improvements = 0
    for i, (keyword, url) in enumerate(real_cases):
        semantic_main = semantic_results.iloc[i]['Main']
        enhanced_main = enhanced_results.iloc[i]['Main']
        
        print(f"\n• {keyword}")
        print(f"  Semantic: {semantic_main}")
        print(f"  Enhanced: {enhanced_main}")
        
        if semantic_main != enhanced_main:
            improvements += 1
            print(f"  ✅ IMPROVED!")
            
    print(f"\n📊 Real Data Results:")
    print(f"Improvements: {improvements}/{len(real_cases)} ({improvements/len(real_cases)*100:.1f}%)")

if __name__ == "__main__":
    test_misclassification_correction()
    test_real_betway_examples()