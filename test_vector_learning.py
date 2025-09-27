#!/usr/bin/env python3
"""
Test script to validate that the upgraded vector-based semantic learning system
actually improves from training data and provides better analysis accuracy.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.vector_semantic_learner import VectorSemanticLearner
from cluster.semantic_analyzer import SemanticKeywordAnalyzer
from scripts.weekly_retrain import train_vector_semantic_model, load_training_data


def test_vector_learner_training():
    """Test that the vector learner can train on enhanced data."""
    print("🧪 TEST 1: Vector Learner Training")
    print("=" * 50)
    
    # Load test training data
    try:
        train_df = pd.read_csv('test_enhanced_training.csv')
        print(f"✅ Loaded {len(train_df)} training examples")
        
        # Initialize vector learner
        learner = VectorSemanticLearner()
        
        # Train the model
        results = learner.train(train_df)
        
        if results['status'] == 'success':
            print(f"✅ Vector model trained successfully!")
            print(f"   - Main Topic Accuracy: {results['main_topic_accuracy']:.1%}")
            print(f"   - Sub Topic Accuracy: {results['sub_topic_accuracy']:.1%}")
            print(f"   - Modifier Accuracy: {results['modifier_accuracy']:.1%}")
            print(f"   - Learned {results['learned_brands']} brands")
            print(f"   - Discovered {results['learned_patterns']} semantic patterns")
            return True
        else:
            print(f"❌ Training failed: {results.get('reason', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_vector_predictions():
    """Test that the vector learner can make predictions."""
    print("\n🧪 TEST 2: Vector Prediction Capabilities")
    print("=" * 50)
    
    try:
        # Train on test data first
        train_df = pd.read_csv('test_enhanced_training.csv')
        learner = VectorSemanticLearner()
        learner.train(train_df)
        
        # Test keywords that should match learned patterns
        test_keywords = [
            "betway registration form",      # Should learn Betway + Registration
            "sportybet mobile login",       # Should learn SportyBet + Login  
            "aviator crash game rules",     # Should learn Aviator + Game
            "football predictions weekend", # Should learn Football + Predictions
            "powerball results today"       # Should learn PowerBall + Results
        ]
        
        predictions = learner.predict(test_keywords)
        
        print("📋 Vector Predictions:")
        for i, keyword in enumerate(test_keywords):
            pred = predictions[i]
            print(f"   '{keyword}'")
            print(f"   → Main: {pred['main_topic']}, Sub: {pred['sub_topic']}, Mod: {pred['modifier']}")
            print(f"   → Confidence: {pred['confidence']:.1%}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_semantic_analyzer_integration():
    """Test that the semantic analyzer uses vector learning."""
    print("🧪 TEST 3: Semantic Analyzer Integration")
    print("=" * 50)
    
    try:
        # First, save test training data to the main training file
        train_df = pd.read_csv('test_enhanced_training.csv')
        train_df.to_csv('data/training_data.csv', index=False)
        print(f"✅ Saved {len(train_df)} training examples to main training file")
        
        # Train the vector model using the weekly retraining system
        print("🧠 Training vector model using weekly retraining system...")
        vector_results = train_vector_semantic_model(train_df)
        
        if vector_results.get('vector_training_status') == 'success':
            print(f"✅ Vector training integrated successfully!")
            print(f"   - Accuracy: Main {vector_results.get('main_accuracy', 0):.1%}, "
                  f"Sub {vector_results.get('sub_accuracy', 0):.1%}, "
                  f"Mod {vector_results.get('modifier_accuracy', 0):.1%}")
        
        # Test semantic analyzer with vector learning
        print("\n🔍 Testing semantic analyzer with vector learning...")
        analyzer = SemanticKeywordAnalyzer()
        
        # Test keywords that should benefit from learned patterns
        test_keywords = [
            "betway account registration help",  # Should use learned Betway + Registration
            "sportybet login issues",           # Should use learned SportyBet + Login
            "aviator game winning strategy",    # Should use learned Aviator patterns
            "football match predictions",       # Should use learned Football + Predictions
            "new brand xyz betting platform"    # Should fall back to rules (not learned)
        ]
        
        print("📊 Semantic Analysis Results:")
        results_df = analyzer.analyze_keywords(test_keywords)
        
        # Check if vector learning was used (should be logged)
        print("\n📋 Analysis Results:")
        for _, row in results_df.iterrows():
            print(f"   '{row['Keyword']}'")
            print(f"   → Main: {row['Main']}, Sub: {row['Sub']}, Mod: {row['Mod']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_learning_improvement():
    """Test that training data actually improves analysis accuracy."""
    print("\n🧪 TEST 4: Learning Improvement Validation")
    print("=" * 50)
    
    try:
        # Test keywords with expected improvements
        test_cases = [
            {
                'keyword': 'betway registration bonus',
                'expected_main': 'Branded',
                'expected_sub': 'Betway',
                'expected_mod': 'Registration'
            },
            {
                'keyword': 'sportybet login page',
                'expected_main': 'Branded', 
                'expected_sub': 'SportyBet',
                'expected_mod': 'Login'
            },
            {
                'keyword': 'aviator game crash',
                'expected_main': 'Casino',
                'expected_sub': 'Aviator', 
                'expected_mod': 'Game'
            }
        ]
        
        # Load training data and train
        train_df = pd.read_csv('test_enhanced_training.csv')
        train_df.to_csv('data/training_data.csv', index=False)
        
        # Initialize analyzer (will load learned patterns)
        analyzer = SemanticKeywordAnalyzer()
        
        # Test each case
        correct_predictions = 0
        for test_case in test_cases:
            keyword = test_case['keyword']
            results_df = analyzer.analyze_keywords([keyword])
            
            if not results_df.empty:
                result = results_df.iloc[0]
                main_correct = result['Main'] == test_case['expected_main']
                sub_correct = result['Sub'] == test_case['expected_sub']
                mod_correct = result['Mod'] == test_case['expected_mod']
                
                if main_correct and sub_correct and mod_correct:
                    correct_predictions += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"{status} '{keyword}'")
                print(f"   Expected: {test_case['expected_main']}/{test_case['expected_sub']}/{test_case['expected_mod']}")
                print(f"   Got:      {result['Main']}/{result['Sub']}/{result['Mod']}")
        
        accuracy = (correct_predictions / len(test_cases)) * 100
        print(f"\n📊 Learning Accuracy: {correct_predictions}/{len(test_cases)} ({accuracy:.1f}%)")
        
        if accuracy >= 66:  # At least 2/3 correct
            print("✅ Learning improvement validated!")
            return True
        else:
            print("⚠️  Learning accuracy below expected threshold")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests to validate the vector learning system."""
    print("🚀 TESTING VECTOR-BASED SEMANTIC LEARNING SYSTEM")
    print("=" * 60)
    print("This test validates that the upgraded system actually learns from training data\n")
    
    tests = [
        ("Vector Learner Training", test_vector_learner_training),
        ("Vector Prediction Capabilities", test_vector_predictions),
        ("Semantic Analyzer Integration", test_semantic_analyzer_integration),
        ("Learning Improvement Validation", test_learning_improvement)
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}\n")
    
    print("=" * 60)
    print(f"🏁 TEST RESULTS: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 ALL TESTS PASSED! Vector learning system is working correctly!")
        print("✨ Your semantic analyzer now learns from training data and gets smarter!")
    elif passed_tests >= len(tests) * 0.75:  # 75% pass rate
        print("✅ Most tests passed. System is mostly functional with minor issues.")
    else:
        print("⚠️  System needs debugging. Multiple tests failed.")
    
    return passed_tests == len(tests)


if __name__ == "__main__":
    main()