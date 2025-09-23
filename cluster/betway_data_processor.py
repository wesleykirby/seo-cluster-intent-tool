"""Process Betway organic keywords dataset to train and validate URL-intent mapping."""
from __future__ import annotations

import pandas as pd
from typing import List, Tuple, Dict
from .semantic_analyzer import SemanticKeywordAnalyzer


class BetwayDataProcessor:
    """Processes Betway organic keywords data for ML enhancement."""
    
    def __init__(self):
        self.analyzer = SemanticKeywordAnalyzer()
    
    def process_betway_csv(self, csv_path: str, sample_size: int = 100) -> Dict:
        """
        Process Betway CSV data to train URL-intent patterns
        
        Args:
            csv_path: Path to Betway CSV file
            sample_size: Number of keywords to process for testing
            
        Returns:
            Analysis results and training insights
        """
        # Read the CSV data
        df = pd.read_csv(csv_path)
        
        # Clean and prepare data
        df = df.dropna(subset=['Keyword', 'Current URL'])
        df = df[df['Current URL'].str.strip() != '']
        
        # Take a sample for initial processing
        sample_df = df.head(sample_size)
        
        # Prepare keyword-URL pairs
        keyword_url_pairs = [(row['Keyword'], row['Current URL']) 
                           for _, row in sample_df.iterrows()]
        
        # Analyze with URL enhancement
        results_df = self.analyzer.analyze_keywords_with_urls(keyword_url_pairs)
        
        # Compare with semantic-only analysis
        keywords_only = [pair[0] for pair in keyword_url_pairs]
        semantic_only_df = self.analyzer.analyze_keywords(keywords_only)
        
        # Analyze improvements
        comparison_results = self._compare_results(sample_df, semantic_only_df, results_df)
        
        return {
            'enhanced_results': results_df,
            'semantic_only_results': semantic_only_df,
            'comparison': comparison_results,
            'sample_data': sample_df
        }
    
    def _compare_results(self, original_df: pd.DataFrame, 
                        semantic_df: pd.DataFrame, 
                        enhanced_df: pd.DataFrame) -> Dict:
        """Compare semantic-only vs URL-enhanced results."""
        
        improvements = []
        url_insights = []
        
        for i, row in original_df.iterrows():
            keyword = row['Keyword']
            url = row['Current URL']
            volume = row['Volume']
            
            # Get classifications
            semantic_main = semantic_df.iloc[i]['Main'] if i < len(semantic_df) else 'Unknown'
            enhanced_main = enhanced_df.iloc[i]['Main'] if i < len(enhanced_df) else 'Unknown'
            
            # Check for improvements
            if semantic_main != enhanced_main:
                improvements.append({
                    'keyword': keyword,
                    'url': url,
                    'volume': volume,
                    'semantic_classification': semantic_main,
                    'url_enhanced_classification': enhanced_main,
                    'improvement_type': self._classify_improvement(semantic_main, enhanced_main, url)
                })
            
            # Extract URL insights
            url_insight = self._extract_url_insight(keyword, url, enhanced_main)
            if url_insight:
                url_insights.append(url_insight)
        
        return {
            'total_improvements': len(improvements),
            'improvement_rate': len(improvements) / len(original_df) * 100,
            'improvements': improvements,
            'url_insights': url_insights
        }
    
    def _classify_improvement(self, semantic_main: str, enhanced_main: str, url: str) -> str:
        """Classify the type of improvement made by URL analysis."""
        
        url_lower = url.lower()
        
        # URL pattern-based improvement classification
        if '/luckynumbers' in url_lower and enhanced_main == 'Lottery':
            return 'Lottery Detection (URL: /luckynumbers)'
        elif '/sport' in url_lower and enhanced_main == 'Sports':
            return 'Sports Detection (URL: /sport)'
        elif '/casino' in url_lower and enhanced_main == 'Casino':
            return 'Casino Detection (URL: /casino)'
        elif '/account/loginform' in url_lower and 'Login' in enhanced_main:
            return 'Login Intent (URL: /loginform)'
        elif '/app' in url_lower and 'App' in enhanced_main:
            return 'App Intent (URL: /app)'
        else:
            return f'Generic Improvement ({semantic_main} → {enhanced_main})'
    
    def _extract_url_insight(self, keyword: str, url: str, classification: str) -> Dict:
        """Extract insights about URL patterns and their intent signals."""
        
        url_analyzer = self.analyzer.url_analyzer
        url_analysis = url_analyzer.analyze_url(url)
        
        if url_analysis['intent_signals']:
            return {
                'keyword': keyword,
                'url': url,
                'intent_signals': url_analysis['intent_signals'],
                'confidence': url_analysis['confidence'],
                'classification': classification,
                'brand': url_analysis.get('brand'),
                'pattern_type': self._get_pattern_type(url)
            }
        
        return None
    
    def _get_pattern_type(self, url: str) -> str:
        """Identify the type of URL pattern."""
        
        url_lower = url.lower()
        
        if '/luckynumbers' in url_lower:
            return 'Lottery Content'
        elif '/sport' in url_lower:
            return 'Sports Content'
        elif '/casino' in url_lower or '/lobby' in url_lower:
            return 'Casino Content'
        elif '/account' in url_lower:
            return 'Account Management'
        elif '/app' in url_lower:
            return 'App/Mobile'
        elif '/promo' in url_lower:
            return 'Promotions'
        else:
            return 'General Content'
    
    def generate_training_report(self, results: Dict) -> str:
        """Generate a training report showing URL enhancement insights."""
        
        comparison = results['comparison']
        improvements = comparison['improvements']
        url_insights = comparison['url_insights']
        
        report = []
        report.append("🎯 BETWAY URL-ENHANCED ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Total Keywords Analyzed: {len(results['sample_data'])}")
        report.append(f"URL Enhancements Made: {comparison['total_improvements']}")
        report.append(f"Improvement Rate: {comparison['improvement_rate']:.1f}%")
        report.append("")
        
        # Top improvements by volume
        if improvements:
            report.append("🚀 HIGH-IMPACT IMPROVEMENTS (by volume):")
            sorted_improvements = sorted(improvements, key=lambda x: x['volume'], reverse=True)
            
            for imp in sorted_improvements[:10]:  # Top 10
                report.append(f"  • {imp['keyword']} (Vol: {imp['volume']:,})")
                report.append(f"    {imp['semantic_classification']} → {imp['url_enhanced_classification']}")
                report.append(f"    {imp['improvement_type']}")
                report.append("")
        
        # URL pattern insights
        if url_insights:
            pattern_counts = {}
            for insight in url_insights:
                pattern = insight['pattern_type']
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            report.append("📊 URL PATTERN DISTRIBUTION:")
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  • {pattern}: {count} keywords")
            report.append("")
        
        # Intent signal effectiveness
        signal_effectiveness = {}
        for insight in url_insights:
            for signal in insight['intent_signals']:
                if signal not in signal_effectiveness:
                    signal_effectiveness[signal] = {'count': 0, 'avg_confidence': 0}
                signal_effectiveness[signal]['count'] += 1
                signal_effectiveness[signal]['avg_confidence'] += insight['confidence']
        
        if signal_effectiveness:
            report.append("🧠 INTENT SIGNAL EFFECTIVENESS:")
            for signal, data in signal_effectiveness.items():
                avg_conf = data['avg_confidence'] / data['count']
                report.append(f"  • {signal}: {data['count']} matches, {avg_conf:.0f}% avg confidence")
        
        return "\n".join(report)


def process_betway_sample():
    """Quick function to process Betway sample data."""
    processor = BetwayDataProcessor()
    
    # Process the uploaded Betway data
    results = processor.process_betway_csv(
        'attached_assets/betway learning data - Sheet1_1758604426441.csv',
        sample_size=50  # Start with 50 keywords
    )
    
    # Generate and return report
    report = processor.generate_training_report(results)
    return results, report