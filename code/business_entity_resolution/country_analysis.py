"""
Per-country data distribution and error pattern analysis (Phase 3).

Tailored models per country for 98%+ F₀.₅ target.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from collections import Counter


class CountryAnalyzer:
    """Analyzes per-country distribution and error patterns."""
    
    def __init__(self, verbose: bool = True):
        """Initialize analyzer."""
        self.verbose = verbose
        self.country_stats = {}
    
    def extract_country(self, address: str) -> Optional[str]:
        """Extract country from address."""
        if not address:
            return 'US'
        
        mappings = {
            'us': 'US', 'usa': 'US', 'united states': 'US',
            'in': 'IN', 'india': 'IN',
            'fr': 'FR', 'france': 'FR',
            'uk': 'UK', 'gb': 'UK', 'united kingdom': 'UK',
            'ca': 'CA', 'canada': 'CA',
            'au': 'AU', 'australia': 'AU',
            'mx': 'MX', 'mexico': 'MX',
            'br': 'BR', 'brazil': 'BR',
            'jp': 'JP', 'japan': 'JP',
            'cn': 'CN', 'china': 'CN',
            'de': 'DE', 'germany': 'DE',
        }
        
        # Try last part (most common country location)
        parts = address.split(',')
        if len(parts) > 0:
            last = parts[-1].strip().lower()
            if last in mappings:
                return mappings[last]
        
        # Try anywhere in address
        addr_lower = address.lower()
        for key, code in mappings.items():
            if key in addr_lower:
                return code
        
        return 'US'
    
    def analyze_distribution(self, s1_df: pd.DataFrame, s2_df: pd.DataFrame,
                            s3_df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze country distribution across sources."""
        if self.verbose:
            print("\n🌍 Analyzing per-country distribution...")
        
        s1_countries = s1_df['business_address'].apply(self.extract_country)
        s2_countries = s2_df['business_address'].apply(self.extract_country)
        s3_countries = s3_df['business_address'].apply(self.extract_country)
        
        stats = {}
        
        for country, count in Counter(s1_countries).most_common():
            if country not in stats:
                stats[country] = {'s1': 0, 's2': 0, 's3': 0}
            stats[country]['s1'] = count
        
        for country, count in Counter(s2_countries).most_common():
            if country not in stats:
                stats[country] = {'s1': 0, 's2': 0, 's3': 0}
            stats[country]['s2'] = count
        
        for country, count in Counter(s3_countries).most_common():
            if country not in stats:
                stats[country] = {'s1': 0, 's2': 0, 's3': 0}
            stats[country]['s3'] = count
        
        for country in stats:
            stats[country]['total'] = (stats[country]['s1'] + stats[country]['s2'] + stats[country]['s3'])
        
        if self.verbose:
            print(f"  ✓ Found {len(stats)} countries")
            for country, dist in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)[:5]:
                print(f"    {country}: S1={dist['s1']:,}, S2={dist['s2']:,}, S3={dist['s3']:,}, Total={dist['total']:,}")
        
        self.country_stats = stats
        return stats
    
    def estimate_error_patterns(self, country_stats: Dict) -> Dict:
        """Estimate error patterns by country."""
        if self.verbose:
            print("\n📊 Estimating error patterns by country...")
        
        patterns = {}
        
        for country, stats in country_stats.items():
            s1_count = stats['s1']
            s2_count = stats['s2']
            s3_count = stats['s3']
            
            ratio = (s2_count + s3_count) / max(s1_count, 1)
            
            patterns[country] = {
                'country': country,
                's1_count': s1_count,
                's2_s3_total': s2_count + s3_count,
                'ratio': ratio,
                'recall_error': 'HIGH' if ratio > 10 else 'MEDIUM' if ratio > 5 else 'LOW',
                'precision_error': 'HIGH' if s1_count < 1000 else 'MEDIUM' if s1_count < 5000 else 'LOW',
                'multilingual': country in ['IN', 'CN', 'JP'],
                'regional_split': s1_count > 50000,
            }
        
        if self.verbose:
            print(f"  ✓ Analyzed {len(patterns)} countries")
        
        return patterns
    
    def create_country_groups(self, country_stats: Dict) -> Dict[str, list]:
        """Group countries by similarity for model strategy."""
        groups = {
            'high_volume': [],      # >10k entities
            'medium_volume': [],    # 1k-10k
            'low_volume': [],       # <1k
            'multilingual': [],     # Need special handling
        }
        
        for country, stats in country_stats.items():
            total = stats['total']
            
            if total > 10000:
                groups['high_volume'].append(country)
            elif total > 1000:
                groups['medium_volume'].append(country)
            else:
                groups['low_volume'].append(country)
            
            if country in ['IN', 'CN', 'JP', 'KR']:
                groups['multilingual'].append(country)
        
        return groups
