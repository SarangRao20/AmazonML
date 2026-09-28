#!/usr/bin/env python3
"""Fast hybrid creation - just combine v3 (US/India) + v4 (France)"""

import pandas as pd
from pathlib import Path

def get_country(entity_id):
    """Get country from entity ID"""
    num = int(entity_id.split('-')[1])
    if 700_000_000 <= num < 800_000_000:
        return 'France'
    elif 400_000_000 <= num < 600_000_000:
        return 'India'
    else:
        return 'US'

print("="*80)
print("FAST HYBRID: v3 (US/India) + v4 (France)")
print("="*80)

# Load
print("\nLoading...")
v3 = pd.read_csv('output_v3_benchmark/matching_results.tsv', sep='\t')
v4 = pd.read_csv('output_v4_author/matching_results.tsv', sep='\t')

# Add country
print("Detecting countries...")
v3['country'] = v3['source1_entity_id'].apply(get_country)
v4['country'] = v4['source1_entity_id'].apply(get_country)

# Combine
print("Combining...")
us_india = v3[v3['country'].isin(['US', 'India'])].copy()
france = v4[v4['country'] == 'France'].copy()

hybrid = pd.concat([us_india, france], ignore_index=True)
hybrid = hybrid.sort_values('source1_entity_id').reset_index(drop=True)
hybrid = hybrid[['source1_entity_id', 'matched_entity_ids']]

# Stats
total = sum(len(str(x).split(',')) for x in hybrid['matched_entity_ids'] if pd.notna(x))
empty = sum(hybrid['matched_entity_ids'].isna())

print(f"\n✅ HYBRID CREATED:")
print(f"   Total matches: {total:,}")
print(f"   Mean: {total/len(hybrid):.3f}")
print(f"   Singletons: {empty:,} ({100*empty/len(hybrid):.2f}%)")

# Save
output_dir = Path('output_final_hybrid')
output_dir.mkdir(exist_ok=True)
hybrid.to_csv(output_dir / 'matching_results.tsv', sep='\t', index=False)

# Copy candidates
import shutil
shutil.copy('output_v3_benchmark/candidate_pairs.tsv', output_dir / 'candidate_pairs.tsv')

print(f"✅ Saved to output_final_hybrid/")
print("="*80)
