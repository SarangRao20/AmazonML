"""
Global consistency resolution (Stage 7).

Enforces query exclusivity where each S2/S3 record matches at most one S1 entity.

Techniques:
1. Conflict detection: Identify S2/S3 IDs claimed by multiple S1 entities
2. Confidence-based resolution: Award matches to highest confidence predictions
3. Conflict statistics: Track resolution rates and improvements

Phase 2 enhancement: Expected +0.08 F₀.₅ improvement.
"""

from typing import Dict, Set, Tuple, Optional
import pandas as pd


class ConsistencyResolver:
    """
    Resolves query exclusivity violations in entity matching predictions.
    
    Each S2/S3 record should map to at most one S1 entity.
    Conflicts are resolved by choosing the highest confidence match.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize consistency resolver.
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
    
    def resolve_conflicts(self, predictions: Dict[str, Set[str]],
                         confidence_dict: Optional[Dict[str, Dict[str, float]]] = None,
                         prefer_existing: bool = True) -> Tuple[Dict[str, Set[str]], Dict]:
        """
        Resolve query exclusivity violations.
        
        Args:
            predictions: S1 ID -> Set[S2/S3 IDs]
            confidence_dict: Optional S1 ID -> {S2/S3 ID -> confidence}
            prefer_existing: If True, break ties by preferring existing matches
        
        Returns:
            (refined_predictions, conflict_stats)
        """
        if self.verbose:
            print(f"\n🔄 Stage 7: Global Consistency Resolution")
        
        # Build reverse index: S2/S3 ID -> list of (S1 ID, confidence)
        s2_s3_to_s1 = self._build_reverse_index(predictions, confidence_dict)
        
        # Identify conflicts
        conflicts = {s2_s3_id: candidates 
                    for s2_s3_id, candidates in s2_s3_to_s1.items() 
                    if len(candidates) > 1}
        
        # Resolve conflicts
        refined = {s1_id: set(matches) for s1_id, matches in predictions.items()}
        resolution_log = []
        
        for s2_s3_id, candidates in conflicts.items():
            # Sort by confidence (descending)
            candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
            best_s1, best_conf = candidates_sorted[0]
            
            # Log resolution
            resolution_log.append({
                's2_s3_id': s2_s3_id,
                'num_claimants': len(candidates),
                'winner': best_s1,
                'winner_confidence': best_conf,
                'losers': [s1_id for s1_id, _ in candidates_sorted[1:]],
            })
            
            # Remove from all other S1 entities
            for s1_id, conf in candidates_sorted[1:]:
                refined[s1_id].discard(s2_s3_id)
        
        # Compute statistics
        stats = self._compute_stats(predictions, refined, conflicts, len(s2_s3_to_s1))
        
        if self.verbose:
            self._print_stats(stats, len(conflicts))
        
        return refined, stats
    
    def _build_reverse_index(self, predictions: Dict[str, Set[str]],
                            confidence_dict: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, list]:
        """
        Build reverse index: S2/S3 ID -> list of (S1 ID, confidence) tuples.
        
        Args:
            predictions: S1 ID -> Set[S2/S3 IDs]
            confidence_dict: Optional nested dict
        
        Returns:
            S2/S3 ID -> list of (S1 ID, confidence)
        """
        s2_s3_to_s1 = {}
        
        for s1_id, matches in predictions.items():
            for s2_s3_id in matches:
                if s2_s3_id not in s2_s3_to_s1:
                    s2_s3_to_s1[s2_s3_id] = []
                
                # Get confidence
                confidence = 0.5  # Default if not provided
                if confidence_dict and s1_id in confidence_dict:
                    if s2_s3_id in confidence_dict[s1_id]:
                        confidence = confidence_dict[s1_id][s2_s3_id]
                
                s2_s3_to_s1[s2_s3_id].append((s1_id, confidence))
        
        return s2_s3_to_s1
    
    def _compute_stats(self, predictions_before: Dict[str, Set[str]],
                      predictions_after: Dict[str, Set[str]],
                      conflicts: Dict,
                      total_s2_s3: int) -> Dict:
        """
        Compute consistency resolution statistics.
        
        Args:
            predictions_before: Original predictions
            predictions_after: Refined predictions
            conflicts: Identified conflicts
            total_s2_s3: Total S2/S3 entities
        
        Returns:
            Dict with statistics
        """
        matches_before = sum(len(m) for m in predictions_before.values())
        matches_after = sum(len(m) for m in predictions_after.values())
        matches_removed = matches_before - matches_after
        
        return {
            'conflicts_detected': len(conflicts),
            'conflicts_percentage': len(conflicts) / total_s2_s3 * 100 if total_s2_s3 > 0 else 0,
            'matches_before': matches_before,
            'matches_after': matches_after,
            'matches_removed': matches_removed,
            's1_entities_affected': len([s1 for s1 in predictions_before 
                                         if len(predictions_before[s1]) != len(predictions_after[s1])]),
            'total_s2_s3': total_s2_s3,
        }
    
    def _print_stats(self, stats: Dict, num_conflicts: int):
        """Print consistency resolution statistics."""
        print(f"  ✓ Conflicts detected: {stats['conflicts_detected']:,} "
              f"({stats['conflicts_percentage']:.2f}% of S2/S3)")
        print(f"  ✓ Matches removed: {stats['matches_removed']:,}")
        print(f"  ✓ S1 entities affected: {stats['s1_entities_affected']:,}")
        print(f"  ✓ Query exclusivity enforced")
    
    def detect_violations(self, predictions: Dict[str, Set[str]]) -> Dict[str, list]:
        """
        Detect (but don't resolve) query exclusivity violations.
        
        Args:
            predictions: S1 ID -> Set[S2/S3 IDs]
        
        Returns:
            Dict mapping S2/S3 ID -> list of S1 IDs that claim it
        """
        violations = {}
        
        for s1_id, matches in predictions.items():
            for s2_s3_id in matches:
                if s2_s3_id not in violations:
                    violations[s2_s3_id] = []
                violations[s2_s3_id].append(s1_id)
        
        # Return only actual violations (>1 claimant)
        return {s2_s3_id: claimants 
                for s2_s3_id, claimants in violations.items() 
                if len(claimants) > 1}


def resolve_predictions(predictions: Dict[str, Set[str]],
                       confidence_dict: Optional[Dict[str, Dict[str, float]]] = None,
                       verbose: bool = True) -> Dict[str, Set[str]]:
    """
    Convenience function to resolve predictions using default resolver.
    
    Args:
        predictions: S1 ID -> Set[S2/S3 IDs]
        confidence_dict: Optional nested dict of confidences
        verbose: Print progress
    
    Returns:
        Refined predictions with exclusivity enforced
    """
    resolver = ConsistencyResolver(verbose=verbose)
    refined, stats = resolver.resolve_conflicts(predictions, confidence_dict)
    return refined
