"""
Embedding-based blocking for entity resolution (Phase 4 optional upgrade).

Uses multilingual-e5-small embeddings + FAISS for semantic blocking.
Complements token-based blocking with semantic similarity.

Integration: Works with Phase 3 country routing for 98%+ F₀.₅ target.

Dependencies:
- sentence-transformers (multilingual-e5-small)
- faiss-cpu or faiss-gpu

Expected improvements:
- +5-10 point recall ceiling
- Better handling of multilingual/transliterated names
- Reduces false negatives from token sparsity

Status: Phase 4 optional upgrade if time permits.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class EmbeddingBlocker:
    """
    Semantic blocking using multilingual embeddings (Phase 4).
    
    Algorithm:
    1. Embed all S2/S3 business names using multilingual-e5-small
    2. Build FAISS index on S2/S3 embeddings
    3. For each S1 entity, retrieve top-K semantic neighbors
    4. Combine with token-based candidates
    
    Phase 4 optional enhancement for 98%+ F₀.₅.
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small",
                 top_k: int = 50, verbose: bool = True):
        """
        Initialize embedding blocker.
        
        Args:
            model_name: HuggingFace model ID for embeddings
            top_k: Number of top candidates to retrieve
            verbose: Print progress
        """
        self.model_name = model_name
        self.top_k = top_k
        self.verbose = verbose
        
        self.model = None
        self.index = None
        self.s2_s3_ids = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Load embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            if self.verbose:
                print("⚠️  sentence-transformers not installed. Embedding blocking disabled.")
            return
        
        if self.verbose:
            print(f"📦 Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            if self.verbose:
                print(f"✓ Model loaded successfully")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to load model: {e}")
            self.model = None
    
    def build_index(self, s2_s3_ids: List[str], s2_s3_names: List[str]):
        """
        Build FAISS index on S2/S3 embeddings.
        
        Args:
            s2_s3_ids: List of S2/S3 entity IDs
            s2_s3_names: List of S2/S3 business names
        """
        if not self.model or not FAISS_AVAILABLE:
            if self.verbose:
                print("⚠️  Embedding index not available")
            return
        
        if self.verbose:
            print(f"\n🔨 Building embedding index for {len(s2_s3_ids):,} S2/S3 entities...")
        
        # Embed S2/S3 names
        embeddings = self.model.encode(s2_s3_names, show_progress_bar=self.verbose)
        embeddings = embeddings.astype(np.float32)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.s2_s3_ids = s2_s3_ids
        
        if self.verbose:
            print(f"✓ Index built: {len(s2_s3_ids):,} vectors, {dimension} dimensions")
    
    def get_candidates(self, s1_names: List[str], s1_ids: List[str],
                      return_scores: bool = True) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve semantic neighbors for S1 entities.
        
        Args:
            s1_names: List of S1 business names
            s1_ids: List of S1 entity IDs
            return_scores: Return distance scores
        
        Returns:
            Dict: S1 ID -> list of (S2/S3 ID, score) tuples
        """
        if not self.model or not self.index:
            return {}
        
        if self.verbose:
            print(f"\n🔍 Retrieving semantic neighbors for {len(s1_ids):,} S1 entities...")
        
        # Embed S1 names
        s1_embeddings = self.model.encode(s1_names, show_progress_bar=self.verbose)
        s1_embeddings = s1_embeddings.astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(s1_embeddings, self.top_k)
        
        # Build result dict
        candidates = {}
        for i, s1_id in enumerate(s1_ids):
            candidates[s1_id] = []
            for j, idx in enumerate(indices[i]):
                if idx >= 0 and idx < len(self.s2_s3_ids):  # Valid index
                    s2_s3_id = self.s2_s3_ids[idx]
                    # Convert L2 distance to similarity (lower distance = higher similarity)
                    distance = distances[i][j]
                    similarity = 1.0 / (1.0 + distance)  # Normalize to [0,1]
                    candidates[s1_id].append((s2_s3_id, similarity))
        
        if self.verbose:
            avg_candidates = np.mean([len(c) for c in candidates.values()])
            print(f"✓ Retrieved avg {avg_candidates:.1f} candidates per S1")
        
        return candidates
    
    def merge_candidates(self, token_candidates: Dict[str, List[Tuple[str, float]]],
                        semantic_candidates: Dict[str, List[Tuple[str, float]]],
                        weight_token: float = 0.6,
                        weight_semantic: float = 0.4,
                        top_k: int = 50) -> Dict[str, List[Tuple[str, float]]]:
        """
        Merge token-based and semantic candidates (Phase 4 Task 2).
        
        Args:
            token_candidates: From token-based blocking
            semantic_candidates: From embedding blocking
            weight_token: Weight for token-based scores
            weight_semantic: Weight for semantic scores
            top_k: Keep top-K after merging
        
        Returns:
            Merged candidates
        """
        merged = {}
        
        # Combine scores
        for s1_id in token_candidates:
            combined_scores = {}
            
            # Add token-based candidates
            for s2_s3_id, score in token_candidates.get(s1_id, []):
                combined_scores[s2_s3_id] = weight_token * score
            
            # Add semantic candidates
            for s2_s3_id, score in semantic_candidates.get(s1_id, []):
                if s2_s3_id in combined_scores:
                    combined_scores[s2_s3_id] += weight_semantic * score
                else:
                    combined_scores[s2_s3_id] = weight_semantic * score
            
            # Sort and keep top-K
            sorted_candidates = sorted(combined_scores.items(), 
                                      key=lambda x: x[1], reverse=True)[:top_k]
            merged[s1_id] = sorted_candidates
        
        return merged
    
    def is_available(self) -> bool:
        """Check if embedding blocking is available."""
        return self.model is not None and self.index is not None


def enable_embedding_blocking(blocking_candidates: Dict[str, List[Tuple[str, float]]],
                              s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                              weight_token: float = 0.6, weight_semantic: float = 0.4,
                              top_k: int = 50, verbose: bool = True) -> Dict[str, List[Tuple[str, float]]]:
    """
    Enhance blocking with embedding-based candidates (optional).
    
    Args:
        blocking_candidates: Token-based candidates
        s1_df: Source 1 dataframe
        s2_df: Source 2 dataframe
        s3_df: Source 3 dataframe
        weight_token: Weight for token scores
        weight_semantic: Weight for semantic scores
        top_k: Keep top-K candidates
        verbose: Print progress
    
    Returns:
        Enhanced candidates combining token + semantic
    """
    if not EMBEDDINGS_AVAILABLE or not FAISS_AVAILABLE:
        if verbose:
            print("⚠️  Embedding blocking disabled (dependencies missing)")
        return blocking_candidates
    
    # Initialize blocker
    blocker = EmbeddingBlocker(verbose=verbose)
    
    # Build index on S2/S3
    s2_s3_df = pd.concat([s2_df, s3_df], ignore_index=True)
    blocker.build_index(
        s2_s3_df['entity_id'].tolist(),
        s2_s3_df['business_name'].tolist()
    )
    
    # Get semantic candidates
    semantic_candidates = blocker.get_candidates(
        s1_df['business_name'].tolist(),
        s1_df['entity_id'].tolist()
    )
    
    # Merge candidates
    merged = blocker.merge_candidates(
        blocking_candidates,
        semantic_candidates,
        weight_token=weight_token,
        weight_semantic=weight_semantic,
        top_k=top_k
    )
    
    if verbose:
        print(f"✓ Embedding blocking complete (+candidates merged)")
    
    return merged
