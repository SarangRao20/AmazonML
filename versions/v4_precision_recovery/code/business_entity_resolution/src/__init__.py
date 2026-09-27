"""Business Entity Resolution pipeline.

Modules
-------
config               Central configuration (paths, blocking, model, thresholds)
data_loader          Loading + entity-level disjoint train/val split
normalize            Text normalization (NFKC, legal suffixes, transliteration)
blocking             Multi-channel candidate generation
features             Pairwise feature extraction
model                Gradient-boosted ensemble training
threshold_optimizer  Macro F_0.5 threshold tuning + singleton gating
evaluate             Macro per-entity F_0.5 metrics + error analysis
consistency          Query-exclusivity conflict resolution
pipeline             End-to-end orchestration
"""

__all__ = [
    "config",
    "data_loader",
    "normalize",
    "blocking",
    "features",
    "model",
    "threshold_optimizer",
    "evaluate",
    "consistency",
    "pipeline",
]
