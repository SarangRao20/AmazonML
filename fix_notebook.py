"""Fix the two defects in retrain_colab.ipynb found by the first Colab run.

1. gdown --folder X -O DIR writes the folder's *contents* into DIR, it does
   not create a subdirectory named after the folder. Passing
   -O dataset therefore put train_source*.tsv directly in dataset/ and the
   retrain died on FileNotFoundError for dataset/train/train_source1.tsv.

2. The loader silently falls back to basic string similarity when anyascii
   and rapidfuzz are absent. That would make Colab's feature values differ
   from the local run's, so the model would be trained against different
   features than the ones used at inference time. Pin the same versions
   the local venv_fresh has.

Also adds an assertion cell so a bad path fails in seconds rather than
after a multi-minute download, and sets the tarfile extraction filter
explicitly (Python 3.14 changes the default).
"""

import json
import pathlib

NB = pathlib.Path(__file__).resolve().parent / "retrain_colab.ipynb"
nb = json.loads(NB.read_text())


def setsrc(i, text):
    nb["cells"][i]["source"] = text.splitlines(keepends=True)


DEPS = """!pip -q install polars pandas numpy scipy scikit-learn gdown
# sparse-dot-topn compiles a C++ extension; Colab ships gcc.
!pip -q install sparse-dot-topn
!pip -q install xgboost lightgbm catboost
# Without these two the feature extractor logs "using fallback" and swaps
# in basic string similarity, so Colab's features would not match the
# local run's. Same versions as the local venv_fresh.
!pip -q install anyascii==0.3.3 rapidfuzz==3.14.6
import anyascii, rapidfuzz, polars, sparse_dot_topn
print("deps ok")"""

# -O must be the train directory itself: gdown does not nest a subdir.
FETCH = """!gdown --folder "https://drive.google.com/drive/folders/1BfqxOy-J6gzgUtiOurfx7GiOo9oe5hsg" -O /content/AmazonML/dataset/train 2>&1 | tail -4
!ls -la /content/AmazonML/dataset/train/
import os
for f in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
          "train_ground_truth.tsv"):
    pth = "/content/AmazonML/dataset/train/" + f
    assert os.path.exists(pth), "missing " + pth
print("train split present and verified")"""

RUN = """import os
for f in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
          "train_ground_truth.tsv"):
    pth = "/content/AmazonML/dataset/train/" + f
    assert os.path.exists(pth), "preflight failed, missing " + pth
print("preflight ok - starting retrain")
"""

RETRAIN = """!cd /content/AmazonML && ER_TRAIN_CHUNK=5000 ER_POOL_SHARD=1000000 ER_POOL_SHARD_CHAR=400000 \\
  python -u train_model.py \\
    --blocker sparse --max-candidates 45 --sample-frac 0.08 \\
    --out /content/AmazonML/models_v2 \\
    --verdict-file /content/AmazonML/logs/retrain_verdict.txt 2>&1 | tail -70
"""

setsrc(2, DEPS)
setsrc(6, FETCH)
setsrc(8, RUN + RETRAIN)

s4 = "".join(nb["cells"][4]["source"]).replace(
    '.extractall(ROOT)',
    '.extractall(ROOT, filter="data")',
)
setsrc(4, s4)

NB.write_text(json.dumps(nb, indent=1))
print("patched", NB, NB.stat().st_size, "bytes")
