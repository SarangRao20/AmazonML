# Team BreakEven — Amazon ML Challenge 2026 Submission Log

**Team Name:** BreakEven  
**Members:** Pranav Sanjay Men, Sarang Rao  
**Hackathon End Time:** 27 Sep 2026, 23:59:00 IST  

---

## 📌 Important Amazon Evaluation Rule
> **"Candidate generation counts toward the final ranking.** We will review your `candidate_pairs.tsv` and the code that produces it when deciding final rankings, alongside your `matching_results.tsv` score. The approach that generates a **smaller candidate set per Source 1 entity** will be ranked higher in the final evaluation beyond the public/private leaderboard."

---

## 📊 Live Leaderboard Submissions Log

| # | Date & Time | Score ($F_{0.5}$) | Rank | Matched Pairs | Mean / S1 | Candidate Pairs | Cand / S1 | Status & Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 27 Sep, 12:32 PM | **0.712498** | ~4,200 | 7,466,211 | 4.309 | 77.9M | 45.0 | Initial baseline run ($\tau=0.65$ relative rule + rank dedup). Over-predicted matches on 10M pool; $4\times$ FP penalty bounded score. |
| **2** | 27 Sep, 01:47 PM | **0.836000** | ~3,500 | 5,767,152 | 3.329 | 77.9M | 45.0 | Bisection calibrated per country. France mean dropped to 2.942 after 1-to-1 dedup (134k collision loss). |
| **3** | 27 Sep, 03:54 PM | **0.972865** | **1275** | 5,718,652 | **3.301** | 78.4M | **45.26** | **Current Best.** Direct-evidence LightGBM (89 feats, premise/unit/floor roles, $\tau=0.80$). Strict 1-to-1 exclusivity (0 collisions, 0 cross-country). |
| **4** | *Pending* | — | — | — | — | — | — | Reserved for planned, verified precision optimization. |
| **5** | *Pending* | — | — | — | — | — | — | Final reserve submission before 23:59 IST deadline. |

---

## 📦 Current Active Final Submission Package
- **Package Name:** `BreakEven_submission.zip`
- **Location:** `/home/sarang/AmazonML/BreakEven_submission.zip`
- **Archive Size:** 454 MB (Under 500 MB portal limit)
- **Matching TSV MD5:** `360f7e1f5efc5959e37ce276dfa25cd8` (`output_sidtech_v3/matching_results.tsv`)
- **Candidate TSV MD5:** Verified superset containing 100% of matches
- **Candidate Size Metric:** **45.26 candidates per Source 1 entity** (>99.9995% search space reduction from 17.2 Trillion pairs).
- **Format Validator:** PASS (exit code 0)
