# Amazon ML Challenge 2026: Business Entity Resolution Challenge

## 1. Overview
In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers, and the challenge of determining which records refer to the same business is known as **Entity Resolution (ER)**. 

The goal of this challenge is to build an ML solution that, given business records from 3 independent data sources with noisy and inconsistent fields, determines which records across sources refer to the same real-world business entity.

- **Reference Source:** Source 1 is the deduplicated reference source.
- **Task:** Find all matching records from Source 2 and Source 3 for each Source 1 entity.
- **Match Multiplicity:** A Source 1 entity may match zero (singleton), one, or many records from Source 2 and Source 3.

---

## 2. File & Data Format Specifications

### File Delimiter
- **All files in this challenge are tab-separated (`.tsv`), and all submissions must be tab-separated too.**
- Tabs are used because business addresses and ID list columns contain commas.
- Example reading in Python:
```python
import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```
*Note: Reading `.tsv` without `sep="\t"` will silently produce a single column containing the whole line.*

---

## 3. Dataset Description

Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) contains the following columns:

| Column | Description |
| :--- | :--- |
| `entity_id` | Unique identifier for the record (prefixed with `S1-`, `S2-`, or `S3-`). |
| `business_name` | Name of the business entity (may contain abbreviations, legal suffixes, typos, transliterations). |
| `business_address` | Address of the business (may contain partial addresses, format variations, missing components, landmark references). |
| `country` | Country label for the record. |

### Country Distribution & Open-Set Handling
- **Training Data:** Covers `US` and `India`.
- **Test Data:** Contains a third country, `France`, which does **not** appear in the training data.
- **Open-Set Requirement:** Treat `country` as an open set of string labels. Do **not** hard-code, filter, or one-hot encode the pipeline to only `{US, India}`. Every test entity — including those from `France` — must appear in the final submission.

### Ground Truth File (`train_ground_truth.tsv`)
| Column | Description |
| :--- | :--- |
| `source1_entity_id` | The `entity_id` of a Source 1 record. |
| `matched_entity_ids` | Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 (empty for singletons). |

### Noise Patterns to Expect
- **Name Variations:** Abbreviations (Corp vs. Corporation, Pvt vs. Private, Ltd vs. Limited), legal suffix inconsistencies, DBA/trade names, punctuation differences (`&` vs `"and"`), word-order transpositions, typos.
- **Address Variations:** Abbreviations (Rd vs. Road, St vs. Street), transliteration variants, missing components (missing PIN code or state), landmark-based references (e.g. "Near SBI ATM"), municipal numbering formats, component reordering.

---

## 4. File Structure & Directory Layout

### Provided Dataset Files
```
dataset/
├── train/
│   ├── train_source1.tsv          # Source 1 training records (reference)
│   ├── train_source2.tsv          # Source 2 training records
│   ├── train_source3.tsv          # Source 3 training records
│   └── train_ground_truth.tsv     # Ground truth matching labels
└── test/
    ├── test_source1.tsv           # Source 1 test records (generate matches for all)
    ├── test_source2.tsv           # Source 2 test records
    └── test_source3.tsv           # Source 3 test records
```

---

## 5. Output Format Specifications

Your solution produces **two** tab-separated files inside the `output/` folder:

### 1. `output/matching_results.tsv` (Scored on Leaderboard)
Your final entity matches.

| Column | Description |
| :--- | :--- |
| `source1_entity_id` | The `entity_id` of a Source 1 record |
| `matched_entity_ids` | Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 |

**Example:**
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**Formatting Rules:**
- Every Source 1 entity in the test set must have exactly one row.
- Leave `matched_entity_ids` empty for singletons (entities with no matches).
- No duplicate entity IDs within a single ID list.
- ID lists must only contain Source 2 (`S2-`) or Source 3 (`S3-`) IDs that exist in the test set.

### 2. `output/candidate_pairs.tsv` (Pipeline Verification)
The candidate set produced by your blocking / candidate-generation stage, *before* your final matching model narrowed it down.

| Column | Description |
| :--- | :--- |
| `source1_entity_id` | The `entity_id` of a Source 1 record |
| `candidate_entity_ids` | Comma-separated list of candidate `entity_id`s from Source 2 and/or Source 3 |

**Example:**
```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

*Note: Final matched IDs in `matching_results.tsv` must be a subset of the candidate IDs in `candidate_pairs.tsv`.*

---

## 6. Local Submission Validation

Run the validation script before submitting:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
- Returns `PASS` (exit 0) when files are valid.
- Returns a numbered list of issues (exit 1) if validation fails.

---

## 7. Final Submission Package Structure

When completing the challenge, teams submit a single zip archive formatted as follows:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Final matches (leaderboard file)
│   └── candidate_pairs.tsv         # Blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # All source code
│       ├── README.md               # End-to-end execution guide
│       └── requirements.txt        # Pinned dependencies / environment
└── Documentation_template.md       # Completed methodology write-up
```

---

## 8. Evaluation Criteria

Submissions are evaluated using **Macro-Averaged \(F_{\beta}\) Score (\(\beta = 0.5\))**:

\[
F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}
\]

- **Precision Weighting:** \(F_{0.5}\) weights precision **2× over recall**, heavily penalizing false merges (matching two distinct businesses).
- **Macro-Averaging:** Computed per Source 1 entity and averaged across **all** Source 1 entities in the test set.
- **Singletons:** Singletons score **1.0** when correctly predicted as an empty list, and **0.0** if any false match is predicted.

### Example Scoring:
- Predicted matches for `S1-00001`: `[S2-00047, S2-00193, S3-00812]`
- Ground truth matches: `[S2-00047, S3-00812]`
- Precision = 2/3 = 0.667, Recall = 2/2 = 1.0
- \(F_{0.5} = \frac{1.25 \times 0.667 \times 1.0}{0.25 \times 0.667 + 1.0} = \mathbf{0.714}\)

---

## 9. Constraints & Rules

1. **Model License & Size:** Final model must be **MIT or Apache 2.0 licensed** and up to **8 Billion parameters**.
2. **Leaderboard Splitting:** 
   - Public Leaderboard: Evaluated on a subset of the test set during the competition.
   - Private Leaderboard: Evaluated on the remaining test set portion for final rankings.
3. **Strictly Prohibited:** External database lookups, commercial entity resolution APIs, government registration searches, geocoding APIs, or internet data augmentation. Any external lookup results in immediate disqualification.

---

## 10. Key Strategies & Recommendations

- **Blocking Strategy:** Invest in high-recall candidate generation (TF-IDF, phonetic encoding, blocking keys) as it caps the maximum achievable recall.
- **Feature Engineering:** Calculate string similarity features (Jaccard, Levenshtein, token overlap) across business names and addresses.
- **Precision Focus:** Tune model thresholds for high precision to maximize \(F_{0.5}\).
- **Singleton Handling:** Pay special attention to singletons to avoid false positive penalties.
