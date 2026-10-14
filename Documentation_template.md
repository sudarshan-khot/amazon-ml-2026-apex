# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Apex
**Team Members:** Sudarshan Khot, Shree Mengshetti, Vallabh Vidyasagar, Saksham Agarwal
**Submission Date:** 02/10/2026

---

## 1. Executive Summary
A four-stage pipeline: noise-reversing normalisation, rarity-weighted composite-key blocking with a learned
re-ranker, a two-stage gradient-boosted pair classifier with "collective" features (agreement with the
entity's other confident matches and with competing entities), and an expected-F0.5 decision rule under a
one-record-one-entity constraint. Out-of-fold macro F0.5 on the full training set (2,206,821 Source-1
entities, scored with the official formula against `train_ground_truth.tsv`) is **0.9832**
(US 0.9862, India 0.9787, singletons 0.9767), stable across 5 folds (0.9830–0.9834).

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the training data (all numbers measured):

* **Scale:** 2.2M Source-1 (S1) entities vs 10.3M Source-2/3 (S2/S3) records in train; 1.73M vs 9.97M in test.
  Exhaustive comparison is impossible (~2×10¹³ pairs).
* **Structure:** every S2/S3 record belongs to at most one S1 entity; **no cross-country matches**;
  26% of S2/S3 records match no S1 entity (realistic distractors, often near-duplicates of a real entity);
  5.6% of S1 entities are singletons; 3.46 matches per entity on average (max 11).
* **Names are weak alone:** 30% of S1 names are duplicated; 38% of the unmatched distractors share their
  name with some S1 entity. Addresses must confirm names.
* **The noise is generated and systematic.** By mining single-token substitutions between true pairs we
  reverse-engineered it:
  - legal words swapped/dropped (`limited→ltd` ×1841), descriptor words replaced by fillers
    (`private→center`), leetspeak (`0→o, 1→l, 5→s`, capital I rendered `l`), typos, word shuffles;
  - **alias names** "`<generated brand> DBA <real name>`": the real name is always on the right (3352/3352);
  - **random brand names** built from a fixed syllable set (`Yumabrix`, `Nexrizaonyx`);
  - **domains / handles** (`paamtraders.com`, `@robustworks`), acronyms (`IMC`);
  - **native Indian scripts** (23% of Indian S2 names; 9 scripts) and native-script state names;
  - addresses: component reordering (14% even in S1), street-type abbreviations (`road→rd`), US state
    codes vs names, Indian state codes/names/native names, French region vs department, house-number
    zero padding / truncation (`21403→2140`), placeholder parts (`<NULL>`, `N/A`), injected floor text,
    PMB / PO-box numbers, city names replaced by neighbouring towns (city agrees in only 84% of true US pairs);
  - Hyderabad addresses labelled Andhra Pradesh in S2 but Telangana in S1 (570 of 604 state disagreements).
* **Test differs from train:** France (15% of test S1) has no training data, and test has more
  distractors per entity (5.75 S2/S3 records per S1 vs 4.68).

### 2.2 Solution Strategy
**Approach Type:** Hybrid – rule-based normalisation + weighted composite-key blocking with a learned
re-ranker + two-stage gradient-boosted classifier + metric-aware decision rule.  
**Core Innovation:** (1) blocking on *rare combinations* rather than rare words, ranked by a
weighted Jaccard that stops information-rich neighbours from out-ranking sparse true matches;
(2) collective second-stage features that judge a candidate against the entity's other confident matches
and against competing entities; (3) an expected-F0.5 decision per entity under a one-to-one constraint.

Validation: 5 folds by S1 entity (stratified by country and match count; every true match follows its
entity; unmatched records stay in every search pool). Every learned component (blocking re-ranker, stage-1
and stage-2 models) is cross-fitted, so each train entity is scored by models that never saw it, and the
reported F0.5 is computed with the official formula over all 2.2M train entities.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** 11 composite key families built from the cleaned fields, per country:
  single rare name word (df ≤ 100), name-word pairs, phonetic-name-word pairs, name word × state,
  name word × address word, glued name (domains), house number × address word, address number × address word,
  address number × state, address-word pairs (very common words excluded), address word × state.
  Each key is weighted by its rarity in the S2/S3 pool of the same country, `w = log(N_pool/df)`
  (keys with df above a family cap, 100–1000, never create candidates).
- **Ranking:** for each S1 record, every pool record sharing ≥1 key is scored; the union of three short
  lists is kept – top 40 by **weighted Jaccard** (shared key weight / union key weight), top 20 by raw shared
  weight, top 10 by name-only weight – and re-ranked by a cross-fitted gradient-boosting model on 15
  language-neutral features (scores, ranks, distance to the entity's best candidate). Top **20** are kept.
- **Candidate pairs generated:** 44,135,733 (train), 34,649,601 (test) – 20 per S1 record.
- **How you ensured true matches were not lost:** every design choice was measured on train pairs:
  single-word keys reach only 44% recall at df ≤ 1000 (US), composite keys reach 99.5%; raw-sum ranking lost
  sparse (empty-address) matches to rich neighbours, weighted Jaccard fixed half of that, the learned
  re-ranker most of the rest. Final out-of-fold recall **98.43%** of all 7,638,365 true pairs
  (US 98.81%, India 97.86%), identical in all folds (98.42–98.44%); F0.5 ceiling with these candidates **0.995**.
  The re-ranker trained on one country and applied to the other kept almost all its gain, which supports
  its use on France. Engineering: pure-NumPy CSR index, shared-memory parallel scoring (identical output
  for any worker count), fits in 8 GB RAM.

---

## 4. Matching Model

**Features used:** 85, all language-neutral (country is not a feature):
- Name features: fuzzy ratio / token-set / token-sort / partial / Jaro-Winkler on the cleaned name,
  phonetic-skeleton similarity (handles transliteration and typos), glued-name similarity and equality
  (domains), acronym match, legal-form agreement, record type (normal / domain / handle / brand / alias /
  acronym), native-script flag, rarity-weighted name-word Jaccard and the rarest shared word.
- Address features: word ratios, rarity-weighted address and street-word Jaccard, number-set Jaccard,
  house number equal / different / missing, **house-number arithmetic** (prefix/suffix truncation vs
  same-length difference = neighbour, log numeric gap), unit, city, state agreement, empty-address flags.
- Other: blocking scores and re-rank probability; competition features (how many entities claim the same
  record and whether this entity is its best claimant); **collective stage-2 features** built from the
  stage-1 out-of-fold probabilities – the entity's number and sum of confident matches, the gap to the best
  competing entity on name/address similarity, and agreement (name, address, numbers, house number) with
  the entity's most confident *other* match; source flag (S2/S3).

**Model type:** XGBoost (GPU, `hist`), depth 8, learning rate 0.08, early stopping; two stages
(stage 1 on 52 pair features; final model on all 85). Each fold model is trained on a 40% entity sample of
the other four folds (learning curve: 10% → 25% → 40% gives AP 0.99873 → 0.99883 → 0.99889). Test is scored
by the mean of the 5 fold models (probability distribution verified identical to a single model's).
Out-of-fold pair AUC 0.99975, average precision 0.99890.  
**Threshold selection method:** decision rules compared by macro F0.5 on out-of-fold train predictions:
plain threshold (best 0.70: 0.98298), one-to-one + threshold (0.98299), expected-F0.5 (0.98317),
**one-to-one + expected-F0.5 (0.98320, chosen)**. The one-to-one step gives each S2/S3 record only to the
entity that scores it highest; the expected-F0.5 step chooses, per entity, how many top candidates to
predict (0, 1, 2, …) to maximise the expected F0.5 given the probabilities, where predicting nothing is
valued at P(no true match) – this handles singletons directly.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9832** out-of-fold on all 2,206,821 train entities (US 0.9862,
  India 0.9787; singletons 0.9767; folds 0.9830–0.9834; pair precision 99.58%, recall 95.89%).
  Progression: baseline 0.9770 → + collective features 0.9801 → + house-number / rarity features 0.9828 →
  + more data and fold ensemble 0.9832.
- **Common false positives (wrong merges):** near-duplicate businesses at the same street with a house
  number one or two apart (`901` vs `904 Center Park Dr`), and records claimed by several same-name entities.
  The house-number arithmetic and sibling-agreement features cut singleton false merges from 3.2% to 2.3%.
- **Common false negatives (missed matches):** sparse records whose name is generic and shared by several
  entities, so no model can safely attach them (48% of the pairs rejected by the decision rule have an empty
  address); about 21% of the remaining loss is pairs that never became candidates (1.57% of true pairs,
  54% of them with an empty address). A second-hop blocking idea was measured and
  rejected (≤20% of those pairs recoverable at ~200M extra candidates).

---

## 6. Conclusion
Reverse-engineering the noise from the training pairs made every stage measurable: cleaning raised
name-similarity AUC from 0.75–0.92 to 0.97, rarity-weighted composite blocking kept 98.4% of true pairs in
20 candidates, and the collective second-stage features plus a metric-aware decision rule took the
out-of-fold macro F0.5 from 0.977 to 0.983. The main lesson: judge a candidate in the context of the
entity's other matches and its competitors, not in isolation.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (see its `README.md` for exact commands and run times):

| file | role |
|---|---|
| `src/data_cleaning.py` | single entry point for cleaning (writes cleaned TSVs) |
| `src/make_folds.py`, `src/evaluate.py` | validation folds, official scorer |
| `src/blocking.py` | blocking + cross-fitted re-ranker → candidate pairs |
| `src/matching.py` | pair / collective / extra features, out-of-fold XGBoost training, test scoring |
| `src/decide.py` | decision rules, tuning, `matching_results.tsv` + `candidate_pairs.tsv` |
| checks | `test_normalize.py` (114 tests), `verify_cleaning.py`, `check_cleaned_files.py`, `check_blocking.py`, `error_analysis.py` |

End-to-end: `data_cleaning → make_folds → blocking → matching (features, train, stage2, extra, train3) → decide`;
about 4.5 hours on a 6-core laptop with a GTX 1650.

### B. Additional Results

Blocking recall vs candidates kept (out-of-fold, train):

| K | recall | F0.5 ceiling | India recall | US recall |
|---|---|---|---|---|
| 5 | 91.3% | 0.982 | 90.2% | 92.0% |
| 10 | 97.8% | 0.993 | 97.0% | 98.4% |
| 20 | 98.4% | 0.995 | 97.9% | 98.8% |

Cleaning quality (true pairs vs hard negatives, AUC raw → clean): US name 0.916 → 0.975,
India name 0.748 → 0.971, US address 0.953 → 0.972, India address 0.958 → 0.968.

Test sanity (no labels): share of S1 entities with ≥1 predicted match – France 0.949, India 0.943,
US 0.943 (train truth 0.944); predicted matches per entity 3.42 / 3.33 / 3.37 (train truth 3.46).

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
