# Business Entity Resolution – reproducible pipeline

Finds, for every Source-1 business, the Source-2/3 records that describe the same business
(ML Challenge 2026). Out-of-fold macro F0.5 on the training set: **0.9832**
(US 0.9862, India 0.9787, singletons 0.9767).

## Environment

* Python 3.12, packages pinned in `requirements.txt` (all open-source, permissive licenses)
* Tested on: Windows 11, Intel i5-11400H (6 cores / 12 threads), 16.8 GB RAM, NVIDIA GTX 1650 (4 GB)
* A CUDA GPU is optional: add `--device cpu` to the `matching.py` training commands to run on CPU (slower).
* No external data, API or look-up service is used anywhere. Everything is learned from the provided files.

```bash
pip install -r requirements.txt
```

## Run end-to-end

Run from this folder (`code/business_entity_resolution/`). `DATA` is the challenge `dataset/` folder
(containing `train/` and `test/`); `OUT` is where the two submission files are written.

```bash
DATA=../../dataset
OUT=../../output

# 1. cleaning: 22.2M records -> artifacts/cleaned_dataset/{train,test}/*.tsv
python src/data_cleaning.py --data $DATA
# 2. validation folds: 5 folds by Source-1 entity -> artifacts/folds.tsv
python src/make_folds.py --data $DATA
# 3. blocking: candidate pairs (top 20 per entity) -> artifacts/blocking/
python src/blocking.py --data $DATA
# 4. matching model
python src/matching.py --data $DATA --phase features                 # 52 pair features
python src/matching.py --data $DATA --phase train                    # stage-1 model (out-of-fold)
python src/matching.py --data $DATA --phase stage2                   # 22 collective features
python src/matching.py --data $DATA --phase extra                    # 11 house-number / rarity features
python src/matching.py --data $DATA --phase train3 --train_frac 0.40 --max_rounds 3000 \
       --model3 artifacts/model_stage3b --test_ensemble                # final model (85 features)
# 5. decisions -> $OUT/matching_results.tsv and $OUT/candidate_pairs.tsv
python src/decide.py --data $DATA --model artifacts/model_stage3b --out $OUT
# validate
python ../../utils/validate_submission.py --matching $OUT/matching_results.tsv \
       --candidate $OUT/candidate_pairs.tsv --test-dir $DATA/test --check-ids
```

Measured run time on the machine above (GPU at full power, laptop plugged in):

| step | time |
|---|---|
| data_cleaning | ~15 min |
| make_folds | ~1 min |
| blocking (train + test, 4 workers) | ~1 h 45 min |
| matching features / stage2 / extra | ~20 / ~12 / ~14 min |
| matching train (stage 1) / train3 (final) | ~30 / ~65 min |
| decide | ~5 min |

Peak memory stays below ~8 GB. Blocking uses `--workers 0` (auto: from CPU cores and free RAM),
so it scales to bigger machines automatically; results are identical for any worker count.

## Pipeline

| file | step | what it does |
|---|---|---|
| `src/data_cleaning.py` | 1 | Repairs encoding damage, lowercases/strips accents, transliterates 9 Indian scripts (dictionary learned from train pairs + phonetic fallback), splits "Brand DBA Real name" aliases, detects domains / handles / generated brand names, extracts legal forms, and parses addresses into words, numbers, house number, unit, city and state (codes, names and native-script names unified). |
| `src/make_folds.py` | 2 | 5 folds stratified by country and number of matches. The unit is the Source-1 entity; every true Source-2/3 match follows its entity and unmatched records stay in every search pool. |
| `src/evaluate.py` | 2 | Official macro-F0.5 scorer (self-test reproduces the README example, 0.714). |
| `src/blocking.py` | 3 | 11 composite key families (name-word pairs, name word x address word, house number x street word, ...), each weighted by rarity. Candidates are the union of top-40 weighted-Jaccard, top-20 raw score and top-10 name-only lists, re-ranked by a cross-fitted gradient-boosting model; top 20 kept. Recall 98.4 %, F0.5 ceiling 0.995. |
| `src/matching.py` | 4 | 52 pair features (fuzzy name/phonetic/glued-name/address similarities, house number/unit/city/state agreement, competition between entities), 22 collective features (agreement with the entity's most confident other match, gaps to competing entities), 11 extra features (house-number truncation vs neighbour, rarity-weighted overlaps, source). XGBoost, 5-fold out-of-fold; test scored by the mean of the 5 fold models. |
| `src/decide.py` | 5 | One-to-one constraint (a record goes only to the entity that scores it highest) + per-entity expected-F0.5 choice of how many candidates to predict. Tuned on out-of-fold predictions. |

Country is never a model feature and no English-only word list is used for matching, so the model
applies unchanged to France (present only in the test set).

## Checks and analysis tools

| file | purpose |
|---|---|
| `src/test_normalize.py` | 114 unit tests of the cleaning on real records |
| `src/verify_cleaning.py` | quality of cleaned fields on train pairs vs hard negatives (AUC raw -> clean) |
| `src/check_cleaned_files.py` | integrity of the cleaned TSV files (rows, order, round trip) |
| `src/check_blocking.py` | integrity of the blocking outputs (ids, countries, duplicates, per-country stats) |
| `src/error_analysis.py` | splits the F0.5 loss into false merges / threshold misses / blocking misses |
| `src/review_samples.py` | prints raw vs cleaned records for manual review |
