# Otto23
Kaggle Tabular Competition - [OTTO – Multi-Objective Recommender System 2023](https://www.kaggle.com/competitions/otto-recommender-system/overview)
<br> - My refactored solution to get top 7% (Bronze) in this Kaggle competition
<br> - Uses cudf and polars to speed up processing

# Leaderboard Scores
Scores should be about the same as below. Slight variations are due to covisit creation
| Architecture     | Private | Public  |
| ---------------- |:-------:| :------:|
| Covisit          | 0.57828 | 0.57807 |
| Covisit + Ranker | 0.58047 | 0.57966 |

## Quick start
For covisit predictions only, run steps 1-5.
<br> For covisit + LGBMRanker predictions, run steps 1-6.
1) Download data. Unzip in folder data: [Kaggle Otto data](https://www.kaggle.com/competitions/otto-recommender-system/data)
2) Run NB 01_01 - preprocess json data into parquets. Also saves aid type dictionaries as pickle files
3) Run NB 01_02 - Creates local train and validation data files. For local val., we train on weeks 1-3, and validate on wk 4. Wk4 validation is created based on "Train/Test Split" on the [Otto Github repo](https://github.com/otto-de/recsys-dataset). Also, the final train set is created by using weeks 2-4 for train, and the original test set (week 5)
4) Run NB 01_03 - Create smaller versions of validation data (5 / 10 / 25 / 50 / 100 %). This lets you iterate faster. Pick smallest val. data that meets your need. 5% is a good start point
5) Run NB 02_01 - Creates covisit matrices (function = preprocess_covisits), and saves them as pickle files. Set CV_NUM to val. data split as preferred. Set DO_LOCAL_VALIDATION (set False if you want to submit to Kaggle Otto). This notebook only submits predictions based on covisitation matrices
6) RUN NB 03_01 - Extends off of NB 02_01. Get predictions with LGBMRanker. NB takes 50 covisitation candidates, and finds the top 20 for each aid type. Creates user and item features. Need to sweep each aid type for optimal HPs. Currently set only for 'carts' and 'orders'. Similar to NB 02_01, need to set CV_NUM and DO_LOCAL_VALIDATION. Final submissions use LGBMRanker predictions

## Todos
- Update NB 02_01 with code changes from NB 03_01. Should be minor code cleanup
- Refactor suggest_carts function into otto_utils.py while maintaining same process speed. Starmap?
- Add a blurb on how validation data is created

## Acknowledgements
Thanks all
