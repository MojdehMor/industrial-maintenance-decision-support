# Industrial maintenance decision-support framework

This repository contains publicly shareable, modular **research code** for an
industrial maintenance decision-support study. It includes algorithmic
implementations for intervention categorization, prediction of numerical
maintenance outcomes, maintenance-action recommendation, conditional
circularity re-ranking, and aggregate evaluation.

**No original industrial data, synthetic datasets, runnable examples,
notebooks, case-level predictions, trained models, annotation workbooks,
whitelists, or inventory records are included.

## Modules

- `modules/categorization.py`: Ordered rule-based action categorization,
  primary/secondary/secondary-required labels, feature preprocessing and
  classification model definitions.
- `modules/regression.py`: Numerical outcome extraction, mixed-feature
  preprocessing and regression model definitions for duration/cost tasks.
- `modules/recommender.py`: Popularity scores, configurable content retrieval,
  SBERT case-based retrieval, internal text+metadata logistic regression,
  weighted fusion and action ranking.
- `modules/circularity.py`: Conditional relative-score-gap reranking.
- `evaluation/cross_validation.py`: Aggregate classification and regression
  cross-validation evaluation.
- `evaluation/ablation_study.py`: Aggregate out-of-fold component comparisons.
- `evaluation/metrics.py`: Ranking and aggregation metrics.

## Methodology and limitations

The recommender research uses a pretrained SBERT sentence encoder for
incident-text retrieval, an optional categorical business-type gate, nearest
neighbour voting (default 15), and configurable content attributes (age,
business type, and affected component). 
Evaluation supports shuffled five-fold KFold or strict stratified folds,
explicitly configured by the researcher, and reports aggregate Top-1,
HR@k, Precision@k, nDCG@k, MRR and coverage. Numeric outcomes can be assessed
using MAE, RMSE and R²; classification uses accuracy and F1 scores.

The circularity module requires caller-supplied replacement and preservation
categories. No confidential original category-to-action mapping is included.


## Requirements and integration

The code targets Python 3.10+ and uses NumPy, pandas, scikit-learn and, for the
recommender, sentence-transformers. Required input column names and interfaces
are defined in module function signatures and docstrings. 


