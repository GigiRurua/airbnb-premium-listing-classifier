# Airbnb NYC Premium Listing Classifier

Predicting whether an Airbnb NYC listing qualifies as a premium (top-quartile price) listing, comparing a tuned Decision Tree against a feedforward neural network. Built as my Break Through Tech AI Program capstone.

## Overview

A platform that manually reviews listings to flag "premium" ones is slow and inconsistent between reviewers. This project builds a model that applies the same standard to every listing automatically — freeing up staff time and giving hosts a consistent, predictable bar to meet.

**Problem type:** Binary classification
**Label:** `price_category` — whether a listing's price falls in the top 25th percentile (`high`) or not (`low`)
**Dataset:** 28,022 NYC Airbnb listings, 53 features after preprocessing

## Methodology

**EDA & Ethical Review**
Identified moderate class imbalance (~3:1 low-to-high), and flagged neighborhood as a potential proxy variable for race and socioeconomic status given NYC's housing segregation history — a concern that surfaced directly in the final feature importances (see Key Findings).

**Data Preparation**
- Dropped `price` (label leakage), identifiers, URLs, and free-text fields
- Mean-imputed missing numeric values with a missingness flag; filled missing categoricals as `'unavailable'`
- One-hot encoded low-cardinality categoricals; dropped high-cardinality ones (e.g. raw neighborhood) to avoid sparse noise
- 80/20 stratified train/test split (22,417 train / 5,605 test)

**Modeling**
1. Baseline Decision Tree
2. Hyperparameter tuning via 5-fold GridSearchCV (`max_depth`, `min_samples_leaf`, optimized for F1)
3. Final tuned Decision Tree
4. Feedforward neural network (Keras/TensorFlow) — 3 hidden layers (64 → 32 → 16, ReLU), sigmoid output, SGD optimizer (lr=0.1), 100 epochs, as a second model for comparison

F1 was tracked alongside accuracy throughout, since a model that always predicted "low" would score ~75% accuracy while being useless — F1 forces the model to actually catch the minority premium class.

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Decision Tree (baseline) | 0.800 | 0.615 |
| Decision Tree (tuned: `max_depth=8`, `min_samples_leaf=5`) | **0.836** | 0.627 |
| Neural Network (3-layer, SGD) | 0.819 | **0.650** |

Neither model dominates outright. The Decision Tree wins on accuracy; the neural network wins on F1 — meaning it does comparatively better at catching the minority "premium" class, which matters more for this business problem, since a missed premium flag (false negative) costs a specific host real income.

**Top 5 Decision Tree feature importances:**
1. `accommodates` (0.363)
2. `bathrooms` (0.089)
3. `room_type: Private room` (0.087)
4. `neighbourhood_group: Manhattan` (0.071)
5. `instant_bookable` (0.055)

## Key Findings

- Capacity and room-type features dominate the model's decisions — a reasonable, defensible signal for "premium."
- **Manhattan location lands as the 4th most important feature.** This is exactly the fairness risk flagged during EDA: the model may be partially leaning on location as a proxy rather than purely on listing quality. Before recommending this model for deployment, I'd want to check the false-negative rate broken out by neighborhood, not just the aggregate F1 score.
- The accuracy/F1 tradeoff between the two models isn't noise — it reflects a real difference in which class each model is better at catching, which matters more than the single point of accuracy the Decision Tree wins by.

## Next Steps
- Validate the accuracy/F1 gap holds under cross-validation rather than a single 80/20 split
- Address class imbalance directly (class weighting or SMOTE) instead of relying on F1 alone
- Engineer features from the text columns dropped during prep (e.g. description sentiment/length)
- Try dropout layers and a wider learning-rate/epoch sweep on the neural network

## Tech Stack
Python · pandas · NumPy · scikit-learn · TensorFlow/Keras · Matplotlib · Seaborn

## Individual Contribution
Completed independently as a Break Through Tech AI Program capstone (individual assignment) — problem definition, EDA, data preparation, both models, evaluation, and ethical analysis are my own work.

## Running This Project
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
jupyter notebook Capstone_Part2_completed.ipynb
```
Data: `airbnbData_train.csv` (included in this repo).
