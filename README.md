# SCORIFY — AI-Powered Lead Scoring Platform

Scorify is a machine learning platform that predicts lead conversion probability and expected profit for incoming sales leads. Built using real CRM data from Go2Africa, a luxury African safari travel company.

## Live Demo

[https://scorify.streamlit.app](https://scorify-pfrfczc4cvsrwznwqmkmxr.streamlit.app/)

## What It Does

Enter lead details (source, device, budget, engagement signals) and get instant predictions:

- **Conversion Probability** — how likely this lead is to convert (0-100%)
- **Predicted Profit** — expected profit if they convert
- **Lead Score** — HOT / WARM / COLD classification
- **Recommended Action** — what the sales team should do next

| Score | Conversion | Action |
|-------|-----------|--------|
| HOT   | 20%+      | Senior rep calls within 2 hours |
| WARM  | 5-20%     | Sales rep follows up within 24 hours |
| COLD  | 0-5%      | Add to email nurture campaign |

## Models Used

| Target | Model | R² Score |
|--------|-------|----------|
| Conversion Probability | XGBoost Regressor | 0.720 |
| Profit Prediction | Random Forest Regressor | 0.759 |

## How to Run Locally

```bash
pip install -r requirements.txt
python train_and_save.py
streamlit run app.py
```

## How to Deploy on Streamlit Cloud

1. Fork or clone this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Select this repo → branch `main` → file `app.py`
5. Click Deploy

## Project Structure

```
scorify/
├── app.py                        # Streamlit web interface
├── train_and_save.py             # Model training script
├── requirements.txt              # Python dependencies
├── scorify_cleaned_dataset.csv   # Cleaned dataset (2,458 leads)
├── model_conv.pkl                # Trained XGBoost (conversion)
├── model_profit.pkl              # Trained Random Forest (profit)
├── encoders.pkl                  # Label encoders for categorical inputs
├── feature_names.pkl             # Feature column names and order
└── README.md
```

## Dataset

- Source: Go2Africa CRM (HubSpot via BigQuery)
- Size: 2,458 leads × 37 columns
- Targets: conversion probability, predicted profit
- Features: marketing source, device, budget, engagement signals, sales pipeline data

## Tech Stack

- Python
- scikit-learn (Random Forest, Label Encoding, metrics)
- XGBoost (gradient boosting regressor)
- Streamlit (web interface)
- pandas / numpy (data processing)
