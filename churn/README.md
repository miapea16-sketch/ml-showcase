# Customer Churn Risk Lab

A next-level machine-learning project built on IBM's public Telco Customer Churn sample. It predicts whether a customer is likely to leave and demonstrates the parts that make applied ML harder than a toy classifier:

- mixed numeric and categorical data
- missing-value handling
- domain feature engineering
- class imbalance
- model comparison
- risk-oriented metrics
- permutation-based explainability
- interactive scoring with Streamlit

## Run it

From the repository root:

```bash
pip install -e ".[dev]"
python -m churn.src.train
streamlit run churn/app.py
pytest churn/tests -q
```

The dataset is included at `data/telco_churn.csv` so the project is reproducible offline.

## What makes this intermediate-level work

The project compares balanced logistic regression and random forest models using the same preprocessing discipline. It selects the model by holdout ROC AUC, but also reports recall and precision because a churn-retention team cares about missed at-risk customers, not just overall accuracy.

The engineered features include:

- `tenure_band`: groups customers by relationship age
- `avg_monthly_spend`: total charges divided by tenure

The app intentionally surfaces the threshold limitation: a real business would choose a threshold using outreach capacity and the cost of false positives versus false negatives.

## Data source and limitations

This is IBM's fictional telco sample dataset, containing 7,043 customer records and a `Churn` label. It is a benchmark for learning, not a claim about a real telecommunications company. Do not use this demo to make consequential decisions about real customers without privacy review, fairness analysis, monitoring, and human oversight.

Source: [IBM Telco Customer Churn sample repository](https://github.com/IBM/telco-customer-churn-on-icp4d). The source repository is archived; the included CSV is retained here for reproducibility.

## Suggested next upgrades

1. Tune the decision threshold against a defined retention budget.
2. Add cross-validation and confidence intervals.
3. Track experiments with MLflow or Weights & Biases.
4. Add fairness checks across demographic groups.
5. Deploy the scorer behind an API with monitoring.
