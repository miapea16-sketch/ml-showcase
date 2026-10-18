# ML Showcase

A two-project machine-learning portfolio showing progression from fundamentals to applied tabular ML.

## Projects

### 1. [Iris Classifier](.) — fundamentals

A clean multiclass classification workflow with a scikit-learn pipeline, holdout evaluation, unit tests, CI, and a Streamlit demo.

### 2. [Customer Churn Risk Lab](churn/) — next level

A realistic applied-ML project using IBM's public Telco Customer Churn sample. It adds mixed-type preprocessing, missing-value handling, domain feature engineering, model comparison, class-imbalance-aware metrics, permutation importance, and an interactive risk-scoring dashboard.

### 3. [Tiny GPT from Scratch](tiny_gpt/) — deep-learning foundations

An educational character-level GPT-style Transformer built with PyTorch primitives. It makes tokenization, causal self-attention, positional embeddings, residual blocks, next-token loss, and autoregressive generation visible in a small laptop-friendly project.

## Why this progression matters

The Iris project shows that the fundamentals are understood. The churn project shows how those fundamentals change when the data is messy and the business question matters: model choice, recall, thresholds, explainability, and limitations all need to be considered.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"

# Fundamentals demo
streamlit run app.py

# Next-level demo
streamlit run churn/app.py
```

Each project includes its own README and tests. See [churn/README.md](churn/README.md) for the full applied-ML walkthrough, or [tiny_gpt/README.md](tiny_gpt/README.md) for the Transformer learning project.
