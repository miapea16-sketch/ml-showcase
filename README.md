# ML Showcase: Iris Classifier

A compact, portfolio-ready machine-learning project that demonstrates the complete supervised-learning workflow:

- clean data loading with scikit-learn's Iris dataset
- stratified train/test splitting
- a reproducible preprocessing + logistic-regression pipeline
- evaluation with accuracy, classification report, and confusion matrix
- an interactive Streamlit demo for live predictions
- unit tests and GitHub Actions CI

## Why this project is useful

This is intentionally small enough to understand in one sitting, but structured like a real ML project. The model never sees the test set during training, preprocessing is part of the pipeline, and every result can be reproduced from a fixed random seed.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
python -m src.train
pytest
streamlit run app.py
```

Then open the local URL printed by Streamlit. Use the sliders to classify a flower from its sepal and petal measurements.

## Project structure

```text
.
├── app.py                    # Interactive demo UI
├── src/
│   ├── model.py              # Dataset, pipeline, metrics, prediction helpers
│   └── train.py              # Reproducible training/evaluation entry point
├── tests/
│   └── test_model.py         # Fast smoke and behavior tests
├── .github/workflows/ci.yml  # Automated checks on every push/PR
├── pyproject.toml            # Dependencies and tool configuration
└── LICENSE
```

## Model snapshot

The Iris dataset has 150 labeled examples across three species: setosa, versicolor, and virginica. The app uses four measurements in centimeters:

| Feature | Meaning |
|---|---|
| `sepal length` | Length of the sepal |
| `sepal width` | Width of the sepal |
| `petal length` | Length of the petal |
| `petal width` | Width of the petal |

The baseline model is a `StandardScaler` followed by multinomial logistic regression. On the standard stratified holdout split it typically reaches about 97–100% accuracy; exact metrics are printed by the training command.

## What to discuss in a portfolio walkthrough

1. **Problem framing:** multiclass classification with four numerical features.
2. **Leakage prevention:** scaling is fitted only inside the training pipeline.
3. **Evaluation:** accuracy is supplemented with per-class precision/recall/F1.
4. **Limitations:** Iris is small, clean, and unusually easy; production data needs monitoring, validation, and governance.
5. **Next step:** replace the built-in dataset with a versioned data source and compare calibrated models.

## License

MIT — see [LICENSE](LICENSE).
