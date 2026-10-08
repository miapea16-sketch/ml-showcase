from churn.src.model import (
    best_result,
    load_data,
    make_features,
    predict_customer,
    train_and_compare,
)


def test_real_dataset_loads_and_target_is_binary():
    frame = load_data()
    assert len(frame) == 7043
    assert set(frame["Churn"].unique()) == {0, 1}
    assert frame["TotalCharges"].isna().sum() == 0


def test_feature_engineering_adds_domain_features():
    frame = load_data()
    features = make_features(frame)
    assert "tenure_band" in features
    assert "avg_monthly_spend" in features
    assert "customerID" not in features


def test_models_train_and_score_a_customer():
    results, _, _ = train_and_compare()
    winner = best_result(results)
    assert len(results) == 2
    assert winner.metrics["roc_auc"] > 0.70
    prediction, probability = predict_customer(
        winner.model,
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
            "TotalCharges": 170.0,
        },
    )
    assert prediction in {0, 1}
    assert 0.0 <= probability <= 1.0
