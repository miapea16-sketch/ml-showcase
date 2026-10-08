"""Interactive Streamlit dashboard for customer churn risk."""

import streamlit as st

from churn.src.model import best_result, predict_customer, train_and_compare

st.set_page_config(page_title="Churn Risk Lab", page_icon="📊", layout="wide")
st.title("Customer Churn Risk Lab")
st.caption("A realistic tabular ML workflow with model comparison and risk-focused evaluation")


@st.cache_resource
def get_models():
    return train_and_compare()


results, x_test, y_test = get_models()
winner = best_result(results)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Selected model", winner.name.replace("_", " ").title())
metric_b.metric("ROC AUC", f"{winner.metrics['roc_auc']:.1%}")
metric_c.metric("Recall", f"{winner.metrics['recall']:.1%}")

with st.expander("Why recall matters"):
    st.write(
        "For retention outreach, missing a customer who is likely to leave can be more costly "
        "than contacting a customer who stays. This demo therefore reports recall alongside "
        "ROC AUC. "
        "The 0.5 threshold is a starting point, not a final business decision."
    )

left, right = st.columns([1, 1.1], gap="large")
with left:
    st.subheader("Score a customer")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    tenure = st.slider("Tenure (months)", 0, 72, 8)
    monthly = st.slider("Monthly charges", 18.0, 120.0, 70.0)
    total = st.slider(
        "Total charges", 0.0, 9000.0, float(max(monthly * max(tenure, 1), 70)), step=10.0
    )
    tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    predicted, probability = predict_customer(winner.model, customer)
    if predicted:
        st.error(f"Elevated churn risk — estimated probability: **{probability:.1%}**")
    else:
        st.success(f"Lower churn risk — estimated probability: **{probability:.1%}**")

with right:
    st.subheader("Model comparison")
    for result in results.values():
        st.write(result.name.replace("_", " ").title())
        st.dataframe(
            {key: [round(value, 3)] for key, value in result.metrics.items()},
            hide_index=True,
            use_container_width=True,
        )
    st.subheader("Portfolio takeaway")
    st.write(
        "This is closer to production ML than a toy classifier: the data contains mixed types, "
        "missing numeric values, class imbalance, business costs, and a need to compare models "
        "rather than trust the first score."
    )
