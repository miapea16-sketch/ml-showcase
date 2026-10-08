"""Interactive Streamlit demo for the Iris classifier."""

import pandas as pd
import streamlit as st

from src.model import predict_one, train_model

st.set_page_config(page_title="Iris ML Showcase", page_icon="🔬", layout="wide")

st.title("Iris Classifier")
st.caption(
    "A small, end-to-end machine-learning project: "
    "data → pipeline → evaluation → prediction"
)

@st.cache_resource
def get_model():
    return train_model()

model, evaluation, class_names = get_model()

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Holdout accuracy", f"{evaluation.accuracy:.1%}")
metric_b.metric("Training examples", "120")
metric_c.metric("Classes", str(len(class_names)))

st.divider()

left, right = st.columns([1, 1.2], gap="large")
with left:
    st.subheader("Try a prediction")
    st.write("Adjust the four measurements in centimeters.")
    measurements = [
        st.slider("Sepal length", 4.3, 7.9, 5.8, 0.1),
        st.slider("Sepal width", 2.0, 4.4, 3.0, 0.1),
        st.slider("Petal length", 1.0, 6.9, 4.3, 0.1),
        st.slider("Petal width", 0.1, 2.5, 1.3, 0.1),
    ]
    predicted_class, probabilities = predict_one(model, measurements)
    st.success(f"Predicted species: **{predicted_class.title()}**")
    st.caption(f"Model confidence: {probabilities.max():.1%}")

with right:
    st.subheader("Class probabilities")
    probability_frame = pd.DataFrame(
        {"species": [name.title() for name in class_names], "probability": probabilities}
    ).set_index("species")
    st.bar_chart(probability_frame)
    st.subheader("What the model learned")
    st.write(
        "The pipeline standardizes measurements before fitting multinomial logistic regression. "
        "Keeping preprocessing and the estimator together prevents train/test leakage."
    )

with st.expander("See evaluation details"):
    report = pd.DataFrame(evaluation.report).T.round(3)
    st.dataframe(report, use_container_width=True)
    st.write("Confusion matrix — rows are true classes, columns are predicted classes")
    st.dataframe(
        pd.DataFrame(evaluation.confusion, index=class_names, columns=class_names),
        use_container_width=True,
    )
