import numpy as np

from src.model import build_pipeline, load_data, predict_one, train_model


def test_dataset_shape_and_labels():
    features, labels, class_names = load_data()
    assert features.shape == (150, 4)
    assert labels.shape == (150,)
    assert len(class_names) == 3


def test_pipeline_trains_and_predicts_probabilities():
    model = build_pipeline()
    features, labels, _ = load_data()
    model.fit(features, labels)
    predicted, probabilities = predict_one(model, [5.1, 3.5, 1.4, 0.2])
    assert predicted == "setosa"
    assert np.isclose(probabilities.sum(), 1.0)


def test_training_is_reproducible():
    _, first, _ = train_model()
    _, second, _ = train_model()
    assert first.accuracy == second.accuracy
    assert np.array_equal(first.confusion, second.confusion)
