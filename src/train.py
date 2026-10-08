"""Command-line training and evaluation entry point."""

from .model import train_model


def main() -> None:
    _, evaluation, class_names = train_model()
    print(f"Accuracy: {evaluation.accuracy:.3f}")
    print("\nClassification report:")
    for label in class_names:
        row = evaluation.report[label]
        print(
            f"  {label:>10} | precision {row['precision']:.3f} "
            f"| recall {row['recall']:.3f} | f1 {row['f1-score']:.3f}"
        )
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(evaluation.confusion)


if __name__ == "__main__":
    main()
