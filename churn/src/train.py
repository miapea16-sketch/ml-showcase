"""Train and compare churn models from the command line."""

from .model import best_result, feature_importance, train_and_compare


def main() -> None:
    results, x_test, y_test = train_and_compare()
    print(f"Holdout rows: {len(x_test)}")
    print("\nModel comparison:")
    for result in results.values():
        metrics = " | ".join(f"{key}={value:.3f}" for key, value in result.metrics.items())
        print(f"  {result.name:>20} | {metrics}")
    winner = best_result(results)
    print(f"\nSelected model by ROC AUC: {winner.name}")
    print("\nTop global permutation signals:")
    for row in feature_importance(winner, x_test, y_test).head(8).itertuples():
        print(f"  {row.feature:>20} | {row.importance:+.4f}")


if __name__ == "__main__":
    main()
