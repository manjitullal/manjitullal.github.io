import csv
from pathlib import Path


DATA_FILE = Path(__file__).with_name("data.csv")


def load_dataset(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = [(float(row["x"]), float(row["y"])) for row in reader]

    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    return xs, ys


def fit_simple_linear_regression(xs, ys):
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return intercept, slope


def predict(intercept, slope, x):
    return intercept + slope * x


def main():
    xs, ys = load_dataset(DATA_FILE)
    intercept, slope = fit_simple_linear_regression(xs, ys)

    print("Dataset: y = x^2 for x = 0..10")
    print(f"Trained linear model: y_hat = {intercept:.2f} + {slope:.2f}x")
    print(f"Intercept: {intercept:.2f}")
    print(f"Slope: {slope:.2f}")
    print()
    print("Interpretation:")
    print(
        "For this fitted straight-line model, a 1-unit increase in x changes "
        f"the predicted mean y by {slope:.2f} units."
    )
    print(
        "The intercept is the model's predicted mean y when x = 0. "
        f"Here it is {intercept:.2f}."
    )
    print()
    print("Example predictions from the fitted linear model:")
    for x in [0, 1, 5, 10]:
        actual = x**2
        predicted = predict(intercept, slope, x)
        error = actual - predicted
        print(
            f"x={x:>2}: actual y={actual:>6.2f}, "
            f"predicted y={predicted:>6.2f}, error={error:>7.2f}"
        )
    print()
    print(
        "Important: the real data-generating rule is curved: y = x^2. "
        "Its true unit change is not constant. From x to x+1, y changes by "
        "2x + 1. The fitted slope is only the best average straight-line "
        "summary over this dataset."
    )


if __name__ == "__main__":
    main()
