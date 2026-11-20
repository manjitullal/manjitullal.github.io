import csv
from pathlib import Path


try:
    from sklearn.linear_model import LinearRegression
except ModuleNotFoundError as error:
    raise SystemExit(
        "scikit-learn is not installed. Install it with:\n"
        "python3 -m pip install -r Concepts/regression/requirements.txt"
    ) from error


DATA_FILE = Path(__file__).with_name("data.csv")


def load_dataset(path):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = [(float(row["x"]), float(row["y"])) for row in reader]

    x_values = [[row[0]] for row in rows]
    y_values = [row[1] for row in rows]
    return x_values, y_values


def main():
    x_values, y_values = load_dataset(DATA_FILE)

    model = LinearRegression()
    model.fit(x_values, y_values)

    intercept = model.intercept_
    slope = model.coef_[0]

    print("Dataset: y = x^2 for x = 0..10")
    print(f"Sklearn linear model: y_hat = {intercept:.2f} + {slope:.2f}x")
    print(f"Intercept: {intercept:.2f}")
    print(f"Slope: {slope:.2f}")
    print()
    print(
        "For this fitted sklearn model, a 1-unit increase in x changes "
        f"the predicted mean y by {slope:.2f} units."
    )
    print(
        "The negative intercept does not reduce the unit change. It shifts "
        "all fitted predictions down by 15, but it appears in both predictions "
        "and cancels when you compare x with x+1."
    )
    print()
    print("Comparing actual y with fitted predictions:")
    for x in [0, 1, 5, 9]:
        next_x = x + 1

        actual_y = x**2
        actual_next_y = next_x**2
        actual_change = actual_next_y - actual_y

        predicted_y = model.predict([[x]])[0]
        predicted_next_y = model.predict([[next_x]])[0]
        predicted_change = predicted_next_y - predicted_y

        print(
            f"x={x:>2} to x={next_x:>2}: "
            f"actual y {actual_y:>6.2f} -> {actual_next_y:>6.2f} "
            f"(change {actual_change:>5.2f}); "
            f"fitted y_hat {predicted_y:>6.2f} -> {predicted_next_y:>6.2f} "
            f"(change {predicted_change:>5.2f})"
        )
    print()
    print(
        "Because the true relationship is y = x^2, this slope is an average "
        "straight-line summary over the training range, not the exact local "
        "change at every x."
    )


if __name__ == "__main__":
    main()
