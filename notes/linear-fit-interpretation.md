# What a Straight Line Says About a Curve

What do the slope and intercept of a linear regression mean when the true relationship is not linear?
A tiny worked example: the data follow `y = x^2` for `x = 0..10`, and we fit

```text
y_hat = intercept + slope * x
```

Code: [train_linear_model.py](code/linear-fit/train_linear_model.py) (closed-form least squares in plain Python),
[train_sklearn_model.py](code/linear-fit/train_sklearn_model.py) (the same fit with scikit-learn), and
[data.csv](code/linear-fit/data.csv).

For `x = 0..10`, the fitted model is:

```text
y_hat = -15 + 10x
```

So:

- `intercept = -15`
- `slope = 10`

The slope means that, according to this fitted straight-line model, a one-unit increase in `x` is associated with a `10` unit increase in the predicted mean of `y`.

The intercept means that, according to the fitted model, when `x = 0`, the predicted mean of `y` is `-15`.

The negative intercept does not mean a 1-unit increase in `x` changes fitted `y` by less than `10`. The intercept shifts both fitted predictions by the same amount, so it cancels when comparing `x` and `x + 1`:

```text
y_hat(x)     = -15 + 10x
y_hat(x + 1) = -15 + 10(x + 1)

y_hat(x + 1) - y_hat(x)
= [-15 + 10(x + 1)] - [-15 + 10x]
= 10
```

Examples using the fitted line:

```text
x = 0: y_hat = -15 + 10(0) = -15
x = 1: y_hat = -15 + 10(1) =  -5
change = -5 - (-15) = 10

x = 5: y_hat = -15 + 10(5) = 35
x = 6: y_hat = -15 + 10(6) = 45
change = 45 - 35 = 10

x = 9:  y_hat = -15 + 10(9)  = 75
x = 10: y_hat = -15 + 10(10) = 85
change = 85 - 75 = 10
```

However, the real rule is `y = x^2`, which is not linear. In the real rule, the unit change in `y` depends on where you are:

```text
from x to x + 1:
(x + 1)^2 - x^2 = 2x + 1
```

Examples:

- From `x = 0` to `x = 1`, `y` changes by `1`
- From `x = 5` to `x = 6`, `y` changes by `11`
- From `x = 9` to `x = 10`, `y` changes by `19`

That is why the fitted slope `10` should be understood as an average linear summary of the curved relationship over the training range, not as the exact change at every value of `x`.
