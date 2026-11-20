# A/B Testing: Statistical Significance Cheat Sheet

**Goal:** decide whether the observed difference between treatment and control is statistically significant.

## Given data

| Parameter | Control (C) | Treatment (T) |
|---|---:|---:|
| Mean saves per user ($\bar X$) | 4.0 | 4.2 |
| Standard deviation ($s$) | 8 | 8 |
| Sample size ($n$) | 50,000 | 50,000 |

Significance level: $\alpha = 0.05$, two-sided.

---

## Step 1: Calculate the observed effect

$$\hat\Delta = \bar X_T - \bar X_C = 4.2 - 4.0 = \mathbf{0.2}$$

Relative lift = $0.2 / 4.0 =$ **5%**.

## Step 2: Calculate the standard error (SE)

**Purpose:** estimate how much the difference between the group means could vary from random chance alone.

$$SE(\hat\Delta) = \sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}} = \sqrt{\frac{8^2}{50{,}000} + \frac{8^2}{50{,}000}} = \sqrt{0.00128 + 0.00128} = \mathbf{0.0506}$$

## Step 3: Calculate the t-statistic

**Purpose:** measure how many standard errors the observed effect is from the null hypothesis of zero effect.

$$t = \frac{\hat\Delta - 0}{SE(\hat\Delta)} = \frac{0.2}{0.0506} = \mathbf{3.95}$$

- For a two-sided 5% test with a large sample, the critical value is about **±1.96**.
- Since $3.95 > 1.96$, **reject the null hypothesis**.
- Two-sided p-value ≈ **0.00008** (exact: 7.7 × 10⁻⁵).

## Step 4: Calculate the 95% confidence interval

**Purpose:** show the range of plausible values for the treatment effect.

$$CI_{95\%} = \hat\Delta \pm 1.96 \times SE(\hat\Delta) = 0.2 \pm 1.96 \times 0.0506 = 0.2 \pm 0.0992$$

$$\mathbf{CI_{95\%} = [0.101,\ 0.299]}$$

The interval **excludes zero**, which agrees with the significant result from Step 3.

---

## Final interpretation

| Result | Value | Meaning |
|---|---|---|
| t-statistic | **3.95** | Above the critical value of 1.96 |
| 95% CI | **[0.101, 0.299]** | Excludes zero |

**Conclusion:** the treatment increases average saves per user by an estimated **0.2 (+5%)**, and the increase is statistically significant at the 5% level. This assumes independent user-level observations.

## Four formulas to remember

| Metric | Formula |
|---|---|
| Treatment effect | $\hat\Delta = \bar X_T - \bar X_C$ |
| Standard error | $SE = \sqrt{s_T^2/n_T + s_C^2/n_C}$ |
| t-statistic | $t = \hat\Delta / SE$ |
| 95% CI | $\hat\Delta \pm 1.96 \cdot SE$ |

**Shortcut:** effect → SE → t-statistic → CI → interpret significance **and** business impact.

The t-statistic tests the null hypothesis. The confidence interval shows the size of the effect and how uncertain it is.

---

## Check it in Python (no libraries needed)

```python
import math

mean_c, mean_t = 4.0, 4.2
sd_c, sd_t = 8, 8
n_c, n_t = 50_000, 50_000

delta = mean_t - mean_c                                # 0.2
se = math.sqrt(sd_t**2 / n_t + sd_c**2 / n_c)          # 0.0506
t = delta / se                                         # 3.95
p_value = math.erfc(abs(t) / math.sqrt(2))             # two-sided, normal approximation: 7.7e-05
ci = (delta - 1.96 * se, delta + 1.96 * se)            # (0.101, 0.299)
```

## Common follow-up questions

- **Why can we use 1.96 when this is a "t" statistic?** With 50,000 per group the t-distribution is essentially normal. With small samples, use the t critical value with Welch's degrees of freedom.
- **Is significant the same as important?** No. Also check practical significance: is +0.2 saves per user worth the cost of shipping? Compare the CI to the minimum effect worth acting on.
- **When does "independent observations" break?** When the metric is per impression but randomization is per user, or when users influence each other (network effects). Use the delta method or cluster-robust standard errors.
- **Is the SD of 8 vs. a mean of 4 strange?** No. Count metrics like saves are heavily right-skewed (most users save little, a few save a lot). Means are still fine at this n because of the CLT, but variance reduction (CUPED) or capping outliers can make the test much more sensitive.
