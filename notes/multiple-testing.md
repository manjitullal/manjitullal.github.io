# Multiple Testing in A/B Experiments: When to Correct, and How

## 1. Start with one hypothesis test

Suppose a content platform launches a new ranking model, and we want to know whether it improves saves per user.

\[
H_0: \mu_T=\mu_C \qquad H_1: \mu_T\ne\mu_C
\]

We choose a significance level \(\alpha=0.05\).

**What that means:** if the new model truly has no effect, the test still has a 5% probability of
incorrectly declaring a statistically significant difference. This is a **Type I error**, or false positive.

| One hypothesis test (null true, test calibrated) | Probability |
|---|---:|
| No false positive | 95% |
| False positive | 5% |

So far, nothing unusual.

## 2. Now test five metrics

Imagine the new model actually improves none of the metrics, but we run five hypothesis tests anyway:

| Metric | Probability of false positive |
|---|---:|
| Saves | 5% |
| Clicks | 5% |
| Shares | 5% |
| Session duration | 5% |
| Hide rate | 5% |

Each test individually has a 5% false-positive probability. Now consider this decision rule:

> Launch the new model if **any one** of the five metrics is statistically significant.

We now have five opportunities to observe a false positive. Assuming independent tests:

\[
P(\text{no false positives})=0.95^5=0.774
\]

\[
P(\text{at least one false positive})=1-0.95^5=\boxed{22.6\%}
\]

The probability grows quickly with the number of tests (all nulls true, independent tests, \(\alpha=0.05\) each):

| Number of tests \(m\) | \(P(\text{at least one false positive}) = 1-0.95^m\) |
|---:|---:|
| 1 | 5.0% |
| 2 | 9.8% |
| 5 | 22.6% |
| 10 | 40.1% |
| 20 | 64.2% |

### Why 22.6% matters

The team may believe it is deciding with a 5% false-positive risk, when the actual probability of
finding at least one false signal is about 22.6%: more than four times as large.

**Careful:** 22.6% is *not* the probability that a particular significant result is false. It is the
probability of at least one false rejection across five tests when all five null hypotheses are true.

Real metrics are usually correlated (clicks, saves, and shares are related behaviors), so the exact
22.6% may not apply. The underlying multiple-testing concern remains.

## 3. The primary metric is where this matters

Compare two experiment designs.

**Scenario A: one primary metric. No automatic correction needed.**
Before the experiment, the team decides that saves per user is the only primary success metric; the
other four are secondary diagnostics or guardrails. Saves is tested at \(\alpha=0.05\). If the experiment
is valid and \(p=0.032\), the primary test is significant at 5%. There is no need to apply Bonferroni
across all five metrics merely because five metrics were reported.

**Scenario B: five chances to claim success. Correction needed.**
The team will declare success if saves, clicks, shares, session duration, *or* hide rate shows a
favorable significant effect. That is five chances to make a success claim, so we need a method that
controls false discoveries across those tests.

Prespecifying one primary metric solves the specific problem of picking whichever metric happens to
look significant. It does **not** solve other issues such as repeated peeking, testing many variants,
or trying many alternative analyses.

## 4. Bonferroni correction

Bonferroni is a simple method for controlling the **family-wise error rate (FWER)**:

\[
\text{FWER} = P(\text{at least one false rejection among a family of tests})
\]

To keep FWER at or below 5% across \(m\) tests, Bonferroni sets

\[
\alpha_{\text{per test}}=\frac{\alpha_{\text{family}}}{m}=\frac{0.05}{5}=0.01
\]

Instead of comparing each p-value against 0.05, compare it against 0.01.

| Metric | p-value | Significant at 0.05? | Significant at 0.01? |
|---|---:|:---:|:---:|
| Saves | 0.032 | Yes | No |
| Clicks | 0.041 | Yes | No |
| Shares | 0.009 | Yes | Yes |
| Session duration | 0.180 | No | No |
| Hide rate | 0.220 | No | No |

Under Bonferroni, only shares remains statistically significant.

### Why dividing by \(m\) works

Each test now has at most a 1% false-positive probability under its null, so by the **union bound**

\[
P(\text{at least one false positive}) \le 0.01+0.01+0.01+0.01+0.01 = 0.05
\]

The union bound holds even when the tests are correlated. The cost: Bonferroni is conservative,
reducing power and making real effects harder to detect.

## 5. When to apply a multiple-testing correction

The key question is not how many metrics you measured. It is:

> How many hypotheses can independently lead to a positive statistical claim or decision?

| Experiment design | Correction? |
|---|---|
| One prespecified primary metric, four descriptive secondary metrics | Not automatically across all five |
| Five metrics, success if any one is significant | Yes, control multiplicity |
| Three equally important primary metrics, success if any one improves | Yes |
| Three required primary metrics, success only if all three improve | Not necessarily; this is a different joint testing rule |
| Ten model variants tested against control | Usually yes, account for multiple comparisons |
| One primary metric checked daily with early stopping | Needs sequential-testing control, not ordinary Bonferroni |

**Other methods:** **Holm's** step-down procedure also controls FWER and is uniformly more powerful
than Bonferroni. **Benjamini–Hochberg** controls the **false discovery rate (FDR)**, the expected share of
false positives among the rejections. FDR is useful when screening many hypotheses, but it is a weaker
guarantee than FWER.

## Related notes

- [A/B test significance, step by step](ab-significance.html)
