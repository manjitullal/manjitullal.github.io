# Sample-Selection Bias Correction Notes

Paper: **Sample Selection Bias Correction Theory**  
URL: https://arxiv.org/abs/0805.2775

## High-Level Idea

Sample-selection bias happens when the labeled training data does not have the
same feature distribution as the population where the model will be used.

```text
Training distribution: P_train(x)
Target distribution:   P_target(x)
```

If these are different, a normal model will optimize too much for regions that
are overrepresented in the training set and too little for regions that are
underrepresented.

The correction idea is to keep the same training rows, features, and labels, but
change how much each row contributes to the loss.

```text
unweighted loss = sum loss(model(x_i), y_i)

weighted loss   = sum w_i * loss(model(x_i), y_i)
```

Rows that are rare in the biased training sample but common in the target
population get larger weights. Rows that are too common in the biased training
sample get smaller weights.

## Importance Weighting

The ideal weight is a density ratio:

```text
w(x) = P_target(x) / P_train(x)
```

If a region is twice as common in the target population as in the training data,
its rows should count about twice as much. If a region is overrepresented in
training, its rows should count less.

In practice, we usually do not know these probabilities, so the paper discusses
ways to estimate useful weights from data.

## Simple Cluster / Frequency Weighting

This is the technique implemented in:

```text
[bias_correction_cluster_weighting.py](code/sample-selection-bias/bias_correction_cluster_weighting.py)
```

The implementation uses an unlabeled target sample plus the biased labeled
training sample.

Steps:

1. Split the feature space into clusters or buckets.
2. Count how often each cluster appears in the target sample.
3. Count how often each cluster appears in the biased training sample.
4. Assign each training row the ratio for its cluster.

Formula:

```text
weight(cluster) =
    target share of cluster
    /
    biased training share of cluster
```

Example:

```text
target share = 0.125
biased share = 0.040

weight = 0.125 / 0.040 = 3.125
```

That means rows in that cluster are underrepresented in the training sample, so
they count about 3.1x during model fitting.

The model then uses the same features and labels as before:

```python
model.fit(x_train, y_train, sample_weight=weights)
```

Only the per-row contribution to the training loss changes.

## Kernel Mean Matching

Kernel Mean Matching, or KMM, estimates one weight per training sample directly.
It does not first put rows into buckets.

The goal is:

```text
weighted mean of training samples in kernel feature space
approximately equals
mean of target samples in kernel feature space
```

More formally:

```text
choose weights w to minimize:

|| (1 / n_train) * sum_i w_i * phi(x_train_i)
   -
   (1 / n_target) * sum_j phi(x_target_j) ||^2
```

`phi(x)` is the feature representation induced by the kernel. We usually do not
compute `phi(x)` directly. Instead, we compute similarities with a kernel such
as an RBF kernel.

After expanding the squared distance, the optimization over weights becomes:

```text
minimize:

1/2 * w^T K w - kappa^T w
```

Where:

```text
K[i, j] = kernel(x_train_i, x_train_j)

kappa[i] =
    how similar training row i is to the target sample overall
```

Constraints keep the learned weights reasonable:

```text
0 <= w_i <= B
sum_i w_i approximately equals n_train
```

So KMM learns:

```text
overrepresented training rows   -> smaller weight
underrepresented training rows  -> larger weight
already representative rows     -> weight near 1
```

Then the supervised model is trained normally, except with those learned
weights:

```python
weights = learn_kmm_weights(x_train, x_target)
model.fit(x_train, y_train, sample_weight=weights)
```

## Difference Between Cluster Weighting And KMM

Cluster weighting:

```text
group rows into buckets
estimate one weight per bucket from frequency ratios
simple and easy to explain
```

KMM:

```text
learn one weight per training row
match weighted training distribution to target distribution in kernel space
more flexible, but requires solving an optimization problem
```

## Key Takeaway

Bias correction here does not change the labels, does not change the features,
and does not synthesize new rows.

It changes the training objective so the model behaves as if the labeled sample
came from the target distribution.

