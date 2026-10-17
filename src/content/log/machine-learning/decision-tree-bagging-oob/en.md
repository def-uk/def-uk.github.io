---
title: "Decision Tree, Bagging and OOB"
date: 2026-09-01T22:59:00+09:00
description: "Why decision trees are unstable, how bagging turns that instability into stability with bootstrap and aggregation, and what OOB really estimates."
tags: ["Bagging", "ML", "MachineLearning", "ensemble"]
draft: false
---
>### Learning goals
To reduce the high variance of a **Decision Tree**, **Bagging** adds randomness to the data, **Random Forest** also adds randomness to the feature candidates, and **Extra Trees** adds randomness all the way down to the split threshold.

A decision tree naturally expresses non-linear relationships and interactions between variables. It needs relatively little preprocessing, and people can follow its rules.

But it has one big weakness: **if the training sample changes even slightly, the tree's structure and predictions can change a lot.**

Starting from this problem, this post answers the following questions.

1. Why does a decision tree wobble?
2. Why does bagging build many trees?
3. What do bootstrap and aggregation each do?
4. Where does OOB come from, and what does it evaluate?

---

## 1. Why are decision trees unstable?

### 1.1 One top-level split changes the whole structure

At each node, a decision tree greedily (Greedy Algorithm) chooses the question that splits the data best.

```text
petal width < 0.8인가?
```

If the best candidates have similar scores, adding or removing just a few training examples can change which split is chosen. When a split near the top changes, the data its child nodes receive changes too, so the branch structure below changes in a cascade.

```text
원본 데이터 D      → Tree A → Decision Boundary A
조금 바뀐 데이터 D' → Tree B → Decision Boundary B
```

In statistical language, this property is described as **high variance**.

Here, variance does not mean the variance of a data column.

> If we drew a new training sample from the same population and retrained the model, how much would the predictions change?

A deep tree can have low bias because it captures complex patterns, but it easily follows chance variation and noise in the sample as well, so its variance can grow.

```text
복잡하고 깊은 Tree: bias가 낮을 수 있지만 variance가 커질 수 있음
단순하고 얕은 Tree: variance는 작아질 수 있지만 bias가 커질 수 있음
```

This is less an absolute law than a typical trade-off that appears when you tune model complexity.

### 1.2 Couldn't we just make one tree simpler?

Constraints like `max_depth` and `min_samples_leaf` make a tree simpler and can reduce the wobble. But restricting it too much can miss important non-linear relationships and cause underfitting.

Bagging asks a different question.

> Without giving up much of each tree's expressive power, can we cancel out the wobble of many trees when we combine them?

The answer to that question is **Bootstrap Aggregating**, or **Bagging** for short.

---

## 2. Bagging: let them wobble differently, then average

Bagging trains base models of the same kind on different bootstrap samples and combines their predictions.

```text
원본 학습 데이터
  ├─ Bootstrap 표본 1 → Tree 1 ─┐
  ├─ Bootstrap 표본 2 → Tree 2 ─┼→ 평균 또는 투표
  └─ Bootstrap 표본 3 → Tree 3 ─┘
```

No tree inherits the result of the previous one. So **they can be trained separately without sequential dependency, which makes them easy to parallelize.**

Saying "the trees are statistically independent" would not be accurate, though. They share the same original data and features, so their prediction errors can be correlated. That correlation problem is what Random Forest, in the next post, deals with.

### 2.1 Copying the same tree is not enough

If you build 100 trees with the same data, the same algorithm and the same settings, you may get identical or very similar trees.

```text
같은 학습 데이터
  ├─ Tree 1 ─┐
  ├─ Tree 2 ─┤ 비슷한 예측
  └─ Tree 3 ─┘
```

Collecting 100 models that make the same mistakes gives averaging only a limited effect. Bagging first changes the training sample, giving each model a different learning experience.

---

## 3. Bootstrap sampling: building different training samples

Bootstrap draws from the original training data **with replacement**.

Suppose the original data looks like this.

```text
[A, B, C, D, E]
```

A bootstrap sample of size 5 might come out like this.

```text
표본 1: [A, A, C, D, E]
표본 2: [B, B, C, C, E]
표본 3: [A, B, D, D, D]
```

Because an observation that was drawn can be drawn again, duplicates appear. Conversely, some observations may never be selected.

### What `max_samples=0.8` actually means

Suppose there are 1,000 training examples and you set the following.

```python
max_samples=0.8
bootstrap=True
```

Each model **draws 800 times with replacement**, 80% of the original size. This does not mean 800 distinct observations are guaranteed to be included. Because of duplicates, the actual number of unique observations is less than 800.

The goal of bootstrap is not necessarily to make any single tree more accurate.

> It is to make the trees wobble differently, creating the conditions for some of their errors to cancel out when combined.

---

## 4. Aggregation: turning wobble into stability

Suppose models `f_1, f_2, ..., f_B` were trained on bootstrap samples `D_1, D_2, ..., D_B`.

For regression, the predictions are averaged.

$$
\hat f_{\text{bag}}(x)=\frac{1}{B}\sum_{b=1}^{B}f_b(x)
$$

- $B$ = number of trees
- $f_b(x)$ = the prediction of the b-th tree for input $x$
- $\sum_{b=1}^{B} f_b(x)$ = the sum of all tree predictions
- $\frac{1}{B}$ = divide by the number of trees
- $\hat f_{\text{bag}}(x)$ = bagging's final prediction



For example, if the true value is 50 and five trees predict the following:

```text
Tree 1 → 43
Tree 2 → 58
Tree 3 → 48
Tree 4 → 55
Tree 5 → 46
평균   → 50
```

As long as the individual trees' errors don't all point the same way, some of the error cancels out in the averaging. That is why bagging is called a classic **variance reduction technique**.

For classification, the way predictions are combined depends on which level you are describing.

- Breiman's original description of bagging: a plurality vote that picks the class with the most votes
- scikit-learn `BaggingClassifier`: averages class probabilities if the base model provides `predict_proba()`, and uses voting otherwise

So rather than memorizing "classification is always hard voting", check how the implementation you use combines predictions.

---

## 5. OOB: an internal estimate that comes naturally from bootstrap

Suppose there are `N` original examples and we build a bootstrap sample of size `N`. The probability that a particular observation is never selected is:

$$
\left(1-\frac{1}{N}\right)^N \xrightarrow[N\to\infty]{} \frac{1}{e}\approx0.368
$$

So a standard bootstrap sample has, on average, the following composition.

```text
한 번 이상 등장한 고유 관측치: 약 63.2%
한 번도 등장하지 않은 관측치: 약 36.8%
```

The observations not included in a tree's bootstrap sample are that tree's **Out-of-Bag (OOB) sample**.

### 5.1 How is the OOB evaluation built?

For each training observation, collect only the predictions of the trees that were trained without seeing it.

```text
관측치 x_i
  ↓
x_i를 학습에 사용하지 않은 Tree만 선택
  ↓
그 Tree들의 예측을 집계
  ↓
y_i와 비교 → OOB 성능 추정치
```

OOB is a convenient signal you get without retraining any models. But its role should not be overstated.

- 36.8% is the approximation for drawing `N` times with replacement from `N` examples.
- If `max_samples` changes, the OOB ratio changes too.
- OOB is an estimate built inside the original training pool.
- It is not the same as a final holdout, the final independent test set.
- Nor is there a rule that it replaces cross-validation in every situation.

It is more accurate to call OOB **an auxiliary internal performance estimate obtained from the bootstrap process** than a "free test set".

---

## 6. Checking bagging and OOB with scikit-learn

### 6.1 Preparing the data

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)
```

### 6.2 A single decision tree

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

print("Decision Tree test accuracy:", tree.score(X_test, y_test))
```

You cannot judge a model's stability from a single accuracy number. You also need to check how much the predictions or performance change across different data splits.

### 6.3 BaggingClassifier

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)

bagging.fit(X_train, y_train)

print("Bagging test accuracy:", bagging.score(X_test, y_test))
print("Bagging OOB accuracy:", bagging.oob_score_)
```

| Parameter | Meaning |
|---|---|
| `estimator` | The base model trained repeatedly |
| `n_estimators` | Number of base models |
| `max_samples` | Number or fraction of samples drawn for each model |
| `bootstrap=True` | Draw samples with replacement |
| `oob_score=True` | Compute an internal score on the OOB samples |
| `n_jobs=-1` | Use all available CPU cores |
| `random_state` | Make the random process reproducible |

In code from before scikit-learn 1.2 you may see `base_estimator` instead of `estimator`. The current API uses `estimator`.

On small, easy data like Iris, the two models may get the same test accuracy on a single run. You should not conclude from that one result that "bagging is always better" or that "it has no effect".

---

## 7. Key takeaways

### Decision Tree

> A high-variance model whose predictions can change a lot when the training sample changes slightly.

### Bagging

> Instead of discarding the unstable model, it makes models wobble differently on different bootstrap samples and aggregates their predictions to reduce the wobble.

### OOB

> An auxiliary performance estimate inside the training pool, built from each model's predictions on the samples it did not see during training.

Bagging can be applied to many base estimators, not just trees. But it is especially useful for high-variance models that are sensitive to changes in the sample, so it is often paired with decision trees. Adding more trees tends to stabilize predictions, but the improvement saturates while the computation cost grows.

However, even with bootstrap, a strong feature can keep the trees choosing similar splits. If the trees make similar errors, the effect of averaging is limited.

In the next post, to solve this problem, I compare **Random Forest, which randomizes the feature candidates at each node**, and **Extra Trees, which randomizes the split thresholds as well**.

---

## References

- [Leo Breiman, “Bagging predictors” (1996)](https://doi.org/10.1007/BF00058655)
- [scikit-learn: BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [scikit-learn: OOB Errors for Random Forests](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)
- [scikit-learn: Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
