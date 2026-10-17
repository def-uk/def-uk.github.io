---
title: "Random Forest and Extra Trees"
date: 2026-09-01T23:14:00+09:00
description: "The tree-to-tree error correlation that bagging leaves behind, and how Random Forest (random features) and Extra Trees (random thresholds) tackle it."
tags: ["Algorithms", "ML", "MachineLearning", "Random Forest", "ensemble", "extra trees"]
draft: false
---
>### Learning goals
To reduce the high variance of a **Decision Tree**, **Bagging** adds randomness to the data, **Random Forest** also adds randomness to the feature candidates, and **Extra Trees** adds randomness all the way down to the split threshold.

In part 1, we looked at how bagging eases a decision tree's high variance through bootstrap and aggregation.

But bootstrap alone does not make every tree different enough. This post starts from the **error correlation between trees** that remains after bagging and lays out the difference between Random Forest and Extra Trees.

---

## 1. The problem left after bagging: the trees are too similar

Even with different bootstrap samples, every tree usually looks for the best split across all features. If one feature is very strong, most trees may pick that same feature at the root or upper nodes.

```text
Bootstrap A → Fare로 첫 분할
Bootstrap B → Fare로 첫 분할
Bootstrap C → Fare로 첫 분할
Bootstrap D → Fare로 첫 분할
```

Even if the training samples differ a little, the tree structures and errors can end up similar. That is why simply adding more models does not keep increasing the effect of averaging.

If we simplify by assuming every pair among `B` models with the same variance `σ²` has the same correlation `ρ`, the variance of the averaged prediction can be written as follows.

$$
\operatorname{Var}(\bar f)
=\sigma^2\left(\rho+\frac{1-\rho}{B}\right)
$$

- If `ρ=0`, the averaged variance drops all the way to `σ²/B`.
- If `ρ` is large, adding more models reduces it only so much.
- Even as `B→∞`, the expression approaches `ρσ²`.

This formula is an intuition under simple assumptions, but it carries an important message.

> In an ensemble, what matters is not just the number of models but how differently the members' errors move.

Random Forest adds randomness to feature selection as well, to lower this correlation.

---

## 2. Random Forest: each node looks at different feature candidates

A typical Random Forest in scikit-learn combines the following three elements.

```text
Bootstrap sample
  +
노드별 Feature 후보 무작위화
  +
Tree 예측의 평균
```

The key is `max_features`. It does not draw one bundle of features per tree and keep it fixed to the end. It selects a new set of feature candidates **every time a node searches for a split**.

```text
루트 노드:      [Age, Fare, Sex] 중 최적 분할 선택
왼쪽 자식 노드: [Pclass, Fare, SibSp] 중 최적 분할 선택
오른쪽 자식:    [Age, Parch, Pclass] 중 최적 분할 선택
```

At nodes where the strong feature is left out of the candidates, other features get a chance to split. As a result, tree structures and error patterns can become more diverse.

### 2.1 The goal of Random Forest is not simply "more randomness"

The number of feature candidates involves a trade-off.

- With many candidates, each tree finds good splits more easily, but the trees may become similar.
- With few candidates, tree diversity can increase, but important features may be missed often.

Breiman's Random Forest paper explains this as a balance between the **strength** of individual trees and the **correlation** between trees.

```text
좋은 Random Forest를 만드는 방향
= 충분한 개별 Tree의 예측력
+ 지나치게 높지 않은 Tree 간 오류 상관
```

So `max_features` is not a parameter where "smaller is better". It is a knob you need to validate for your data.

### 2.2 Bagged trees vs Random Forest

```text
Bagged Trees
Bootstrap sample
  → 보통 모든 Feature에서 최적 분할 탐색
  → Tree 생성
  → 예측 집계

Random Forest
Bootstrap sample
  → 매 노드에서 Feature 일부를 무작위 선택
  → 선택된 후보 안에서 최적 분할 탐색
  → Tree 생성
  → 예측 집계
```

That said, scikit-learn's general `BaggingClassifier` also supports feature sampling through `max_features` and `bootstrap_features`. The difference is that Random Forest re-selects the feature candidates **at each node of the tree**.

### 2.3 Be careful with the phrase "independent trees"

In a Random Forest, no tree inherits the result of the previous one, so they can be trained separately and in parallel. But that does not mean their prediction errors are statistically independent.

In fact, the problem Random Forest is trying to solve is exactly the **correlation of tree errors**. So the following is more accurate.

> The trees can be trained without sequential dependency, and feature randomness is a device designed to make them move less alike.

---

## 3. Extra Trees: randomizing the split threshold too

**Extra Trees (Extremely Randomized Trees)** puts even stronger randomness into the splitting process than Random Forest.

```text
Random Forest
→ Feature 일부를 무작위 선택
→ 각 후보 Feature에서 좋은 Threshold 탐색
→ 후보 안에서 최적 분할 선택

Extra Trees
→ Feature 일부를 무작위 선택
→ 각 후보 Feature의 Threshold를 무작위 생성
→ 무작위로 생성된 분할 중 가장 좋은 것 선택
```

For example, if `Age` is a candidate feature, Random Forest compares the possible split points and finds the value that reduces impurity the most. Extra Trees generates a threshold like `Age < 31.7` at random and compares it with the random splits generated for the other candidate features.

The important point is that Extra Trees does not just accept any split as is.

> It generates a random threshold for each candidate feature, but among those random candidates it picks the best split.

### 3.1 The trade-off of stronger randomness

Extra Trees' randomization can further lower the correlation of tree errors and the ensemble variance. Training may also be faster because it does not search every possible threshold.

On the other hand, if it misses a useful optimal split, bias can increase.

```text
무작위성 증가
  → Tree 다양성 증가 가능
  → 오류 상관과 variance 감소 가능
  → 개별 Tree의 최적 분할을 놓쳐 bias 증가 가능
```

This direction is **a general tendency, not a performance guarantee or a fixed ranking.**

### 3.2 An important difference in scikit-learn defaults

In scikit-learn, the two models have different defaults.

- `RandomForestClassifier`: `bootstrap=True`
- `ExtraTreesClassifier`: `bootstrap=False`

So by default, each tree in Extra Trees uses the full training sample. The differences between trees come mainly from the randomness of per-node feature candidates and thresholds.

Because of this, calling Extra Trees "bootstrap bagging" in the strict sense is not accurate. If you want to talk about Bagging, Random Forest and Extra Trees together, this phrasing is safe:

> An ensemble that builds many randomized trees without sequential dependency and averages them

Or, as the scikit-learn documentation puts it, a **perturb-and-combine randomized tree ensemble**.

---

## 4. Comparing Random Forest and Extra Trees

| | Random Forest | Extra Trees |
|---|---|---|
| Training relationship | No sequential dependency between trees | No sequential dependency between trees |
| Samples | Bootstrap by default | Full training sample by default |
| Feature candidates | Random subset per node | Random subset per node |
| Threshold | Searches for a good value within candidate features | Picks the best among random values per feature |
| Randomness | Strong | Stronger |
| Computation | Cost of exact split search | Can be faster |
| Bias–variance | Depends on data and settings | Variance may be lower and bias higher |
| OOB | Available with default settings | Requires switching to `bootstrap=True` |

The bias–variance row shows a direction. Attaching a definitive arrow ranking like the one below would be an oversimplification.

```text
Bagging variance ↓
Random Forest variance ↓↓
Extra Trees variance ↓↓↓
```

Actual results depend on the number of samples, signal strength, number of features, noise, tree depth, `max_features` and more.

---

## 5. Comparing them with scikit-learn

### 5.1 Shared data

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

### 5.2 RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)

random_forest.fit(X_train, y_train)

print("RF test accuracy:", random_forest.score(X_test, y_test))
print("RF OOB accuracy:", random_forest.oob_score_)
```

`max_features="sqrt"` means that when searching for a split at each node, roughly the square root of the total number of features is considered as candidates. The candidate set is re-drawn at random for every node.

The scikit-learn documentation states that it may effectively inspect more than `max_features` features when needed to find a valid split. So it is safer not to say "it looks at exactly that many in every situation".

### 5.3 ExtraTreesClassifier

```python
from sklearn.ensemble import ExtraTreesClassifier

extra_trees = ExtraTreesClassifier(
    n_estimators=300,
    max_features="sqrt",
    bootstrap=False,
    n_jobs=-1,
    random_state=42,
)

extra_trees.fit(X_train, y_train)

print("Extra Trees test accuracy:", extra_trees.score(X_test, y_test))
```

`bootstrap=False` just makes scikit-learn's default behavior explicit. Even when every tree uses the full training sample, feature and threshold randomness can produce different trees.

If you need an OOB score, turn bootstrap on explicitly.

```python
extra_trees_oob = ExtraTreesClassifier(
    n_estimators=300,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)
```

### 5.4 Comparing under the same evaluation conditions

Rather than a single train/test split, it is better to look at the mean and per-fold variability under the same cross-validation conditions.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in {
    "Random Forest": random_forest,
    "Extra Trees": extra_trees,
}.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name:14s}: {scores.mean():.3f} ± {scores.std():.3f}")
```

The standard deviation here is the **variability of the performance score** measured on that fold configuration. It is not the same as model variance in the bias–variance decomposition.

---

## 6. Things to watch out for when choosing

- The values to tune first are `n_estimators`, `max_features`, `max_depth` and `min_samples_leaf`.
- There is no law that a smaller `max_features` monotonically reduces variance or improves performance.
- "RF for clean data, Extra Trees for noisy data" is only a hypothesis to compare, not a model selection rule.
- Feature randomness can lower the correlation of tree errors, but it can also lower the predictive power of individual trees.
- Adding more trees tends to stabilize predictions, but the improvement saturates while the computation cost grows.
- Extra Trees defaults to `bootstrap=False`, so it is not the same as typical bootstrap bagging.
- scikit-learn's Random Forest classifier averages the class probabilities of each tree to decide the final class.

> Theory provides hypotheses to compare; the actual choice is decided by cross-validation results under the same evaluation conditions.

---

## 7. Key takeaways

### Random Forest

> Randomizes the feature candidates at each node so the trees move less alike, trying to strengthen the effect of averaging.

### Extra Trees

> Randomizes not only the features but the split thresholds too, creating a stronger bias–variance trade-off.

### Model selection

> Which of Random Forest and Extra Trees is better depends on the data, so compare performance and cost under the same cross-validation conditions.

Bagging, Random Forest and Extra Trees all build trees without sequential dependency and average their predictions. In the next post, I look at **Boosting**, which asks a different question.

> How can the next tree make up for the direction in which the current model is still wrong?

---

## References

- [Leo Breiman, “Random Forests” (2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Geurts, Ernst & Wehenkel, “Extremely randomized trees” (2006)](https://people.montefiore.uliege.be/ernst/uploads/news/id63/extremely-randomized-trees.pdf)
- [scikit-learn: Ensembles user guide](https://scikit-learn.org/stable/modules/ensemble.html)
- [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn: ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
