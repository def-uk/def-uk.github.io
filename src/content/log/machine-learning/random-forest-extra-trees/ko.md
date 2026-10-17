---
title: "Random Forest와 Extra Trees"
date: 2026-09-01T23:14:00+09:00
description: "Bagging 뒤에도 남는 Tree 간 오류 상관을 Random Forest(Feature 무작위화)와 Extra Trees(Threshold 무작위화)가 각각 어떻게 다루는지 비교한다."
tags: ["Algorithms", "ML", "MachineLearning", "Random Forest", "ensemble", "extra trees"]
draft: false
---
>### 학습지표 
**Decision Tree**의 높은 variance를 줄이기 위해 **Bagging**은 데이터에 randomness를 주고, **Random Forest**는 feature 후보에도 randomness를 추가하며, **Extra Trees**는 분할 threshold까지 randomness를 추가한다.

1편에서는 Decision Tree의 높은 variance를 Bagging이 Bootstrap과 Aggregation으로 완화하는 과정을 살펴봤다.

하지만 Bootstrap만으로 모든 Tree가 충분히 달라지는 것은 아니다. 이번 글에서는 Bagging 뒤에도 남는 **Tree 간 오류 상관**에서 출발해 Random Forest와 Extra Trees의 차이를 정리한다.

---

## 1. Bagging 뒤에도 남는 문제: Tree들이 너무 비슷하다

Bootstrap 표본이 다르더라도 모든 Tree는 보통 전체 Feature에서 가장 좋은 분할을 찾는다. 하나의 Feature가 매우 강력하다면 대부분의 Tree가 루트나 상위 노드에서 같은 Feature를 선택할 수 있다.

```text
Bootstrap A → Fare로 첫 분할
Bootstrap B → Fare로 첫 분할
Bootstrap C → Fare로 첫 분할
Bootstrap D → Fare로 첫 분할
```

학습 표본은 조금씩 달라도 Tree 구조와 오류가 비슷해질 수 있다. 모델 수만 늘리는 것으로 평균 효과가 계속 커지지 않는 이유다.

같은 variance `σ²`을 가진 모델 `B`개의 모든 쌍이 같은 상관계수 `ρ`를 가진다고 단순화하면, 평균 예측의 variance는 다음처럼 쓸 수 있다.

$$
\operatorname{Var}(\bar f)
=\sigma^2\left(\rho+\frac{1-\rho}{B}\right)
$$

- `ρ=0`이면 평균 variance가 `σ²/B`까지 감소한다.
- `ρ`가 크면 모델 수를 늘려도 감소 폭이 제한된다.
- `B→∞`여도 식은 `ρσ²`에 가까워진다.

이 식은 단순한 가정 아래의 직관이지만 중요한 메시지를 준다.

> 앙상블에서는 모델 수뿐 아니라 구성원 오류가 얼마나 다르게 움직이는지도 중요하다.

Random Forest는 이 상관을 낮추기 위해 Feature 선택에도 무작위성을 넣는다.

---

## 2. Random Forest: 노드마다 다른 Feature 후보를 본다

scikit-learn의 전형적인 Random Forest는 다음 세 요소를 결합한다.

```text
Bootstrap sample
  +
노드별 Feature 후보 무작위화
  +
Tree 예측의 평균
```

핵심은 `max_features`다. Tree를 만들 때 Feature 묶음을 한 번만 뽑아 끝까지 고정하는 방식이 아니다. **각 노드에서 분할을 찾을 때마다** Feature 후보 집합을 새로 선택한다.

```text
루트 노드:      [Age, Fare, Sex] 중 최적 분할 선택
왼쪽 자식 노드: [Pclass, Fare, SibSp] 중 최적 분할 선택
오른쪽 자식:    [Age, Parch, Pclass] 중 최적 분할 선택
```

강력한 Feature가 후보에서 빠진 노드에서는 다른 Feature가 분할 기회를 얻는다. 그 결과 Tree 구조와 오류 패턴이 더 다양해질 수 있다.

### 2.1 Random Forest의 목적은 단순히 “더 랜덤하게”가 아니다

Feature 후보 수에는 trade-off가 있다.

- 후보가 많으면 개별 Tree는 좋은 분할을 찾기 쉽지만 Tree들이 비슷해질 수 있다.
- 후보가 적으면 Tree 다양성은 커질 수 있지만 중요한 Feature를 자주 놓칠 수 있다.

Breiman의 Random Forest 논문은 이를 개별 Tree의 **strength**와 Tree 사이의 **correlation** 사이의 균형으로 설명한다.

```text
좋은 Random Forest를 만드는 방향
= 충분한 개별 Tree의 예측력
+ 지나치게 높지 않은 Tree 간 오류 상관
```

따라서 `max_features`는 “작을수록 좋다”는 파라미터가 아니다. 데이터에 맞게 검증해야 하는 조절 장치다.

### 2.2 Bagged Trees와 Random Forest의 차이

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

단, scikit-learn의 범용 `BaggingClassifier`도 `max_features`와 `bootstrap_features`를 통해 Feature sampling을 지원한다. 차이는 Random Forest가 **Tree의 각 노드에서** Feature 후보를 다시 선택한다는 점이다.

### 2.3 “독립적인 Tree”라는 표현을 조심해야 한다

Random Forest의 Tree들은 앞 Tree의 결과를 다음 Tree가 이어받지 않으므로 별도·병렬 학습이 가능하다. 그러나 예측 오류가 통계적으로 독립이라는 뜻은 아니다.

오히려 Random Forest가 해결하려는 문제가 바로 **Tree 오류의 상관**이다. 따라서 다음 표현이 더 정확하다.

> Tree들은 순차 의존 없이 학습할 수 있으며, Feature Randomness는 Tree들이 덜 비슷하게 움직이도록 설계된 장치다.

---

## 3. Extra Trees: 분할 임계값까지 무작위화한다

**Extra Trees(Extremely Randomized Trees)**는 Random Forest보다 분할 과정에 더 강한 무작위성을 넣는다.

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

예를 들어 `Age`가 후보 Feature라면 Random Forest는 가능한 분할점들을 비교해 불순도를 가장 많이 줄이는 값을 찾는다. Extra Trees는 `Age < 31.7`과 같은 임계값을 무작위로 생성하고, 다른 후보 Feature에서 생성한 무작위 분할과 비교한다.

중요한 점은 Extra Trees가 아무 분할이나 그대로 채택하는 것은 아니라는 것이다.

> 후보 Feature마다 무작위 Threshold를 만들지만, 그 무작위 후보들 중에서는 가장 좋은 분할을 선택한다.

### 3.1 더 강한 무작위성의 trade-off

Extra Trees의 무작위화는 Tree 오류의 상관과 ensemble variance를 더 낮출 수 있다. 모든 가능한 Threshold를 탐색하지 않아 학습이 빨라질 수도 있다.

반면 유용한 최적 분할을 놓치면 bias가 커질 수 있다.

```text
무작위성 증가
  → Tree 다양성 증가 가능
  → 오류 상관과 variance 감소 가능
  → 개별 Tree의 최적 분할을 놓쳐 bias 증가 가능
```

이 방향은 **일반적인 경향이지 성능 보장이나 고정 순위가 아니다.**

### 3.2 scikit-learn의 중요한 기본값 차이

scikit-learn에서 두 모델의 기본값은 다음과 다르다.

- `RandomForestClassifier`: `bootstrap=True`
- `ExtraTreesClassifier`: `bootstrap=False`

따라서 Extra Trees는 기본적으로 각 Tree가 전체 학습 표본을 사용한다. Tree 차이는 주로 노드별 Feature 후보와 Threshold 무작위성에서 나온다.

이 때문에 Extra Trees까지 엄밀한 의미의 “Bootstrap Bagging”으로 부르면 정확하지 않다. Bagging, Random Forest, Extra Trees를 함께 묶어 말하고 싶다면 다음 표현이 안전하다.

> 순차 의존 없이 여러 randomized Tree를 만들고 평균하는 ensemble

또는 scikit-learn 문서의 표현처럼 **perturb-and-combine 방식의 randomized tree ensemble**이라고 부를 수 있다.

---

## 4. Random Forest와 Extra Trees 비교

| 구분 | Random Forest | Extra Trees |
|---|---|---|
| 학습 관계 | Tree 간 순차 의존 없음 | Tree 간 순차 의존 없음 |
| 표본 | 기본값 Bootstrap | 기본값 전체 학습 표본 |
| Feature 후보 | 노드마다 무작위 일부 | 노드마다 무작위 일부 |
| Threshold | 후보 Feature 안에서 좋은 값 탐색 | Feature별 무작위 값 중 최선 선택 |
| 무작위성 | 강함 | 더 강함 |
| 계산 | 정확한 분할 탐색 비용 | 더 빠를 수 있음 |
| bias–variance | 데이터·설정에 따라 결정 | variance가 더 낮고 bias가 더 높아질 수 있음 |
| OOB | 기본 설정에서 사용 가능 | `bootstrap=True`로 바꿔야 사용 가능 |

표의 bias–variance 행은 방향성이다. 다음처럼 확정적인 화살표 순위를 붙이면 과도한 단순화가 된다.

```text
Bagging variance ↓
Random Forest variance ↓↓
Extra Trees variance ↓↓↓
```

실제 결과는 표본 수, 신호 강도, Feature 수, 잡음, Tree 깊이, `max_features` 등에 따라 달라진다.

---

## 5. scikit-learn으로 비교하기

### 5.1 공통 데이터

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

`max_features="sqrt"`는 각 노드의 분할을 찾을 때 대략 전체 Feature 수의 제곱근만큼을 후보로 고려한다는 뜻이다. 후보 집합은 노드마다 다시 무작위로 선택된다.

scikit-learn은 유효한 분할을 찾기 위해 필요한 경우 실질적으로 `max_features`보다 더 많은 Feature를 검사할 수 있다고 문서에 명시한다. 따라서 “어떤 상황에서도 정확히 그 개수만 본다”라고 표현하지 않는 편이 안전하다.

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

`bootstrap=False`는 scikit-learn의 기본 동작을 명시한 것이다. 모든 Tree가 전체 학습 표본을 사용하더라도 Feature와 Threshold 무작위성 때문에 서로 다른 Tree가 만들어질 수 있다.

OOB 점수가 필요하다면 Bootstrap을 명시적으로 켠다.

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

### 5.4 같은 평가 조건에서 비교하기

한 번의 train/test split보다 같은 Cross Validation 조건에서 평균과 fold별 변동성을 함께 보는 편이 낫다.

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

여기서 표준편차는 해당 fold 구성에서 측정한 **성능 점수의 변동성**이다. bias–variance 분해에서 말하는 모델 variance와 같은 값은 아니다.

---

## 6. 선택할 때 주의할 점

- 먼저 조정할 값은 `n_estimators`, `max_features`, `max_depth`, `min_samples_leaf`다.
- `max_features`가 작을수록 variance가 단조롭게 줄거나 성능이 좋아진다는 법칙은 없다.
- “Clean이면 RF, Noisy면 Extra Trees”는 비교 가설일 뿐 모델 선택 규칙이 아니다.
- Feature Randomness는 Tree 오류 상관을 낮출 수 있지만 개별 Tree의 예측력도 낮출 수 있다.
- Tree 수를 늘리면 예측은 안정되는 경향이 있지만 개선은 포화되고 계산 비용은 증가한다.
- Extra Trees는 `bootstrap=False`가 기본이므로 전형적인 Bootstrap Bagging과 같지 않다.
- scikit-learn의 Random Forest 분류는 Tree별 class probability를 평균해 최종 class를 정한다.

> 이론은 비교할 가설을 제공하고, 실제 선택은 동일한 평가 조건의 Cross Validation 결과가 결정한다.

---

## 7. 이번 글의 핵심

### Random Forest

> 노드별 Feature 후보 무작위화로 Tree들이 덜 비슷하게 움직이도록 만들고 평균의 효과를 키우려 한다.

### Extra Trees

> Feature뿐 아니라 분할 Threshold까지 무작위화해 더 강한 bias–variance trade-off를 만든다.

### 모델 선택

> Random Forest와 Extra Trees의 우열은 데이터에 따라 달라지므로 같은 Cross Validation 조건에서 성능과 비용을 비교한다.

Bagging, Random Forest, Extra Trees는 Tree들을 순차 의존 없이 만들고 예측을 평균하는 방식이다. 다음 글에서는 이들과 다른 질문을 던지는 **Boosting**을 살펴본다.

> 현재 모델이 아직 틀리는 방향을 다음 Tree가 어떻게 보완할까?

---

## 참고 자료

- [Leo Breiman, “Random Forests” (2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
- [Geurts, Ernst & Wehenkel, “Extremely randomized trees” (2006)](https://people.montefiore.uliege.be/ernst/uploads/news/id63/extremely-randomized-trees.pdf)
- [scikit-learn: Ensembles user guide](https://scikit-learn.org/stable/modules/ensemble.html)
- [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [scikit-learn: ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)
