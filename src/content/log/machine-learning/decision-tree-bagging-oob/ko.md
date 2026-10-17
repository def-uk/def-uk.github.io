---
title: "Decision Tree, Bagging과 OOB"
date: 2026-09-01T22:59:00+09:00
description: "Decision Tree가 흔들리는 이유, Bagging이 Bootstrap과 Aggregation으로 그 흔들림을 안정성으로 바꾸는 방식, OOB가 실제로 추정하는 것."
tags: ["Bagging", "ML", "MachineLearning", "ensemble"]
draft: false
---

>### 학습지표 
**Decision Tree**의 높은 variance를 줄이기 위해 **Bagging**은 데이터에 randomness를 주고, **Random Forest**는 feature 후보에도 randomness를 추가하며, **Extra Trees**는 분할 threshold까지 randomness를 추가한다.

Decision Tree는 비선형 관계와 변수 사이의 상호작용을 자연스럽게 표현한다. 전처리가 비교적 적고, 사람이 규칙을 따라갈 수 있다는 장점도 있다.

그러나 한 가지 약점이 크다. **학습 표본이 조금만 달라져도 Tree 구조와 예측이 크게 달라질 수 있다.**

이번 글에서는 이 문제에서 출발해 다음 질문에 답한다.

1. Decision Tree는 왜 흔들리는가?
2. Bagging은 왜 Tree를 여러 개 만드는가?
3. Bootstrap과 Aggregation은 각각 무슨 역할을 하는가?
4. OOB는 어디서 생기고 무엇을 평가하는가?

---

## 1. Decision Tree는 왜 불안정할까?

### 1.1 상위 분할 하나가 전체 구조를 바꾼다

Decision Tree는 각 노드에서 데이터를 가장 잘 나누는 질문을 탐욕적(Greedy Algorithm)으로 선택한다.

```text
petal width < 0.8인가?
```

최적 후보들의 점수가 비슷하다면 학습 데이터 몇 개만 추가되거나 빠져도 선택되는 분할이 달라질 수 있다. 상위 노드의 분할이 바뀌면 자식 노드가 받는 데이터도 달라지므로 이후 가지 구조가 연쇄적으로 변한다.

```text
원본 데이터 D      → Tree A → Decision Boundary A
조금 바뀐 데이터 D' → Tree B → Decision Boundary B
```

이 성질을 통계적인 언어로는 **variance가 높다**고 표현한다.

여기서 variance는 데이터 열의 분산이 아니다.

> 같은 모집단에서 학습 표본을 다시 뽑아 모델을 재학습한다면 예측이 얼마나 달라지는가?

깊은 Tree는 복잡한 패턴을 잡아 bias가 낮을 수 있지만, 표본의 우연한 변화와 잡음까지 따라가기 쉬워 variance가 커질 수 있다.

```text
복잡하고 깊은 Tree: bias가 낮을 수 있지만 variance가 커질 수 있음
단순하고 얕은 Tree: variance는 작아질 수 있지만 bias가 커질 수 있음
```

이는 절대적인 법칙이라기보다 모델 복잡도를 조절할 때 나타나는 대표적인 trade-off다.

### 1.2 Tree 하나를 단순하게 만들면 되지 않을까?

`max_depth`, `min_samples_leaf` 같은 제약으로 Tree를 단순하게 만들면 흔들림을 줄일 수 있다. 그러나 지나치게 제한하면 중요한 비선형 관계까지 놓쳐 underfitting이 생길 수 있다.

Bagging은 다른 질문을 던진다.

> 개별 Tree의 표현력을 크게 포기하지 않으면서 여러 Tree의 흔들림을 결합 과정에서 상쇄할 수는 없을까?

이 질문에 대한 답이 **Bootstrap Aggregating**, 줄여서 **Bagging**이다.

---

## 2. Bagging: 서로 다르게 흔들리게 한 뒤 평균낸다

Bagging은 같은 종류의 기본 모델을 서로 다른 Bootstrap 표본에 학습하고 예측을 결합한다.

```text
원본 학습 데이터
  ├─ Bootstrap 표본 1 → Tree 1 ─┐
  ├─ Bootstrap 표본 2 → Tree 2 ─┼→ 평균 또는 투표
  └─ Bootstrap 표본 3 → Tree 3 ─┘
```

Tree들은 앞 Tree의 결과를 다음 Tree가 이어받지 않는다. 따라서 **순차 의존 없이 별도로 학습할 수 있고 병렬화하기 쉽다.**

이를 “Tree들이 통계적으로 독립적이다”라고 표현하면 정확하지 않다. 같은 원본 데이터와 Feature를 공유하므로 예측 오류는 서로 상관될 수 있다. 이 상관 문제는 다음 글의 Random Forest가 다룬다.

### 2.1 같은 Tree를 복사하는 것만으로는 부족하다

같은 데이터, 같은 알고리즘, 같은 설정으로 Tree를 100개 만들면 동일하거나 매우 비슷한 Tree가 만들어질 수 있다.

```text
같은 학습 데이터
  ├─ Tree 1 ─┐
  ├─ Tree 2 ─┤ 비슷한 예측
  └─ Tree 3 ─┘
```

같은 실수를 하는 모델 100개를 모아도 평균의 효과는 제한된다. Bagging은 먼저 학습 표본을 바꿔 각 모델에 서로 다른 학습 경험을 제공한다.

---

## 3. Bootstrap Sampling: 서로 다른 학습 표본 만들기

Bootstrap은 원본 학습 데이터에서 **복원 추출(with replacement)**하는 방법이다.

원본 데이터가 다음과 같다고 하자.

```text
[A, B, C, D, E]
```

크기가 5인 Bootstrap 표본은 다음처럼 만들어질 수 있다.

```text
표본 1: [A, A, C, D, E]
표본 2: [B, B, C, C, E]
표본 3: [A, B, D, D, D]
```

한 번 뽑은 관측치를 다시 뽑을 수 있으므로 중복이 생긴다. 반대로 어떤 관측치는 한 번도 선택되지 않을 수 있다.

### `max_samples=0.8`의 정확한 의미

학습 데이터가 1,000개이고 다음과 같이 설정했다고 하자.

```python
max_samples=0.8
bootstrap=True
```

각 모델은 원본의 80%에 해당하는 **800번을 복원 추출**한다. 서로 다른 관측치 800개가 반드시 포함된다는 뜻은 아니다. 중복 때문에 실제 고유 관측치 수는 800개보다 적다.

Bootstrap의 목적은 개별 Tree 하나를 반드시 더 정확하게 만드는 것이 아니다.

> Tree들을 서로 다르게 흔들리게 만들어, 결합할 때 오차가 일부 상쇄될 조건을 만드는 것이다.

---

## 4. Aggregation: 흔들림을 안정성으로 바꾸기

Bootstrap 표본 `D_1, D_2, ..., D_B`로 모델 `f_1, f_2, ..., f_B`를 학습했다고 하자.

회귀에서는 예측값을 평균한다.

$$
\hat f_{\text{bag}}(x)=\frac{1}{B}\sum_{b=1}^{B}f_b(x)
$$

- $B$ = Tree 개수
- $f_b(x)$ = b번째 Tree가 입력 $x$에 대해 낸 예측값
- $\sum_{b=1}^{B} f_b(x)$ = 모든 Tree 예측을 다 더함
- $\frac{1}{B}$ = Tree 개수로 나눔
- $\hat f_{\text{bag}}(x)$ = Bagging의 최종 예측



예를 들어 실제값이 50이고 다섯 Tree의 예측이 다음과 같다면:

```text
Tree 1 → 43
Tree 2 → 58
Tree 3 → 48
Tree 4 → 55
Tree 5 → 46
평균   → 50
```

개별 Tree의 오차 방향이 완전히 같지 않다면 평균 과정에서 일부 오차가 상쇄된다. 그래서 Bagging을 대표적인 **variance reduction 기법**이라고 부른다.

분류의 결합 방식은 설명 층위를 구분할 필요가 있다.

- Breiman의 원래 Bagging 설명: 가장 많은 표를 받은 클래스를 선택하는 plurality vote
- scikit-learn `BaggingClassifier`: 기본 모델이 `predict_proba()`를 제공하면 클래스 확률을 평균하고, 제공하지 않으면 voting 사용

즉 “분류는 무조건 hard voting”이라고 외우기보다 사용하는 구현의 결합 방식을 확인해야 한다.

---

## 5. OOB: Bootstrap에서 자연스럽게 생기는 내부 추정치

원본 데이터가 `N`개이고 크기가 `N`인 Bootstrap 표본을 만든다고 하자. 특정 관측치 하나가 한 번도 선택되지 않을 확률은 다음과 같다.

$$
\left(1-\frac{1}{N}\right)^N \xrightarrow[N\to\infty]{} \frac{1}{e}\approx0.368
$$

따라서 표준적인 Bootstrap 표본 하나에는 평균적으로 다음과 같은 구성이 나타난다.

```text
한 번 이상 등장한 고유 관측치: 약 63.2%
한 번도 등장하지 않은 관측치: 약 36.8%
```

한 Tree의 Bootstrap 표본에 포함되지 않은 관측치를 그 Tree의 **Out-of-Bag(OOB) 표본**이라고 한다.

### 5.1 OOB 평가는 어떻게 만들어질까?

각 학습 관측치에 대해 그 관측치를 보지 않고 학습한 Tree들의 예측만 모은다.

```text
관측치 x_i
  ↓
x_i를 학습에 사용하지 않은 Tree만 선택
  ↓
그 Tree들의 예측을 집계
  ↓
y_i와 비교 → OOB 성능 추정치
```

OOB는 별도의 모델 재학습 없이 얻을 수 있는 편리한 신호다. 그러나 역할을 과장하면 안 된다.

- 36.8%는 `N`개에서 `N`번 복원 추출할 때의 근사값이다.
- `max_samples`가 달라지면 OOB 비율도 달라진다.
- OOB는 원래 학습 풀 내부에서 만든 추정치다.
- 최종 독립 테스트 세트인 final holdout과 같지 않다.
- 모든 상황에서 Cross Validation을 대체한다는 규칙도 아니다.

OOB는 “공짜 테스트 세트”보다 **Bootstrap 과정에서 얻는 내부 보조 성능 추정치**라고 부르는 편이 정확하다.

---

## 6. scikit-learn으로 Bagging과 OOB 확인하기

### 6.1 데이터 준비

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

### 6.2 단일 Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

print("Decision Tree test accuracy:", tree.score(X_test, y_test))
```

한 번의 정확도만으로 모델의 안정성을 판단할 수는 없다. 여러 데이터 분할에서 예측이나 성능이 얼마나 달라지는지도 확인해야 한다.

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

| 파라미터 | 의미 |
|---|---|
| `estimator` | 반복해서 학습할 기본 모델 |
| `n_estimators` | 기본 모델 수 |
| `max_samples` | 각 모델을 위해 뽑을 표본 수 또는 비율 |
| `bootstrap=True` | 표본을 복원 추출 |
| `oob_score=True` | OOB 표본으로 내부 점수 계산 |
| `n_jobs=-1` | 사용 가능한 CPU 코어 활용 |
| `random_state` | 무작위 과정 재현 |

scikit-learn 1.2 이전 코드에서는 `estimator` 대신 `base_estimator`가 보일 수 있다. 현재 API에서는 `estimator`를 사용한다.

Iris처럼 작고 쉬운 데이터에서는 두 모델의 한 번의 test accuracy가 같을 수도 있다. 이 결과 하나로 “Bagging이 항상 더 좋다”거나 “효과가 없다”고 결론 내리면 안 된다.

---

## 7. 이번 글의 핵심

### Decision Tree

> 학습 표본이 조금 달라질 때 예측이 크게 바뀔 수 있는 고분산 모델이다.

### Bagging

> 불안정한 모델을 버리는 대신 서로 다른 Bootstrap 표본에서 다르게 흔들리게 만들고, 예측을 집계해 흔들림을 줄인다.

### OOB

> 각 모델이 학습 중 보지 않은 표본의 예측만 모아 만든 학습 풀 내부의 보조 성능 추정치다.

Bagging은 Tree뿐 아니라 여러 기본 추정기에 적용할 수 있다. 다만 표본 변화에 민감한 고분산 모델에서 특히 유용해 Decision Tree와 자주 결합한다. Tree 수를 늘리면 예측은 안정되는 경향이 있지만 개선은 포화되고 계산 비용은 증가한다.

그러나 Bootstrap을 사용해도 강력한 Feature 때문에 Tree들이 계속 비슷한 분할을 선택할 수 있다. Tree들의 오류가 비슷하면 평균 효과도 제한된다.

다음 글에서는 이 문제를 해결하기 위해 **노드별 Feature 후보를 무작위화하는 Random Forest**와 **분할 임계값까지 무작위화하는 Extra Trees**를 비교한다.

---

## 참고 자료

- [Leo Breiman, “Bagging predictors” (1996)](https://doi.org/10.1007/BF00058655)
- [scikit-learn: BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [scikit-learn: OOB Errors for Random Forests](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)
- [scikit-learn: Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
