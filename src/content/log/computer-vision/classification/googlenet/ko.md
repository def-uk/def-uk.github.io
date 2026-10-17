---
title: "GoogLeNet (Inception v1)"
date: 2026-09-28T16:02:00+09:00
description: "Inception 모듈과 1×1 병목, GAP, 보조 분류기로 깊이는 늘리고 비용은 줄인 GoogLeNet 정리."
tags: ["DL", "DeepLearning", "GoogleNet", "Inception-v1"]
draft: false
---
> 2014, Google (Szegedy et al.) · ILSVRC 2014 분류 부문 우승 (Top-5 error 6.67%)

---

## 1. 왜 나왔나? — Problem

2012년 AlexNet 이후 CNN 연구의 흐름은 **"네트워크를 더 깊고 넓게 만들면 성능이 오른다"** 였다. 하지만 이 방식에는 분명한 한계가 있었다.

- **연산량과 파라미터의 폭발**: 층과 채널을 늘리면 연산량이 제곱 단위로 늘어난다. 모델이 커질수록 학습·추론 비용이 감당하기 어려워진다.
- **과적합**: 파라미터가 많을수록 데이터가 충분하지 않으면 과적합되기 쉽다. 특히 마지막 Fully Connected 층에 파라미터가 몰려 있었다.
- **필터 크기 선택의 딜레마**: 이미지 속 객체의 크기는 제각각이다. 작은 필터(3×3)는 세밀한 특징을, 큰 필터(5×5)는 넓은 문맥을 잘 잡는데, 층마다 어떤 크기를 쓸지 사람이 정해야 했다.
- **기울기 소실**: 깊어질수록 앞쪽 층까지 gradient가 잘 전달되지 않는다.

즉, **"깊게 쌓고 싶은데 비용은 감당할 수 있게"** 라는 문제를 풀어야 했다.

---

## 2. 다른 모델과 뭐가 다른가?

| | AlexNet (2012) | VGG-16 (2014) | **GoogLeNet (2014)** |
|---|---|---|---|
| 깊이 | 8층 | 16층 | **22층** |
| 파라미터 수 | 약 6,000만 | 약 1억 3,800만 | **약 500만** |
| 설계 철학 | CNN의 가능성 증명 | 3×3 conv를 단순하게 깊이 쌓기 | **여러 크기 필터를 병렬로, 효율적으로** |
| 네트워크 형태 | 한 줄로 쌓음 | 한 줄로 쌓음 | **층 안에서 가지가 갈라졌다 합쳐짐** |
| 마지막 분류부 | 큰 FC 층 3개 | 큰 FC 층 3개 | **Global Average Pooling + FC 1개** |

VGG가 "단순함"으로 깊이를 얻었다면, GoogLeNet은 **"구조 설계"로 깊이와 효율을 동시에** 얻었다. VGG보다 깊으면서도 파라미터는 약 1/27이다.

---

## 3. 그래서 어떻게 풀었나? — Solve

### ① Inception 모듈 → 필터 크기 딜레마 해결

한 층에서 필터 크기를 하나만 고르지 않고, **1×1 / 3×3 / 5×5 conv와 3×3 max pooling을 병렬로 수행한 뒤 채널 방향으로 이어 붙인다.** 어떤 스케일의 정보가 중요한지는 네트워크가 학습으로 알아서 조합한다. 이 모듈을 9개 쌓아 네트워크 몸통을 만든다.

```
            ┌─ 1×1 conv ─────────────────┐
            ├─ 1×1 conv → 3×3 conv ──────┤
입력 ───────┤                            ├─ Concat → 출력
            ├─ 1×1 conv → 5×5 conv ──────┤
            └─ 3×3 maxpool → 1×1 conv ───┘
```

### ② 1×1 Convolution 병목 → 연산량 폭발 해결

병렬로 여러 필터를 돌리면 연산량이 커지므로, 3×3·5×5 연산 **앞에** 1×1 conv를 두어 채널 수를 먼저 줄인다.

| 입력 28×28×192 → 5×5 conv로 32채널 출력 | 곱셈 연산량 |
|---|---|
| 5×5 conv 바로 적용 | 약 1억 2,000만 회 |
| 1×1 conv로 16채널 축소 → 5×5 conv | 약 1,240만 회 |

약 **10배** 절감된다. 1×1 conv 뒤에도 ReLU가 붙어 비선형성이 추가되는 효과도 있다.

### ③ Global Average Pooling → 파라미터·과적합 해결

마지막 feature map을 채널별로 평균 내어 1×1×1024 벡터로 만든다. 거대한 FC 층을 없애서 파라미터를 크게 줄이고 과적합도 완화했다.

### ④ 보조 분류기(Auxiliary Classifier) → 기울기 소실 완화

깊은 신경망은 마지막 층에서 계산한 오차가 앞쪽 층까지 전달되는 과정에서 학습 신호가 약해질 수 있다. 이를 완화하기 위해 중간 층에 작은 분류기를 붙여 추가 loss를 계산한다.

```
전체 loss = 메인 loss + 0.3 × (보조 loss1 + 보조 loss2)
```

추론할 때는 떼어낸다. (이후 Batch Normalization이 등장하면서 필요성은 줄었다.)

---

## 4. 코드로 어떻게 활용하나?

### ① Inception 모듈 직접 구현 (PyTorch)

```python
import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, in_ch, c1, c3_red, c3, c5_red, c5, pool_proj):
        super().__init__()
        # 1×1
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, c1, 1), nn.ReLU(inplace=True))
        # 1×1 병목 → 3×3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, c3_red, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c3_red, c3, 3, padding=1), nn.ReLU(inplace=True))
        # 1×1 병목 → 5×5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, c5_red, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c5_red, c5, 5, padding=2), nn.ReLU(inplace=True))
        # 3×3 maxpool → 1×1
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        # 네 갈래의 출력을 채널 방향(dim=1)으로 이어 붙임
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

# 논문의 inception(3a) 설정
m = Inception(192, 64, 96, 128, 16, 32, 32)
x = torch.randn(1, 192, 28, 28)
print(m(x).shape)  # torch.Size([1, 256, 28, 28])  ← 64+128+32+32
```

padding으로 공간 크기(28×28)를 맞춰두었기 때문에 채널 방향으로 concat이 가능하다.

### ② 사전학습 모델로 전이학습 (실무에서 가장 흔한 방식)

```python
import torch.nn as nn
from torchvision import models
from torchvision.models import GoogLeNet_Weights

weights = GoogLeNet_Weights.DEFAULT
model = models.googlenet(weights=weights)   # ImageNet 사전학습, 보조 분류기는 제거된 상태
preprocess = weights.transforms()           # 학습 때와 같은 전처리

# 백본 고정 (데이터가 적을 때)
for p in model.parameters():
    p.requires_grad = False

# 마지막 분류기만 내 클래스 수에 맞게 교체
model.fc = nn.Linear(model.fc.in_features, 10)  # in_features = 1024
```

### ③ 처음부터 학습할 때: 보조 분류기 loss 활용

```python
model = models.googlenet(weights=None, aux_logits=True,
                         num_classes=10, init_weights=True)
criterion = nn.CrossEntropyLoss()

model.train()
out = model(images)  # GoogLeNetOutputs(logits, aux_logits2, aux_logits1)
loss = criterion(out.logits, labels) \
     + 0.3 * (criterion(out.aux_logits1, labels) + criterion(out.aux_logits2, labels))

model.eval()
pred = model(images)  # 추론 시에는 메인 출력만 반환
```

---

## 한 줄 정리

> GoogLeNet은 **여러 크기의 필터를 병렬로 쓰는 Inception 모듈**과 **1×1 conv 병목**으로, 깊이는 늘리면서 연산량과 파라미터는 줄인 모델이다. 이 병목 설계는 이후 ResNet의 bottleneck block, MobileNet 같은 경량 모델로 이어진다.
