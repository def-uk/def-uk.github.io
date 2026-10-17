---
title: "ResNet (Residual Network)"
date: 2026-09-28T16:48:00+09:00
description: "깊을수록 오히려 학습이 안 되는 Degradation 문제를, 차이만 배우는 Residual Block과 Skip Connection으로 푼 ResNet 정리."
tags: ["DL", "DeepLearning", "Resnet", "residual network"]
draft: false
---
> He et al., *Deep Residual Learning for Image Recognition*, 2015 (ILSVRC 2015 우승)

---

## 1. 배경: 왜 나왔나? (Problem)

VGG 이후로 "층을 깊게 쌓을수록 성능이 좋아진다"는 게 거의 정설이었다. 그런데 실제로 계속 쌓아보니 이상한 일이 생겼다.

- 56층 네트워크가 20층 네트워크보다 **test error뿐 아니라 train error도 더 높게** 나왔다.
- train error까지 높다는 건 **overfitting이 아니라는 뜻**이다. 모델이 아예 학습을 제대로 못 하고 있는 것.
- Batch Normalization 등으로 vanishing gradient는 어느 정도 잡은 상태였는데도 이런 현상이 나타났다.

이걸 **Degradation Problem**이라고 부른다.

논리적으로 보면 말이 안 된다. 20층 모델에 **아무것도 안 하는 층(identity)** 36개만 붙여도 56층 모델은 최소한 20층만큼은 나와야 한다. 즉, 문제는 표현력이 아니라 **"깊은 네트워크가 identity 같은 단순한 함수조차 학습하기 어렵다"는 최적화 문제**였다.

* identity: 항등함수


---

## 2. 다른 구조와 뭐가 다른가?

| 구분 | Plain Network (VGG 등) | ResNet |
|---|---|---|
| 학습 대상 | 원하는 출력 `H(x)`를 직접 학습 | 차이값 `F(x) = H(x) - x`만 학습 |
| 층의 출력 | `y = F(x)` | `y = F(x) + x` |
| identity가 최적일 때 | 여러 비선형 층으로 identity를 근사해야 함 (어려움) | `F(x) → 0`으로 만들면 끝 (쉬움) |
| gradient 흐름 | 층을 지날 때마다 곱해지며 약해짐 | shortcut을 통해 직접 전달되는 경로가 있음 |

비슷한 시기의 **Highway Network**도 shortcut을 썼지만, 여기엔 학습 가능한 gate가 있어서 파라미터가 늘고 gate가 닫히면 shortcut이 막힐 수 있다. ResNet의 identity shortcut은 **파라미터 0개, 항상 열려 있다**는 점이 다르다.

Highway Network와 ResNet 모두 입력을 뒤쪽으로 전달하는 **shortcut(skip connection)**을 사용한다. 차이는 shortcut을 조절하는 방식이다. Highway Network는 $y=T(x)H(x)+(1-T(x))x$처럼 학습 가능한 gate를 곱해 원래 입력 $x$를 얼마나 전달할지 결정한다. Gate 값에 따라 shortcut이 거의 막힐 수도 있다. 반면 ResNet의 identity shortcut은 $y=F(x)+x$처럼 입력을 그대로 더한다. Shortcut 자체에는 학습할 파라미터가 없고, gate에 의해 닫히지도 않는다.

---

## 3. 그래서 어떻게 풀었나? (Solve)

### 핵심: Residual Block + Skip Connection

```
x ──────────────────┐
│                   │ (identity shortcut)
▼                   │
Conv → BN → ReLU    │
│                   │
Conv → BN           │
│                   │
▼                   │
(+) ◄───────────────┘
│
ReLU
▼
y = F(x) + x
```

### 왜 잘 되나?

1. **학습 목표가 쉬워진다**
   출력 전체를 새로 만드는 대신 입력에서 "얼마나 바꿀지"만 배운다. 필요 없는 층은 가중치를 0 근처로 두면 그냥 통과시킬 수 있다.

2. **gradient가 잘 흐른다**
   $y = F(x) + x$를 미분하면
   $$\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + 1$$
   $+1$ 덕분에 gradient가 아무리 깊어도 사라지지 않고 앞쪽 층까지 전달된다.

### 세부 설계

- **차원이 안 맞을 때**: stride나 채널 수가 바뀌어 `x`와 `F(x)`의 shape이 다르면, shortcut에 `1x1 conv`(projection)를 넣어 맞춘다.
- **Basic Block** (ResNet-18/34): `3x3 → 3x3`
- **Bottleneck Block** (ResNet-50/101/152): `1x1(채널 축소) → 3x3 → 1x1(채널 복원)`
  깊게 쌓으면서도 연산량을 줄이기 위한 구조.

결과적으로 **152층**까지 안정적으로 학습시켰고, ImageNet top-5 error 3.57%로 당시 1위를 차지했다.

---

## 4. 코드로는 어떻게 쓰나?

### (1) Residual Block 직접 구현 (PyTorch)

```python
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # shape이 달라지면 1x1 conv로 shortcut을 맞춰줌
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # 핵심: F(x) + x
        return self.relu(out)
```

### (2) 사전학습 모델 가져와서 Fine-tuning

직접 구현하기보다 ImageNet으로 학습된 가중치를 가져와 **분류기(fc)만 바꿔서** 쓰는 경우가 대부분이다.

```python
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)

# (선택) 백본은 고정하고 분류기만 학습
for p in model.parameters():
    p.requires_grad = False

num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 새 분류기는 학습됨
```

데이터가 충분하면 백본 일부(`layer4` 등)를 풀어서 함께 학습시키면 성능이 더 오른다.

### (3) 개념은 어디에나 있다

Skip connection은 이미지 분류를 넘어서 이후 거의 모든 딥러닝 구조의 기본 부품이 됐다.

- **Transformer**: 각 sublayer마다 `x + Attention(x)`, `x + FFN(x)`
- **U-Net**: encoder-decoder 사이 skip connection
- **DenseNet**: 더하기 대신 이어붙이기(concat)로 확장

> 한 줄 요약: **"깊게 쌓으면 학습이 안 된다" → "차이만 배우게 하고 입력은 그대로 더해주자"**
