---
title: "VGGNet"
date: 2026-09-28T15:56:00+09:00
description: "3×3 합성곱만 깊게 쌓아 \"깊이가 성능을 좌우하는가\"에 답한 VGGNet의 아이디어, 한계, PyTorch 활용법."
tags: ["DL", "DeepLearning", "VGGNet"]
draft: false
---
## 1. 왜 나왔나? (Problem)
2012년 AlexNet이 ImageNet에서 큰 성공을 거두면서, 연구자들은 **"CNN을 어떻게 설계해야 성능이 더 좋아질까?"**를 고민하기 시작했다.

당시에는 필터 크기, stride, 층 수 등을 이것저것 바꿔보는 식이어서, **어떤 요소가 성능을 결정하는지** 명확하지 않았다. 특히 이런 질문이 남아 있었다.

- 네트워크를 **깊게** 만들면 성능이 좋아지는가?
- 그런데 층을 늘리면 파라미터와 연산량이 폭발하는데, 이걸 어떻게 감당하나?

2014년 옥스퍼드 대학 Visual Geometry Group(Simonyan & Zisserman)은 다른 조건은 최대한 단순하게 고정하고 **깊이 하나만 바꿔가며** 이 질문에 답하려 했다.

## 2. 다른 모델과 뭐가 다른가?

| 구분 | AlexNet (2012) | VGGNet (2014) | GoogLeNet (2014) |
|---|---|---|---|
| 필터 크기 | 11×11, 5×5, 3×3 혼용 | **3×3으로 통일** | 1×1, 3×3, 5×5 병렬 (Inception) |
| 깊이 | 8층 | **16~19층** | 22층 |
| 구조 | 층마다 설정이 제각각 | **단순하고 규칙적** | 복잡한 모듈 구조 |
| 파라미터 | 약 6천만 | 약 1억 3,800만 | 약 700만 |

- **AlexNet 대비**: 큰 필터를 버리고 작은 필터를 깊게 쌓았다.
- **GoogLeNet 대비**: 성능은 약간 낮았지만(ILSVRC 2014 분류 2위), 구조가 훨씬 단순해서 이해·구현·변형이 쉬웠다. 그래서 오히려 더 널리 쓰였다.

## 3. 어떻게 풀었나? (Solve)

### 핵심: 3×3 합성곱을 여러 번 쌓는다
3×3 합성곱을 연속으로 쌓으면 큰 필터와 같은 **수용 영역(receptive field)**을 얻는다.

| 구성 | 수용 영역 | 파라미터 수 (채널 C 기준) |
|---|---|---|
| 7×7 합성곱 1층 | 7×7 | 49C² |
| 3×3 합성곱 3층 | 7×7 | **27C²** |

이렇게 하면 깊이를 늘리면서도 다음 이점을 얻는다.
1. **파라미터 감소**: 같은 수용 영역을 더 적은 파라미터로 얻는다.
2. **비선형성 증가**: 층마다 ReLU가 들어가므로 표현력이 풍부해진다.

### 전체 구조
입력 224×224 → 합성곱 블록 5개 → FC 3개 → Softmax

- 각 블록 = 3×3 합성곱(stride 1, padding 1) 여러 개 + 2×2 Max Pooling(stride 2)
- 풀링마다 공간 크기는 절반, 채널은 두 배 (64 → 128 → 256 → 512 → 512)
- 마지막은 FC 4096 → FC 4096 → FC 1000
- **VGG16**: 합성곱 13층 + FC 3층 / **VGG19**: 합성곱 16층 + FC 3층

### 결과와 남은 문제
깊이가 깊어질수록 성능이 좋아진다는 것을 실험으로 보여주었다. 다만 한계도 분명했다.
- 파라미터 대부분이 FC층에 몰려 있어 **모델이 무겁다** (첫 FC층만 약 1억 개).
- 19층 이상 쌓으면 **기울기 소실**로 학습이 어려워진다 → 이후 **ResNet**이 잔차 연결로 해결.

## 4. 코드로 어떻게 활용하나? (PyTorch)

### 4-1. 구조 직접 구현해보기
블록 구조가 규칙적이라 설정 리스트만으로 네트워크를 만들 수 있다.

```python
import torch.nn as nn

# 숫자 = 3×3 합성곱의 출력 채널, 'M' = Max Pooling
VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
             512, 512, 512, 'M', 512, 512, 512, 'M']

def make_features(cfg):
    layers, in_ch = [], 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1),
                       nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = make_features(VGG16_CFG)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### 4-2. 전이학습 (가장 흔한 활용)
실무에서는 직접 학습하기보다 ImageNet으로 사전학습된 모델을 가져와 내 데이터에 맞게 마지막 층만 바꾼다.

```python
import torch.nn as nn
from torchvision import models

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# 특징 추출부는 고정
for p in model.features.parameters():
    p.requires_grad = False

# 마지막 FC층을 내 클래스 수에 맞게 교체 (예: 5개 클래스)
model.classifier[6] = nn.Linear(4096, 5)
```

### 4-3. 특징 추출기로 활용 (Perceptual Loss 등)
VGG의 중간층 출력은 이미지의 질감·형태를 잘 담고 있어서, 스타일 트랜스퍼나 초해상도 모델에서 **"두 이미지가 사람 눈에 얼마나 비슷한가"**를 측정하는 데 쓰인다.

```python
import torch.nn as nn
from torchvision import models

vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
for p in vgg.parameters():
    p.requires_grad = False

def perceptual_loss(pred, target):
    # 픽셀 값 대신 VGG 특징 맵끼리 비교
    return nn.functional.mse_loss(vgg(pred), vgg(target))
```

## 정리
- **Problem**: CNN에서 깊이가 성능에 어떤 영향을 주는지 불분명했다.
- **차이점**: 큰 필터 대신 3×3 필터만 쓰고, 구조를 단순·규칙적으로 만들었다.
- **Solve**: 작은 필터를 깊게 쌓아 파라미터는 줄이고 비선형성은 늘렸다.
- **활용**: 지금은 분류 모델 자체보다 전이학습 백본, 특징 추출기로 주로 쓰인다.
