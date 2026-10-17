---
title: "GoogLeNet (Inception v1)"
date: 2026-09-28T16:02:00+09:00
description: "GoogLeNet went deeper while cutting cost with Inception modules, 1×1 bottlenecks, global average pooling and auxiliary classifiers."
tags: ["DL", "DeepLearning", "GoogleNet", "Inception-v1"]
draft: false
---
> 2014, Google (Szegedy et al.) · Winner of ILSVRC 2014 classification (Top-5 error 6.67%)

---

## 1. Why did it appear? — Problem

After AlexNet in 2012, the trend in CNN research was **"make the network deeper and wider and performance goes up"**. But that approach had clear limits.

- **Exploding computation and parameters**: adding layers and channels grows computation quadratically. The bigger the model, the harder training and inference costs are to bear.
- **Overfitting**: the more parameters, the easier it is to overfit without enough data. Parameters were especially concentrated in the final fully connected layers.
- **The filter-size dilemma**: objects in images come in all sizes. Small filters (3×3) capture fine details and large filters (5×5) capture wider context, but someone had to decide which size each layer would use.
- **Vanishing gradients**: the deeper the network, the worse the gradient reaches the early layers.

In short, the problem was **"we want to stack deeply, but at a cost we can afford"**.

---

## 2. How is it different from other models?

| | AlexNet (2012) | VGG-16 (2014) | **GoogLeNet (2014)** |
|---|---|---|---|
| Depth | 8 layers | 16 layers | **22 layers** |
| Parameters | ~60M | ~138M | **~5M** |
| Design philosophy | Prove what CNNs can do | Stack 3×3 convs simply and deeply | **Several filter sizes in parallel, efficiently** |
| Network shape | Stacked in a single line | Stacked in a single line | **Branches split and merge inside a layer** |
| Final classifier | Three large FC layers | Three large FC layers | **Global Average Pooling + one FC** |

If VGG gained depth through "simplicity", GoogLeNet gained **depth and efficiency at the same time through "architecture design"**. It is deeper than VGG yet has about 1/27 of the parameters.

---

## 3. So how did it solve it? — Solve

### ① Inception module → solving the filter-size dilemma

Instead of picking one filter size per layer, it **runs 1×1 / 3×3 / 5×5 convolutions and 3×3 max pooling in parallel, then concatenates them along the channel axis.** The network learns on its own how to combine whichever scales matter. Nine of these modules are stacked to form the body of the network.

```
            ┌─ 1×1 conv ─────────────────┐
            ├─ 1×1 conv → 3×3 conv ──────┤
입력 ───────┤                            ├─ Concat → 출력
            ├─ 1×1 conv → 5×5 conv ──────┤
            └─ 3×3 maxpool → 1×1 conv ───┘
```

### ② 1×1 convolution bottleneck → solving the computation explosion

Running several filters in parallel increases computation, so a 1×1 conv is placed **before** the 3×3 and 5×5 operations to reduce the number of channels first.

| Input 28×28×192 → 5×5 conv producing 32 channels | Multiplications |
|---|---|
| Apply the 5×5 conv directly | ~120 million |
| Reduce to 16 channels with a 1×1 conv → 5×5 conv | ~12.4 million |

That is about a **10×** reduction. A ReLU also follows the 1×1 conv, which adds extra non-linearity.

### ③ Global Average Pooling → solving parameters and overfitting

It averages the last feature map per channel into a 1×1×1024 vector. Removing the huge FC layers greatly reduced the parameters and also eased overfitting.

### ④ Auxiliary classifiers → easing vanishing gradients

In a deep network, the learning signal can weaken as the error computed at the last layer travels back to the early layers. To ease this, small classifiers are attached to intermediate layers to compute additional losses.

```
전체 loss = 메인 loss + 0.3 × (보조 loss1 + 보조 loss2)
```

They are removed at inference time. (Once Batch Normalization appeared, the need for them decreased.)

---

## 4. How do you use it in code?

### ① Implementing the Inception module yourself (PyTorch)

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

Because padding keeps the spatial size (28×28) the same, the outputs can be concatenated along the channel axis.

### ② Transfer learning with a pretrained model (the most common approach in practice)

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

### ③ When training from scratch: using the auxiliary classifier losses

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

## One-line summary

> GoogLeNet uses **Inception modules that run several filter sizes in parallel** and **1×1 conv bottlenecks** to increase depth while reducing computation and parameters. This bottleneck design carried on into ResNet's bottleneck block and lightweight models like MobileNet.
