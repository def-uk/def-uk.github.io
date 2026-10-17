---
title: "VGGNet"
date: 2026-09-28T15:56:00+09:00
description: "VGGNet answered \"does depth matter?\" by stacking only 3×3 convolutions — the idea, its limits, and how it is used today in PyTorch."
tags: ["DL", "DeepLearning", "VGGNet"]
draft: false
---
## 1. Why did it appear? (Problem)
After AlexNet's big success on ImageNet in 2012, researchers started asking **"How should we design a CNN to get better performance?"**

At the time people tried changing filter sizes, strides and layer counts more or less at random, so it was unclear **which factors actually determined performance**. In particular, these questions remained:

- Does making the network **deeper** improve performance?
- But adding layers makes parameters and computation explode — how do we handle that?

In 2014 the Visual Geometry Group at Oxford (Simonyan & Zisserman) tried to answer this by keeping every other condition as simple and fixed as possible and **changing only the depth**.

## 2. How is it different from other models?

| | AlexNet (2012) | VGGNet (2014) | GoogLeNet (2014) |
|---|---|---|---|
| Filter size | Mix of 11×11, 5×5, 3×3 | **All 3×3** | 1×1, 3×3, 5×5 in parallel (Inception) |
| Depth | 8 layers | **16–19 layers** | 22 layers |
| Structure | Different settings per layer | **Simple and regular** | Complex module structure |
| Parameters | ~60M | ~138M | ~7M |

- **Versus AlexNet**: it dropped large filters and stacked small filters deeply instead.
- **Versus GoogLeNet**: slightly lower accuracy (2nd in ILSVRC 2014 classification), but a much simpler structure that was easy to understand, implement and modify. That is exactly why it ended up being used more widely.

## 3. How did it solve it? (Solve)

### The core idea: stack 3×3 convolutions several times
Stacking 3×3 convolutions back to back gives the same **receptive field** as a large filter.

| Configuration | Receptive field | Parameters (for C channels) |
|---|---|---|
| One 7×7 conv layer | 7×7 | 49C² |
| Three 3×3 conv layers | 7×7 | **27C²** |

This lets you increase depth while gaining the following:
1. **Fewer parameters**: the same receptive field with fewer parameters.
2. **More non-linearity**: each layer adds a ReLU, so the representation becomes richer.

### Overall architecture
224×224 input → 5 conv blocks → 3 FC layers → Softmax

- Each block = several 3×3 convolutions (stride 1, padding 1) + 2×2 max pooling (stride 2)
- Each pooling halves the spatial size and doubles the channels (64 → 128 → 256 → 512 → 512)
- The end is FC 4096 → FC 4096 → FC 1000
- **VGG16**: 13 conv layers + 3 FC layers / **VGG19**: 16 conv layers + 3 FC layers

### Results and remaining problems
It showed experimentally that performance improves as depth increases. The limits were clear, though.
- Most parameters sit in the FC layers, so **the model is heavy** (about 100M in the first FC layer alone).
- Beyond 19 layers, **vanishing gradients** make training difficult → later solved by **ResNet** with residual connections.

## 4. How do you use it in code? (PyTorch)

### 4-1. Implementing the architecture yourself
Because the block structure is regular, you can build the network from a configuration list alone.

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

### 4-2. Transfer learning (the most common use)
In practice, rather than training from scratch, you take a model pretrained on ImageNet and replace only the last layer to fit your data.

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

### 4-3. Using it as a feature extractor (perceptual loss, etc.)
VGG's intermediate outputs capture an image's texture and shape well, so style transfer and super-resolution models use them to measure **"how similar two images look to the human eye"**.

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

## Summary
- **Problem**: it was unclear how depth affects CNN performance.
- **Difference**: use only 3×3 filters instead of large ones, and keep the structure simple and regular.
- **Solve**: stack small filters deeply to reduce parameters and increase non-linearity.
- **Use**: today it is mostly used as a transfer-learning backbone or a feature extractor rather than as a classifier itself.
