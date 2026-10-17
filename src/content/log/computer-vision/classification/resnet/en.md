---
title: "ResNet (Residual Network)"
date: 2026-09-28T16:48:00+09:00
description: "Why deeper plain networks got worse (the degradation problem) and how ResNet fixed it by learning only the residual with skip connections."
tags: ["DL", "DeepLearning", "Resnet", "residual network"]
draft: false
---
> He et al., *Deep Residual Learning for Image Recognition*, 2015 (Winner of ILSVRC 2015)

---

## 1. Background: why did it appear? (Problem)

After VGG, "the deeper you stack, the better the performance" was almost conventional wisdom. But when people actually kept stacking, something strange happened.

- A 56-layer network had **higher train error as well as higher test error** than a 20-layer network.
- A higher train error means **it is not overfitting**. The model simply is not learning properly.
- This happened even though vanishing gradients had been largely handled by things like Batch Normalization.

This is called the **Degradation Problem**.

Logically, it makes no sense. If you just attach 36 **layers that do nothing (identity)** to the 20-layer model, the 56-layer model should be at least as good as the 20-layer one. So the problem was not expressive power but **an optimization problem: "a deep network struggles to learn even a simple function like the identity."**

* identity: the identity function


---

## 2. How is it different from other architectures?

| | Plain network (VGG, etc.) | ResNet |
|---|---|---|
| What it learns | The desired output `H(x)` directly | Only the difference `F(x) = H(x) - x` |
| Layer output | `y = F(x)` | `y = F(x) + x` |
| When identity is optimal | Several non-linear layers must approximate the identity (hard) | Just push `F(x) → 0` (easy) |
| Gradient flow | Multiplied at every layer and gets weaker | Has a path that passes directly through the shortcut |

The **Highway Network** from around the same time also used shortcuts, but it had learnable gates, which add parameters and can block the shortcut when a gate closes. ResNet's identity shortcut differs in that it has **zero parameters and is always open**.

Both Highway Networks and ResNet use a **shortcut (skip connection)** that passes the input further down the network. The difference is how the shortcut is controlled. A Highway Network multiplies by a learnable gate, as in $y=T(x)H(x)+(1-T(x))x$, to decide how much of the original input $x$ to pass on. Depending on the gate value, the shortcut can be almost closed. ResNet's identity shortcut, on the other hand, simply adds the input, as in $y=F(x)+x$. The shortcut itself has no parameters to learn and is never closed by a gate.

---

## 3. So how did it solve it? (Solve)

### The core: Residual Block + Skip Connection

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

### Why does it work?

1. **The learning target becomes easier**
   Instead of building the whole output from scratch, the layer only learns "how much to change" the input. A layer that is not needed can just pass the input through by keeping its weights near 0.

2. **Gradients flow well**
   Differentiating $y = F(x) + x$ gives
   $$\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + 1$$
   Thanks to the $+1$, the gradient does not vanish no matter how deep the network is, and reaches the early layers.

### Design details

- **When dimensions don't match**: if the stride or channel count changes so that `x` and `F(x)` have different shapes, a `1x1 conv` (projection) is added to the shortcut to match them.
- **Basic Block** (ResNet-18/34): `3x3 → 3x3`
- **Bottleneck Block** (ResNet-50/101/152): `1x1 (reduce channels) → 3x3 → 1x1 (restore channels)`
  A structure for reducing computation while stacking deeply.

As a result it trained stably up to **152 layers** and took first place with an ImageNet top-5 error of 3.57%.

---

## 4. How do you use it in code?

### (1) Implementing a residual block yourself (PyTorch)

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

### (2) Fine-tuning a pretrained model

Rather than implementing it yourself, most of the time you load weights trained on ImageNet and **replace only the classifier (fc)**.

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

If you have enough data, unfreezing part of the backbone (e.g. `layer4`) and training it as well improves performance further.

### (3) The idea is everywhere

Skip connections went beyond image classification to become a basic building block of almost every deep learning architecture that followed.

- **Transformer**: `x + Attention(x)` and `x + FFN(x)` in every sublayer
- **U-Net**: skip connections between encoder and decoder
- **DenseNet**: extends the idea by concatenating instead of adding

> One-line summary: **"Stacking deeper stops training from working" → "Let the layers learn only the difference, and add the input back as is."**
