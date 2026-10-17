---
title: "What Has Each Optimizer Been Fixing?"
date: 2026-09-07T13:23:00+09:00
description: "From plain gradient descent to Momentum, AdaGrad, RMSProp, Adadelta and Adam — implemented by hand in PyTorch, framed by what state each one stores."
tags: ["AdaGrad", "DL", "DeepLearning", "Gradient Descent", "Mini batch", "Optimizer", "RMSProp", "SGD", "adam", "momentum", "경사하강법", "옵티마이저", "확률적 경사 하강법"]
draft: false
---
> ## Learning goals
>
> After reading this post, you will be able to:
>
> 1. Explain how the gradient computed by `loss.backward()` is used to update parameters.
> 2. Tell Batch GD, SGD and Mini-batch GD apart in code by `batch_size`.
> 3. Implement the core updates of Momentum, AdaGrad, RMSProp, Adadelta and Adam yourself.
> 4. Compare PyTorch optimizers by swapping them in the same training loop.

---

## Contents

1. Shared practice code
2. Gradient Descent — updating parameters directly with the gradient
3. Batch GD / SGD / Mini-batch — seeing it through `batch_size`
4. Problems with basic gradient descent — reproducing them in code
5. Momentum — storing the previous direction
6. AdaGrad — building per-parameter accumulated step sizes
7. RMSProp / Adadelta — forgetting old gradients
8. Adam — storing direction and magnitude together
9. Running the same experiment with `torch.optim`
10. Trade-offs and the overall flow

<br>

<div align="center">
  <strong>
    In the end, the difference between optimizers is<br>
    what state they store and how they update the parameters.
  </strong>
</div>

<br>

## 1. Shared practice code

First, create simple linear regression data that every optimizer will use.

The true relationship is roughly $y=3x+2$, and the model learns the weight and bias of this relationship.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# y = 3x + 2에 작은 noise를 추가한 데이터
X = torch.linspace(-2, 2, 256).unsqueeze(1)
y = 3 * X + 2 + 0.2 * torch.randn_like(X)

dataset = TensorDataset(X, y)


def make_model():
    torch.manual_seed(42)
    return nn.Linear(1, 1)


criterion = nn.MSELoss()
```

The model's prediction is:

$$
\hat{y}=wx+b
$$

The parameters to learn are `model.weight` and `model.bias`.

```python
model = make_model()

for name, parameter in model.named_parameters():
    print(name, parameter.shape)

# weight torch.Size([1, 1])
# bias   torch.Size([1])
```

Now let's change only the update code on the same model and data and see how the optimizers differ.

---

## 2. Gradient Descent — updating parameters directly with the gradient

The gradient descent update rule is:

$$
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta J(\theta_t)
$$

This rule translates into PyTorch code almost as is.

```python
model = make_model()

pred = model(X)
loss = criterion(pred, y)

# 각 parameter의 gradient를 계산해 .grad에 저장한다.
loss.backward()

learning_rate = 0.1

with torch.no_grad():
    for parameter in model.parameters():
        parameter -= learning_rate * parameter.grad

# 다음 step의 gradient가 누적되지 않도록 비운다.
model.zero_grad()
```

The key is three lines.

```python
loss.backward()                              # gradient 계산
parameter -= learning_rate * parameter.grad # gradient 반대 방향으로 이동
model.zero_grad()                            # 이전 gradient 제거
```

### What is in `parameter.grad`?

Calling `loss.backward()` stores the partial derivative of the loss with respect to each parameter in `.grad`.

```python
model = make_model()
loss = criterion(model(X), y)
loss.backward()

print(model.weight.grad)  # ∂J / ∂w
print(model.bias.grad)    # ∂J / ∂b
```

Collecting the two values into one vector gives the gradient.

$$
\nabla J(w,b)
=
\begin{bmatrix}
\frac{\partial J}{\partial w}\\
\frac{\partial J}{\partial b}
\end{bmatrix}
$$

> **The gradient is the direction in which the objective function $J$ increases fastest from the current position.**

To reduce the loss we have to move in the opposite direction, which is why the update code has a `-`.

```python
parameter -= learning_rate * parameter.grad
```

### Why recompute the gradient at every step?

When the parameters change, the model's predictions and loss change, and so does the gradient at the new position.

So the training code repeats the following process.

```python
for step in range(100):
    pred = model(X)               # 현재 parameter로 예측
    loss = criterion(pred, y)     # 현재 위치의 Loss

    loss.backward()               # 현재 위치의 Gradient

    with torch.no_grad():
        for parameter in model.parameters():
            parameter -= 0.1 * parameter.grad

    model.zero_grad()             # 다음 step을 위해 초기화
```

![Why the gradient is computed at every step](./image/01.png)

---

## 3. Batch GD / SGD / Mini-batch — seeing it through `batch_size`

Batch GD, SGD and Mini-batch share the same update principle. The most visible difference in code is the `DataLoader`'s `batch_size`.

```python
# Batch Gradient Descent: 전체 데이터로 1번 update
batch_loader = DataLoader(
    dataset,
    batch_size=len(dataset),
    shuffle=True,
)

# Stochastic Gradient Descent: 데이터 1개마다 update
sgd_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)

# Mini-batch Gradient Descent: 32개마다 update
mini_batch_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)
```

You can also check the number of updates per epoch right away.

```python
print(len(batch_loader))       # 1
print(len(sgd_loader))         # 256
print(len(mini_batch_loader))  # 8
```

The training loop is the same in all three cases.

```python
def train_one_epoch(model, loader, optimizer):
    model.train()

    for x_batch, y_batch in loader:
        optimizer.zero_grad()

        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        loss.backward()
        optimizer.step()
```

The difference is which data the gradient used by a single `optimizer.step()` was computed from.

| Method | `batch_size` | Data per update | Characteristics |
| --- | ---: | ---: | --- |
| Batch GD | Size of the full dataset | All | Stable but each update is expensive |
| SGD | `1` | 1 | Updates often but is very noisy |
| Mini-batch | `32`, `64`, `128`, etc. | Some | A compromise between stability and computational efficiency |

### How is the mini-batch gradient built?

If a batch has $B$ examples, the average of each sample's gradient is used.

$$
g_B=\frac{1}{B}\sum_{i=1}^{B}g_i
$$

In PyTorch, the default setting of `nn.MSELoss()`, which averages the loss, and `backward()` handle this computation.

```python
x_batch, y_batch = next(iter(mini_batch_loader))

pred = model(x_batch)
loss = criterion(pred, y_batch)  # batch의 평균 MSE
loss.backward()                  # 평균 Loss에 대한 Gradient
```

By averaging several samples, mini-batches are more stable than pure SGD and can use the GPU efficiently through matrix operations.

On the other hand, if the batch is too large, memory usage grows and the number of updates per epoch drops. If it is too small, gradient noise grows.

> **Batch size, too, is a trade-off between computational efficiency and gradient stability.**

---

## 4. Problems with basic gradient descent — reproducing them in code

### 4.1 When the learning rate is too small or too large

Let's change only the learning rate on a simple function.

$$
J(w)=(w-3)^2
$$

The minimum of this function is $w=3$.

```python
def run_gradient_descent(lr, steps=10):
    w = torch.tensor(0.0, requires_grad=True)
    history = []

    for step in range(steps):
        loss = (w - 3) ** 2
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad

        history.append({
            "step": step,
            "w": w.item(),
            "loss": loss.item(),
        })

        w.grad.zero_()

    return history


slow = run_gradient_descent(lr=0.05)      # 조금씩 이동
oscillating = run_gradient_descent(lr=0.8) # minimum 양쪽을 오가며 수렴
diverging = run_gradient_descent(lr=1.1)   # 진폭이 커지며 발산
```

The gradient tells you the direction, but the actual step size is decided by `lr`.

```python
update = learning_rate * gradient
```

- If `lr` is too small, training is slow.
- If `lr` is too large, it can overshoot the minimum and oscillate or diverge.

### 4.2 Is it always a minimum when the gradient is 0?

A saddle point looks like a minimum in one direction and a maximum in another.

$$
f(x,y)=x^2-y^2
$$

Computing the gradient at the origin in code gives 0 for both components.

```python
point = torch.tensor([0.0, 0.0], requires_grad=True)
x, y = point[0], point[1]

value = x**2 - y**2
value.backward()

print(point.grad)  # tensor([0., 0.])
```

But the value increases along $x$ and decreases along $y$, so the origin is not a minimum.

### 4.3 The same learning rate is applied to every parameter

The basic SGD code multiplies every parameter by the same `lr`.

```python
for parameter in model.parameters():
    parameter -= lr * parameter.grad
```

But each parameter's gradient magnitude and update frequency can differ. AdaGrad, RMSProp and Adam later store extra state to give each parameter a different effective step size.

---

## 5. Momentum — storing the previous direction

In mini-batch SGD the gradient changes from batch to batch, so it can zigzag on a narrow, elongated loss surface.

Basic SGD uses only the current gradient.

```python
parameter -= lr * parameter.grad
```

Momentum stores the direction of movement so far in `velocity`.

$$
v_t=\beta v_{t-1}+g_t
$$

$$
\theta_{t+1}=\theta_t-\eta v_t
$$

Implementing it directly looks like this.

```python
model = make_model()
parameters = list(model.parameters())

# parameter마다 같은 모양의 velocity를 하나씩 만든다.
velocity = {
    parameter: torch.zeros_like(parameter)
    for parameter in parameters
}


@torch.no_grad()
def momentum_step(parameters, velocity, lr=0.01, beta=0.9):
    for parameter in parameters:
        velocity[parameter].mul_(beta).add_(parameter.grad)
        parameter.add_(velocity[parameter], alpha=-lr)
```

In the training loop, call your own `momentum_step()` where `optimizer.step()` would go.

```python
for x_batch, y_batch in mini_batch_loader:
    model.zero_grad()

    loss = criterion(model(x_batch), y_batch)
    loss.backward()

    momentum_step(parameters, velocity, lr=0.01, beta=0.9)
```

The key part of the code is this.

```python
velocity[parameter].mul_(beta).add_(parameter.grad)
```

- When the gradient keeps pointing the same way, it accumulates in `velocity`.
- When the gradient flips direction, it partly cancels the previous value.

> **Momentum reinforces movement in a consistent direction and dampens repeated oscillation.**

However, if `beta` is too large, the inertia becomes strong and can cause overshooting past the minimum.

In PyTorch, the same behavior is used like this.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
)
```

---

## 6. AdaGrad — building per-parameter accumulated step sizes

AdaGrad's starting point is simple.

> **Move parameters that have moved often and far more slowly, and parameters that have rarely moved relatively more.**

To do this, it stores the accumulated sum of past squared gradients for each parameter.

$$
G_t=G_{t-1}+g_t^2
$$

$$
\theta_{t+1}
=
\theta_t-\frac{\eta}{\sqrt{G_t}+\epsilon}g_t
$$

```python
model = make_model()
parameters = list(model.parameters())

squared_grad_sum = {
    parameter: torch.zeros_like(parameter)
    for parameter in parameters
}


@torch.no_grad()
def adagrad_step(parameters, squared_grad_sum, lr=0.01, eps=1e-10):
    for parameter in parameters:
        grad = parameter.grad

        # G_t = G_{t-1} + g_t²
        squared_grad_sum[parameter].addcmul_(grad, grad)

        # parameter마다 서로 다른 실제 보폭
        denominator = squared_grad_sum[parameter].sqrt().add_(eps)
        parameter.addcdiv_(grad, denominator, value=-lr)
```

The key is that the `denominator` differs per parameter.

```python
denominator = squared_grad_sum[parameter].sqrt() + eps
effective_lr = lr / denominator
```

For a parameter whose gradient has stayed large, `squared_grad_sum` grows and the effective learning rate shrinks.

The problem is that this value only ever increases.

```python
squared_grad_sum[parameter] += grad * grad
```

Squared gradients are never negative, so the accumulated value never decreases. As training goes on, the effective learning rate can get close to 0.

> **AdaGrad remembers the past for too long, so learning can slow down too quickly.**

In PyTorch, it is used as follows.

```python
optimizer = torch.optim.Adagrad(
    model.parameters(),
    lr=0.01,
)
```

---

## 7. RMSProp / Adadelta — forgetting old gradients

If the problem with AdaGrad is that it adds up every past gradient, the fix is to reduce the influence of old information.

### RMSProp

Instead of a plain running sum, RMSProp stores an exponential moving average of the squared gradients.

$$
v_t=\beta v_{t-1}+(1-\beta)g_t^2
$$

```python
model = make_model()
parameters = list(model.parameters())

square_avg = {
    parameter: torch.zeros_like(parameter)
    for parameter in parameters
}


@torch.no_grad()
def rmsprop_step(parameters, square_avg, lr=0.001, beta=0.9, eps=1e-8):
    for parameter in parameters:
        grad = parameter.grad

        # v_t = beta * v_{t-1} + (1 - beta) * g_t²
        square_avg[parameter].mul_(beta).addcmul_(
            grad,
            grad,
            value=1 - beta,
        )

        denominator = square_avg[parameter].sqrt().add_(eps)
        parameter.addcdiv_(grad, denominator, value=-lr)
```

Compared with AdaGrad, the key is a one-line difference.

```python
# AdaGrad: 과거 값을 그대로 유지하며 계속 더한다.
accumulator += grad**2

# RMSProp: 과거 값은 beta만큼만 남긴다.
square_avg = beta * square_avg + (1 - beta) * grad**2
```

For example, with `beta=0.9`, the influence of an old gradient shrinks over time like this.

$$
1\rightarrow0.9\rightarrow0.9^2\rightarrow0.9^3\rightarrow\cdots
$$

> **RMSProp adjusts per-parameter step sizes while weighting the magnitude of recent gradients more heavily.**

The PyTorch code is:

```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.9,
)
```

### Adadelta

Adadelta also uses an exponential moving average of the squared gradients. On top of that, it also stores the mean of the squared recent parameter updates.

```python
model = make_model()
parameters = list(model.parameters())

square_avg = {
    parameter: torch.zeros_like(parameter)
    for parameter in parameters
}
acc_delta = {
    parameter: torch.zeros_like(parameter)
    for parameter in parameters
}


@torch.no_grad()
def adadelta_step(parameters, square_avg, acc_delta, rho=0.9, eps=1e-6):
    for parameter in parameters:
        grad = parameter.grad

        # 최근 Gradient²의 이동 평균
        square_avg[parameter].mul_(rho).addcmul_(
            grad,
            grad,
            value=1 - rho,
        )

        # 최근 update²와 Gradient²를 이용해 update 크기 결정
        std = square_avg[parameter].add(eps).sqrt()
        delta = acc_delta[parameter].add(eps).sqrt().div(std).mul(grad)

        parameter.sub_(delta)

        # 방금 적용한 update²의 이동 평균
        acc_delta[parameter].mul_(rho).addcmul_(
            delta,
            delta,
            value=1 - rho,
        )
```

Where RMSProp adjusts the step size around a single `square_avg`, Adadelta uses `square_avg` and `acc_delta` together.

```python
optimizer = torch.optim.Adadelta(
    model.parameters(),
    rho=0.9,
)
```

| AdaGrad | RMSProp / Adadelta |
| --- | --- |
| Keeps accumulating gradient² | Reduces the influence of old gradients |
| Effective learning rate keeps decreasing | Adjusts step size around recent information |
| Can become too slow in long training | Eases AdaGrad's decay problem |

---

## 8. Adam — storing direction and magnitude together

Let's look again at the state stored in the code so far.

```python
# Momentum
velocity       # Gradient 방향을 누적

# RMSProp
square_avg     # Gradient²의 이동 평균
```

Adam uses both ideas together.

```python
first_moment   # Gradient의 지수 이동 평균
second_moment  # Gradient²의 지수 이동 평균
```

### First moment

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

Like Momentum, it stabilizes the direction of the gradient.

### Second moment

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Like RMSProp, it adjusts the step size per parameter.

### Implementing bias correction yourself

Both moments are initialized to 0, so early in training they are biased toward 0.

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

The full update code using the corrected values is:

```python
model = make_model()
parameters = list(model.parameters())

first_moment = {
    parameter: torch.zeros_like(parameter)
    for parameter in parameters
}
second_moment = {
    parameter: torch.zeros_like(parameter)
    for parameter in parameters
}


@torch.no_grad()
def adam_step(
    parameters,
    first_moment,
    second_moment,
    step,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
):
    for parameter in parameters:
        grad = parameter.grad

        # 1차 모멘트: Gradient의 이동 평균
        first_moment[parameter].mul_(beta1).add_(
            grad,
            alpha=1 - beta1,
        )

        # 2차 모멘트: Gradient²의 이동 평균
        second_moment[parameter].mul_(beta2).addcmul_(
            grad,
            grad,
            value=1 - beta2,
        )

        # 0으로 초기화되어 생기는 초반 편향 보정
        m_hat = first_moment[parameter] / (1 - beta1**step)
        v_hat = second_moment[parameter] / (1 - beta2**step)

        parameter.addcdiv_(
            m_hat,
            v_hat.sqrt().add_(eps),
            value=-lr,
        )
```

In the training loop, `step` is incremented starting from 1.

```python
step = 0

for x_batch, y_batch in mini_batch_loader:
    model.zero_grad()

    loss = criterion(model(x_batch), y_batch)
    loss.backward()

    step += 1
    adam_step(
        parameters,
        first_moment,
        second_moment,
        step,
    )
```

Connecting the code to the roles:

| State in code | Formula | What it remembers | Role |
| --- | --- | --- | --- |
| `first_moment` | $m_t$ | Gradient | Stabilize direction |
| `second_moment` | $v_t$ | Gradient² | Adjust per-parameter step size |
| `m_hat`, `v_hat` | $\hat{m}_t$, $\hat{v}_t$ | Corrected moments | Correct the initial bias toward 0 |

In PyTorch, the same core logic is a single line.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
)
```

> **Adam tracks both the direction and the magnitude of the gradient and performs an adaptive update for each parameter.**

---

## 9. Running the same experiment with `torch.optim`

The update rules we implemented differ, but the PyTorch training loop is almost the same.

```python
def train(model, loader, optimizer, epochs=100):
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(x_batch)

        losses.append(epoch_loss / len(loader.dataset))

    return losses
```

You can split out only the optimizer creation into a function and swap it as you run.

```python
def make_optimizer(name, model):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.01)

    if name == "momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
        )

    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=0.01)

    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=0.001,
            alpha=0.9,
        )

    if name == "adadelta":
        return torch.optim.Adadelta(
            model.parameters(),
            rho=0.9,
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
        )

    raise ValueError(f"Unknown optimizer: {name}")
```

Compare them with the same initial parameters and the same mini-batch conditions.

```python
results = {}

for name in [
    "sgd",
    "momentum",
    "adagrad",
    "rmsprop",
    "adadelta",
    "adam",
]:
    model = make_model()
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    optimizer = make_optimizer(name, model)

    losses = train(model, loader, optimizer, epochs=100)

    results[name] = {
        "final_loss": losses[-1],
        "weight": model.weight.item(),
        "bias": model.bias.item(),
    }

for name, result in results.items():
    print(name, result)
```

What matters in this experiment is not settling a ranking once. Each optimizer has a different suitable range of learning rates, so a single value cannot decide a fair winner.

Instead, it is good to use this code to check the following.

- How fast does the initial loss decrease?
- Does the loss oscillate a lot?
- After the same number of epochs, how close is it to $w\approx3$, $b\approx2$?
- How sensitive are the results to changing the learning rate?

### Checking the state an optimizer actually stores

Looking at `optimizer.state` shows what each algorithm remembers. You need to run `optimizer.step()` at least once before checking.

```python
parameter = next(model.parameters())
state = optimizer.state[parameter]

print(state.keys())
```

The typical states are:

| Optimizer | Main state | Meaning |
| --- | --- | --- |
| SGD | None | Uses only the current gradient |
| SGD + Momentum | `momentum_buffer` | Accumulated previous direction |
| AdaGrad | `sum` | Running sum of gradient² |
| RMSProp | `square_avg` | Moving average of gradient² |
| Adadelta | `square_avg`, `acc_delta` | Moving averages of gradient² and update² |
| Adam | `exp_avg`, `exp_avg_sq`, `step` | First and second moments, and the step |

In the end, inside the same `optimizer.step()`, each optimizer reads and updates different state.

---

## 10. Trade-offs and the overall flow

Adam's code uses the most information, but that does not make it the best for every problem.

| Optimizer | Extra state | What you gain | Watch out for |
| --- | ---: | --- | --- |
| SGD | 0 | Simple update, little memory | Sensitive to learning rate and oscillation |
| Momentum SGD | 1 per parameter | Accumulated direction, less oscillation | Strong inertia can overshoot |
| AdaGrad | 1 per parameter | Per-parameter learning rate | Step size can keep shrinking |
| RMSProp | 1 per parameter | Step size driven by recent gradients | Needs a decay rate setting |
| Adadelta | 2 per parameter | Reflects gradient and update magnitudes | Needs more state |
| Adam | 2+ per parameter | Adjusts direction and step size together | More memory, not always the best |

Connecting the evolution of optimizers to the change in state in the code:

```text
Gradient Descent / SGD
└─ parameter.grad만 사용
   │
   │ Mini-batch Gradient가 noisy하고 진동한다.
   ▼
Momentum
└─ velocity 추가
   │
   │ 모든 parameter에 같은 Learning Rate를 사용한다.
   ▼
AdaGrad
└─ squared_grad_sum 추가
   │
   │ 과거 Gradient²가 끝없이 누적된다.
   ▼
RMSProp / Adadelta
└─ square_avg, acc_delta로 오래된 정보의 영향 감소
   │
   │ 방향 정보도 함께 사용하면 어떨까?
   ▼
Adam
├─ first_moment: Gradient 방향의 이동 평균
└─ second_moment: Gradient²의 이동 평균
```

Seen again in code, the difference is even clearer.

```python
# SGD
parameter -= lr * grad

# Momentum
velocity = beta * velocity + grad
parameter -= lr * velocity

# AdaGrad
grad_sum += grad**2
parameter -= lr * grad / (sqrt(grad_sum) + eps)

# RMSProp
square_avg = beta * square_avg + (1 - beta) * grad**2
parameter -= lr * grad / (sqrt(square_avg) + eps)

# Adam
first_moment = beta1 * first_moment + (1 - beta1) * grad
second_moment = beta2 * second_moment + (1 - beta2) * grad**2
parameter -= lr * m_hat / (sqrt(v_hat) + eps)
```

> **The evolution of optimizers is not simply a race to descend faster.**  
> **It is the process of deciding what extra information to store and reflect in the update, depending on the problems met during training.**

---

## Reference

- Sebastian Ruder, *An overview of gradient descent optimization algorithms*, [arXiv:1609.04747](https://arxiv.org/abs/1609.04747).
- [PyTorch Optimizer documentation](https://pytorch.org/docs/stable/optim.html)
