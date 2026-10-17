---
title: Soccer Broadcast Semantic Segmentation
date: 2026-10-17T17:26:00+09:00
description: Taking a ball that covers 0.1% of pixels from IoU 0 to 0.77 — a U-Net++ project that narrowed down the cause of class imbalance through experiments.
kind: [code]
tools: [python, pytorch]
role: Data checks · Experiment design · Modeling · Evaluation
accent: "#15803d"
cover: ./media/cover.png
featured: true
draft: false
---

> **TL;DR**
> I trained a pixel-level segmentation model on 100 soccer broadcast frames, and **classes making up less than 1% of pixels** — the ball, the referee, a goalkeeper — swung between IoU 0 and 0.8 depending on the fold. A seed experiment showed the cause was not the data split but **unstable training**, and switching the loss to **CE + Dice + class weights** raised test mIoU from **0.770 → 0.873** and ball IoU from **0 → 0.771**.

## 1. Problem: the ball isn't detected

The data is **100 frames at 1920×1080** taken from 2017 UEFA Super Cup highlights, with 10 classes (team A/B, goalkeeper A/B, referee, ball, goal bar, ground, audience, advertisements).

The class ratios in the dataset documentation were by instance count. But the segmentation loss is computed per pixel, so I looked again by pixel area, and that is where the problem showed up.

![Instance ratio vs pixel area ratio](./media/class-ratio.png)

| Class | Instance ratio | Pixel area ratio |
|---|---|---|
| Ground | 9.4% | **40.5%** |
| Ball | 7.0% | **0.1%** |
| Referee | 2.6% | **0.4%** |

The ball is 7% by instances but 0.1% by pixels — 1/400 of the ground.

## 2. Approach: suspect the data before the model

Before training, I closed off the places where scores could get inflated.

- **Exact duplicate images (MD5)**: 1 pair → kept on the same side
- **Near-duplicate images (pHash)**: 32 of 100 → frames extracted from video include many look-alike scenes. All fixed to train
- **Split design** → from the rest, searched for a split where instance count, pixel area and number of images containing each class were all close to 7:1.5:1.5

## 3. Experiments: narrowing the cause in three steps

| Step | What I did | What I learned |
|---|---|---|
| LAB 1 | U-Net++ (ResNet34) + BCE + Dice, 5-fold cross-validation | Large classes were stable above 0.9, but **only the three small classes** swung between IoU 0 and 0.8 per fold. AP above 0.7, yet IoU 0 |
| LAB 2 | Retrained the same fold changing **only the seed** | With the same data, referee IoU ranged from 0 to 0.93 → the cause was not the data split but **unstable training** |
| LAB 3 | BCE → **CE**, plus **class weights** inversely proportional to pixel frequency | Mean mIoU +0.126, standard deviation across folds **0.099 → 0.024** |

The decisive clue was the pattern **"high AP but IoU 0"**. The model did know something about the ball; it was losing the final argmax competition to larger classes.

![Val mIoU per fold and small-class IoU comparison](./media/results.png)

## 4. Results

The 15 test images were never used for training or model selection and were evaluated **exactly once**.

| Model | mIoU | mDice | Ball | Referee | Goalkeeper B |
|---|---|---|---|---|---|
| BCE + Dice | 0.770 | 0.828 | **0.000** | 0.689 | 0.767 |
| **CE + Dice + weights** | **0.873** | **0.931** | **0.771** | **0.878** | **0.806** |

![Original · ground truth · prediction](./media/predictions.png)

## 5. Lessons and limitations

- Look at data in the **unit the loss is computed in**. Had I only looked at instance ratios, I would have spent a long time wondering why the ball wasn't detected.
- I ran the **cheapest experiment that separates the two hypotheses** (changing only the seed) first.
- LAB 3 switched to CE and added class weights **at the same time**, so their individual contributions weren't separated. I plan to run them separately next.
- The boundary metric (mBIoU 0.43) is still low, and using this for real broadcast tracking would also require checking stability across consecutive frames and speed.

> The full experiment record — the data checks, per-fold numbers and analysis of missed objects — is in the [Log post](/log/computer-vision/segmentation/soccer-segmentation-class-imbalance).
