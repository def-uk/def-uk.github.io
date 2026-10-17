---
title: "Cat and Dog Face Detection: Comparing SSDLite320 and YOLOv8n Under the Same Conditions"
date: 2026-10-17T23:21:00+09:00
description: "Comparing SSDLite320 and YOLOv8n on Oxford-IIIT Pet face boxes with matched training conditions. The gap was not in mAP@0.5 but in box precision at high IoU, and in inference speed."
tags: ["Oxford-IIIT Pet", "SSDLite320", "YOLOv8n"]
draft: false
---
> **On the same Oxford-IIIT Pet face-box data, which model finds faces more accurately and faster: SSDLite320 or YOLOv8n?**

This is not a classification problem of guessing 37 breeds. It is an object detection problem: find where the animal's face (head ROI) is in the photo, with two classes, `cat_face` and `dog_face`. While comparing two lightweight detectors, I kept asking "am I really comparing them under the same conditions?", and along the way I redesigned the comparison once.

## TL;DR

- YOLOv8n, with its training conditions (optimizer, learning-rate candidates, EMA, post-processing) matched to SSDLite320, was more accurate: holdout mAP@0.5:0.95 **0.7533 → 0.8401** (mean of 3 seeds).
- mAP@0.5 was similar for both models, around 0.99. The gap came from **box precision at high IoU thresholds**.
- YOLOv8n was also faster at inference. Median on this Mac: MPS **11.5 → 7.2 ms**, CPU **108.1 → 11.0 ms**.

## Dataset

| Item | Details |
| --- | --- |
| Source | Oxford-IIIT Pet Dataset (Parkhi et al., CVPR 2012) |
| Images | 7,390 (12 cat breeds, 25 dog breeds, 37 breeds total) |
| Labels | 3,686 face-box files in PASCAL VOC XML format |
| Classes | `0 = cat_face`, `1 = dog_face` |
| Official split | `trainval` 3,680 images, `test` 3,669 images |

![](./image/01.png)

*12 random images from train/validation. Orange boxes are the XML ground truth. Face size, position, and background vary a lot.*

Here the first problem shows up: **the official test set has no face XML at all.** So AP cannot be computed on test, and I had to re-split the labeled `trainval` into train/validation/holdout.

## Checking the data: what can be trusted before training

### README vs. actual files

Most of the README matched the actual files. But three points could not be taken at face value.

| README says | Actual data | Decision |
| --- | --- | --- |
| BREED ID: cats 1\~25, dogs 1\~12 | `list.txt` has **cats 1\~12, dogs 1\~25** | Treated as a README typo; used `list.txt` values |
| `list.txt` lists all images | 7,349 of the 7,390 images are in `list.txt` | 41 images outside the list |
| Every image has a head ROI label | Only 3,686 XML files; 0 for official test | test not used for AP evaluation |

### File matching and image quality

| Check | Result |
| --- | --- |
| `trainval` images with both image and XML | **3,671** (9 without XML) |
| Images outside the list | 41 (15 of them have XML) |
| Filename/XML mismatch, parse failures, duplicate IDs across splits, metadata conflicts | 0 for all |
| Decode failures | 0 / 7,390 |
| Files with a `.jpg` extension but a different real format | 10 (6 GIF, 4 PNG) — none have XML |
| Format of the 3,671 training candidates | All RGB JPEG; XML size matches actual size for all |

![](./image/02.png)

*Width, height, and aspect-ratio distribution of the training candidates.* Most cluster around 500×375 (median aspect ratio 1.33), but the short side ranges from 108 to 2,448px and the aspect ratio from 0.52 to 2.56. Because of this tail, I had to check separately how much face shapes change when shrinking to 320×320.

### Box coordinates

The 3,686 XML files yielded 3,687 boxes. Malformed coordinates, sizes of 0 or less, out-of-image coordinates, unknown classes, and conflicts with official species info were all 0. The only image with two boxes is `Bengal_105` (two cats), so the dataset and evaluator were built to accept multiple boxes per image.

The coordinate minimum was 1, not 0, and some boxes had `xmax` equal to the image width. I read these as PASCAL VOC-style "1-based, endpoint-inclusive" coordinates and converted them to continuous `xmin-1, ymin-1, xmax, ymax` coordinates for the models.

Passing the coordinate rules does not mean a box actually covers the face, so I checked that separately. I looked at 34 distinct images by eye: 12 random ones, the 8 smallest and 8 largest boxes by relative area, 6 aspect-ratio extremes, and the two-box image.

![](./image/03.png)

*The 8 boxes with the smallest relative area (0.4–1.0% of the image).* Even the small boxes pointed precisely at the faces of distant animals. Among the aspect-ratio extremes, only `beagle_181` had a box extending into part of the body, but I did not fix labels based on a single case.

### Duplicate images and the split

I used MD5 hashes to find byte-identical files. Of 7,390 images there were 7,360 unique hashes, with 29 groups (59 images) sharing a hash.

- Groups containing 2+ candidates: **0**
- Groups shared by a candidate and an official test image: **0**
- 30 of the 41 out-of-list images were copies of in-list images.

So **I excluded the 41 out-of-list images from the split.** They have no official species/breed info, and if copies cross split boundaries, evaluation can come out optimistic.

I split the remaining 3,671 candidates 70:15:15, stratified by breed (split seed 42).

| Split | Images | Boxes | Cat ratio | Purpose |
| --- | --- | --- | --- | --- |
| train | 2,567 | 2,568 | 0.321 | Training |
| validation | 554 | 554 | 0.323 | Learning rate, best epoch, model selection |
| holdout | 550 | 550 | 0.322 | One final evaluation after settings are fixed |

There were 0 shared `image_id`s and MD5s across splits. **The holdout was never used for later preprocessing decisions, visual box checks, or tuning.**

## Preprocessing: letterbox or just stretch?

Both models needed a common 320×320 input. There were two options.

| Method | Median box short side | Short side < 16px | Face shape |
| --- | --- | --- | --- |
| Letterbox (keep ratio + padding) | 99.8px | 0% | Preserved, but on average 26.4% of the canvas is padding |
| Direct resize (stretch) | 108.2px | 0% | Distorted by the image aspect ratio (0.67–1.51 in the 5–95% range, up to 2.56) |

The difference in face size was small. So I chose the option that preserves face shape: **letterbox**.

![](./image/04.png)

*Left column: letterbox only. Right three columns: examples with augmentation added.* Even images with an aspect ratio of 2.56 (middle row) or 0.53 (bottom row) keep their faces undistorted; only the padding grows.

| Augmentation (train only) | Value |
| --- | --- |
| Horizontal flip | p = 0.5 |
| Brightness, contrast, saturation | ±0.3 |
| Hue | ±0.02 |
| Padding color | 114 (augmentation applied before letterboxing so the padding color stays fixed) |
| Excluded | Zoom, shift, crop, YOLO Mosaic |

By leaving out augmentations that only one model is used to, like Mosaic, both models saw **the same inputs**.

## Before training: verify the evaluator and the training path

The built-in evaluation functions of the two models define things slightly differently. So I built a **shared evaluator** and scored both models with it.

| Evaluation setting | Value |
| --- | --- |
| Primary metric | mAP@0.5:0.95 (secondary: mAP@0.5, per-class AP) |
| IoU thresholds | 10 thresholds, 0.50–0.95 in steps of 0.05 |
| Precision interpolation | COCO-style 101 points |
| Max detections | 100 per image per class |
| Score filter | None (for quantitative evaluation) |

The evaluator passed 30 checks, including hand-computable cases (perfect predictions, no predictions, IoU 0.667, wrong class, duplicate predictions, etc.). Cross-checked against pycocotools in a separate environment on 4 random scenarios, the mAP difference was at most 6e-17.

Before real training, I trained repeatedly on just 16 training images without augmentation to see whether the model could memorize them.

![](./image/05.png)

*SSDLite320 overfitting check.* Over 60 epochs the loss dropped from 7.95 to 3.54, and mAP@0.5:0.95 on the same 16 images rose from 0.023 to 0.759. This is not evidence of generalization; it is only evidence that **loading → coordinate conversion → loss → prediction → coordinate restoration → evaluation** is connected end to end. YOLOv8n went from 0.176 to 1.000 on the same check.

I also verified early stopping with best-epoch restoration, stopping on NaN loss, and that interrupting and resuming training gives the same result, before starting the main training.

## Experiment design: fix the budget before seeing results

Both models followed the same rules.

| Item | Value |
| --- | --- |
| Pretraining | COCO (only the classification head re-initialized for 2 classes) |
| Batch / max epochs | 16 / 30 |
| Early stopping | Stop if validation mAP@0.5:0.95 does not improve for 5 epochs |
| LR schedule | Linear warmup for 100 steps, then cosine decay |
| Experiment budget | 3 learning-rate candidates × seed 0 → seeds 1 and 2 added for the chosen rate (5 runs per model) |
| Selection criterion | Best-epoch validation mAP@0.5:0.95 |

I fixed the budget in advance so I would not add candidates after seeing results. Final numbers are reported as **mean ± sample standard deviation over 3 seeds**, not the single best seed.

## Experiment 1: SSDLite320

Using SGD (momentum 0.9, weight decay 4e-5), I searched learning rates 1e-3, 3e-4, and 1e-4.

> Partway through, I added a reproducibility guard that puts data, split, and annotation hashes into the run ID. SSDLite320 results run before that are kept as preliminary records; the numbers below are from retraining with the fixed code.

![](./image/06.png)

*Left: validation mAP@0.5:0.95. Right: training loss (comparable across learning rates since it is the same model).* The lower the learning rate, the higher the loss where it stopped. Within the 30-epoch budget, these look undertrained.

| Learning rate | Best / stop epoch | mAP@0.5:0.95 | mAP@0.5 | Final training loss |
| --- | --- | --- | --- | --- |
| **1e-3** | 13 / 18 | **0.7580** | 0.9757 | 0.92 |
| 3e-4 | 22 / 27 | 0.7293 | 0.9644 | 1.72 |
| 1e-4 | 21 / 26 | 0.6420 | 0.9078 | 3.07 |

I picked 1e-3 and added seeds 1 and 2; all three seeds stopped early.

| Seed | 0 | 1 | 2 | Mean ± SD |
| --- | --- | --- | --- | --- |
| mAP@0.5:0.95 | 0.7580 | 0.7655 | 0.7605 | **0.7613 ± 0.0038** |

## Experiment 2: YOLOv8n default recipe — is this a fair comparison?

At first I ran YOLOv8n close to the Ultralytics default training recipe: AdamW, EMA, gradient clipping, NMS IoU 0.7, with learning-rate candidates 2e-3, 8e-4, and 3e-4.

| Learning rate | Best / stop epoch | mAP@0.5:0.95 |
| --- | --- | --- |
| 2e-3 | 30 / 30 | 0.8014 |
| 8e-4 | 30 / 30 | 0.8377 |
| **3e-4** | 25 / 30 | **0.8534** |

Repeating seeds at 3e-4 gave **0.8511 ± 0.0025**, nearly 0.09 higher than SSDLite320.

But I could not compare these numbers as they were. The optimizer, EMA, post-processing NMS threshold, and learning-rate range were **all different**. As it stood, I could not tell whether the gap came from model architecture or from the training recipe.

## Experiment 3: YOLOv8n with conditions matched to SSDLite320

So I retrained YOLOv8n with its training recipe matched to SSDLite320 as closely as possible.

| Item | SSDLite320 | YOLOv8n default (reference) | **YOLOv8n matched** |
| --- | --- | --- | --- |
| Optimizer | SGD, momentum 0.9, wd 4e-5 | AdamW, wd 5e-4 | **SGD, momentum 0.9, wd 4e-5** |
| LR candidates | 1e-3, 3e-4, 1e-4 | 2e-3, 8e-4, 3e-4 | **1e-3, 3e-4, 1e-4** |
| EMA | No | Yes | **No** |
| Gradient clipping | No | Norm 10 | **No** |
| Backprop loss | Per-image mean | Batch sum | **Per-image mean** |
| Validation NMS IoU | 0.55 | 0.7 | **0.55** |
| Data, augmentation, evaluator, seeds | — | Same | Same |

Loss functions, model architecture, and the re-initialized portion differ by model, so the conditions are not perfectly identical. But **every training setting that could be changed was matched.**

![](./image/07.png)

*Learning-rate search for matched YOLOv8n.* SSDLite320's best score spread from 0.64 to 0.76 depending on learning rate, while YOLOv8n's three rates stayed within 0.827–0.844. Over the same candidate range, it was less sensitive to learning rate.

| Learning rate | Best / stop epoch | mAP@0.5:0.95 | mAP@0.5 |
| --- | --- | --- | --- |
| **1e-3** | 20 / 25 | **0.8442** | 0.9916 |
| 3e-4 | 10 / 15 | 0.8281 | 0.9836 |
| 1e-4 | 16 / 21 | 0.8272 | 0.9800 |

| Seed | 0 | 1 | 2 | Mean ± SD |
| --- | --- | --- | --- | --- |
| mAP@0.5:0.95 | 0.8442 | 0.8451 | 0.8306 | **0.8399 ± 0.0081** |

With matched conditions, YOLOv8n's score dropped 0.0112 from the default recipe (0.8511). Still, it remained well above SSDLite320. Since optimizer, EMA, and learning-rate range changed together, I did not isolate which of them accounts for the 0.0112.

## Validation comparison

![](./image/08.png)

*Blue: SSDLite320, orange: YOLOv8n (matched), 3 seeds each.* After the first few epochs, the two models' curves never overlap. The gap between models is far larger than the seed-to-seed variation.

| Model | mAP@0.5:0.95 | mAP@0.5 | Train time per epoch (s) | Params (M) |
| --- | --- | --- | --- | --- |
| SSDLite320 | 0.7613 ± 0.0038 | 0.9760 ± 0.0008 | 34 | 2.22 |
| **YOLOv8n (matched)** | **0.8399 ± 0.0081** | 0.9890 ± 0.0023 | 24 | 3.01 |

- The mean difference is **+0.0786**.
- Even YOLOv8n's lowest seed (0.8306) beats SSDLite320's highest seed (0.7655).
- Yet **mAP@0.5 is similar for both models, 0.97–0.99.**

This was the most important clue in the experiment. Both models are nearly saturated at finding roughly where the face is. The gap opens at IoU 0.75 or 0.9, in **how tightly the box is drawn**. I did not, however, break box error down into position and size.

![](./image/09.png)

*Per-class AP@0.5:0.95 (3-seed mean ± SD).* For both models, cats score higher than dogs: 0.803 vs 0.720 for SSDLite320, 0.885 vs 0.795 for YOLOv8n. The pattern held in every run across learning rates and seeds. I did not experiment on the cause.

Drawing both models' predictions on the same 8 validation images makes the difference visible (only scores ≥ 0.5 shown, for visualization).

![](./image/10.png)

*SSDLite320.* Solid orange is ground truth; dashed blue is the prediction.

![](./image/11.png)

*YOLOv8n (matched).* The same 8 images.

In most images, YOLOv8n's dashed box hugs the solid ground-truth box more closely. Each model also has one miss. SSDLite320 had no box ≥ 0.5 on the side-facing `Ragdoll_190`. YOLOv8n had no box ≥ 0.5 on `saint_bernard_185` in the snow, though SSDLite320 also produced a box well off the ground truth (0.63) on that image. This is an 8-image sample drawn with a 0.5 display threshold, so it should be kept separate from the quantitative evaluation (no score filter).

### Extra check: does the conclusion hold with YOLOv8n at 3e-4?

Since matched YOLOv8n at 1e-3 showed seed variation (0.8306–0.8451), I also ran seeds 1 and 2 for the runner-up candidate, 3e-4.

| YOLOv8n matched | mAP@0.5:0.95 per seed | Mean ± SD |
| --- | --- | --- |
| lr 1e-3 (selected) | 0.8442 / 0.8451 / 0.8306 | 0.8399 ± 0.0081 |
| lr 3e-4 (post-hoc check) | 0.8281 / 0.8334 / 0.8328 | 0.8314 ± 0.0029 |

The two learning rates' ranges overlap, so neither can be declared better. Still, even at 3e-4 YOLOv8n was 0.0701 above SSDLite320, so **the direction of the model comparison held**. Because this check spent extra budget on YOLOv8n only, after seeing results, I kept 1e-3 as the representative learning rate per the pre-set rule.

## Final evaluation: holdout and inference speed

With all settings fixed, I evaluated each model's seed 0, 1, and 2 checkpoints (6 in total) once on the 550 holdout images (177 cats, 373 dogs). Each checkpoint was first checked to reproduce its saved validation score exactly (all 6 passed), and only then scored on holdout.

![](./image/12.png)

*Holdout mAP@0.5:0.95, cat AP, dog AP. Bars are 3-seed means; dots are individual seeds.*

| Model | mAP@0.5:0.95 | mAP@0.5 | cat AP | dog AP |
| --- | --- | --- | --- | --- |
| SSDLite320 | 0.7533 ± 0.0015 | 0.9872 ± 0.0032 | 0.7854 ± 0.0028 | 0.7211 ± 0.0005 |
| **YOLOv8n (matched)** | **0.8401 ± 0.0082** | 0.9895 ± 0.0056 | **0.8667 ± 0.0142** | **0.8134 ± 0.0022** |

On holdout the gap was **+0.0868**, in the same direction as validation. YOLOv8n's lowest seed (0.8341) beats SSDLite320's highest seed (0.7544).

![](./image/13.png)

*The dashed line is validation = holdout.* All six points sit near the diagonal. Moving from validation to holdout, the mean change was −0.0080 for SSDLite320 and +0.0001 for YOLOv8n. Even though learning rate and epoch were chosen on validation, holdout scores did not collapse.

Speed was measured separately from accuracy.

| Measurement setup | Value |
| --- | --- |
| Scope | One in-memory RGB image → letterbox → device transfer → inference → NMS → restore original coordinates (excludes file reading and model loading) |
| Settings | Batch 1, float32, 200 holdout images, 20 warmup images, 3 repeats per model (model order interleaved) |
| Post-processing | Score 0.001, NMS IoU 0.55, max 300 |
| Environment | Apple Silicon Mac (MPS, CPU 12 threads) |

![](./image/14.png)

*Bars are medians; whiskers are P95.*

| Device | Model | Median (ms) | P95 (ms) | FPS |
| --- | --- | --- | --- | --- |
| MPS | SSDLite320 | 11.5 | 13.9 | 83.1 |
| MPS | **YOLOv8n** | **7.2** | 8.8 | **134.7** |
| CPU | SSDLite320 | 108.1 | 110.8 | 9.2 |
| CPU | **YOLOv8n** | **11.0** | 11.5 | **90.3** |

SSDLite320 (2.22M) has fewer parameters than YOLOv8n (3.01M), yet YOLOv8n was about 1.6× faster on MPS and about 10× faster on CPU. This confirmed that parameter count alone is not a good guide to speed.

## Conclusion

The selection rule was set before the experiments: **"the fastest model among candidates within 1%p of the best model's validation mAP@0.5:0.95."**

- SSDLite320 is 7.86%p below YOLOv8n on validation, so it is not within 1%p. Holdout pointed the same way.
- YOLOv8n was also faster, so there was no accuracy–speed trade-off at all.
- The final model is **YOLOv8n (matched, learning rate 1e-3)**, and the settings were not changed after seeing the holdout results.

Looking back, two decisions were decisive.

1. **Not using the YOLO default-recipe result as is.** There was a better number, 0.8511, but with a different optimizer, EMA, and post-processing, it could not be called a model comparison. Matching conditions narrowed the gap (0.8511 → 0.8399), and what remains after narrowing is closer to a difference between the models.
2. **Looking at mAP@0.5 and mAP@0.5:0.95 side by side.** Had I looked only at mAP@0.5, I would have concluded the two models were similar. The difference was in how precisely the box fits.

## Limitations and next steps

- **The conditions are not perfectly identical.** Loss functions, model architecture, and the re-initialized portion could not be matched. SGD at 1e-3 to 1e-4 is a range chosen for SSDLite320, so this was not an experiment that found YOLOv8n's best settings either.
- **The learning-rate search hit the edge of its range.** For both SSDLite320 and matched YOLOv8n, the largest candidate, 1e-3, was selected. A larger learning rate might do better.
- **Learning-rate selection opportunities were asymmetric.** Only YOLOv8n got post-hoc seed repeats at 3e-4.
- **Only 3 seeds and 1 split seed (42).** Near-duplicates (resized or recompressed copies) were not checked, and holdout has only 13–15 images per breed, so per-breed interpretation is hard.
- **Speed figures are reference values for this Mac.** They may differ from the originally planned Colab GPU or a real deployment environment.
- **Per-image failure cases and memory usage were not analyzed.** Since the models differed at high IoU, the next step is to collect cases where boxes are off and split the error into position and size. Why dog AP was lower than cat AP in every run is also still unverified.
