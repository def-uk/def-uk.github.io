---
title: "SSD300 and YOLO"
date: 2026-10-05T23:20:00+09:00
description: "Comparing SSD300 and YOLOv1 on two axes — grids and boxes — and following how each YOLO version from v1 to YOLO26 fixed the previous one's weakness."
tags: ["DL", "DeepLearning", "SSD", "YOLO", "YOLOv1", "YOLOv2", "model", "yolov3"]
draft: false
---
## Introduction

SSD300 and YOLO are both 1-stage detectors that find objects by looking at the image only once. In the end, their differences come down to two things: **how many grids they use and at what resolution**, and **whether they draw boxes from scratch or adjust pre-placed templates**.

This post compares SSD300 and YOLOv1 along these two axes, then follows which weakness of the previous version each YOLO release, from v1 to YOLO26, fixed and how. It assumes you already know basic object detection concepts such as bounding boxes and IoU.

## Basics: a grid is a feature map

YOLO and SSD are both CNNs. The image is not cut into pieces; it goes in whole, and **the width and height of the feature map that comes out of the CNN is the "grid"**.

Take YOLOv1: a 448×448×3 image becomes a 7×7 feature map after convolutions and pooling. Since 448 ÷ 7 = 64, one position in the feature map corresponds to roughly a 64×64-pixel region of the original image. What people call "one cell" is one position in the feature map, and the bundle of numbers at that position is the prediction.

YOLOv1's final output is 7×7×30. The 30 numbers in a cell are 2 boxes × (x, y, w, h, confidence) = 10, plus 20 class probabilities (for VOC). Strictly speaking, v1 reshapes the output of its last fully connected layer into 7×7×30; from v2 on, the convolutional feature map is used directly as the grid.

Feature maps behave differently depending on where they come from in the CNN. This difference runs through everything that follows.

|  | Early layers | Deep layers |
| --- | --- | --- |
| Feature map size | Large (dense grid) | Small (sparse grid) |
| Field of view per position | Narrow | Wide |
| Good at | Details like edges and textures | Semantic information like "this is a bird" |
| Analogy | A rookie with sharp eyes but no experience | A veteran with lots of experience but poor eyesight |

![](./image/01.png)


## YOLOv1 vs SSD300: there are two differences

### Difference 1. Feature maps: one vs several resolutions

YOLOv1 predicts objects of every size from a single 7×7 grid. SSD300 uses six feature maps taken from different points in the CNN, each for its own predictions. Each position holds 4–6 default boxes of different shapes.

| SSD300 feature map | Field of view per position (300px input) | Default boxes per position | Boxes |
| --- | --- | --- | --- |
| 38×38 | ~8px | 4 | 5,776 |
| 19×19 | ~16px | 6 | 2,166 |
| 10×10 | 30px | 6 | 600 |
| 5×5 | 60px | 6 | 150 |
| 3×3 | 100px | 4 | 36 |
| 1×1 | 300px | 4 | 4 |
| Total |  |  | 8,732 |

YOLOv1 has 7 × 7 × 2 = 98 boxes. So it is tempting to think "SSD catches small objects well because it has many boxes", but the count is not the point. Even if you raised YOLOv1 to 178 boxes per cell for a total of 8,722, it would still struggle with small objects. There are two reasons.

- **Resolution**: 7×7 is the feature map of the deepest layer, so a small 20px object has already been smeared out. You cannot catch an object you cannot see, no matter how many boxes you draw.
- **One set of classes per cell**: v1 predicts classes only once per cell. If the centers of several small objects fall into one cell, it can effectively handle only one of them.

SSD solves this with **division of labor by size**. Small templates on the less-compressed 38×38 feature map handle small objects, while large templates on the wide-view 3×3 and 1×1 maps handle large objects.

> The difference is not the number of boxes but whether grids at multiple resolutions split the work by object size.

### Difference 2. Boxes: drawing from scratch vs adjusting templates

Say a person's box is 50 pixels wide and 150 pixels tall.

- **YOLOv1**: has to hit the values "50 wide, 150 tall" themselves, with no reference point. Early in training it starts from arbitrary values.
- **SSD300**: a tall default box (e.g. 60×180) is already sitting there. It only has to learn small corrections like "shrink the width a bit, shrink the height a bit".

Because the range of values to predict is narrower, training is more stable.

![](./image/02.png)


### Summary: v1's two weaknesses and the tools that fix them

| v1's weakness | Fix | SSD | YOLO |
| --- | --- | --- | --- |
| Misses small objects | Grids at multiple resolutions (feature maps) | From the start | Introduced in v3 |
| Box learning is hard | Pre-placed templates | Default boxes | Anchors introduced in v2 |

SSD came out with both tools from the start, and YOLO caught up one at a time across v2 and v3.

SSD has a weakness too, though. The 38×38 map that handles small objects comes from an early layer, so it sees position well but lacks the semantic information needed to decide what the object is. It is like leaving the rookie to judge alone. This weakness comes up again in YOLOv3.

## YOLOv2: changing the templates

**Problem solved**: v1 predicted boxes without templates, which made training hard.

**What changed**: it placed 5 anchors chosen by k-means in every cell of a 13×13 feature map (for a 416×416 input), and predicted only corrections to those templates.

SSD's default boxes had shapes chosen by the researchers. But if most objects in your data are long and thin, like utility poles or trains, predefined templates don't fit well. v2 clustered the real boxes in the training data with k-means to pick representative shapes. As the distance it used **1 − IoU** instead of Euclidean distance (how many pixels apart). With Euclidean distance, large boxes get a big distance even when they differ only slightly, which is unfair.

A cell is one position, but an anchor is a template placed on the original image centered on that cell, so it can be much larger than the cell. It is like drawing several circles of different sizes around a single pin stuck in a map. The deep feature of a cell was built from a region much wider than 32×32 through many convolutions, so it can judge large boxes too.

The output is 13×13×125 (for VOC).

- One anchor = 4 position corrections + 1 objectness + 20 class probabilities = 25
- One cell = 25 × 5 anchors = 125

v1 had only one set of class probabilities per cell, but v2 has one **per anchor**. So a single cell can talk about different objects at once, such as a person and a bicycle.

For reference, v2 also used a passthrough layer that brought in details from the earlier 26×26 map, but it still predicted from only one place, the 13×13 map. The grid problem carries over to v3.

![](./image/03.png)


## YOLOv3: changing the grid

**Problem solved**: v2 used only one feature map for prediction, so it was weak on small objects.

**What changed**: it predicts from three feature maps, 13×13, 26×26 and 52×52 (for a 416×416 input), and merges information from deep layers into earlier layers by upsampling it.

It keeps v2's k-means approach to pick 9 anchors and assigns 3 to each grid according to size. It effectively combines SSD's division of labor by size with v2's data-fitted templates.

There is also a part where it goes one step beyond SSD. SSD leaves small objects to the early layer (the rookie) alone, but YOLOv3 enlarges the judgment of the deep layer (the veteran) and combines it with the early layer's details to make its predictions. This structure comes from the idea behind FPN (Feature Pyramid Network).

You can see which way information flows from the shape of the operations. Upsampling enlarges a small feature map. To merge 13×13 with 26×26 they must be the same size, so the deep 13×13 map is doubled and merged into 26×26. 26×26 is merged into 52×52 the same way. In other words, information flows **from deep layers → to early layers**.

![](./image/04.png)


## YOLOv8: dropping anchors

**Problem solved**: k-means anchors were trapped in the data. They had to be recomputed whenever the dataset changed, and predictions still started from only a fixed handful of shapes.

**What changed**: without anchors (anchor-free), each grid point directly predicts four distances to the object's left, top, right and bottom edges.

It kept the grids at three resolutions. For a 640×640 input it predicts from 80×80, 40×40 and 20×20 feature maps. Points on the dense 80×80 grid handle only small objects, and points on the sparse 20×20 grid handle only large ones.

So why isn't training hard, if it predicts without templates like v1? v1 had reference points (cells) too. The difference is how dense the reference points are and how narrow each one's range is. In v1 one point handled every size, but in v8, thanks to the division of labor by size, the distances each point predicts shrink to a narrow range that matches its grid size.

> The size-based grid split that started in v3 stands in for the "roughly this big" starting point, so v8 can learn the distance from each point to the edges stably without anchors that have to be refitted for every dataset.

Two caveats:

- The first anchor-free attempt in the YOLO family was not v8 but YOLOX (2021).
- v8 changed many parts at once besides going anchor-free. So you cannot conclude that "dropping anchors improved performance". "The architecture became simpler" is the accurate way to put it.

![](./image/05.png)


## YOLOv10 / YOLO26: dropping NMS

**Problem solved**: NMS can delete one of two tightly packed objects of the same class, and it is slow because it is separate post-processing outside the model.

**What changed**: instead of removing duplicate boxes afterwards, the model is trained from the start to output only one box per object, eliminating NMS.

### How NMS works

1. **Confidence filter (before NMS)**: discard boxes whose score is below a threshold (e.g. 0.25).
2. **Sort**: line up the remaining boxes by score within each class.
3. **Suppress**: delete boxes whose IoU with the top box exceeds a threshold (e.g. 0.5). Pick the top box among the rest and repeat.

The score decides "who becomes the representative", and IoU decides "who is a duplicate of that representative". A box is deleted not because its score is low, but because **it overlaps heavily with a box that scores higher**. For example, if the same cat has boxes scoring 0.9, 0.8 and 0.3, the 0.8 box that overlaps the 0.9 box at IoU 0.7 is deleted, while the 0.3 box at IoU 0.1 is treated as a different object and kept.

### NMS's weaknesses

- **Crowded objects**: if two people with arms around each other have boxes scoring 0.9 and 0.85 with an IoU of 0.6, a threshold of 0.5 makes the second person, who really exists, disappear. Raise the threshold to 0.7 and duplicate boxes survive instead. Having a person find the right value for every dataset is the same kind of problem as anchors.
- **Speed and deployment**: it compares boxes one by one in a loop, so it gets slower as boxes increase and becomes an obstacle when deploying models to edge devices.

### NMS-free

During training, YOLOv10 learns both "several points answer for one object" and "only the single best point answers for one object", and at inference it uses only the latter, eliminating NMS.

YOLO26, released by [Ultralytics](https://docs.ultralytics.com/models) in January 2026, continues in this direction. According to the Ultralytics roadmap, the YOLO26 paper centers on an NMS-free end-to-end architecture and a new optimizer called MuSGD. The next-generation YOLO27 was announced at the YOLO Vision 2026 event in September 2026, but as of the official GitHub the model has not been released yet.

![](./image/06.png)


## Summary at a glance

Read as "a process of fixing the previous version's weaknesses one at a time", YOLO's evolution becomes a single story.

| Version | Previous problem | What changed | What was changed |
| --- | --- | --- | --- |
| YOLOv1 | (starting point) | Predicts 2 boxes per cell without templates from one 7×7 feature map | — |
| SSD300 | YOLOv1's two weaknesses | Division of labor by size across 6 feature maps + default box corrections | Grid, templates |
| YOLOv2 | Predicting boxes without templates made training hard | Corrects 5 anchors chosen by k-means (1 − IoU) | Templates |
| YOLOv3 | One prediction feature map, weak on small objects | 3 feature maps + merging deep-layer information into early layers (FPN-style) | Grid |
| YOLOv8 | Anchors trapped in the data | Anchor-free, predicts distances from each point to the edges | Templates removed |
| YOLOv10 / YOLO26 | NMS deletes crowded objects and is slow | Trained for one box per object, NMS removed | Post-processing removed |

There is one rule for deciding which structure was a weakness: **if later versions kept a structure, it was a successful idea; if they dropped it, it was a weakness.** Grids at multiple resolutions have been kept from v3 to today, while anchors and NMS were eventually dropped.

## References

- [Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016 (arXiv 1512.02325)](https://arxiv.org/pdf/1512.02325)
- [Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016 (arXiv 1506.02640)](https://arxiv.org/pdf/1506.02640)
- [Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger", CVPR 2017 (arXiv 1612.08242)](https://arxiv.org/pdf/1612.08242)
- [Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018 (arXiv 1804.02767)](https://arxiv.org/pdf/1804.02767)
- [Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017 (arXiv 1612.03144)](https://arxiv.org/pdf/1612.03144)
- [Ge et al., "YOLOX: Exceeding YOLO Series in 2021" (arXiv 2107.08430)](https://arxiv.org/pdf/2107.08430)
- [Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", 2024 (arXiv 2405.14458)](https://arxiv.org/pdf/2405.14458)

Official documentation

- [Ultralytics supported models](https://docs.ultralytics.com/models)
- [Ultralytics roadmap](https://www.ultralytics.com/roadmap)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLO Vision 2026 highlights](https://www.ultralytics.com/blog/key-highlights-from-ultralytics-yolo-vision-2026)
