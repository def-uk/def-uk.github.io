---
title: "How Do Self-Driving Cars Understand the Scene Around Them?"
date: 2026-10-11T21:32:00+09:00
description: "Semantic, instance and panoptic segmentation compared — what each one answers, how panoptic combines them, and a Mask2Former example."
tags: ["Panoptic Segmentation", "image segmentation", "instance segmentation", "semantic segmentation"]
draft: false
---
A self-driving car has to tell the cars and pedestrians around it apart while also making sense of areas such as the road and the sidewalk. One technique that can be used for this kind of scene understanding is **Panoptic Segmentation**.

## 1. First, what is image segmentation?

Segmentation means "dividing". **Image segmentation is the technique of classifying or grouping the pixels of an image to draw the boundaries of objects and regions.**

Unlike object detection, which marks where a car is with a rectangle, segmentation also captures the shape of the pixels the car occupies. The result that marks a particular region is called a **mask**.

Depending on how the pixels are divided, it splits into semantic, instance and panoptic segmentation.

## 2. Semantic segmentation — what is this pixel?

**It assigns a class, that is, a kind, to each pixel.** Person pixels get the person color, car pixels the car color, road pixels the road color.

The same class uses the same color. Even if there are two cars, both are marked as "car" and individual cars are not distinguished. This does not mean it thinks the two cars are the same object.

![](./image/01.png)

### Where is it used?

For problems where the layout of regions in a scene matters. Examples include analyzing the area of farmland, cities and forests in satellite imagery, or finding the region a tumor occupies in medical images.

### Representative models

- **FCN:** a representative early model that predicts a class for each pixel using convolutions.
- **U-Net:** connects encoder and decoder features to restore regions and boundaries, and is widely used for medical image segmentation.
- **DeepLab series:** predicts per-pixel classes using context information at multiple ranges.
- **SegNet:** reuses the encoder's pooling indices in the decoder to restore the segmentation.

## 3. Instance segmentation — even within the same kind, which one is which?

**It distinguishes individual objects even when they belong to the same class.**

If there are two cars, both are in the "car" class but each gets its own mask. People are distinguished one by one as well.

![](./image/02.png)

### Where is it used?

For counting the people in a store, or separating individual cells in cell images. Combined with tracking, it can also be used to follow a specific car or person through a video.

**Distinguishing objects within one image and tracking the same object across frames are separate problems.** Having an instance ID does not mean the same ID is kept in the next frame.

### Representative models

- **Mask R-CNN:** predicts a mask for each object along with the detection result.
- **YOLO segmentation models:** predict each object's class and mask, and are used where fast inference is needed.
- **YOLACT / YolactEdge:** build masks by combining shared mask patterns with per-object coefficients.

## 4. Why do we need panoptic segmentation?

Picture a road scene: two cars, two pedestrians, the road, the sidewalk and the sky.

Semantic segmentation explains the kind of every pixel but **cannot tell the cars and pedestrians apart individually.** Instance segmentation distinguishes individual objects but **usually does not segment background regions like the road or the sky.**

To understand the whole scene, you need both kinds of information.

- Cars and pedestrians must be distinguished one by one.
- The road, sidewalk and sky must be distinguished by region class.

**Panoptic segmentation is the concept that unifies per-pixel classification and per-object segmentation into a single task.**

## 5. How do the three differ?

| | Semantic | Instance | Panoptic |
| --- | --- | --- | --- |
| Core question | What is this pixel? | Which object is this? | What is this pixel, and which object does it belong to? |
| Distinguishes objects of the same class | No | Yes | Yes |
| Segments regions like road and sky | Included | Usually excluded | Included |
| Output | Class per pixel | Class and mask per object | Class per pixel plus object identity |

In a scene with two cars, semantic marks both cars as the same class. Instance marks the two cars with different masks. Panoptic distinguishes each car while also marking the road and the sky.

![](./image/03.png)

## 6. How is it solved?

### 6.1. Separating things and stuff

Panoptic segmentation divides classes into two groups.

- **Things:** classes split into individual objects, like people and cars.
- **Stuff:** classes treated as regions rather than objects, like road, sky and grass.

It is easy to think of them as "countable" and "uncountable", but in practice the split follows the class definitions of the dataset you use.

### 6.2. Representing class and object information together

For example, the result can be represented like this.

| Region | Class | Object identity |
| --- | --- | --- |
| First car | Car | Car A |
| Second car | Car | Car B |
| First pedestrian | Person | Person A |
| Second pedestrian | Person | Person B |
| Road | Road | Not split into objects |
| Sky | Sky | Not split into objects |

**Things are distinguished by object even within the same class, while stuff is represented as one region per class.** An ideal panoptic result gives each pixel exactly one class and region, so the final regions never overlap. In real inference, some pixels may be left unassigned because of confidence thresholds and the like.

### 6.3. How the result is produced differs by model

Conceptually it combines the goals of semantic and instance segmentation, but that does not necessarily mean running two separate models and merging their outputs.

- **Merging results:** predict object masks and the background segmentation, then resolve overlapping regions and so on.
- **Grouping by center points:** predict each pixel's class and the direction to its object's center, then group pixels belonging to the same object.
- **Predicting classes and masks together:** predict the classes and masks of many regions and build the final segmentation map in post-processing.

### Representative models

- **Panoptic FPN**: shares features to predict instance and background segmentation together, then merges them.
- **Panoptic-DeepLab**: predicts classes plus object centers and offsets to group pixels by object.
- **Mask2Former**: uses attention focused on predicted mask regions and supports all three segmentation tasks.
- **OneFormer**: performs semantic, instance and panoptic segmentation with a single model by specifying the task type.

## 7. If the representative models differ, does the model decide the task?

The representative models were introduced per task above, but **the three tasks are defined by their output format and are not tied to a particular architecture.** U-Net, introduced as a representative semantic model, can also be used for instance segmentation depending on what it is trained to predict.

### 7.1. Why is it hard as is?

In semantic segmentation the classes are fixed, like "car" and "road", so one output channel can correspond to one class. Object IDs, on the other hand, have no fixed meaning.

- Swapping the names of car A and car B gives the same correct answer.
- The number of objects differs from image to image.

So you cannot give object IDs directly as targets to an architecture that predicts per-pixel classes with a fixed number of channels. Approaches split into two broad branches depending on how they solve this.

- **Top-down (detect, then segment):** first find the objects, then predict a mask for each one. The detection stage handles the varying number of objects. Mask R-CNN is the typical example.
- **Bottom-up (predict pixels, then group):** instead of object IDs, predict a value per pixel that does not depend on numbering, then group pixels of the same object in post-processing. Encoder-decoder architectures like U-Net can be used as they are.

### 7.2. Ways to distinguish objects with U-Net

| Value predicted per pixel | How pixels are grouped into objects | Example |
| --- | --- | --- |
| Foreground / boundary / background | Separate each region cut off by boundaries | The original U-Net paper (weights on boundaries between touching cells) |
| Distance to the object center | Watershed | Cell and nucleus segmentation |
| Radial distances from the center to the boundary | Reconstruct as a polygon | StarDist |
| Flow vectors pointing to the center | Group pixels that converge along the flow | Cellpose |
| Per-pixel embedding vectors | Clustering | Discriminative Loss |
| Class + center + offset | Assign to the nearest center | Panoptic-DeepLab |

**The architecture stays almost the same; only the outputs and the loss function change.** The "grouping by center points" approach from 6.3 belongs to this line as well.

### 7.3. Still, each approach suits certain problems

- Bottom-up assigns only one object to each pixel, so it struggles to represent overlapping objects with complete masks. Top-down predicts a separate mask for each object, so masks can overlap.
- In scenes that mix objects of very different sizes, detection-based approaches are often more stable.
- In cell images packed with objects of similar size, bottom-up approaches are widely used.
- Bottom-up results can change depending on post-processing settings (boundary thresholds, clustering criteria, etc.).

### 7.4. Handling all three tasks with one architecture

Mask2Former predicts a fixed number of masks and a class for each, and during training compares predictions and ground truth by **pairing them one-to-one (Hungarian matching)**. Because it finds the best-matching pairs regardless of order, the problem of object ID ordering disappears. So the same architecture can be trained on semantic, instance or panoptic data and used for all three tasks.

**The lists of representative models show which models first performed well on each task; tasks and models do not map one-to-one.**


## 8. How is it used in code?

Let's segment a single image with a pretrained Mask2Former. The code below is an example based on the Hugging Face documentation and the API of a COCO panoptic checkpoint.

This checkpoint uses COCO's class scheme. If your goal is road and sidewalk segmentation specifically for self-driving, choose a checkpoint trained on a dataset that includes those classes.

### 8.1. Setup

```bash
pip install torch torchvision transformers pillow matplotlib numpy
```

Prepare the image to segment as `scene.jpg`. The first run needs an internet connection to download the model files.

### 8.2. Image input → model inference → post-processing

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

checkpoint = "facebook/mask2former-swin-small-coco-panoptic"
processor = AutoImageProcessor.from_pretrained(checkpoint)
model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
model.eval()

image = Image.open("scene.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)

result = processor.post_process_panoptic_segmentation(
    outputs,
    target_sizes=[(image.height, image.width)],
    label_ids_to_fuse=set(),
)[0]

segmentation = result["segmentation"].cpu().numpy()
segments_info = result["segments_info"]

for segment in segments_info:
    label = model.config.id2label[segment["label_id"]]
    pixel_count = int((segmentation == segment["id"]).sum())
    print(
        f"영역 ID={segment['id']}, 클래스={label}, "
        f"점수={segment['score']:.3f}, 픽셀 수={pixel_count}"
    )
```

```powershell
#output
[transformers] Mask2FormerForUniversalSegmentation LOAD REPORT from: facebook/mask2former-swin-small-coco-panoptic
Key                                                                                                                      | Status     | 
-------------------------------------------------------------------------------------------------------------------------+------------+-
model.pixel_level_module.encoder.swin.encoder.layers.{0, 1, 2, 3}.blocks.{0...17}.attention.self.relative_position_index | UNEXPECTED | 
model.pixel_level_module.encoder.swin.layernorm.bias                                                                     | MISSING    | 
model.pixel_level_module.encoder.swin.layernorm.weight                                                                   | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
영역 ID=1, 클래스=car, 점수=0.998, 픽셀 수=140758
영역 ID=2, 클래스=wall-stone, 점수=0.910, 픽셀 수=17587
영역 ID=3, 클래스=person, 점수=0.999, 픽셀 수=12782
영역 ID=4, 클래스=pavement-merged, 점수=0.994, 픽셀 수=164265
영역 ID=5, 클래스=grass-merged, 점수=0.861, 픽셀 수=4038
영역 ID=6, 클래스=mountain-merged, 점수=0.876, 픽셀 수=4870
영역 ID=7, 클래스=person, 점수=0.999, 픽셀 수=12482
영역 ID=8, 클래스=car, 점수=0.999, 픽셀 수=20824
영역 ID=9, 클래스=sky-other-merged, 점수=1.000, 픽셀 수=190229
영역 ID=10, 클래스=tree-merged, 점수=0.998, 픽셀 수=591433
영역 ID=11, 클래스=road, 점수=0.997, 픽셀 수=413505
```

`label_ids_to_fuse` is used to merge masks of the same stuff class.

### 8.3. Coloring each region

```python
# facebook/mask2former-swin-small-coco-panoptic 기준
# 클래스 ID 0~79: Things(개별 객체)
thing_ids = set(range(80))

object_colors = np.array([
    [255, 40, 40],    # 빨강
    [0, 80, 255],    # 파랑
    [255, 150, 0],   # 주황
    [210, 0, 255],   # 보라
    [0, 210, 110],   # 초록
    [255, 0, 150],   # 분홍
], dtype=np.uint8)

background_colors = np.array([
    [180, 205, 220],
    [195, 215, 185],
    [220, 205, 180],
    [205, 195, 220],
], dtype=np.uint8)

original = np.array(image)
overlay = original.copy()
object_index = 0

for segment in segments_info:
    mask = segmentation == segment["id"]
    label_id = segment["label_id"]

    if label_id in thing_ids:
        # 객체마다 다른 선명한 색
        color = object_colors[object_index % len(object_colors)]
        object_index += 1
        alpha = 0.75
    else:
        background_colors = np.array([
            [0, 220, 255],    # 하늘색
            [160, 255, 40],   # 연두색
            [255, 210, 0],    # 노란색
            [180, 100, 255],  # 보라색
        ], dtype=np.uint8)

        color = background_colors[
            (label_id - 80) % len(background_colors)
        ]
        alpha = 0.65

    overlay[mask] = (
        (1 - alpha) * original[mask] + alpha * color
    ).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("Original")
axes[1].imshow(overlay)
axes[1].set_title("Panoptic segmentation")

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.savefig("panoptic_result.png", dpi=150)
plt.show()
```
![](./image/04.png)


From this result you can extract the mask of a specific class, count the predicted car regions, or compute how many pixels a region occupies. Converting pixel areas into real distances or areas requires extra information such as camera calibration and depth.

## References

- The original Panoptic Segmentation paper
- The original Mask2Former paper
- Hugging Face Mask2Former documentation
- The model checkpoint used in the example
