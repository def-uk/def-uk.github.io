---
title: "Conditional Image Generation on Fashion-MNIST: Does Adding a VAE Training Signal to a cGAN Help? (cGAN vs cVAE-GAN)"
date: 2026-10-23T13:42:00+09:00
description: "Comparing a cGAN with a cVAE-GAN that adds an encoder and a reconstruction loss to the same generator. The result was a trade-off: the cGAN wins on realism, the cVAE-GAN on diversity."
tags: []
draft: false
---
> **Problem**: If you add a VAE-style training signal (encoder + reconstruction loss) to a generator with the same architecture, does conditional image generation get better?

I trained a model that generates clothing images of a given class from a Fashion-MNIST label in two ways and compared them. The two models share the exact same generation path (generator, discriminator, label injection, loss function); only the cVAE-GAN adds an encoder and a reconstruction loss. So what changes in this experiment is **not the generator's architecture but how the generator is trained**.

"Better" is vague, so I split it into four sub-questions.

| Sub-question | Metric |
| --- | --- |
| Q1. Does it follow the label better? | Conditional accuracy |
| Q2. Is it more realistic? | FID, Precision |
| Q3. Is it more diverse? | Recall, within-class diversity ratio |
| Q4. Does it train more stably? | Loss and D-output trends, fluctuation of the FID curve |

## Summary

- The VAE-style training signal **did not make conditional generation better overall.** As expected, it was a trade-off.
- **The cGAN wins on realism**: test FID **10.35 vs 15.26**, Precision 0.854 vs 0.826, and per-class FID is lower for the cGAN in all 10 classes.
- **The cVAE-GAN wins on diversity**: Recall **0.738 → 0.802**, diversity ratio 0.962 → 1.004. In exchange, training time per epoch is about 2×.

## Dataset

| Item | Details |
| --- | --- |
| Data | Fashion-MNIST (Zalando), 28×28 grayscale |
| Size | train 60,000 / test 10,000 |
| Classes | T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot (10) |
| Split | The original train set split with stratification into train 48,000 / val 12,000. Test used only for the final evaluation |
| Preprocessing | Normalize [0, 255] → [-1, 1] (to match the tanh range of the generator's output layer) |

## 1. Checking the data

I looked at the data before training for three reasons: to ground the preprocessing, to build a reference point for the generated results, and to set up hypotheses in advance for interpreting the results.

### Integrity and distribution

- Image and label counts match, values are uint8 0–255, and there are 0 NaN or blank images. Nothing to clean.
- All 10 classes have exactly 6,000 train / 1,000 test images. So for generation and evaluation, sampling labels evenly at the same count per class (1,000) is enough.
- 50.2% of pixels are background (0). Normalizing to [-1, 1] puts the background at exactly -1.
- The per-class pixel mean differs between train and test by at most 0.006, and the standard deviation by at most 0.005. Test can serve as the real-data reference for FID.

### Class prototypes and diversity

![](https://velog.velcdn.com/images/def_uk/post/15cd08d8-e957-4bcc-92ac-858b6c739789/image.png)

The top row is the mean image per class and the bottom row is the standard-deviation image. The brighter the bottom row, the more varied the shapes within that class.

| Rank | Class | Mean pixel std |
| --- | --- | --- |
| 1 | Bag | **0.266** |
| 2 | Shirt | 0.229 |
| 3 | Pullover | 0.223 |
| ... | ... | ... |
| 9 | Trouser | 0.143 |
| 10 | Sneaker | **0.132** |

- Bag, Shirt and Pullover have the most varied shapes. If a generator produces only a few shapes, I expected diversity to drop here first (mode-collapse candidates).
- Trouser and Sneaker have consistent shapes, so I expected them to be easy to generate.

### Similarity between classes: which ones get confused?

![](https://velog.velcdn.com/images/def_uk/post/e8967c51-57be-4725-a7cb-524b3ea5d319/image.png)

In t-SNE (right) the classes split into three large clumps. The tops (T-shirt/top, Pullover, Coat, Shirt, Dress) stick together, and among them **Pullover, Coat and Shirt are mixed so much that no boundary is visible.** The three kinds of footwear gather separately, and Trouser and Bag form their own clusters. In 2-D PCA (left), the tops mostly overlap.

![](https://velog.velcdn.com/images/def_uk/post/50c4b717-5720-41ed-8a8a-368ce00cfb47/image.png)

Cosine similarity between class mean images. The top pairs are below.

| Rank | Class pair | Cosine similarity |
| --- | --- | --- |
| 1 | Coat – Shirt | **0.987** |
| 2 | Pullover – Coat | 0.984 |
| 3 | Pullover – Shirt | 0.983 |
| 4 | T-shirt/top – Shirt | 0.949 |
| 5 | Sandal – Ankle boot | 0.927 |

Half the image is background, so the values are high overall (minimum 0.44); I read the ranking rather than the absolute values. Shirt appears in 3 of the top 4 pairs, so I expected **Shirt to be the most confusable class**. This led to hypothesis H3 (see the hypothesis table below).

### Duplicates and leakage: does test overlap with train?

Since FID and Precision are measured against test, scores would be optimistic if test images were nearly identical to train images. I checked in two steps.

- **Exact duplicates (md5)**: 0 within train, 0 within test, and 0 between train and test. There are also 0 cases of the same image with different labels.
- **Near duplicates**: At first I planned to use a perceptual hash (pHash) Hamming distance ≤ 4. But at 28×28, pHash only distinguishes roughly at the level of silhouettes.

![](https://velog.velcdn.com/images/def_uk/post/ece36be2-ac06-4979-b9e3-f43f75c96132/image.png)

The left shows the pHash Hamming distance from each test image to its nearest train image; the right shows the pixel RMS distance for the pHash candidate pairs. Cutting at Hamming distance ≤ 4 alone flags 28% of test, but in pixel distance most pairs cluster at RMS 30–60. They share a silhouette but are actually different images.

So I used pHash only to select candidates and made the final decision with pixel RMS distance.

| Step | Test images | Share | Pairs with different labels |
| --- | --- | --- | --- |
| pHash ≤ 4 (candidates) | 2,842 | 28.42% | 6,072 |
| + RMS ≤ 5 | 6 | 0.06% | 0 |
| **+ RMS ≤ 10 (adopted)** | **13** | **0.13%** | **0** |
| + RMS ≤ 20 | 153 | 1.53% | 0 |

![](https://velog.velcdn.com/images/def_uk/post/e6508a9d-bf85-4f6b-9e2a-5f48141de628/image.png)

The top four rows are near-duplicate pairs (test | train) confirmed at RMS ≤ 10, and the last row is a pair rejected just outside the threshold (RMS > 10). The confirmed pairs are visibly the same product. Nearly identical images are only 0.13% of test, so any optimistic bias in test-based scores is negligible.

### Hypotheses

Hypotheses based on the data check and the general properties of the two models.

| Hypothesis | Statement |
| --- | --- |
| H1 | The cGAN is sharper but at greater risk of mode collapse, so its per-class FID varies more |
| H2 | The cVAE-GAN is more diverse and trains more stably, but is slightly blurrier |
| H3 | For both models, Shirt / T-shirt / Pullover / Coat have the lowest conditional accuracy |

## 2. Comparing the two models fairly

### Model setup

| Component | cGAN | cVAE-GAN |
| --- | --- | --- |
| Generator (= Decoder) | DCGAN style, z(100) + label embedding(50) concat, tanh output | Same |
| Discriminator | DCGAN-style conv, label input, output is a logit without sigmoid | Same (also returns intermediate features) |
| Encoder | - | (x, y) → μ, logσ² |
| Loss | BCEWithLogits, non-saturating G loss | Same + KL + feature reconstruction |

The cVAE-GAN is Larsen et al. (2016)'s VAE-GAN with only a label condition added. The reconstruction loss is measured not as pixel MSE but as **MSE between the discriminator's intermediate features**, because pixel MSE produces the blurry images typical of VAEs. Each module gets a different loss.

| Module | Loss used for its update |
| --- | --- |
| Encoder | KL + feature reconstruction |
| Decoder | γ(=1.0) × feature reconstruction + GAN (fool D into calling both reconstructions x̃ and prior samples real) |
| Discriminator | GAN (real → 1, reconstruction x̃ → 0, prior sample → 0) |

### Fair-comparison conditions

| Item | Setting |
| --- | --- |
| Generator / Discriminator | Same architecture and initial weights (G 2,409,204 / D 473,380 parameters) |
| z_dim, label injection | 100, Embedding concat |
| optimizer | Adam, lr 2e-4, β1 0.5 |
| batch size | 128 |
| Training budget | Up to 50 epochs, same seed (42) |
| Early stopping | On val FID; stop after no improvement for 10 epochs (2 evaluations) |

Because of its encoder (1,288,684), the cVAE-GAN has about 1.45× more parameters. But generation doesn't use the encoder, so the generation path is the same.

I checked that the conditions really are the same using the untrained state. With both models only initialized and run through the same evaluation, **all 48 generation-path metrics came out identical (max difference 0.0)**.

## 3. Building the evaluation tools

### Evaluation classifier

Conditional accuracy, Precision/Recall and the diversity ratio all depend on a classifier. So I built the classifier first and finalized it after checking where it makes mistakes.

- Model: ResNet-18 adapted for 28×28 (1 input channel, first conv 3×3 stride 1, maxpool removed), trained from scratch
- First run (v0): val accuracy 0.9395. But train accuracy hit 1.000 from epoch 12, and val loss bottomed out at epoch 4 and then rose until the end. **It was overfitting and overconfident** (ECE 0.044).

I added candidates to reduce overfitting one at a time, adopting one only if val accuracy rose by 0.002 or more.

![](https://velog.velcdn.com/images/def_uk/post/82194fc4-dd23-4822-aa48-2357757a793b/image.png)

In the val loss on the left, only v0 (blue) rises after epoch 4. The other candidates, which include RandomCrop, keep going down to the end.

| Candidate | val acc | Gain | Final train acc | ECE | Shirt acc | Adopted |
| --- | --- | --- | --- | --- | --- | --- |
| v0 | 0.9395 | - | 1.0000 | 0.0440 | 0.819 | Baseline |
| **crop** | **0.9493** | **+0.0098** | 0.9750 | **0.0171** | **0.853** | **O** |
| crop + weight decay | 0.9494 | +0.0001 | 0.9754 | 0.0190 | 0.861 | X |
| crop + flip | 0.9493 | +0.0000 | 0.9607 | 0.0103 | 0.847 | X |
| crop + rotation 10° | 0.9460 | -0.0033 | 0.9540 | 0.0104 | 0.846 | X |

- Crop alone removed the overfitting. Train accuracy stopped at 0.975 and ECE fell by more than half.
- Since crop gave a big gain, I took a lack of data diversity as the cause of overfitting and also tried flip and rotation. Val loss and ECE dropped a bit more, but accuracy didn't rise, and Shirt accuracy actually fell. Crop had already fixed the overfitting, so more augmentation only made training harder.
- I ruled out label smoothing from the start. It compresses feature clusters, which would distort the diversity ratio and Precision/Recall measured with this classifier.

![](https://velog.velcdn.com/images/def_uk/post/ec0a86ab-d7ea-4e0a-8504-f5c7abe13e67/image.png)

The difference in confusion matrices between crop and v0 (red = increased with crop). On the diagonal, Shirt improved the most at +3.3%p, and Shirt misread as T-shirt fell by 1.4%p.

The final classifier's val accuracy of **0.9493** is effectively the ceiling for conditional accuracy on generated images. Per class, Shirt is lowest at 0.853, followed by T-shirt/top 0.910, Pullover 0.927 and Coat 0.929. The classifier, too, makes the most mistakes on the four top classes predicted in the data check.

### Metrics and baselines

| Metric | Sub-question | Measured in | Reading |
| --- | --- | --- | --- |
| Conditional accuracy | Q1 | Classifier prediction | Higher is better |
| FID (overall and per class) | Q2 | Inception | Lower is better |
| Precision / Recall (k=3) | Q2 / Q3 | Classifier features | Measures quality and diversity separately |
| Within-class diversity ratio | Q3 | Classifier features | Mean distance among generated ÷ mean distance among real. Much below 1 means mode collapse |
| Nearest-train distance ratio | memorization | Classifier features | Much below 1 suggests copying train |

- FID was measured with Inception features so it can be compared with other work. The other metrics were measured in the 512-dimensional feature space of a classifier trained on Fashion-MNIST, since Inception may not capture differences between 28×28 grayscale clothing well.
- Conditional accuracy can hit 100% even if the model repeats the same image. So I always read it together with the diversity ratio, Recall and FID.

To get a feel for each metric's range, I put the untrained G (lower bound) and real val images (upper side) in place of generated images and computed the same metrics.

| Metric | Untrained G | Real val |
| --- | --- | --- |
| FID | 457.71 | 2.51 |
| Conditional accuracy | 0.1000 | 0.9495 |
| Precision / Recall | 0.003 / 0.000 | 0.880 / 0.883 |
| Diversity ratio | 0.217 | 0.997 |
| Nearest-train distance ratio | 2.49 | 1.00 |

- Even for real images, Precision and Recall are 0.88, not 1. This is a limit of approximating the manifold with k-NN, so I treat 0.88 as the effective ceiling.
- Per-class FID on real val is 10–21, far larger than the overall FID (2.51). Estimating a 2048-dimensional covariance from 1,000 images per class biases the value upward. So I used per-class FID only for relative comparison between the two models.

### Not choosing the model with test

At first, the evaluation during training (choosing the best checkpoint, early stopping) also used test. That amounts to **choosing the model with test**, which biases the final test score optimistically. So I moved in-training evaluation to 1,000 val images per class (10,000 total, the same as test, so FID magnitudes are comparable) and used test only once, for the final evaluation of the best checkpoint.

### Smoke test

Before the main run, I checked the whole pipeline with a reduced setup (10 train batches, 2 epochs, 50 images per class). I kept the hyperparameters and only cut the amount of data and epochs. Both models passed all 11 steps.

| Check | Result |
| --- | --- |
| No duplicated sigmoid on D output | 82–86% of initial logits outside [0, 1] |
| Label injection | Mean L2 distance 0.60 between outputs with the same z and only the label changed |
| Update paths | Each optimizer changes only its own module; no gradient to G or E during the D update |
| cVAE-GAN small-scale overfit | Feature reconstruction 0.407 → 0.194; KL does not collapse to 0 |
| Checkpoint resume | Loss differs by at most 1.7e-3 (nondeterministic MPS ops); RNG and fixed noise match exactly |

The iteration times measured here were 22 ms for the cGAN and 49 ms for the cVAE-GAN. The cVAE-GAN is about 2.2× slower because of its encoder and reconstruction path.

## 4. Main training

### cGAN

![](https://velog.velcdn.com/images/def_uk/post/1fad5543-1766-4665-b0a1-7ffffd4ff3ec/image.png)

The val FID at bottom left keeps falling through epoch 50. In the D output at top center, D(x) (blue) rises and D(G(z)) (orange) falls, drifting further apart.

| epoch | 5 | 10 | 20 | 30 | 40 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| val FID | 60.31 | 24.73 | 14.16 | 12.55 | 11.30 | **10.53** |
| Conditional accuracy | 0.792 | 0.866 | 0.913 | 0.908 | 0.913 | 0.916 |

- It ran all 50 epochs with no early stop. **The best epoch is the last one (50), so it hasn't converged yet.**
- D(x) went 0.71 → 0.79 and D(G(z)) 0.29 → 0.21, and G loss rose 1.99 → 2.54. As training goes on, D gradually gains the upper hand.
- The gap in D output between real train and val images is 0.074 at the end. There's little sign of D memorizing train.

### cVAE-GAN

![](https://velog.velcdn.com/images/def_uk/post/11c5247f-2e4e-4342-af29-6bb23eee2a25/image.png)

Val FID (top right) flattens out at 15–16 after epoch 25. In the D output (top, second), D(x) stays near 0.5. KL (bottom, third) rises slowly and doesn't drop to 0.

| epoch | 5 | 10 | 20 | 30 | 35 | 45 |
| --- | --- | --- | --- | --- | --- | --- |
| val FID | 50.99 | 28.46 | 17.32 | 15.97 | **15.52** | 15.71 |
| Conditional accuracy | 0.794 | 0.853 | 0.883 | 0.897 | 0.902 | 0.908 |

- It stopped early at epoch 45, and the best is epoch 35.
- At epoch 5 its FID was lower than the cGAN's (50.99 vs 60.31), but from epoch 10 the order flipped (28.46 vs 24.73).
- D(x) stayed near 0.5, between 0.43 and 0.58. D didn't tilt to one side as it did for the cGAN.
- KL rose 0.034 → 0.080. No posterior collapse.
- The feature reconstruction loss rose 0.236 → 0.362. As D trains, the feature space used as the reference itself changes, so I did not read this increase as worse reconstruction. I checked reconstruction quality separately with test PSNR and SSIM.

### Training summary

| Item | cGAN | cVAE-GAN |
| --- | --- | --- |
| Epochs run | 50 | 45 (early stop) |
| Best epoch | 50 | 35 |
| Best val FID (selection criterion) | 10.53 | 15.52 |
| Test FID | **10.35** | 15.26 |
| Training time | 8.1 min | 14.8 min |
| Training time per epoch | 9.7s | 19.8s |

Test FID differs from the best val FID by only 0.2–0.3. The checkpoint chosen on val holds up on test.

## 5. Final evaluation (test, 1,000 per class)

| Metric | Untrained | cGAN | cVAE-GAN | Real val | Sub-question |
| --- | --- | --- | --- | --- | --- |
| FID ↓ | 457.71 | **10.35** | 15.26 | 2.51 | Q2 |
| Conditional accuracy ↑ | 0.1000 | **0.9161** | 0.9016 | 0.9495 | Q1 |
| Precision ↑ | 0.003 | **0.854** | 0.826 | 0.880 | Q2 |
| Recall ↑ | 0.000 | 0.738 | **0.802** | 0.883 | Q3 |
| Diversity ratio (closer to 1 is better) | 0.217 | 0.962 | **1.004** | 0.997 | Q3 |
| Per-class FID spread ↓ | 16.39 | **5.19** | 9.09 | 2.83 | H1 |
| Nearest-train distance ratio | 2.49 | 1.10 | 1.16 | 1.00 | memorization |
| Reconstruction PSNR / SSIM | 7.76 dB / 0.037 | - | 16.80 dB / 0.733 | - | cVAE-GAN only |

The nearest-train distance ratio is above 1 for both models. Neither copied train images.

### Sample comparison

![](https://velog.velcdn.com/images/def_uk/post/0299c615-262b-43da-844b-ef6bf1d6e088/image.png)

The cGAN (generated on the right) has clean outlines and simple patterns. Its tone is similar to the real images (left).

![](https://velog.velcdn.com/images/def_uk/post/718a32a7-d59c-41ac-ace6-44fb4d0e3411/image.png)

Contrary to expectations, the cVAE-GAN is **not blurry.** Instead, many Dress, Pullover, Shirt and Bag images have excessive blotchy patterns and rough texture. Shapes are more varied, but the texture is messier than in real images.

### Per-class results

| Class | Classifier ceiling | Accuracy cGAN | Accuracy cVAE-GAN | FID cGAN | FID cVAE-GAN | Diversity cGAN | Diversity cVAE-GAN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-shirt/top | 0.910 | 0.826 | **0.849** | **31.0** | 36.8 | 0.955 | 0.944 |
| Trouser | 0.991 | **0.993** | 0.990 | **17.2** | 19.4 | 0.948 | 1.025 |
| Pullover | 0.927 | **0.862** | 0.823 | **27.1** | 32.1 | 0.938 | 1.036 |
| Dress | 0.943 | 0.911 | **0.920** | **26.6** | 32.1 | 0.916 | 0.960 |
| Coat | 0.929 | 0.794 | **0.817** | **26.1** | 30.7 | 1.004 | 0.994 |
| Sandal | 0.991 | 0.982 | 0.982 | **34.2** | 42.8 | 0.998 | 1.033 |
| Shirt | 0.853 | **0.842** | 0.711 | **29.8** | 36.7 | 0.886 | 0.918 |
| Sneaker | 0.977 | 0.965 | **0.967** | **26.8** | 36.1 | 1.036 | 1.122 |
| Bag | 0.993 | **1.000** | 0.978 | **36.4** | 54.6 | 0.971 | 0.965 |
| Ankle boot | 0.981 | **0.986** | 0.979 | **27.9** | 31.3 | 0.972 | 1.041 |

- Per-class FID is **lower for the cGAN in all 10 classes.** The biggest gap is Bag (36.4 vs 54.6).
- Conditional accuracy is mixed across classes. Most of the overall gap (0.0145) comes from the cVAE-GAN's **Shirt (0.711)**. Excluding Shirt, the averages are 0.924 vs 0.923, almost the same.

![](https://velog.velcdn.com/images/def_uk/post/9ad76f32-56f5-411a-b1e0-cf9bdac1f5a4/image.png)

![](https://velog.velcdn.com/images/def_uk/post/41314c25-b891-45c3-b8b4-1eab3a8505a3/image.png)

Top is the cGAN and bottom the cVAE-GAN conditional confusion matrix (rows = conditioning label). Off-diagonal values concentrate in the block of the four top classes. For the cGAN, 14.1% of the Coat row leaks into Shirt; for the cVAE-GAN, the Shirt row spreads evenly into T-shirt (9.5%), Coat (9.1%) and Pullover (7.8%).

## 6. Testing the hypotheses

| Hypothesis | Result | Verdict |
| --- | --- | --- |
| H1. The cGAN is sharper but at greater risk of mode collapse, so its per-class FID varies more | Realism: the cGAN leads on FID, Precision and per-class FID → O<br>Mode collapse: diversity ratio 0.962 (0.886–1.036 per class). Shirt is lowest as expected, but not at collapse level → X<br>Per-class FID spread: cGAN 5.19 < cVAE-GAN 9.09, **the opposite** | Partly supported |
| H2. The cVAE-GAN is more diverse and trains more stably, but is slightly blurrier | Diversity: Recall 0.802 vs 0.738, diversity ratio 1.004 vs 0.962 → O<br>Stability: D(x) stays near 0.5, val D(x) fluctuation std 0.077 vs 0.101 → O. But the cGAN also didn't diverge or collapse, so the practical difference is small<br>Blur: not blurry; if anything the texture is excessive → X | Partly supported |
| H3. For both models, Shirt / T-shirt / Pullover / Coat have the lowest conditional accuracy | For both models the bottom four classes are **exactly these four**. The confusion directions also match the top similarity pairs from the data check (Coat→Shirt 0.141 / 0.111, T-shirt→Shirt 0.132 / 0.107) | Supported |

- H2's "blur" being wrong seems to be the effect of measuring the reconstruction loss on D features rather than pixels. That's the effect Larsen et al. (2016) intended with this approach. The cVAE-GAN's higher FID comes not from blur but from unnatural texture.
- Putting H3 next to the classifier ceiling, the cGAN's Shirt (0.842 vs a ceiling of 0.853) is almost at the classifier's limit. Where the generator itself has a big problem is the cGAN's **Coat** (-0.135 from the ceiling) and the cVAE-GAN's **Shirt** (-0.142). Without the classifier ceiling, I would have mistaken the two models' weaknesses for the same thing.

## 7. Conclusion

**In this setup (γ = 1, up to 50 epochs, one seed), adding a VAE-style training signal did not improve conditional image generation overall.**

| Sub-question | Ahead | Evidence |
| --- | --- | --- |
| Q1. Does it follow the label better? | cGAN (slightly) | 0.916 vs 0.902. Most of the gap is the cVAE-GAN's Shirt |
| Q2. Is it more realistic? | **cGAN** | FID 10.35 vs 15.26, Precision 0.854 vs 0.826, per-class FID in all 10 |
| Q3. Is it more diverse? | **cVAE-GAN** | Recall 0.802 vs 0.738, diversity ratio 1.004 vs 0.962 |
| Q4. Does it train more stably? | cVAE-GAN (slightly) | D output balanced near 0.5. But the cGAN wasn't unstable either |

- The reconstruction signal makes the generator **cover the real distribution more widely** (Recall ↑). The price is lower realism and label adherence for individual samples (FID ↑, Precision ↓, Shirt accuracy ↓).
- The cVAE-GAN takes about 2× as long per epoch.
- Choose the cGAN if realistic images matter, and the cVAE-GAN if diversity that covers the real distribution without gaps matters.
- What was decisive for interpreting the results was **putting the classifier ceiling and the untrained baseline in the same table**. Without the ceiling, I couldn't have told whether low Shirt accuracy came from the classifier or from the generator.

## 8. Limitations and next steps

- **The cGAN hasn't converged yet.** Its best epoch is the last one (50) and val FID was still falling. The cVAE-GAN, by contrast, plateaued after epoch 25. More training could widen the FID gap. → Compare again with more epochs
- **Only one seed.** The gap in conditional accuracy (0.0145) or stability may be within seed-to-seed variation. → Repeat with 3+ seeds and compare mean ± standard deviation
- **Only γ = 1.0 was tried.** The cVAE-GAN's excessive texture and Shirt confusion may relate to the reconstruction weight. → Tune γ first
- **Evaluation feature space.** Precision/Recall, the diversity ratio and nearest distance are based on features from a classifier I trained, and even real images top out at 0.88.
- **Per-class FID.** Based on 1,000 images, so it's biased upward. I interpreted it only as a relative comparison between models.

The next experiments will change one variable at a time so they don't mix with the base experiment: tune γ → spectral normalization / hinge loss (stability) → projection discriminator (label adherence) → one-sided label smoothing → add CAS evaluation (train a classifier on generated images and evaluate on real test).

**References**

- Larsen et al., Autoencoding beyond pixels using a learned similarity metric (2016)
- Kynkäänniemi et al., Improved Precision and Recall Metric for Assessing Generative Models (2019)
