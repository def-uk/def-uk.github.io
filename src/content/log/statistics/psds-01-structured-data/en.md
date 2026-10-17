---
title: "Practical Statistics for Data Scientists (1)"
date: 2026-09-25T22:19:00+09:00
description: "Notes on Practical Statistics for Data Scientists, part 1: types of structured data, tabular data, and estimates of location and variability."
tags: ["EDA", "데이터", "정형데이터", "탐색적 데이터 분석", "통계"]
draft: false
---
## Elements of structured data
Data arrives unstructured, from sensor measurements, images, video, text, events and more. **Data science** has to turn this **_explosive volume of raw (unprocessed) data_** into a structured form that can be used.
The most common form of structured data is a **table** with columns.

### Types of structured data
#### Numeric data
- **Continuous data**: wind speed or duration
- **Discrete data**: how often an event occurs
#### Categorical data
Values from a fixed set, such as city names (Daejeon, Busan, Seoul)
- **Binary data**: a special case of categorical data that takes one of two values, such as true/false, 0 and 1, or yes/no
- **Ordinal data**: ratings (1, 2, 3, 4, 5) are the typical example

### Why convert to structured data?
Because the type of data plays an important role in visualization, interpretation and choosing a statistical model when analyzing data and building predictive models.

---

## Tabular data
The most commonly used form of object is tabular data, like a table in Excel or a database. It is basically a two-dimensional matrix of rows representing records and columns representing features (variables).

> An indicator variable can also become an outcome variable.

### Non-tabular data structures
- **Time series data**: holds successive measurements of the same variable.
- **Spatial data**: when representing objects, the focus is an object (a house) and its spatial coordinates. Field data, on the other hand, focuses on small units of space and an appropriate measurement value (pixel brightness).
- **Graph (or network) data**: used to represent physical, social and somewhat abstract relationships

---

## Estimates of location
The most basic step in exploring data is finding a "typical value" for each feature (variable): an estimate of where most of the values are located (central tendency).

- **Mean**: the sum of all values divided by the number of values
- **Weighted mean**: the sum of values multiplied by weights, divided by the sum of the weights
- **Median**: the value in the very middle of the data; a robust estimate of location
- **Percentile**: the value below which $P\%$ of the data lies (a statistic giving the value at a particular position when the sorted data is divided into 100 equal parts)
- **Weighted median****: after sorting the data, the value where the running sum of the weights from the top reaches half of the total
- **Trimmed mean**: the mean of the values remaining after dropping a fixed number of extreme values
- **Robust**: not sensitive to extreme values
- **Outlier**: a data value very different from most of the others (synonym: extreme value)

> **Exponentially weighted mean**: a way of computing a moving average in which the influence of older data decays exponentially

---

## Estimates of variability
Describes dispersion: how tightly packed or spread out the data values are.

- **Deviation**: the difference between an observed value and the estimate of location (synonyms: error, residual)
- **Variance**: the sum of squared deviations from the mean divided by n-1, where n is the number of data points (synonym: mean squared error)
- **Standard deviation**: the square root of the variance
- **Mean absolute deviation**: the mean of the absolute deviations from the mean (synonyms: L1 norm, Manhattan norm, mean absolute error)
- **Median absolute deviation from the median (MAD)**: the median of the absolute deviations from the median
- **Order statistics**: metrics based on the data values sorted from smallest to largest

#### Source
Practical Statistics for Data Scientists (pp. 19–38)
