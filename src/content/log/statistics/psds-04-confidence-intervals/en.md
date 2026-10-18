---
title: "Practical Statistics for Data Scientists (4)"
date: 2026-10-18T20:23:00+09:00
description: "Notes on Practical Statistics for Data Scientists, part 4: confidence intervals, the normal distribution and z-scores, QQ plots, long-tailed distributions and Student's t-distribution."
tags: ["QQ그림", "신뢰구간", "정규분포", "표본분포"]
draft: false
---
## Data and sampling distributions
### Confidence intervals

Just because the mean computed from a sample is 100 doesn't mean we can say the population mean is exactly 100. A different sample could have produced a different mean.

Instead of presenting a single estimate, a **confidence interval** expresses the uncertainty of the estimate as a range.

- **Point estimate**: estimating a quantity of interest, such as the population mean, as a single number
- **Confidence interval**: an interval estimate of a parameter, built from the sample
- **Confidence level**: the proportion of intervals that contain the true parameter when sampling and interval computation are repeated with the same procedure
- **Interval endpoints**: the lower and upper bounds of the confidence interval

Say a 95% confidence interval for the mean is computed as `[90, 110]`. What does the 95% mean?

> If we repeatedly drew samples from the same population and built confidence intervals the same way, about 95% of those intervals would contain the true population mean.

From the frequentist point of view, the population mean is a fixed value; what varies from sample to sample is the interval. So this must be kept apart from interpreting one already-computed interval as "there is a 95% probability that the population mean lies in here."

Also, **a 95% confidence interval does not mean it contains 95% of the data.** A confidence interval for the mean doesn't describe how widely individual data points are spread; it describes how precisely the population mean has been estimated.

### Confidence intervals with the bootstrap

The bootstrap from the previous post can also produce confidence intervals. The method that uses quantiles of the resampled statistics is called the **percentile bootstrap confidence interval**.

For a sample of size $n$, the procedure is:

1. Draw a sample of size $n$ with replacement from the original sample.
2. Compute the statistic of interest, such as the mean of the resample.
3. Repeat this enough times to collect many statistics.
4. Sort the statistics and trim both ends according to the desired confidence level.
5. Use the extreme values of what remains as the endpoints of the confidence interval.

For a **90% confidence interval**, trim the bottom 5% and the top 5%, so the 5th and 95th percentiles of the bootstrap statistics become the endpoints. A **95% confidence interval** uses the 2.5th and 97.5th percentiles.

This works not only for the mean but for other statistics such as the median. However, since the bootstrap still computes from the original sample, **it does not fix the bias of a sample that lacks representativeness.**

### Width of a confidence interval

The width of a confidence interval shows the uncertainty of the estimate. With everything else equal, the following relationships hold.

- **As the confidence level rises**, the interval widens.
- **As the sample size grows**, the interval generally narrows.
- **As the variability of the data grows**, the interval generally widens.

A 99% confidence interval computed from the same sample is wider than a 95% one. To get a higher coverage rate, you have to take a correspondingly wider range.

The standard error of a mean computed from independent observations is estimated as follows.

$$
\widehat{SE}(\bar{x})=\frac{s}{\sqrt{n}}
$$

Here $s$ is the sample standard deviation and $n$ is the sample size. If the sample standard deviation stays about the same, quadrupling the sample size roughly halves the standard error.

### The normal distribution

The normal distribution is a bell-shaped distribution, symmetric around its mean. The mean $\mu$ sets the location of the center and the standard deviation $\sigma$ sets the spread.

- **Error**: the difference between an observation and a reference or predicted value
- **Standardization**: transforming data by subtracting the mean and dividing by the standard deviation
- **z-score**: how far a value is from the mean, expressed in units of standard deviations
- **Standard normal distribution**: a normal distribution with mean 0 and standard deviation 1
- **QQ plot**: a plot comparing the quantiles of two distributions; also used to compare a sample's distribution with the normal distribution

In a normal distribution, the following proportions hold around the mean.

| Range | Proportion included |
| --- | --- |
| $\mu \pm \sigma$ | about 68% |
| $\mu \pm 2\sigma$ | about 95% |
| $\mu \pm 3\sigma$ | about 99.7% |

These proportions are a property of the normal distribution. They are not a rule you can apply to every data set as is.

Plenty of data, such as income or purchase amounts, is skewed to one side. The importance of the normal distribution is not that all raw data follows it. What matters for statistical inference is that **under suitable conditions, the distribution of the sample mean can be approximated by a normal distribution.**

The central limit theorem from the previous post also doesn't mean the original data itself turns normal. It means that under conditions such as independence and finite variance, as the sample size grows, the distribution of the standardized sample mean gets close to normal.

### The standard normal distribution and z-scores

Standardizing a value $x$ from a distribution with mean $\mu$ and standard deviation $\sigma$ gives:

$$
z=\frac{x-\mu}{\sigma}
$$

If the population mean and standard deviation are unknown, you can standardize with the sample mean $\bar{x}$ and the sample standard deviation $s$.

$$
z_i=\frac{x_i-\bar{x}}{s}
$$

For example, scoring 90 on a test with a mean of 70 and a standard deviation of 10 gives a z-score of 2. It means the score sits two standard deviations above the mean.

> Standardizing data does not make it normally distributed.

Standardization changes the center and the scale. If the original distribution is asymmetric, that asymmetric shape survives standardization. You get the standard normal distribution when a variable that is already normally distributed is standardized with the population mean and standard deviation.

### QQ plots

One way to see how close data is to a normal distribution is the **QQ plot** (quantile-quantile plot).

A normal QQ plot compares two things:

- **Horizontal axis**: the theoretical quantiles of the normal distribution
- **Vertical axis**: the sample values sorted in order, or their standardized values

If the data is close to normal, the points fall roughly along a straight line. If the points instead bend systematically or stray far at both ends, that signals features that differ from the normal distribution.

For example, if the points in the middle stay close to the reference line but the left end drops below it and the right end rises above it, you can suspect a distribution in which extreme values show up in both tails more often than under a normal distribution.

With small samples, though, points can stray by chance. A QQ plot is a tool for seeing the character of a distribution, not a way to settle normality from a single picture.

### Long-tailed distributions

Real data can show values far larger or smaller than a normal distribution would, and show them often. The parts of a distribution far from its center are called the **tails**.

- **Tail**: the parts at both ends of a distribution where extreme values appear
- **Skewness**: a measure of how asymmetric a distribution is from left to right
- **Long-tailed distribution**: in this section, a distribution in which extreme values appear relatively more often than in a normal distribution

Tail behavior and asymmetry should be kept apart. Not only skewed distributions but **symmetric distributions can also have heavier tails than the normal distribution.**

The book uses a QQ plot of Netflix's daily stock returns as an example. The values in the middle stay fairly close to the reference line, but the values at both ends stray far from it. This shows that extreme swings can happen more often than a normal distribution would predict.

Describing such data with a normal distribution alone can underestimate rare but large swings. You need to look not only at the center, where most of the data sits, but also at **what happens in the tails**.

And an observed extreme value shouldn't be dismissed as an error right away. Check the context to see whether it is an input error or something that can really happen.

### Student's t-distribution

Student's t-distribution is a symmetric bell shape like the standard normal distribution, but with heavier tails. It is used for inference about a sample mean when the population standard deviation is unknown.

- **$n$**: sample size
- **Degrees of freedom**: the value that determines the shape of the distribution; for t inference on the mean of one sample it is usually $n-1$
- **t-statistic**: the difference between the sample mean and the population mean, divided by the estimated standard error

When a sample is drawn independently from a normal population, the following statistic follows a t-distribution with $n-1$ degrees of freedom.

$$
T=\frac{\bar{X}-\mu}{S/\sqrt{n}}
$$

Here $\bar{X}$ is the sample mean and $S$ is the sample standard deviation.

Because the population standard deviation $\sigma$ is unknown, the $S$ computed from the sample is used instead. Since the standard error itself has to be estimated, extra uncertainty comes in. The heavy tails of the t-distribution reflect that uncertainty.

**The sample mean itself does not always follow a t-distribution.** Keep in mind that it is the distribution of the statistic standardized by the sample standard deviation, as above.

The fewer the degrees of freedom, the heavier the tails; as the degrees of freedom grow, it approaches the standard normal distribution. So using the t-distribution on a small sample gives a wider confidence interval than using normal critical values at the same confidence level.

### Confidence intervals with the t-distribution

A two-sided $100(1-\alpha)\%$ confidence interval for the population mean is computed as follows.

$$
\bar{x}\pm t_{1-\alpha/2,\,n-1}\frac{s}{\sqrt{n}}
$$

Here $t_{1-\alpha/2,\,n-1}$ is the $1-\alpha/2$ quantile of the t-distribution with $n-1$ degrees of freedom. For a 95% confidence interval, $\alpha=0.05$, so the 0.975 quantile is used.

As an illustration, say 25 independent observations have a mean of 100 and a sample standard deviation of 10. Assuming a normal population, there are 24 degrees of freedom, and the t critical value for a 95% confidence interval is about 2.064.

$$
\widehat{SE}(\bar{x})=\frac{10}{\sqrt{25}}=2
$$

$$
100\pm 2.064\times 2
\approx [95.87,\ 104.13]
$$

For a normal population, the theoretical basis of this t interval holds exactly. Even when the population isn't normal, it can often be used as an approximation if the sample is large enough. But when the sample is small and heavily skewed, or extreme values have a large influence, the conditions for using it deserve a closer look.

The bootstrap estimates the variability of a statistic by resampling, while the t-distribution approach uses a theoretical distribution and the standard error. Both are **used to express how much uncertainty there is in an estimate obtained from a sample.**
