---
title: "Practical Statistics for Data Scientists (5)"
date: 2026-10-25T19:25:00+09:00
description: "Notes on Practical Statistics for Data Scientists, part 5: the binomial, chi-square and F distributions, and the Poisson, exponential and Weibull distributions."
tags: ["F분포", "베이불 분포", "이항분포", "지수분포", "카이제곱분포", "푸아송 분포"]
draft: false
---
## Data and sampling distributions

### Binomial distribution

When you repeat a trial that has two possible outcomes, such as whether someone buys or clicks, the number of successes follows the **binomial distribution**.

- **Trial** : an experiment or observation that produces one outcome
- **Success** : the outcome of interest in a trial
- **Binary outcome** : an outcome that splits two ways, such as yes/no or 0/1
- **Bernoulli trial** : a single trial with two outcomes, success or failure
- **Binomial distribution** : the probability distribution of the number of successes over $n$ independent trials with the same probability of success

Counting the number of heads in 10 coin flips is the classic example. The two outcomes don't have to be 50/50; if the probability of success is $p$, the probability of failure is $1-p$.

In statistics, a success is not necessarily a good outcome. An event of interest in the analysis, such as an error or a fraudulent transaction, can be defined as the success.

#### Mean and variance

With $n$ trials and a probability of success $p$, the mean and variance are as follows.

$$
mean=np
$$

$$
variance=np(1-p)
$$

For example, if you repeat 100 independent trials each with a click probability of 0.1, the expected number of clicks is 10.

You need to distinguish **the probability of exactly 2 successes** from **the probability of 2 or fewer successes**. The former is the probability of that one count; the latter adds up the probabilities of 0 through 2.

When the number of trials is large and both the expected successes $np$ and the expected failures $n(1-p)$ are large enough, the binomial distribution can be approximated by a normal distribution.

### Chi-square distribution

Used to judge how far the observed frequencies in each category depart from the expected frequencies.

- **Observed frequency** : the count actually observed in the data
- **Expected frequency** : the count expected if the null hypothesis is true
- **Null hypothesis** : the hypothesis the test is built on, such as no difference or no association
- **Chi-square statistic** : a single number summarizing the difference between observed and expected frequencies

The chi-square statistic is computed by squaring the difference between each observed and expected frequency, dividing by the expected frequency, and summing.

$$
\chi^2=\sum\frac{(observed-expected)^2}{expected}
$$

A small value means the observed frequencies are close to the expected ones; a large value means they depart a lot from what was expected. Under suitable conditions, you compare this statistic with the chi-square distribution to judge whether the difference is statistically significant.

It is typically used for the following tests.

- **Goodness-of-fit test** : checks whether the observed category frequencies match an assumed distribution
- **Test of independence** : checks whether two categorical variables are independent

For example, you could check whether acquisition channel and purchase are related. The shape of the chi-square distribution depends on its degrees of freedom, and a test should also check that observations are independent and expected frequencies are large enough.

### F distribution

When comparing the means of several groups, you need to check how large the differences between groups are relative to the variation within groups. The statistic used for this is the **F statistic**.

- **Between-group variation** : how far apart the group means are
- **Within-group variation** : how spread out the observations are within the same group
- **Analysis of variance (ANOVA)** : a method that tests differences among group means by comparing between-group and within-group variation

In ANOVA, the F statistic is the ratio of mean squares, each variation adjusted by its degrees of freedom.

$$
F=\frac{between\text{-}group\ mean\ square}{within\text{-}group\ mean\ square}
$$

For example, you could compare crop yields across three groups that used different fertilizers. Yields vary even within the same group, so differences in group means have to be judged together with the internal variation.

Under the null hypothesis that all groups share the same population mean, and when conditions such as normality and equal variances hold, the F statistic follows the F distribution. The shape of the F distribution depends on the degrees of freedom of the numerator and denominator.

> A significant ANOVA result does not mean every group mean differs from every other. It is evidence that at least one group's population mean differs, and finding which groups differ needs further comparisons.

The F statistic is used not only in ANOVA but also to test the significance of regression models.

### The Poisson distribution and related distributions

Events that occur in time or space, such as website visits, calls to a support center, or defects in a given area, can be described with probability distributions.

- **Rate** : the average number of events per unit of time or space
- **Poisson distribution** : the distribution of the number of events in a fixed interval
- **Exponential distribution** : the distribution of waiting times between events in a Poisson process with a constant rate
- **Weibull distribution** : a distribution that can describe lifetimes whose failure risk increases or decreases over time

### Poisson distribution

A distribution for how many times an event occurs in a fixed amount of time or space. It can model the number of requests to a server or the number of support calls per minute.

The Poisson parameter $\lambda$ means **the average number of events in the interval being observed**.

$$
mean=\lambda,\qquad variance=\lambda
$$

If an average of 2 support calls come in per minute, the number of calls in one minute can be modeled with a Poisson distribution where $\lambda=2$. If the rate is constant, the average over 5 minutes is 10 calls, so $\lambda=10$.

So when using $\lambda$, you also need to check **which unit of time or space it refers to**.

### Exponential distribution

Where the Poisson distribution deals with the number of events in an interval, the exponential distribution deals with the time from one event to the next.

For example, it can model the time you wait for the next call after a call comes into a support center.

If the rate is $r$, the mean waiting time is as follows.

$$
mean\ waiting\ time=\frac{1}{r}
$$

If an average of 0.2 calls come in per minute, the mean waiting time is 5 minutes.

Generating 100 time intervals from an exponential distribution means **generating 100 waiting times**. It does not mean the result of observing for 100 minutes.

When linking a Poisson process to the exponential distribution, the assumption of a constant rate matters. If traffic varies a lot by time of day, you can split the data into periods where the rate is relatively constant.

![](https://velog.velcdn.com/images/def_uk/post/7c4ac810-c8bd-4e22-831e-3ea1df06bfa2/image.png)

### Estimating failure rates

When failures are rare, there are few chances to observe the time between them, so the rate is hard to estimate precisely.

But observation time with no failures is also information. You can assume a rate, compute the probability of going a long time without a single failure, and evaluate how well that assumption fits what was observed.

> Observing no failures does not mean the failure rate is 0. The observation time and the assumed rate have to be considered together.

When data is scarce, you can check the fit with the observations through probability calculations or simulations under several assumed rates.

### Weibull distribution

The risk of equipment failing can change over time. A typical case is failure risk increasing with wear the longer something is used.

The Weibull distribution is used to model this kind of lifetime data.

- **Shape parameter $\beta$** : determines how the hazard rate changes over time
- **Scale parameter $\eta$** : sets the time scale of the lifetime, also called the characteristic life

The hazard rate changes with the shape parameter as follows.

- $\beta<1$ : the hazard rate decreases over time
- $\beta=1$ : the hazard rate is constant, the same as the exponential distribution
- $\beta>1$ : the hazard rate increases over time

Here the hazard rate is **the instantaneous rate of failure given that there has been no failure up to that point**.

When fitting a Weibull distribution to data, you estimate the shape and scale parameters. The scale parameter reflects the time scale but is not always equal to the mean lifetime.

![](https://velog.velcdn.com/images/def_uk/post/c9650b6a-7db5-4917-90f2-7a79c724e52b/image.png)

### Wrapping up

Even with a lot of data, estimates can be biased if the sample doesn't properly represent the population. So you need to look at how the sample was collected and whether the data fits the question, not just the amount of data.

Probability distributions help you understand how data is generated and explain variation due to chance. The right distribution depends on whether the observation is a count of successes, a count of events per interval, or the time between events.

The bootstrap estimates the variability of a statistic by repeatedly resampling the observed data with replacement. It does not, however, automatically fix a lack of representativeness or bias in the original sample.
