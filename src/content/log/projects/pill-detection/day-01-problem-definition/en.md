---
title: "Pill Detection Project, Day 1: Defining the Problem and Setting Up Collaboration"
date: 2026-11-01T18:54:00+09:00
description: "Day one of the team project: defining the problem for a medication-log service built on pill recognition, and setting up the team's development environment and collaboration."
tags: ["객체탐지", "바운딩박스"]
draft: false
---
On the first day of the project, we defined the problem for a medication-log service based on pill recognition and set up the team's development environment and collaboration. In the morning we discussed the service's core user experience; in the afternoon we fixed environment setup problems and then did EDA. In the wrap-up meeting we shared the results of the data fixes and feedback on our approach.

## 1. Schedule

| Time | Activity |
| --- | --- |
| After orientation, 10–11 a.m. | Problem definition and discussion of service direction |
| After the morning discussion until 3 p.m. | Setting up the GitHub repository, development environment, and fixing problems |
| After 3 p.m. | EDA and data quality checks |
| Around 6 p.m. | Sharing results and feedback |

## 2. Problem definition: reducing the burden of remembering and logging doses

The team focused on **the difficulty users who take several kinds of medication have in remembering and logging which pills they took today**. The concern was that when you have to manage dose times and quantities, a tedious logging process or an unreliable memory can lead to double doses or missed doses.

Among several proposals, we centered the discussion on those submitted by Sohee and another teammate: a service that automatically records a dose at the time a photo of the medication is taken. The goal is to spare users from memorizing or manually entering every dose.

We also shared a draft problem definition for the README, written from the pre-project mentoring materials. Framed as the team's service direction, it reads as follows.

> Users who take several kinds of medication have to manage the dose time and quantity set for each one. Our team aims to reduce the burden of keeping a medication log by connecting a pill object-detection model to an app, and to help users easily check their own dose history.

We set the model's basic inputs and outputs as follows.

| Item | Details |
| --- | --- |
| Model input | Pill image |
| Ground truth for training | Bounding boxes (bbox) and class labels |
| Model output | Predicted bounding boxes, confidence scores, class names |

However, a photo only shows the medication at the moment it was taken, so **how to link a photo record to whether the dose was actually taken remained a separate design problem**.

## 3. Key discussion: photographing the bottle vs. photographing the pills

As we fleshed out the service direction, the biggest question was **whether users should photograph the medicine bottle or the pills they are about to take**.

### Photographing the bottle: minimizing the user's logging burden

Those who proposed photographing the bottle emphasized convenience. Their view was that photographing the bottle and getting a reminder is simpler than spreading the pills out on your palm or a surface to photograph them.

In particular, someone shared that for users who take many kinds and quantities of medication, gathering all the pills in one place for a single photo can itself be a burden. This made the team consider that a tedious photo process could make the service hard to use.

### Photographing the pills: tying the log to preparing the dose

Those who supported photographing the pills focused on the act of taking out and preparing the medication you're actually about to take. Their view was that taking the pills out and photographing them ties in more closely with the act of taking them than photographing a bottle or checking a reminder.

Some argued that we should also consider cases where people move pills into other containers or lose the bottle. There were also views on differences in how medication is packaged and stored in Korea and abroad, which we'll need to confirm by researching how target users actually take their medication.

### Outcome

On day one we settled on **a direction centered on photographing the pills**. But not every teammate was fully convinced, so it wasn't a final consensus.

We agreed to keep examining how much we can reduce the user's photo burden, whether photographing helps the act of taking medication, and how far the log can be trusted.

## 4. Development environment and collaboration setup

### GitHub repository and shared environment

The team decided to use `uv` to pin versions and set up virtual environments. The project structure was based on an earlier pill detection project.

As team lead, Dongwook set up the repository and shared development environment while answering teammates' questions about environment setup and how to collaborate. Since everyone except the team lead lacked relevant experience, the first day took a lot of time just to lay the groundwork for starting work.

### Problems during environment setup

A typical problem was the kernel actually running the code differing from the virtual environment we had agreed on. While loading data, someone used libraries and a kernel suggested by an AI and picked a different kernel that included a `.venv` instead of the configured environment, causing a mismatch.

The point of this case wasn't the specific kernel name but **changing settings without checking whether the current runtime matched the environment the team had agreed on**.

Through this experience, the team confirmed the importance of the following.

- Check the shared environment setup documented in the README first.
- Make sure the environment where libraries are installed matches the kernel running the code.
- Review whether settings or commands suggested by an AI fit the current project.

The detailed errors and fixes were recorded separately on an environment-setup troubleshooting page.

### Choosing a task management tool

We considered managing issues and to-dos on a kanban board. At first we planned to use Notion, but once GitHub Projects came up as an option we started comparing the two.

We didn't settle on a tool on the first day; **the choice of task management tool was pushed to day two.**

## 5. EDA and data quality checks

After 3 p.m., teammates who had finished their setup did EDA. The main work was checking the data structure, looking for missing bounding boxes and outliers, and fixing label errors.

| Owner | Main work |
| --- | --- |
| Teammate who fixed bounding boxes | Found missing bounding boxes for some pills, added and verified them |
| Younghoon | Data loading and basic EDA |
| Jaemin | Searching for bounding-box size outliers and fixing label errors |

### Missing bounding boxes for some pills

We found training images with missing bounding boxes. Looking at the actual images, it wasn't that whole images lacked labels; rather, **only some of the 4 pills in an image were missing boxes**.

Deleting such an image would also throw away the correctly labeled pills in it. The teammate in charge judged that deleting would lose 16 existing ground-truth boxes, so instead of deleting the images they added and verified the missing boxes. OpenCV was used for the fix.

This work was meaningful in that it checked the actual images and labels before bulk-deleting missing data. We also saw the need to write down the correction criteria and verification method clearly enough to explain them.

### Bounding-box size outliers

While looking at bounding-box sizes, Jaemin found cases that counted as outliers by the interquartile rule. Checking the actual images, these were cases where the pill itself was large, so the box was large too.

From this, the team confirmed that **a statistical outlier doesn't necessarily mean a data error and has to be judged together with the actual image**.

### Fixing a label error

Jaemin found one label error, removed the existing bounding box, and replaced it with a new one.

When the results were shared, there was feedback that we need to fully understand the cause of an error and the reason for a fix before using AI. The point was that besides producing a fix, we should be able to explain why it's needed.

## 6. Wrap-up meeting and team feedback

In the wrap-up meeting around 6 p.m., we shared the EDA results and how the data was fixed.

Team lead Dongwook gave the feedback that, before jumping straight to fixing bounding boxes, **it would have been better to first look at what the `NaN` values in the dataset's features mean and why they occur**.

He also stressed that teammates who used OpenCV to fix bounding boxes need to fully understand the functions they used and how they work. The team took this feedback on board and agreed that even when using AI and libraries, we should be able to check the following ourselves.

- What problem occurred?
- Why is a fix needed?
- What role does the tool we used play?
- How did we verify that the fix is valid?

## 7. Retrospective from the team's perspective

On the first day we spent more time than expected on environment setup and guiding collaboration. Just choosing a shared development environment wasn't enough; we needed a process to help everyone understand the same standards and apply them to real work.

In the service discussion, we confirmed that **user convenience and the reliability of the medication log have to be considered together**. Before building a pill-photo feature, we need to examine whether users can handle that process every day.

In the data work, it was meaningful that we told missing data and outliers apart by checking the actual images. At the same time, we learned that we need the habit of finding the cause before fixing data and recording the reasons for and verification of each fix.

From the team lead's perspective, too, project management wasn't just about allocating schedules and tasks. On the first day, an important part of the role was clearing problems that blocked teammates and helping them understand the shared environment and how to collaborate.

## 8. Follow-up tasks for day 2

| Task | What to check |
| --- | --- |
| Decide on a task management tool | Choose between Notion and GitHub Projects for the team |
| Recheck the shared environment | Verify that the virtual environment, kernel and install environment match the README |
| Refine the service direction | Review the photo burden and how photo records are handled |
| Analyze causes of missing values | Check what feature `NaN` values mean and how they relate to missing labels |
| Document label fixes | Record data before and after, reasons for fixes, and verification methods |
| Understand the tools used | Check how OpenCV functions and AI-suggested code work |
