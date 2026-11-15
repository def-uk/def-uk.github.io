---
title: "Pill Photos That Differ Only in Angle: Are They Really the Same Data?"
date: 2026-11-08T19:57:00+09:00
description: "Pill detection project, day 2: switching task management to GitHub issues, and debating — from a validation-leakage angle — whether pill images that differ only in camera angle count as the same data."
tags: ["객체탐지"]
draft: false
---
On day two of the project, we switched task management from personal to-dos in Notion to GitHub issues. In the morning we shared how to use the new kanban board and fixed setup problems, and then everyone did their own EDA. In the wrap-up meeting, each teammate shared the data issues they had found, and a discussion followed on whether pill images that differ only in camera angle should be treated as the same data.

## 1. Schedule

| Time | Activity |
| --- | --- |
| 10 a.m. | Morning meeting: announcing the change in task management |
| After the morning meeting through the afternoon | Kanban board walkthrough, fixing setup problems, individual EDA |
| 6 p.m. | Daily wrap-up meeting: sharing EDA results and discussing open questions |

## 2. Switching task management: from Notion to-dos to GitHub issues

### Why we switched

On day one we tracked tasks with to-dos on the Notion project page and pushed the choice between Notion and GitHub Projects to day two. Personal to-dos were enough for organizing each person's own work. But it was hard to immediately tell the whole team about important things found during EDA.

### The new structure

From day two we decided to manage work around GitHub issues. The role of the old to-dos was replaced by per-person issue tracking, and the overall schedule was reorganized in Notion.

| Area | Tool | Role |
| --- | --- | --- |
| Overall schedule | Notion | Managing the whole project schedule |
| Individual work | GitHub issues, kanban board | Tracking assignments and progress |
| Important findings | GitHub issues, Discord notifications | Sharing anything unusual found during EDA with the whole team |

In particular, I announced that anything unusual found in EDA should be filed as an issue so every teammate gets notified. Since EDA was just starting, we didn't define tasks in advance. Rather than splitting up predefined tasks, the point was to first set up a structure where problems we find pile up as issues and get shared.

### Difficulties in adopting it

Between the morning and afternoon meetings, the team lead spent a lot of time walking people through the new kanban board and fixing various setup problems. Teammates differed in how familiar they were with issue tools and settings, so explanations and fixes took time.

As with the shared development environment on day one, just picking a tool didn't get the team working the same way. We also needed a process to help everyone understand how the tool is used and apply it to real work.

## 3. Sharing EDA results

In the 6 p.m. wrap-up meeting, everyone presented their EDA in terms of issues. For each issue found, we sorted out what was fixed that day, what was handed to another teammate, and what to deal with in the long term.

| Owner | Issue found | Follow-up |
| --- | --- | --- |
| Teammate A | Class imbalance in the training data | Long-term task |
| | Images with the same pill combination that differ only in camera angle | Handed to Teammate C |
| | Rare pills that appear only in those angle-only variants | Long-term task (train/val split problem) |
| | Data where pills are so close that their bounding boxes overlap | Fixed |
| | Data with bounding boxes drawn on empty space | Fixed |
| Teammate B | Missing values (NaN) in the JSON files found together across 18 images | Continuing EDA focused on missing values |
| Teammate C | Found 3 images with completely overlapping bounding boxes | Fixed; took over Teammate A's angle issue |
| Teammate D | Found bounding boxes that cut off part of the pill | Fixed; researching how to handle loose boxes |

### A mismatch in the count of missing values

While looking at missing values in the JSON files, Teammate B found clusters of missing values in 18 images. The number of missing bounding boxes shared with the team so far had been 16. That gave a lead to suspect the actual number of missing boxes might be 18. This work also followed from the day-one wrap-up feedback: "Before fixing bounding boxes, first check what NaN means and why it occurs."

### Mismatched bounding-box areas

Teammate D found and fixed bounding boxes that cut off part of the pill. Along the way, they also found the opposite: boxes drawn looser than the pill. Loose boxes can make IoU look higher than it really is, so we decided to research how to handle these cases. After fixing the area mismatches, we also decided to check whether any boxes that originally had different labels ended up with the same label.

## 4. Key discussion: are images that differ only in angle the same data?

The topic we discussed longest in the wrap-up meeting was the angle issue raised by Teammate A.

### The situation

There were several images of the same pill combination (for example, pills A, B and C together in one photo) taken with only the camera angle changed, at 70, 75 and 90 degrees. The problem was that rare pills that appear only a few times were included only in images like these.

### View: they're effectively the same data

Teammate A explained that these images differ only in the camera angle at the time they were taken, so they're effectively the same data. They stressed that what matters is how we handle this data.

Teammate B also saw them as effectively the same photo to the human eye. Their view was that even photos whose arrangement looks different become identical when rotated 180 degrees. So rather than keeping three similar photos, they suggested it would be simpler to keep just one and apply augmentation.

### View: first check whether they're really the same

The team lead argued that we first need to check whether these images are really the same data. The feedback was that although each image has the same set of pill classes, the arrangement differs, so to the model they may effectively be different data.

### The link to validation leakage

This discussion wasn't just about whether to delete duplicate images. It was directly tied to how we split train and validation.

If effectively the same scene ends up split across train and validation, the model gets validated on images almost identical to what it saw in training. Validation performance can then look higher than the true generalization performance. Especially since the rare classes were concentrated in the angle-only variants, splitting randomly by image count alone could make the validation results for rare classes hard to trust.

### Outcome

Opinions on this question were split among the other teammates too. So we didn't settle it on day two. Instead, Teammate A will research the grounds for deciding whether to treat these images as the same data or different data. We also agreed to include Teammate B's view, that rotation makes them the same photo, in the criteria to check.

## 5. Retrospective from the team's perspective

On day two, changing how we manage work changed how EDA results get shared. By organizing problems into issues and splitting them into fixed, handed over, and long-term, the whole team could see not only who's doing what but also what's wrong with the data. Adopting the tool, though, took a lot of time spent explaining and helping with setup. As we saw on day one, recurring questions need to be written up as shared guidance.

On the data side, we confirmed that besides class imbalance, we have to consider similar shots being split between train and validation. Whether images that look the same to a person are also the same from the model's point of view couldn't be assumed until we checked directly.

From the team lead's perspective, an important part of the role when different interpretations come up is not to rush to a conclusion but to set shared criteria for what we'd need to check to decide. That said, since each teammate defined "the same data" differently, the next discussion would have gone more smoothly if we had at least drafted the criteria together.

## 6. Follow-up tasks for day 3

| Task | What to check |
| --- | --- |
| Criteria for similar images | Review criteria for distinguishing same-shoot groups from similar-shoot groups using rotation and image-similarity analysis |
| Splitting rare classes | Review how to split images containing rare classes across train/val |
| Checking class distribution | Check the per-class distribution and the degree of imbalance |
| Analyzing causes of missing values | Check how the 18 images with missing values relate to the 16 previously known missing boxes |
| Bounding-box criteria | Decide how to handle loose boxes and check label consistency |
| Prep for weekend work | Summarize EDA results and research model candidates |
