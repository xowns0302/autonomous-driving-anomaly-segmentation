# Autonomous Driving Anomaly Segmentation - Shared Project Status

> Shared working document for ChatGPT, Claude, and the project owner.  
> Keep this file updated after every major decision, paper review, code experiment, or dataset setup.

---

## 0. Project Repository

Recommended repository name:

```text
autonomous-driving-anomaly-segmentation
```

Short description:

```text
Camera-based road anomaly / OoD object segmentation study for autonomous driving.
```

---

## 1. Project Summary

This graduation project studies **camera-based road anomaly segmentation for autonomous driving**.

The project focuses on detecting and segmenting **out-of-distribution (OoD) / anomalous objects** in road scenes, such as unexpected obstacles, animals, fallen cargo, cones, rocks, or unfamiliar objects that are not included in the predefined semantic classes of ordinary segmentation models.

The project will not build a physical model car due to time constraints. Instead, it will focus on:

1. understanding the research flow of road anomaly segmentation,
2. reviewing benchmark datasets and representative methods,
3. reproducing and analyzing a main paper,
4. comparing it against related methods where feasible,
5. documenting results, limitations, and future extensions.

---

## 2. Current Project Direction

### Main Theme

```text
Camera-based autonomous driving road scene OoD / anomaly object detection and segmentation.
```

### Tentative Korean Title

```text
카메라 기반 자율주행 도로 장면에서 OoD 이상객체 탐지 기법의 비교 분석 및 재현
```

### Tentative English Title

```text
Comparative Analysis and Reproduction of Camera-Based Road Anomaly Segmentation for Autonomous Driving
```

---

## 3. Current Paper Roles

### Main Paper Candidate

#### DetSeg

```text
Beyond Pixel Uncertainty: Bounding the OoD Objects in Road Scenes
ICCV 2025
```

Role in project:

- Main reproduction and analysis target.
- Focuses on road scene OoD object detection.
- Introduces object-level understanding to reduce false positives from pixel-wise uncertainty methods.
- Uses open-world object detection, ID object suppression, and segmentation/refinement variants.

Main concepts to study:

- Grounding DINO-based object proposals
- ID queries and Universal queries
- IE-NMS: In-distribution Enhanced Non-Maximum Suppression
- DetSeg-R: refinement of anomaly score maps
- DetSeg-S: threshold-free anomaly mask generation using box-prompted segmentation such as SAM
- VPHM: Vanishing Point Guided Hungarian Matching for temporal smoothing

Why this is currently the strongest main candidate:

- Published at ICCV 2025.
- Directly about road scene OoD objects.
- Clear problem setting: pixel-wise uncertainty methods create false positives and require threshold search.
- Has official code.
- Can be positioned as an object-level improvement over pixel-wise anomaly score approaches.

---

### Comparison / Supporting Paper

#### S2M

```text
Segment Every Out-of-Distribution Object
CVPR 2024
```

Role in project:

- Important comparison and related work.
- Converts anomaly score maps into prompts for a promptable segmentation model.
- Helps explain the research transition from pixel-level anomaly scores to object-level masks.

Main concepts to study:

- Anomaly score map
- Box prompt generation
- SAM / promptable segmentation
- Threshold-free or threshold-reduced mask generation
- Fragmented mask problem

Relationship with DetSeg:

```text
S2M:
anomaly score map -> box prompt -> promptable segmentation mask

DetSeg:
open-world object detection -> suppress ID objects -> OoD boxes -> refinement or prompt-based mask
```

S2M is useful because it provides a clear bridge between classical anomaly score maps and DetSeg-style object-level reasoning.

---

### Latest Trend / Future Direction Paper

#### ClimaOoD

```text
ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data
arXiv 2026 / venue status still needs confirmation
```

Role in project:

- Related work and latest trend.
- Not currently selected as the main reproduction target because code/data availability and official venue status are uncertain.
- Useful for discussing data-centric improvements in road anomaly segmentation.

Main concepts to study:

- Synthetic OoD driving data
- Weather-diverse road scenes
- Physically realistic anomaly placement
- ClimaDrive generation framework
- Data diversity and open-world robustness

How to use it in the report:

```text
Recent work also attempts to improve road anomaly segmentation not only through model architecture, but also through physically realistic synthetic data generation under diverse weather and driving scenarios.
```

---

## 4. Research Flow to Present in the Graduation Project

The project should not look like a single-paper review. The target structure is a research-flow analysis:

### 4.1 Problem and Benchmarks

Road anomaly segmentation aims to detect and localize unknown or OoD objects in road scenes. This is important for autonomous driving because real-world roads may contain unexpected hazards that are not covered by predefined semantic classes.

Representative datasets / benchmarks:

- LostAndFound
- Fishyscapes
- RoadAnomaly
- Segment-Me-If-You-Can, SMIYC
- RoadObstacle / RoadAnomaly tracks

Purpose of this section:

- Explain why the problem matters.
- Explain what is evaluated.
- Explain commonly used metrics.

---

### 4.2 Pixel-wise Anomaly Score Methods

Earlier and representative methods often produce a pixel-wise anomaly score map based on semantic segmentation uncertainty, energy, rejection score, or related ideas.

Representative methods:

- PEBAL
- RPL
- RbA
- Mask2Anomaly
- UNO
- SynBoost
- DenseHybrid

Typical pipeline:

```text
input road image
-> semantic segmentation model
-> pixel-wise anomaly score map
-> thresholding
-> binary anomaly mask
```

Limitations:

- Threshold selection is sensitive.
- Boundary regions often cause false positives.
- Masks can become fragmented or incomplete.
- Pixel-wise scores do not necessarily capture whole-object structure.
- Frame-to-frame inconsistency can confuse downstream planning.

---

### 4.3 Promptable Segmentation Direction: S2M

S2M addresses the weakness of thresholding anomaly score maps by converting anomaly scores into prompts and using a promptable segmentation model to obtain cleaner masks.

Key idea:

```text
anomaly score map
-> prompt generator
-> box prompts
-> SAM / promptable segmentation model
-> OoD object mask
```

Strengths:

- Improves mask quality.
- Reduces dependence on manual thresholding.
- Uses promptable segmentation models such as SAM.

Limitation to analyze:

- It still depends on the quality of the initial anomaly score map.
- If anomaly scores do not highlight the object well, prompt generation may fail.

---

### 4.4 Object-Level Understanding Direction: DetSeg

DetSeg argues that pixel-wise anomaly score methods lack object-level understanding. It detects object candidates first, suppresses known in-distribution objects, and keeps potential OoD object boxes.

Simplified DetSeg idea:

```text
input road image
-> open-world object detection
-> detect ID objects and all objects
-> suppress ID boxes
-> retain OoD object boxes
-> DetSeg-R or DetSeg-S
```

DetSeg variants:

```text
DetSeg-R:
Use OoD boxes to suppress anomaly scores outside candidate regions.
Goal: reduce scattered false positives.

DetSeg-S:
Use OoD boxes with a box-prompted segmentation module such as SAM.
Goal: produce binary anomaly masks without complex threshold search.
```

Temporal smoothing:

```text
VPHM:
Vanishing Point Guided Hungarian Matching uses road geometry and consecutive frames to smooth predictions.
```

Why DetSeg is the current main paper:

- It naturally follows the limitations of pixel-wise anomaly score methods.
- It can be compared with S2M.
- It is directly about road scene OoD objects.
- It has a clear implementation target and official code.

---

### 4.5 Data-Centric Direction: ClimaOoD

ClimaOoD focuses on the lack of diverse anomaly training data. It argues that existing datasets are limited in weather, scene type, and anomaly diversity.

Key idea:

```text
semantic guidance + diffusion/inpainting + weather diversity + physically plausible anomaly placement
-> synthetic OoD driving data
-> improved anomaly segmentation robustness
```

Use in this project:

- Related work / latest trend.
- Possible future work.
- Not the current main reproduction target unless official code and data become clearly available.

---

## 5. Current Decisions

### Confirmed

- The project will focus on **camera-based** autonomous driving anomaly segmentation.
- Physical model car implementation is excluded due to time constraints.
- The project should not be a shallow single-paper reproduction.
- The project will present the broader research flow and then focus on DetSeg.
- DetSeg is currently the main reproduction target.
- S2M is the most important comparison / supporting paper.
- ClimaOoD is useful as a latest trend / future direction, but not the main implementation target for now.
- Repository name recommendation: `autonomous-driving-anomaly-segmentation`.

### Still Open

- Whether to reproduce only DetSeg-R or both DetSeg-R and DetSeg-S.
- Whether to run S2M code directly or only analyze it as related work.
- Which dataset to use first:
  - RoadAnomaly
  - Fishyscapes
  - SMIYC
  - LostAndFound
- Which metrics to implement/report:
  - AUROC
  - AP / AUPRC
  - FPR95
  - IoU
  - mean F1
  - sIoU / component-level metrics if using SMIYC
- Whether VPHM temporal smoothing will be reproduced. This depends on video data availability and project time.

---

## 6. Suggested Minimum Viable Project Scope

### MVP Version

```text
1. Study and summarize road anomaly segmentation research flow.
2. Analyze S2M and DetSeg in detail.
3. Set up DetSeg official code.
4. Run DetSeg inference/evaluation on one dataset.
5. Compare original anomaly score method vs DetSeg-R, or compare DetSeg-S with S2M numbers from paper.
6. Analyze qualitative success/failure cases.
7. Write final report.
```

### Recommended Dataset Start

Start with one of:

```text
RoadAnomaly
Fishyscapes Lost & Found
```

Reason:

- Smaller and more manageable than trying all benchmarks at once.
- Easier to use for initial reproduction and visualization.

### Stretch Goals

```text
1. Run S2M code and compare with DetSeg-S.
2. Add SMIYC benchmark evaluation.
3. Try LostAndFound video data and test VPHM.
4. Analyze failure cases involving small objects, boundary false positives, and fragmented masks.
```

---

## 7. Code and Repository Structure

Suggested repository structure:

```text
autonomous-driving-anomaly-segmentation/
  README.md
  PROJECT_STATUS.md
  PAPER_REVIEW.md
  papers/
    README.md
  notes/
    01_problem_and_benchmarks.md
    02_s2m_review.md
    03_detseg_review.md
    04_climaood_review.md
    05_related_work.md
  experiments/
    exp_plan.md
    result_log.md
    failure_cases.md
  src/
    README.md
  assets/
    figures/
    qualitative_results/
```

---

## 8. Immediate Next Tasks

### Task 1. Create Repository

Repository name:

```text
autonomous-driving-anomaly-segmentation
```

Initialize with:

```text
README.md
PROJECT_STATUS.md
.gitignore
```

### Task 2. Add Papers

Place the papers in:

```text
papers/
```

Recommended filenames:

```text
papers/detseg_iccv2025.pdf
papers/s2m_cvpr2024.pdf
papers/climaood_arxiv2026.pdf
```

### Task 3. DetSeg Paper Review

Create:

```text
notes/03_detseg_review.md
```

Initial outline:

```markdown
# DetSeg Review

## 1. Problem Definition
## 2. Prior Method Limitations
## 3. Main Idea
## 4. Architecture
## 5. DetSeg-R
## 6. DetSeg-S
## 7. IE-NMS
## 8. VPHM
## 9. Experiments
## 10. What to Reproduce
## 11. Risks
```

### Task 4. Check Official Code

Need to verify:

```text
- official GitHub repository
- environment setup
- pretrained weights
- dataset requirements
- evaluation scripts
- example inference command
```

### Task 5. Decide First Experiment

Recommended first experiment:

```text
Run DetSeg inference/evaluation on RoadAnomaly or Fishyscapes Lost & Found.
```

---

## 9. Prompt for Claude or Another AI

Use the following prompt when moving to Claude or another assistant:

```text
Below is the shared project status for my graduation project. Please use it as the source of truth and continue from the latest decisions.

The project is about camera-based autonomous driving road anomaly / OoD object segmentation. The main paper is DetSeg (ICCV 2025), the key comparison paper is S2M (CVPR 2024), and ClimaOoD is used as latest trend / related work. We are not building a physical model car. The project should not be a single-paper summary; it should present the research flow from benchmarks and pixel-wise anomaly score methods to S2M and DetSeg.

Your next task is to help me prepare the repository and start the DetSeg paper/code analysis. Keep all new decisions compatible with this PROJECT_STATUS.md. At the end, summarize updates under:
1. Newly confirmed decisions
2. Changed assumptions
3. New TODOs
4. Summary to pass back to ChatGPT
```

---

## 10. Work Log

### 2026-09-29

- Decided to focus on camera-based road anomaly segmentation.
- Excluded physical model car implementation due to time constraints.
- Compared S2M, ClimaOoD, and DetSeg.
- Selected DetSeg as the main reproduction candidate.
- Selected S2M as the key comparison paper.
- Selected ClimaOoD as latest trend / related work.
- Decided repository name should clearly show autonomous driving and anomaly segmentation.
- Recommended repository name: `autonomous-driving-anomaly-segmentation`.
- Created this shared status document for ChatGPT-Claude collaboration.
