# UMBC Framework for Humanoid Locomotion Behavior Isolation

Welcome to the official repository for the **Unified Multi-modal Behavior Cloned (UMBC)** framework for humanoid robot locomotion.

This repository provides the necessary scripts and resources for **data collection, unified policy training, and evaluation** of the proposed UMBC framework using NVIDIA Isaac Lab.

---

## Prerequisites & Setup Guide

To ensure compatibility and avoid underlying API conflicts, this codebase has been **tested and developed specifically for Isaac Lab 2.0.2**, which is the recommended version for this repository.

Please follow the steps below carefully to set up the environment, prepare the dataset, train the model, and evaluate the resulting policy.

---

## Step 1: Install Isaac Lab

If you have not already installed Isaac Lab, download and install **Isaac Lab 2.0.2** by following the official NVIDIA Isaac Lab installation instructions.

Before proceeding, ensure that your Isaac Lab installation is functioning correctly.

---

## Step 2: Clone the Repository

Clone this repository inside the `IsaacLab/scripts/` directory to ensure proper path resolution with the simulator scripts.

```bash
cd path/to/isaaclab/scripts/
git clone https://github.com/fxrarz/UMBC.git
```

After cloning, the directory structure should look like:

```text
IsaacLab/
└── scripts/
    └── UMBC/
        ├── 1.data_collection/
        ├── 2.unified_policy/
        ├── 3.test/
        └── ...
```

---

## Step 3: Data Collection *(Optional)*

If you prefer to generate your own training data from scratch, you can use the provided data collection scripts.

Run the following commands from the **root `IsaacLab` directory**.

### Flat Terrain

```bash
./isaaclab.sh -p scripts/UMBC/1.data_collection/flat/phase1.py
```

### Stairway Terrain

```bash
./isaaclab.sh -p scripts/UMBC/1.data_collection/stairway/phase1.py
```

> **Note:** Data collection is optional. If you use the pre-generated dataset, you can skip this step and proceed directly to [Step 4](#step-4-alternative-dataset-setup-kaggle-download).

---

## Step 4: Alternative Dataset Setup (Kaggle Download)

To avoid the potentially time-consuming data collection process, you can use the **pre-generated UMBC training dataset** available on Kaggle.

### Download the Dataset

The dataset is publicly available at:

[UMBC Dataset — Kaggle](https://www.kaggle.com/datasets/towhitebot/umbc-dataset?utm_source=chatgpt.com)

After downloading the dataset:

1. Create a folder named `dataset` inside `scripts/UMBC/`.

2. Place the downloaded dataset archive inside the `dataset` folder.

3. Extract the archive into the `dataset` folder.

The resulting directory structure should resemble:

```text
IsaacLab/
└── scripts/
    └── UMBC/
        ├── dataset/
        │   └── 2_Terrain_Trajectory_Dataset/
        ├── 1.data_collection/
        ├── 2.unified_policy/
        ├── 3.test/
        └── logs/
```

> **Note:** If you download and prepare the pre-generated dataset, you can skip the data collection step and proceed directly to training the unified policy.

### Dataset Contents

The dataset contains the terrain-specific trajectory data required for training the proposed UMBC framework, including data collected for:

* **Flat terrain**
* **Stairway terrain**

The dataset can be used directly with the training script provided in this repository.

---

## Step 5: Train the Unified Policy

Once the dataset has been prepared, run the following command from the **root `IsaacLab` directory**:

```bash
python3 scripts/UMBC/2.unified_policy/train.py
```

The trained model checkpoints and corresponding logs will be stored in the designated `logs` directory.

---

## Step 6: Testing and Simulation

If you want to skip the training process, pre-trained model checkpoints are provided in:

```text
scripts/UMBC/logs/model
```

You can directly evaluate the pre-trained policy using the following scripts.

### Test on Flat Terrain

```bash
python3 scripts/UMBC/3.test/unified_flat.py
```

### Test on Stairway Terrain

```bash
python3 scripts/UMBC/3.test/unified_stairway.py
```

---

## Workflow Overview

The complete workflow can be summarized as follows:

```text
Install Isaac Lab 2.0.2
        │
        ▼
Clone UMBC Repository
        │
        ▼
Prepare Dataset
   ┌────┴────┐
   │         │
   ▼         ▼
Collect     Download
Data        Dataset
   │         │
   └────┬────┘
        ▼
Train Unified Policy
        │
        ▼
Evaluate / Simulate
   ┌────┴─────┐
   │          │
   ▼          ▼
  Flat     Stairway
 Terrain    Terrain
```

---

## Repository Structure

```text
UMBC/
├── 1.data_collection/
│   ├── flat/
│   └── stairway/
│
├── 2.unified_policy/
│   └── train.py
│
├── 3.test/
│   ├── unified_flat.py
│   └── unified_stairway.py
│
├── dataset/
│   └── 2_Terrain_Trajectory_Dataset/
│
└── logs/
    └── model/
```

---

## Quick Start

If you already have Isaac Lab 2.0.2 installed and want to use the pre-generated dataset and pre-trained model, the quickest workflow is:

```bash
# Clone repository
cd path/to/isaaclab/scripts/
git clone https://github.com/fxrarz/UMBC.git

# Train using the prepared dataset
cd path/to/isaaclab/
python3 scripts/UMBC/2.unified_policy/train.py

# Test on flat terrain
python3 scripts/UMBC/3.test/unified_flat.py

# Test on stairway terrain
python3 scripts/UMBC/3.test/unified_stairway.py
```

Alternatively, you can skip training and directly run the testing scripts using the provided pre-trained checkpoints.

---

## Compatibility

| Component     | Version                        |
| ------------- | ------------------------------ |
| **Isaac Lab** | **2.0.2**                      |
| **Python**    | As required by Isaac Lab 2.0.2 |
| **Simulator** | NVIDIA Isaac Lab               |

> **Important:** For reproducibility and compatibility, we strongly recommend using **Isaac Lab 2.0.2** as specified above.

