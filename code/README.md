# Team 17 - Deepfake Detection (NYCU 2025 Deep Learning Final Project)

**Strongly recommended to read this document using a Markdown viewer!**

## 📌 TL;DR
1. **Setup**: `pip install -r requirements.txt` (Environment: Python 3.10)
2. **Data**: Place dataset in `code/dataset/` (`train/` and `test/`)
3. **Inference**: Run `python src/infer.py` to generate `submission.csv`

---

## ℹ️ Project Info
*   **Topic**: Deepfake Detection using OpenCLIP (ConvNext XXLarge) + LoRA
*   **Key Files**: `src/train.py`, `src/infer.py`
*   **Reference**: `src/vn_laion_soup_fin.ipynb`

---

## 👥 Team Members & Credits

Member 1 was responsible for the core solution design and implementation, including the OpenCLIP + LoRA training pipeline, experiments, and final submission generation.

Other members contributed through independent exploration, discussion, experiments, and feedback during the project.

Member numbers **do not** reflect contribution ranking.

| Member | Student ID | Name |
| --- | --- | --- |
| 1 | 314511063 | 李兆翔 |
| 2 | 314511043 | 張恬嘉 |
| 3 | A141113 | 李京罡 |
| 4 | A141245 | 朱庭暄 |
| 5 | A141251 | 廖振宇 |
| 6 | M093088 | 施欣宜 |
| 7 | Z141507 | 黃鈺翔 |



---

## 🛠 Environment Setup

```bash
conda create -n team17 python=3.10 -y
conda activate team17
pip install -r requirements.txt
```

**Key Dependencies**:
- `torch==2.9.1`
- `torchvision`
- `open_clip_torch`
- `pytorch_lightning`
- `peft`
- `scikit-learn`
---

## 📂 Dataset Setup

1.  Download the dataset from the [Kaggle Competition](https://www.kaggle.com/competitions/nycu2025dlfp/data).
2.  Place the data in the `code/dataset/` directory.
3.  The directory structure should look like this:

    ```text
    team_17/
    ├── code/
    │   ├── dataset/
    │   │   ├── train/
    │   │   │   ├── fake/  (Training fake images)
    │   │   │   └── real/  (Training real images)
    │   │   └── test/      (Test images for inference)
    │   ├── src/
    │   │   ├── train.py
    │   │   ├── infer.py
    │   │   └── common.py
    │   ├── requirements.txt
    │   └── README.md
    └── team_17_submission.csv
    ```

---

## 🚀 How to Run

### 1. Training
 Run the training script to fine-tune the model. This will create logs and save the best checkpoint in `./lightning_logs/`.

```bash
python src/train.py
```
*   **Log Location**: `./lightning_logs/version_*/`
*   **Best Checkpoint**: `./lightning_logs/version_*/checkpoints/best-checkpoint.ckpt`

### 2. Inference
Run the inference script to generate predictions on the test set. It automatically finds the best checkpoint from `./lightning_logs`.

```bash
python src/infer.py
```
*   **Output**: Generates `submission.csv`.

---

## 🧠 Model Architecture & Technical Details

We utilize a **ConvNeXt XXLarge** backbone pre-trained on LAION-2B (OpenCLIP), adapted with **LoRA (Low-Rank Adaptation)** for efficient fine-tuning.

*   **Backbone**: `laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup`
    *   We use the **visual trunk** of the CLIP model.
    *   **LoRA** is applied to the `fc1` and `fc2` layers of the MLP blocks in the Transformer/ConvNeXt stages.
*   **Classifier Head**:
    *   Reference: [AIDE: A Sanity Check for AI-generated Image Detection (ICLR 2025)](https://github.com/shilinyan99/AIDE)
    *   Architecture: `Global Avg Pool` -> `Linear(3072, 256)` -> `Linear(256, 1)` (Binary Classification)
*   **Training Strategy**:
    *   **Split**: 85% Training, 15% Validation (Stratified Split using Scikit-Learn).
    *   **Optimizer**: AdamW.
    *   **Loss Function**: Focal Loss (Sigmoid).
    *   **Framework**: PyTorch Lightning for organized training loops and checkpointing.

### Performance & Specs
*   **VRAM Usage**: ~10GB (Tested on NVIDIA RTX 5070 Ti).
*   **Storage**: ~8GB required (Pretrained Model [~4GB] + Checkpoints [~4GB]).
*   **Execution Time**: ~1 hour (Training + Inference).

---

## 💻 Code Overview

The project code is organized into the `src/` directory:

*   **[common.py](src/common.py)**: Contains shared components:
    *   `LitDeepfakeModule`: The PyTorch Lightning module defining the model architecture, forward pass, and training steps.
    *   `CustomDataset`: A unified dataset class for loading images.
    *   Configuration constants (Hyperparameters, Model Name, Seed).
*   **[train.py](src/train.py)**: Handles the training pipeline:
    *   Loads dataset and performs train/val split.
    *   Initializes the model and trainer.
    *   Runs the training loop.
*   **[infer.py](src/infer.py)**: Handles the inference pipeline:
    *   Locates the best checkpoint.
    *   Loads the test set.
    *   Runs prediction and generates the CSV submission file.
*   **[vn_laion_soup_fin.ipynb](src/vn_laion_soup_fin.ipynb)**: The original notebook as reference.