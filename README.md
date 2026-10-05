# AIE-CASE - Color-Invariant Saree Design Recognition

## 1. Project Overview

**AIE-CASE - Color-Invariant Saree Design Recognition** is a computer vision system designed to identify saree designs based on their surface patterns and motifs rather than their color palette.

The central idea is similar to face recognition for textiles: two images should be considered similar when they contain the same underlying design structure, even if their colors are substantially different.

The system supports two tasks:

1. **Verification** - determine whether two saree images represent the same design.
2. **Identification / Retrieval** - given a query saree image, rank a gallery of known designs by visual similarity.

The project is implemented in **PyTorch** using a pretrained **ResNet18** backbone and a contrastive-learning embedding model.

---

## 2. Objective

The main requirement is to learn a representation in which:

- The same design with different color palettes should have similar embeddings.
- Different designs should have dissimilar embeddings, even when their colors are similar.
- The learned embedding can be used for both pairwise verification and gallery-based identification.

This formulation follows the project brief, which describes saree design recognition as a pattern-recognition problem where design identity should be independent of color palette.

---

## 3. Approach

### 500-Character Approach Note

> We learn a color-invariant saree design embedding with a pretrained ResNet18 and contrastive loss. Each training image is augmented into multiple synthetic palette variants while preserving motif structure. Positive pairs are variants of the same source design; negatives come from different design groups. A 256-D L2-normalized embedding supports verification by Euclidean distance and gallery identification by nearest-neighbor ranking. Evaluation uses disjoint train/validation/test design groups and reports verification and retrieval metrics.

---

## 4. Dataset

Two datasets were considered according to the project brief.

### 4.1 Indian Fabric Patterns Dataset

The Indian Fabric Patterns dataset contains four saree/fabric pattern categories:

- Banarasi
- Bandhani
- Ikat
- Pichwai

The dataset contains **1,468 images** after extraction.

| Split      | Banarasi | Bandhani |    Ikat | Pichwai |     Total |
| ---------- | -------: | -------: | ------: | ------: | --------: |
| Train      |      432 |      279 |     303 |     279 |     1,293 |
| Validation |       43 |       22 |      26 |      24 |       115 |
| Test       |       14 |       15 |      13 |      18 |        60 |
| **Total**  |  **489** |  **316** | **342** | **321** | **1,468** |

### 4.2 DeepLure Saree Corpus

The DeepLure saree corpus was also inspected as an additional source.

The local copy used during development contained:

- `handloom_sarees`: 165 JPG images
- `normal_sarees`: 0 images

The DeepLure data did not provide sufficiently clear design labels for the contrastive pair construction used in the final controlled experiment. Therefore, the final training/evaluation protocol uses the labeled Indian Fabric Patterns dataset.

**Important:** The DeepLure dataset is proprietary and must not be redistributed. It is excluded from the Git repository.

---

## 5. Dataset Inspection

Several preprocessing and inspection steps were performed:

- Image count and dimension inspection
- Contact-sheet generation
- Duplicate inspection
- Metadata creation
- Group inspection
- Train/validation/test overlap checking

The final split organization contains:

- **431 training design groups**
- **115 validation design groups**
- **60 test design groups**

No group overlap was found between the train, validation, and test splits.

---

## 6. Color-Invariance Strategy

The original dataset does not provide enough real-world examples of the exact same saree design photographed or manufactured in substantially different colorways.

Therefore, a controlled synthetic color-variant generation step was introduced.

For each source image, three color variants are generated while preserving the spatial arrangement and motif structure.

The transformation changes:

- Hue
- Saturation
- Brightness
- Contrast

Each source design therefore has:

```text
Original
├── Color Variant 1
├── Color Variant 2
└── Color Variant 3
```

These transformations are intended to simulate palette changes while keeping the underlying design structure unchanged.

### Important Limitation

The color variants are **synthetically generated**. They are not independent photographs or physically manufactured sarees of the same design in different colorways.

Therefore, the current experiment demonstrates controlled color-invariance rather than proving complete real-world colorway invariance.

---

## 7. Color-Variant Generation

The script:

```text
src/create_color_variants.py
```

generates three synthetic color variants for every source design group.

Generated data:

| Split      | Design Groups | Generated Variants |
| ---------- | ------------: | -----------------: |
| Train      |           431 |              1,293 |
| Validation |           115 |                345 |
| Test       |            60 |                180 |

Each group contains four images:

```text
original.jpg
color_variant_1.jpg
color_variant_2.jpg
color_variant_3.jpg
```

The generated variants are stored under:

```text
data/color_variants/
```

This directory is ignored by Git.

---

## 8. Pair Construction

For each design group containing four versions, all six possible image combinations are treated as positive pairs.

For four images:

```text
4 choose 2 = 6 positive pairs
```

Negative pairs are created by pairing images belonging to different design groups. The number of negative pairs is balanced with the number of positive pairs.

### Final Pair Counts

| Split      | Positive | Negative | Total |
| ---------- | -------: | -------: | ----: |
| Train      |    2,586 |    2,586 | 5,172 |
| Validation |      690 |      690 | 1,380 |
| Test       |      360 |      360 |   720 |

The pair-generation script also checks that positive pairs do not accidentally contain images from different design groups.

---

## 9. Model Architecture

The model uses a pretrained **ResNet18** as the feature extractor.

```text
Input RGB Image
       |
       v
Resize to 224 x 224
       |
       v
Pretrained ResNet18
       |
       v
Feature Vector
       |
       v
Linear Layer: 512
       |
       v
ReLU
       |
       v
Dropout: 0.2
       |
       v
Linear Layer: 256
       |
       v
L2 Normalization
       |
       v
256-D Saree Design Embedding
```

The final embedding has **256 dimensions**.

The ResNet18 classification layer is replaced with an embedding head, and the final embedding is L2-normalized.

---

## 10. Contrastive Learning

For a pair of embeddings:

```text
z1 = embedding(image1)
z2 = embedding(image2)
```

the Euclidean distance between the embeddings is calculated.

For a positive pair:

```text
same design -> embeddings should be close
```

For a negative pair:

```text
different design -> embeddings should be separated
```

The training uses a contrastive-loss margin of:

```text
Margin = 1.0
```

---

## 11. Training Configuration

| Parameter           | Value            |
| ------------------- | ---------------- |
| Backbone            | ResNet18         |
| Pretrained weights  | ImageNet         |
| Embedding dimension | 256              |
| Loss                | Contrastive Loss |
| Margin              | 1.0              |
| Optimizer           | AdamW            |
| Learning rate       | 1e-4             |
| Weight decay        | 1e-4             |
| Batch size          | 8                |
| Epochs              | 5                |
| Input size          | 224 × 224        |
| Dropout             | 0.2              |

During the final experiment, the pretrained ResNet18 backbone was frozen and the embedding head was trained.

---

## 12. Training Result

The final color-invariant model was trained on **5,172 training pairs**.

| Epoch | Average Loss |
| ----: | -----------: |
|     1 |       0.1883 |
|     2 |       0.1324 |
|     3 |       0.1156 |
|     4 |       0.1052 |
|     5 |       0.0969 |

The trained model is saved as:

```text
models/saree_color_invariant_model.pth
```

---

## 13. Verification Evaluation

Verification asks:

> Are these two saree images from the same design?

A Euclidean embedding-distance threshold of:

```text
0.5
```

was used for the reported verification results.

If:

```text
distance <= 0.5
```

the pair is classified as the same design.

Otherwise:

```text
distance > 0.5
```

the pair is classified as different designs.

---

## 14. Validation Results

The validation set contains:

- 690 positive pairs
- 690 negative pairs
- 1,380 total pairs

| Metric                    |     Result |
| ------------------------- | ---------: |
| Accuracy                  | **95.58%** |
| Precision                 | **92.10%** |
| Recall                    | **99.71%** |
| F1 Score                  | **95.76%** |
| ROC-AUC                   | **0.9958** |
| Average Positive Distance | **0.2153** |
| Average Negative Distance | **0.8230** |

Confusion matrix:

```text
[[631, 59],
 [  2, 688]]
```

---

## 15. Test Verification Results

The held-out test set contains:

- 360 positive pairs
- 360 negative pairs
- 720 total pairs

| Metric                    |     Result |
| ------------------------- | ---------: |
| Accuracy                  | **95.56%** |
| Precision                 | **93.62%** |
| Recall                    | **97.78%** |
| F1 Score                  | **95.65%** |
| ROC-AUC                   | **0.9930** |
| Average Positive Distance | **0.2327** |
| Average Negative Distance | **0.8158** |

Confusion matrix:

```text
[[336, 24],
 [  8, 352]]
```

These results show strong separation between same-design and different-design pairs under the controlled synthetic color-variation setting.

---

## 16. Identification / Retrieval

Verification alone does not answer:

> Which known saree design is this query image?

Therefore, a gallery-based retrieval experiment was performed.

### Gallery

The gallery contains:

- 60 held-out test design groups
- One original image per design group

### Queries

For each of the 60 design groups:

- 3 synthetic color variants are used as queries

Total:

```text
60 x 3 = 180 queries
```

For each query, its embedding is compared with every gallery embedding and the gallery designs are ranked by Euclidean distance.

---

## 17. Retrieval Results

| Metric                          |     Result |
| ------------------------------- | ---------: |
| Number of Gallery Designs       |         60 |
| Number of Queries               |        180 |
| Recall@1                        | **81.11%** |
| Recall@5                        | **97.22%** |
| Average Correct-Design Distance | **0.2357** |

### Interpretation

**Recall@1 = 81.11%**

For approximately 81% of synthetic color-variant queries, the correct design was ranked first.

**Recall@5 = 97.22%**

For approximately 97% of queries, the correct design appeared within the top five retrieved gallery designs.

This indicates that the learned embedding captures useful design-level information while remaining reasonably robust to the synthetic color transformations used in this experiment.

---

## 18. Verification vs Identification

### Verification

Input:

```text
Image A + Image B
```

Output:

```text
Same Design / Different Design
```

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

### Identification

Input:

```text
Query Image + Gallery
```

Output:

```text
Ranked list of candidate designs
```

Metrics:

- Recall@1
- Recall@5
- Retrieval distance

The project evaluates both capabilities.

---

## 19. Data Leakage Prevention

Data leakage is important because multiple images can originate from the same source design.

The dataset was split at the **design-group level** before creating synthetic variants.

Therefore:

```text
Train groups
    !=
Validation groups
    !=
Test groups
```

Synthetic variants of a source image remain within the same split.

Final split:

- 431 training groups
- 115 validation groups
- 60 test groups

No train/validation/test group overlap was found.

---

## 20. Project Structure

```text
saree-design-recognition/
│
├── data/
│   ├── indian_saree_patterns/
│   ├── sarees_dataset/
│   └── color_variants/
│
├── models/
│   └── saree_color_invariant_model.pth
│
├── notebooks/
│
├── results/
│
├── src/
│   ├── inspect_dataset.py
│   ├── create_contact_sheet.py
│   ├── find_duplicates.py
│   ├── create_metadata.py
│   ├── inspect_groups.py
│   ├── color_augmentation.py
│   ├── dataset.py
│   ├── pair_dataset.py
│   ├── pair_torch_dataset.py
│   ├── test_pair_dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_retrieval.py
│   ├── create_color_variants.py
│   ├── create_color_pairs.py
│   ├── color_pair_torch_dataset.py
│   ├── test_color_pair_dataset.py
│   ├── train_color_invariance.py
│   ├── evaluate_color_invariance.py
│   └── evaluate_color_retrieval.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The `data/` directory is ignored by Git so that datasets are not uploaded to the repository.

The proprietary DeepLure images must not be committed or redistributed.

---

## 21. Installation

### Clone the repository

```bash
git clone https://github.com/sunil11122251/saree-design-recognition.git
cd saree-design-recognition
```

### Create a Python environment

Python 3.10 was used during development.

```bash
py -3.10 -m venv venv
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 22. Running the Pipeline

### Step 1 - Inspect the dataset

```bash
python src/inspect_dataset.py
```

### Step 2 - Create metadata

```bash
python src/create_metadata.py
```

### Step 3 - Inspect source groups

```bash
python src/inspect_groups.py
```

### Step 4 - Generate synthetic color variants

```bash
python src/create_color_variants.py
```

### Step 5 - Create positive and negative pairs

```bash
python src/create_color_pairs.py
```

### Step 6 - Test the PyTorch pair dataset

```bash
python src/test_color_pair_dataset.py
```

### Step 7 - Train the color-invariant model

```bash
python src/train_color_invariance.py
```

### Step 8 - Evaluate verification

```bash
python src/evaluate_color_invariance.py
```

### Step 9 - Evaluate identification/retrieval

```bash
python src/evaluate_color_retrieval.py
```

---

## 23. Requirements

```text
torch
torchvision
torchaudio
numpy
pandas
Pillow
scikit-learn
matplotlib
tqdm
```

The project is implemented in PyTorch as required by the project brief.

---

## 24. Reproducibility

The repository contains code for:

- Dataset inspection
- Metadata generation
- Synthetic color-variant creation
- Pair generation
- PyTorch dataset loading
- Model definition
- Contrastive-loss training
- Verification evaluation
- Retrieval evaluation

The trained model checkpoint is included separately from the datasets.

Dataset files are intentionally not included in the repository.

---

## 25. Limitations

### 25.1 Synthetic color variants

The main color-invariance evaluation uses controlled synthetic transformations rather than independently photographed or manufactured sarees with the same design in different colorways.

Therefore, the reported results demonstrate robustness to controlled palette changes, not complete real-world colorway invariance.

### 25.2 Limited labeled design groups

The final retrieval evaluation contains 60 held-out test design groups.

A larger gallery with more unique designs would provide a stronger measure of practical identification performance.

### 25.3 Retrieval performance

The final retrieval performance is:

```text
Recall@1 = 81.11%
Recall@5 = 97.22%
```

The correct design is therefore often retrieved among the top candidates, but it is not always ranked first.

### 25.4 Real-world variation

Future evaluation should include real images of the same design under:

- Different colorways
- Different lighting
- Different camera devices
- Different scales
- Different viewpoints
- Fabric folds
- Background variation

---

## 26. Future Improvements

1. Use real same-design/different-color image pairs.
2. Fine-tune the ResNet18 backbone.
3. Compare stronger pretrained backbones.
4. Use triplet loss or supervised contrastive loss.
5. Add hard-negative mining.
6. Use multiple gallery images per design.
7. Increase the number of unique design identities.
8. Evaluate cross-dataset generalization.
9. Add precision-recall and ROC curves.
10. Benchmark inference speed and model size.
11. Investigate texture-specific feature extractors.
12. Use segmentation or cropping to focus on the saree surface rather than background regions.

---

## 27. Conclusion

This project implements a complete prototype for **color-invariant saree design recognition** using metric learning.

The system:

- Extracts visual design embeddings using a pretrained ResNet18.
- Uses a 256-dimensional normalized embedding space.
- Trains with contrastive loss.
- Generates controlled synthetic color variants to encourage color invariance.
- Performs pairwise design verification.
- Performs gallery-based design identification.
- Uses group-level train/validation/test separation to avoid source-image leakage.

### Final Held-Out Verification

```text
Accuracy  : 95.56%
Precision : 93.62%
Recall    : 97.78%
F1 Score  : 95.65%
ROC-AUC   : 0.9930
```

### Final Controlled Color-Variant Retrieval

```text
Recall@1 : 81.11%
Recall@5 : 97.22%
```

The results demonstrate that the learned embedding captures useful saree design similarity and provides strong robustness to the synthetic color transformations used in this experiment.
