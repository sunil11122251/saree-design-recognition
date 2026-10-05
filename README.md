# AIE-CASE - Color-Invariant Saree Design Recognition

A deep metric learning system for recognizing saree surface designs while reducing the effect of color and palette variations.

## 1. Project Overview

This project addresses saree design recognition where the same surface design may appear in different color palettes.

The goal is to learn an image embedding in which images with the same underlying visual structure remain close to each other even when their color appearance changes, while visually different designs remain farther apart.

The system supports two related tasks:

1. **Identification** - rank gallery images according to similarity to a query image.
2. **Verification** - determine whether two images represent the same underlying visual instance or different instances.

The project is implemented using PyTorch and a ResNet18-based embedding network.

## 2. Objective

The main objective is to learn a color-robust representation of saree surface patterns.

The desired behavior is:

```text
Same visual structure + different color
                ↓
        Similar embeddings
                ↓
         Small distance
```

while:

```text
Different visual structure
                ↓
        Different embeddings
                ↓
         Larger distance
```

The project therefore uses metric learning rather than treating the problem only as a conventional four-class classification task.

## 3. Project Requirements

- PyTorch-based implementation
- End-to-end training and inference pipeline
- Image embedding generation
- Verification using embedding distance
- Gallery-based identification/retrieval
- Evaluation on held-out validation and test splits
- Color augmentation to encourage color invariance
- Pretrained ResNet18 backbone
- Reproducible project structure

## 4. Dataset

### 4.1 Indian Fabric Patterns Dataset

The Indian Fabric Patterns dataset contains four pattern categories:

- Banarasi
- Bandhani
- Ikat
- Pichwai

The dataset contains 1,468 images in total and is organized into train, validation, and test folders.

The dataset README reports automatic orientation correction and resizing to 640 x 640 pixels.

### 4.2 DeepLure Saree Corpus

The project also provides access to a proprietary DeepLure saree corpus.

The DeepLure data was inspected locally during development but is **not included in this repository** because it is proprietary and must not be redistributed.

## 5. Dataset Preparation

The Indian Fabric Patterns dataset was used for the main controlled training and evaluation experiment.

The train, validation, and test groups were checked to ensure that there was no group overlap.

```text
Train groups:       431
Validation groups:  115
Test groups:         60

Train ∩ Validation: 0
Train ∩ Test:       0
Validation ∩ Test:  0
```

## 6. Color Invariance Strategy

Since verified real-world pairs containing the same saree design in different colorways were not available in the provided labeled data, controlled color transformations were used to simulate palette changes.

The training pipeline uses:

```text
ColorJitter(
    brightness=0.25,
    contrast=0.25,
    saturation=0.6,
    hue=0.08
)
```

Random horizontal flipping is also applied during training.

These transformations encourage the model to focus more strongly on structural and textural information rather than exact RGB values.

## 7. Model Architecture

The model uses a pretrained ResNet18 backbone.

```text
Input Image
     |
     v
Resize 224 x 224
     |
     v
ResNet18 Backbone
     |
     v
512-dimensional feature
     |
     v
Linear(512 -> 512)
     |
     v
ReLU
     |
     v
Dropout(0.2)
     |
     v
Linear(512 -> 256)
     |
     v
L2 Normalization
     |
     v
256-dimensional embedding
```

The final embedding dimension is **256**.

## 8. Metric Learning

The model learns an embedding space instead of directly predicting only the four dataset classes.

For two images:

```text
Image 1 -> Embedding 1
Image 2 -> Embedding 2
```

their Euclidean distance is calculated.

- Positive pair -> small distance
- Negative pair -> larger distance

## 9. Contrastive Loss

The project uses contrastive loss with a margin of **1.0**.

For a positive pair, the loss encourages the embedding distance to approach zero.

For a negative pair, the loss penalizes distances that remain below the margin.

## 10. Pair Construction

### Training

```text
Positive pairs: 1293
Negative pairs: 1293
Total pairs:    2586
```

### Validation

```text
Positive pairs: 115
Negative pairs: 115
Total pairs:    230
```

### Test

```text
Positive pairs: 60
Negative pairs: 60
Total pairs:    120
```

For validation and test, positive pairs consist of the same image under independent controlled color transformations.

## 11. Training

The training configuration was:

```text
Backbone:           ResNet18
Embedding size:     256
Batch size:         8
Epochs:             5
Learning rate:      0.0001
Optimizer:          AdamW
Weight decay:       0.0001
Contrastive margin: 1.0
```

The pretrained ResNet18 backbone was frozen for this baseline experiment while the embedding head was trained.

Training loss:

```text
Epoch 1: 0.1907
Epoch 2: 0.1486
Epoch 3: 0.1304
Epoch 4: 0.1180
Epoch 5: 0.1101
```

The trained model was saved as:

```text
models/saree_embedding_model.pth
```

## 12. Verification

Verification determines whether two images should be considered similar.

The Euclidean distance between their normalized embeddings is calculated.

```text
Threshold = 0.5
```

Decision rule:

```text
distance < 0.5  -> Same
distance >= 0.5 -> Different
```

## 13. Verification Results

### Validation Set

```text
Positive pairs: 115
Negative pairs: 115

Average positive distance: 0.1769
Average negative distance: 0.8036

Accuracy:  95.22%
Precision: 91.94%
Recall:    99.13%
F1-score:  95.40%
ROC-AUC:   0.9935
```

Validation confusion matrix:

```text
                 Predicted
                 Different  Same

Actual Different     105      10
Actual Same            1     114
```

### Test Set

```text
Positive pairs: 60
Negative pairs: 60

Average positive distance: 0.1620
Average negative distance: 0.7974

Accuracy:  97.50%
Precision: 95.24%
Recall:    100.00%
F1-score:  97.56%
ROC-AUC:   0.9892
```

Test confusion matrix:

```text
                 Predicted
                 Different  Same

Actual Different      57       3
Actual Same            0      60
```

### Verification Summary

| Metric    | Validation |        Test |
| --------- | ---------: | ----------: |
| Accuracy  |     95.22% |  **97.50%** |
| Precision |     91.94% |  **95.24%** |
| Recall    |     99.13% | **100.00%** |
| F1-score  |     95.40% |  **97.56%** |
| ROC-AUC   |     0.9935 |  **0.9892** |

## 14. Identification / Retrieval

Identification compares a query embedding against a gallery of embeddings and ranks gallery images by Euclidean distance.

For the controlled test experiment:

- Gallery contains the 60 test images.
- A color-transformed version of each test image is used as the query.
- The correct original image is expected to appear at the highest ranking.

## 15. Identification Results

```text
Number of queries: 60

Recall@1: 100.00%
Recall@5: 100.00%

Average query-to-correct-gallery distance: 0.1337
```

Every color-transformed query retrieved its corresponding original image at Rank 1 in this controlled evaluation.

## 16. Important Evaluation Limitation

The available labeled dataset does not provide verified real-world identity labels for independent photographs of the same saree design in different color palettes.

Therefore, the current evaluation uses controlled color transformations of the same test image to test color robustness.

The reported results should therefore be interpreted as performance on controlled color-invariance and instance-level retrieval experiments.

They should **not** be interpreted as proof of 100% real-world recognition of independently photographed sarees with the same design but different colorways.

A stronger future evaluation would require a dataset containing verified design identities with multiple independently photographed color variants for each design.

## 17. Project Structure

```text
saree-design-recognition/
│
├── src/
│   ├── inspect_dataset.py
│   ├── create_contact_sheet.py
│   ├── find_duplicates.py
│   ├── create_metadata.py
│   ├── inspect_groups.py
│   ├── color_augmentation.py
│   ├── dataset.py
│   ├── test_dataset.py
│   ├── pair_dataset.py
│   ├── pair_torch_dataset.py
│   ├── test_pair_dataset.py
│   ├── model.py
│   ├── test_model.py
│   ├── loss.py
│   ├── test_loss.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_retrieval.py
│   └── check_split_overlap.py
│
├── models/
│   └── saree_embedding_model.pth
│
├── results/
├── notebooks/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

The dataset itself is not included in the repository.

## 18. Installation

Clone the repository:

```bash
git clone https://github.com/sunil11122251/saree-design-recognition.git
```

Move into the project:

```bash
cd saree-design-recognition
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 19. Dataset Setup

The required datasets must be obtained separately according to their respective access and licensing conditions.

Place available datasets under:

```text
data/
```

The proprietary DeepLure dataset must not be redistributed through this repository.

## 20. Training

After preparing the dataset and metadata:

```bash
python src/train.py
```

The trained model is saved as:

```text
models/saree_embedding_model.pth
```

## 21. Verification Evaluation

Run:

```bash
python src/evaluate.py
```

This reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average positive distance
- Average negative distance
- Confusion matrix

## 22. Identification Evaluation

Run:

```bash
python src/evaluate_retrieval.py
```

This reports:

- Recall@1
- Recall@5
- Average query-to-correct-gallery distance

## 23. Reproducibility

Dependencies are listed in:

```text
requirements.txt
```

The repository contains source code for:

- Dataset processing
- Pair construction
- Color augmentation
- Model definition
- Contrastive loss
- Training
- Verification
- Identification/retrieval

## 24. Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Contrastive Learning
- Metric Learning
- Image Embeddings
- Euclidean Distance
- ColorJitter
- Scikit-learn
- Pandas
- NumPy
- Pillow

## 25. Future Improvements

1. Collect verified same-design/different-color saree pairs.
2. Train with multiple independent photographs per design.
3. Fine-tune the ResNet18 backbone.
4. Add harder negative pairs from visually similar designs.
5. Evaluate with larger galleries.
6. Report additional retrieval metrics such as mAP.
7. Test additional metric-learning objectives such as Triplet Loss or ArcFace-style objectives.
8. Evaluate cross-dataset generalization.
9. Measure inference time and embedding generation efficiency.

## 26. Conclusion

This project implements a deep metric learning pipeline for color-robust saree surface design recognition.

A ResNet18 backbone with a 256-dimensional normalized embedding was trained using contrastive loss and color augmentation.

On the controlled held-out test experiment, the model achieved:

```text
Verification Accuracy: 97.50%
Verification Precision: 95.24%
Verification Recall: 100.00%
Verification F1-score: 97.56%
Verification ROC-AUC: 0.9892

Identification Recall@1: 100.00%
Identification Recall@5: 100.00%
```

The results indicate that the learned embedding successfully maintains similarity under the controlled color transformations used in this experiment.

The main limitation is the absence of verified real-world same-design/different-color identity labels in the available evaluation data. Future work should evaluate the approach on a dataset containing independently photographed sarees with verified design identities and multiple colorways.

## 27. Project Status

**Status: Completed baseline end-to-end implementation and controlled evaluation.**
