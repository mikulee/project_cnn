# Tree Species Classification - Complete Walkthrough

This document explains the entire deep learning pipeline, from data loading to final evaluation.

---

## Project Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TREE SPECIES CLASSIFICATION                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Dataset: 23 tree species, ~4,800 images from urban streets                     │
│                                                                                  │
│  Primary Comparison:   ResNet50 vs ResNet18 (different sized architectures)     │
│  Supporting Analysis:  Transfer Learning vs Training from Scratch (TreeNet)     │
│                                                                                  │
│  Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix, N-Best          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│   DATA   │──▶│TRANSFORMS│──▶│  MODELS  │──▶│ TRAINING │──▶│EVALUATION│
│  LOADING │   │  & AUG   │   │          │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## 1. Data Loading

### Dataset Structure

```
data/classification_tree/
├── train/              # ~80% of data (~3,800 images)
│   ├── acer_palmatum/
│   ├── cinnamomum_camphora/
│   ├── ginkgo_biloba/
│   └── ... (23 species)
├── val/                # ~10% of data (~480 images)
└── test/               # ~10% of data (~480 images)
```

### Why Three Splits?

| Split | Purpose | When Used |
|-------|---------|-----------|
| **Train** | Learn patterns | Every epoch during training |
| **Validation** | Tune hyperparameters, early stopping | After each epoch |
| **Test** | Final unbiased evaluation | Only once at the end |

### Code

```python
from torchvision import datasets

train_dataset = datasets.ImageFolder(root="data/classification_tree/train", 
                                      transform=train_transforms)
val_dataset = datasets.ImageFolder(root="data/classification_tree/val",
                                    transform=val_transforms)
test_dataset = datasets.ImageFolder(root="data/classification_tree/test",
                                     transform=val_transforms)

classes = train_dataset.classes  # ['acer_palmatum', 'cinnamomum_camphora', ...]
num_classes = len(classes)       # 23
```

---

## 2. Data Transforms & Augmentation

### Why 224x224 Pixels?

The ResNet models were pretrained on ImageNet at 224x224 resolution. Using the same size ensures:
- Pretrained features work correctly
- Good balance between detail and computation

For higher-resolution source images, consider 384x384 or 512x512 for more detail.

### Training Transforms (with augmentation)

```python
train_transforms = transforms.Compose([
    # Resize larger than target for random cropping
    transforms.Resize((256, 256)),
    
    # Random crop - introduces position variance
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    
    # Geometric augmentations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    
    # Color augmentations - handle lighting variations
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                          saturation=0.2, hue=0.1),
    
    # Convert to tensor and normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225]),
])
```

### Validation/Test Transforms (NO augmentation)

```python
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

### Why Augmentation?

```
Original Image          Augmented Versions
    ┌─────┐            ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
    │ 🌳  │    ───▶    │ 🌳  │ │ 🌳↔ │ │ 🌳🔆│ │ 🌳↻ │
    └─────┘            └─────┘ └─────┘ └─────┘ └─────┘
                       original flipped brighter rotated
```

- Trees look different in different seasons, lighting, angles
- Augmentation simulates this variation
- Effectively multiplies dataset size
- Reduces overfitting

---

## 3. Class Weights for Imbalanced Data

### The Problem

```
cinnamomum_camphora:  292 images  ████████████████████
lagerstroemia_indica:  81 images  █████
```

Without correction, the model might just predict the majority class.

### The Solution

```python
# Inverse frequency weighting
class_weights = total_samples / (num_classes * count_per_class)

# Example weights:
# cinnamomum_camphora (292 images): weight ≈ 0.8
# lagerstroemia_indica (81 images): weight ≈ 2.9

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

Rare species get higher loss weight, forcing the model to learn them.

---

## 4. DataLoaders

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=128,           # Process 128 images at once
    shuffle=True,             # Randomize order each epoch
    num_workers=16,           # Parallel data loading
    pin_memory=True,          # Faster CPU→GPU transfer
    persistent_workers=True   # Keep workers alive
)
```

### Platform-Specific Settings

| Platform | batch_size | num_workers | persistent_workers |
|----------|------------|-------------|-------------------|
| WSL2/Linux | 128 | 16 | True |
| Windows | 64 | 0 | False |

---

## 5. Model Architectures

### 5.1 TreeNet (Custom CNN - From Scratch)

```
Input: [3, 224, 224]
    ↓
┌─────────────────────────────────────┐
│ Stage 1: Conv→BN→ReLU, MaxPool     │  → [32, 112, 112]
├─────────────────────────────────────┤
│ Stage 2: Conv→BN→ReLU, MaxPool     │  → [64, 56, 56]
├─────────────────────────────────────┤
│ Stage 3: Conv→BN→ReLU, MaxPool     │  → [128, 28, 28]
├─────────────────────────────────────┤
│ Stage 4: Conv→BN→ReLU, MaxPool     │  → [256, 14, 14]
├─────────────────────────────────────┤
│ Stage 5: Conv→BN→ReLU, MaxPool     │  → [512, 7, 7]
├─────────────────────────────────────┤
│ AdaptiveAvgPool → Flatten          │  → [512]
├─────────────────────────────────────┤
│ FC → ReLU → Dropout → FC           │  → [23]
└─────────────────────────────────────┘
Output: [23] (logits for each class)

Parameters: ~2.5M
Training: From scratch (random initialization)
```

### 5.2 ResNet18 / ResNet50 (Transfer Learning)

```
┌─────────────────────────────────────┐
│         PRETRAINED BACKBONE         │  ← ImageNet weights (frozen initially)
│   (Learned from 14M images)         │
│                                     │
│   ResNet18: 512 features            │
│   ResNet50: 2048 features           │
├─────────────────────────────────────┤
│         CUSTOM HEAD                 │  ← Random init, learns our task
│   FC(512) → BN → ReLU → Dropout     │
│   FC(256) → BN → ReLU → Dropout     │
│   FC(23)                            │
└─────────────────────────────────────┘

ResNet18: ~11.7M parameters
ResNet50: ~25.6M parameters
```

### Transfer Learning Strategy

```
Phase 1 (Epochs 1-5): Backbone FROZEN
    - Only train the classifier head
    - Backbone learning rate: 0 (frozen)
    - Head learning rate: 0.005

Phase 2 (Epochs 6+): Backbone UNFROZEN
    - Fine-tune entire network
    - Backbone learning rate: 0.00005 (100x smaller)
    - Head learning rate: 0.005
```

This prevents destroying pretrained features while adapting to tree classification.

---

## 6. Training Loop

### Single Epoch

```python
for images, labels in train_loader:
    # 1. Move to GPU
    images, labels = images.to(device), labels.to(device)
    
    # 2. Zero gradients
    optimizer.zero_grad()
    
    # 3. Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # 4. Backward pass
    loss.backward()
    
    # 5. Update weights
    optimizer.step()
```

### Mixed Precision Training

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Uses FP16 for faster computation while maintaining FP32 precision where needed.

### Learning Rate Scheduling

```python
# OneCycleLR: Warmup → Peak → Decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.005,
    epochs=50,
    steps_per_epoch=len(train_loader)
)

# Step after each batch
scheduler.step()
```

---

## 7. Hyperparameter Optimization with Optuna

### What Optuna Does

```python
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 0.0001, 0.01, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Create model with these hyperparameters
    model = create_model(dropout=dropout)
    
    # Train for N epochs
    best_val_acc = train(model, lr, weight_decay)
    
    return best_val_acc  # Optuna maximizes this

# Run 20 trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### Optimized Hyperparameters (Example Results)

| Parameter | TreeNet | ResNet50 |
|-----------|---------|----------|
| Learning Rate | 0.000427 | 0.009267 (head) |
| Dropout | 0.35 | 0.60 |
| Weight Decay | 0.00099 | 0.000011 |
| Label Smoothing | 0.08 | 0.006 |

---

## 8. Evaluation Metrics

### 8.1 Basic Metrics

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    labels, predictions, average='macro'
)
```

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | Correct / Total | Overall performance |
| **Precision** | TP / (TP + FP) | "Of predicted X, how many correct?" |
| **Recall** | TP / (TP + FN) | "Of actual X, how many found?" |
| **F1-Score** | 2 × (P × R) / (P + R) | Balance of precision and recall |

### 8.2 Confusion Matrix

```
                    PREDICTED
              oak  maple  pine  birch
         oak  45    3      1     0     │ 49 actual oaks
TRUE   maple   2   41      0     4     │ 47 actual maples
        pine   1    0     38     2     │ 41 actual pines
       birch   0    5      1    42     │ 48 actual birches

Diagonal = Correct predictions
Off-diagonal = Errors (shows which species are confused)
```

### 8.3 N-Best (Top-K) Classification

```python
def calculate_topk_accuracy(probs, labels, k):
    """
    Top-K accuracy: Is the true label in the top K predictions?
    """
    topk_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(1 for i, label in enumerate(labels) 
                  if label in topk_preds[i])
    return correct / len(labels)
```

**Example Results:**

| K | TreeNet | ResNet18 | ResNet50 |
|---|---------|----------|----------|
| 1 | 66.53% | ~82% | 88.77% |
| 3 | ~85% | ~93% | ~96% |
| 5 | ~92% | ~97% | ~99% |
| 23 | 100% | 100% | 100% |

**Key Insight:** Maximum model separation occurs at K=1 (standard accuracy).

---

## 9. Image Quality Analysis

### Resolution Analysis

```python
# Analyze actual image resolutions in dataset
for img_path in image_files:
    with Image.open(img_path) as img:
        width, height = img.size
        aspect_ratio = width / height
        file_size = os.path.getsize(img_path)
```

**What to Check:**
- Are images smaller than model input (224px)? They'll be upscaled.
- Are images much larger? Consider using larger input size.
- Consistent aspect ratios? Inconsistent may cause distortion.

### Class Distribution Analysis

```python
# Count samples per class
class_counts = Counter([label for _, label in train_dataset.samples])

# Identify underrepresented classes
mean_samples = np.mean(list(class_counts.values()))
underrepresented = [cls for cls, count in class_counts.items() 
                    if count < mean_samples * 0.5]
```

**Correlation:** Classes with fewer training samples often have lower accuracy.

---

## 10. Error Analysis

### Most Confused Pairs

```python
# From confusion matrix, find largest off-diagonal values
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > threshold:
            print(f"{classes[i]} often confused with {classes[j]}")
```

**Common Reasons for Confusion:**
- Visually similar species (e.g., different maple varieties)
- Similar tree structure (crown shape, bark texture)
- Seasonal appearance (leaf-off confusion)
- Poor image quality

### Confidence Analysis

```python
# Find high-confidence errors
error_confidences = probs[error_indices].max(axis=1)
high_conf_errors = error_confidences > 0.8

# These indicate:
# - Ambiguous images
# - Potential labeling errors
# - Need for more distinguishing features
```

---

## 11. Preprocessing Recommendations

Based on error analysis, the notebook provides recommendations:

### 1. Image Resolution
```
Current: Various sizes, resized to 224x224
Recommendation: If source images are high-res, try 384x384 input
Benefit: More detail for bark texture, leaf shape
```

### 2. Class Balance
```
Issue: Imbalance ratio > 3x between largest and smallest class
Solution: 
  - Collect more images for underrepresented classes
  - Use stronger augmentation for rare classes
  - Apply class weights (already implemented)
```

### 3. Feature Extraction
```
Current Features Used:
  ✓ Overall tree structure (crown shape)
  ✓ Bark patterns (texture, color)
  ⚠️ Leaf details (may need higher resolution)
  
Recommendations:
  - Add multi-scale inputs
  - Use attention mechanisms
  - Consider seasonal-specific models
```

---

## 12. Files Generated

### Model Checkpoints

```
checkpoints/
├── TreeNet_V3_best.pth       # Best TreeNet weights
├── ResNet50_V3_best.pth      # Best ResNet50 weights
├── resnet18_final.pth        # Best ResNet18 weights
└── best_hyperparameters.json # Optuna results
```

### Visualizations

```
results/
├── confusion_matrix_TreeNet.png
├── confusion_matrix_ResNet18.png
├── confusion_matrix_ResNet50.png
├── nbest_analysis_all_models.png
├── image_resolution_analysis.png
├── class_distribution_analysis.png
└── error_analysis.png
```

### Data Files

```
results/
├── final_results.json       # All metrics
├── nbest_analysis.csv       # Top-K accuracy table
└── per_class_metrics.csv    # Per-class P/R/F1
```

---

## 13. Model Loading for Inference

```python
# Load trained model
checkpoint = torch.load('checkpoints/ResNet50_V3_best.pth')
model = create_resnet('resnet50', num_classes=23, dropout=0.5)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict on new image
def predict(image_path):
    image = Image.open(image_path)
    tensor = val_transforms(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor.to(device))
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1)
    
    return classes[pred_class], probs.max().item()

species, confidence = predict("my_tree.jpg")
print(f"Predicted: {species} ({confidence:.1%})")
```

---

## Summary: Complete Pipeline

```
1. SETUP
   Import libraries, configure GPU, set seeds
                    ↓
2. LOAD DATA
   ImageFolder for train/val/test splits
                    ↓
3. TRANSFORMS
   Training: Augmentation + Normalization
   Val/Test: Normalization only
                    ↓
4. CLASS WEIGHTS
   Handle imbalanced dataset
                    ↓
5. DATALOADERS
   Batch data, parallel loading
                    ↓
6. DEFINE MODELS
   TreeNet (scratch) + ResNet18/50 (transfer)
                    ↓
7. OPTUNA SEARCH (optional)
   Find optimal hyperparameters
                    ↓
8. TRAIN
   Forward → Loss → Backward → Update
                    ↓
9. EVALUATE
   Accuracy, Precision, Recall, F1
   Confusion matrices, N-best analysis
                    ↓
10. ANALYZE
    Image quality, class distribution
    Error patterns, preprocessing recommendations
                    ↓
11. SAVE & DEPLOY
    Model checkpoints, results files
```

---

*Document for Tree Species Classification Project*
