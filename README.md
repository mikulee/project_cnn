# Tree Species Classification with CNNs

A deep learning project for classifying 23 tree species from urban street images using transfer learning (ResNet) and custom CNNs (TreeNet).

## 🎯 Project Goals

1. **Primary Comparison:** Compare two different sized ResNet architectures (ResNet18 vs ResNet50)
2. **Supporting Analysis:** Compare transfer learning (ResNet) against training from scratch (TreeNet)
3. **Comprehensive Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrices, N-best analysis

## 📊 Results Summary

| Model | Parameters | Training Method | Test Accuracy |
|-------|------------|-----------------|---------------|
| **ResNet50** | 25.6M | Transfer Learning | **88.77%** |
| **ResNet18** | 11.7M | Transfer Learning | ~82% |
| **TreeNet** | ~2.5M | From Scratch | 66.53% |

**Key Finding:** Transfer learning provides +22% accuracy improvement over training from scratch.

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `Tree_Species_Final_Evaluation.ipynb` | **Main deliverable** - Complete evaluation with all metrics |
| `Tree_Species_Classification_V3.ipynb` | Training notebook with LR Finder and Optuna optimization (removed for clarity) |
| `Tree_Species_Classification_V3_1.ipynb` | Confusion matrix analysis and improved TreeNet (removed for clarity)|
| `Tree_Species_Classification_V2.ipynb` | Two-phase transfer learning approach (removed for clarity)|
| `Tree_Species_Classification_CNN.ipynb` | Original notebook (V1)(removed for clarity) |
| `requirements.txt` | Python dependencies |
| `Notebook_Walkthrough.md` | Detailed explanation of the pipeline |
| `Neural_Networks_Phase1.md` | Neural network learning materials |
| `Tree_Species_Recognition_PRD.md` | Project requirements document |

### Notebook Progression

```
V1 (Original)     → Basic implementation, identified transfer learning issues
        ↓
V2 (Two-Phase)    → Proper differential learning rates, frozen backbone
        ↓
V3 (Optimized)    → LR Finder, Optuna hyperparameter search
        ↓
V3.1 (Analysis)   → Confusion matrix analysis, improved TreeNet
        ↓
Final Evaluation  → Complete metrics, N-best analysis, all comparisons
```

---

## 🖥️ Hardware Requirements

**Minimum:**
- GPU with 8GB VRAM
- 16GB RAM
- 4 CPU cores

**Recommended (this project was optimized for):**
- NVIDIA RTX 4090 (24GB VRAM)
- 192GB RAM
- 32 CPU cores

---

## 🔧 Installation

### Option 1: WSL2 Ubuntu (Recommended for Windows Users)

```bash
# In WSL Ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv -y

# Create project directory
mkdir -p ~/tree_classification && cd ~/tree_classification

# Create virtual environment
python3 -m venv tree_env
source tree_env/bin/activate

# Install PyTorch (check your CUDA version with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Verify
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Native Windows

```powershell
# Create virtual environment
python -m venv tree_env
.\tree_env\Scripts\activate

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Native Linux

```bash
python3 -m venv tree_env
source tree_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

### Download
1. Go to: https://www.kaggle.com/datasets/erickendric/tree-dataset-of-urban-street-classification-tree
2. Download and extract `classification_tree.zip`

### Expected Structure
```
data/classification_tree/
├── train/           (~3,800 images)
│   ├── acer_palmatum/
│   ├── ginkgo_biloba/
│   └── ... (23 species folders)
├── val/             (~480 images)
└── test/            (~480 images)
```

### Update Path in Notebook
```python
DATA_ROOT = Path("data/classification_tree")  # Update to your path
```

---

## ⚙️ Configuration

The notebooks auto-detect your platform:

| Platform | BATCH_SIZE | NUM_WORKERS | PERSISTENT_WORKERS |
|----------|------------|-------------|-------------------|
| WSL2/Linux | 128 | 16 | True |
| Windows | 64 | 0 | False |

---

## 🚀 Quick Start

### For Final Evaluation (Recommended)

1. Open `Tree_Species_Final_Evaluation.ipynb`
2. Update `DATA_ROOT` to your dataset path
3. Run all cells

**What it does:**
- Loads trained models (or trains if not found)
- Computes all metrics (Accuracy, Precision, Recall, F1)
- Generates confusion matrices for all models
- Performs N-best (Top-K) classification analysis
- Analyzes image quality and class distribution
- Provides preprocessing recommendations

### For Training from Scratch

1. Open `Tree_Species_Classification_V3.ipynb`
2. Run LR Finder → Quick Test → Optuna Search → Full Training

---

## 📈 Model Architectures

### ResNet50 / ResNet18 (Transfer Learning)
- Pretrained on ImageNet (14 million images)
- Custom classifier head added for 23 tree species
- Differential learning rates (backbone: 0.00005, head: 0.005)
- Input size: 224x224 pixels (ImageNet standard)

### TreeNet (Custom CNN - From Scratch)
- 5-stage convolutional architecture
- No pretrained weights - learns only from tree dataset
- Demonstrates the value of transfer learning

---

## 📊 Key Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted class X, how many were correct |
| **Recall** | Of actual class X, how many were found |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Top-K Accuracy** | True label in top K predictions |

### N-Best Analysis

The project analyzes at which value of K the models show maximum separation:
- **K=1:** Standard accuracy (maximum model difference)
- **K=3-5:** Practical for user interfaces showing top predictions
- **K=23:** All models converge to 100%

---

## 📁 Output Files

After running the Final Evaluation notebook:

```
checkpoints/
├── TreeNet_V3_best.pth          # TreeNet model weights
├── ResNet50_V3_best.pth         # ResNet50 model weights  
├── resnet18_final.pth           # ResNet18 model weights

results/
├── final_results.json           # All metrics in JSON
├── nbest_analysis.csv           # Top-K accuracy table
├── per_class_metrics.csv        # Per-class precision/recall/F1
├── confusion_matrix_*.png       # Confusion matrices
├── nbest_analysis_all_models.png
├── image_resolution_analysis.png
└── class_distribution_analysis.png
```

---

## 🔍 Troubleshooting

### "CUDA not available"
```bash
nvidia-smi  # Verify driver
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Windows multiprocessing errors
```python
NUM_WORKERS = 0
PERSISTENT_WORKERS = False
```

### Out of Memory
```python
BATCH_SIZE = 32  # Reduce from 128
```

---

## 📚 Additional Resources

- `Notebook_Walkthrough.md` - Detailed explanation of every section
- `Tree_Species_Recognition_PRD.md` - Project requirements

---

## 📝 License

This project is for educational purposes.

Dataset source: [Kaggle - Urban Street Tree Classification](https://www.kaggle.com/datasets/erickendric/tree-dataset-of-urban-street-classification-tree)

