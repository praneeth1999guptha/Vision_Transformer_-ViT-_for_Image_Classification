# Vision_Transformer_-ViT-_for_Image_Classification
Use a Vision Transformer to solve the Cats and Dogs Dataset. 
This repository contains the notebook **`notebooks/Vision Transformer (ViT) for Image Classification.ipynb`**, which trains a Vision Transformer for image classification using PyTorch.

## ✨ Overview
- **Dataset:** Custom dataset via ImageFolder
- **Backbone:** ViT-B/16 style backbone (patch=16, 224×224)
- **Input:** 224×224 input, patch size 16
- **Augmentations:** RandomHorizontalFlip
- **Training:** 5 epochs · batch size 32 · optimizer AdamW (lr=3e-05, weight_decay=0.01) · scheduler: CosineAnnealingLR
- **Mode:** Fine‑tuning entire backbone
- **Loss:** CrossEntropyLoss
- **Metrics:** **Validation Top‑1 Accuracy:** ~0.99% (from notebook logs)

> The notebook includes data loading, transforms, model definition, training loop, evaluation, and plots (loss/accuracy, confusion matrix).

---

## 📦 Data

If you are using **CIFAR‑10/100**, the notebook can download it automatically via `torchvision.datasets`.  
For a **custom dataset**, organize images as an ImageFolder:

```
data/
└── train/
    ├── class_0/
    │   ├── img001.jpg
    │   └── ...
    └── class_1/
        ├── img101.jpg
        └── ...
└── val/
    ├── class_0/
    └── class_1/
```



## 🛠️ Setup

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_vit.txt
```

### 2) Run
```bash
jupyter notebook notebooks/Vision\ Transformer\ (ViT)\ for\ Image\ Classification.ipynb
```
Run cells in order. If using GPU, PyTorch will auto‑detect it (`cuda`).

---


---

## ⚙️ Reproducibility / Tips
- Set random seeds for `torch`, `numpy`, and dataloader workers.
- If OOM on GPU, lower `batch_size` or use mixed precision (`torch.cuda.amp.autocast`).
- Try stronger augments (RandAugment/AutoAugment) for extra regularization.
- Switch between **linear probe** (freeze backbone) and **full fine‑tune** (unfreeze) by toggling `requires_grad` on ViT parameters.
- Consider **label smoothing** and **AdamW + cosine schedule** (common for ViTs).

.
