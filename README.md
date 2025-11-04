# Artificial Intelligence Model Design & Application – Coursework Log

This repository collects my practical work for the 2025 NCKU “Artificial Intelligence Model Design and Application” course. Each lab folder contains executable notebooks, experiment reports, and artifacts that document the end-to-end training and model compression pipelines I implemented on CIFAR-10.

## Repository layout

```
2025EAI_Project/
├── README.md                     # Course journal (this file)
├── Lab1 Material/                # Lab 1 notebooks, reports, and checkpoints
├── Lab1 Material_backup/         # Original templates provided by the course
├── EAI_Lab2/                     # Lab 2 sparsity training & pruning workflow
└── EAI_Lab2_backup/              # Unmodified backup notebooks for Lab 2
```

> Model checkpoints (`*.pth`) and the CIFAR-10 dataset are excluded from version control to keep the repository lightweight. See **How to reproduce** for regeneration steps.

## Lab highlights

### Lab 1 – ResNet-18 training pipeline
| Component | Key references | Highlights |
| --- | --- | --- |
| Baseline training | `Lab1-Task1.ipynb`, `Lab1-Task1-Report.md` | Clean PyTorch implementation of ResNet-18 with standard training loop, achieving 83.98% test accuracy. |
| Advanced training | `Lab1-Task2.ipynb`, `Lab1-Task2-Report.md` | Introduced extensive data augmentation, Kaiming initialization, AMP, cosine LR schedule with warmup, label smoothing, and early stopping. |
| Result | `Lab1-Task2-Report.md` | Boosted test accuracy to **94.83%** (↑ 7.8% over baseline) with ~11.2M parameters and ~557M FLOPs. |

### Lab 2 – Network slimming & structured pruning (ResNet-56)
| Stage | Notebooks & reports | Deliverables |
| --- | --- | --- |
| Architecture adaptation | `models/resnet.py`, `REPORT_resnet.md` | Refactored the bottleneck ResNet-56 to accept per-layer channel configurations (`cfg`) while enforcing residual shape compatibility. |
| Sparsity training | `sparsity_train.ipynb`, `REPORT_sparsity_train.md` | L1 regularisation on BatchNorm γ (λ=1e-4) to drive channel sparsity. Logged training/test curves and γ distributions across λ ∈ {0, 1e-5, 1e-4}. Best test accuracy: **91.15%**. |
| Channel pruning | `modelprune.ipynb`, `REPORT_modelprune.md` | Constructed pruned models by masking γ, cloning weights with per-layer index mapping, and evaluating 50% & 90% pruning ratios. Raw 90% pruned model achieved **11.35%** accuracy with **3.97M** params. |
| Fine-tuning | `train_prune_model.ipynb`, `REPORT_train_prune_model.md` | Fine-tuned pruned networks (with AMP disabled for numerical stability). Restored the 90% pruned model to **90.56%** test accuracy while staying under **4M** parameters. |

### Consolidated metrics

| Model variant | Params | FLOPs | Best test accuracy |
| --- | ---: | ---: | ---: |
| ResNet-56 (sparsity training) | ~23.5M | ~3.7G | 91.15% |
| ResNet-56, 50% pruned + FT | ~12.8M | ~1.9G | 92.53% |
| ResNet-56, 90% pruned + FT | **~3.97M** | **~0.6G** | **90.56%** |

## How to reproduce

```bash
# clone project
git clone https://github.com/livejiaquan/2025EAI_Project.git
cd 2025EAI_Project

# create environment (example with conda)
conda create -n eai2025 python=3.9 -y
conda activate eai2025

# install core dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib tqdm thop torchsummary

# (optional) enable notebook workflow
pip install jupyterlab ipywidgets

# launch notebooks
jupyter lab
```

Run the notebooks inside `Lab1 Material/` or `EAI_Lab2/` in the order documented in each report. Regenerate datasets via `torchvision.datasets.CIFAR10` (download flag). Model checkpoints are saved under each lab’s `checkpoints/` subfolder once training completes.

## Key learnings & troubleshooting notes

- Enforcing consistent bottleneck output channels is critical; otherwise residual additions fail during pruning because tensor shapes diverge.
- Applying L1 regularisation directly to BatchNorm scales (`γ`) effectively highlights redundant channels while preserving accuracy when λ is tuned conservatively (1e-4 here).
- When cloning weights into pruned architectures, both input and output channel indices must be masked to keep convolution kernels aligned with the slimmed topology.
- Mixed-precision training can trigger `cublasLtMatmul` issues after aggressive pruning; forcing FP32 forward passes during fine-tuning resolved the instability without modifying the backbone code.

## Acknowledgements

Course: *Artificial Intelligence Model Design and Application*, National Cheng Kung University, 2025.

Maintainer: **livejiaquan** — repository intended for academic use and personal study reference.
