# RetinaFlow — Complete Agent Knowledge & Handoff Master Document (AGENTS.md)

> **CRITICAL DIRECTIVE FOR ANY AI AGENT:** 
> This single file contains the complete, authoritative truth of the **RetinaFlow** project.
> If you are an AI agent or developer assigned to this repository, reading this document provides **100% of the context, technical architecture, empirical journey, GitHub sync state, and strict operational constraints** required to continue development without any missing links.

---

## 📍 1. Executive Summary & Current Operational State

* **GitHub Repository:** [`https://github.com/markegyptian55-cloud/RetinaFlow.git`](https://github.com/markegyptian55-cloud/RetinaFlow.git)
* **Local Workspace Directory:** `D:\projects\EyeDisease` (Windows local workspace)
* **Active Git Branch:** `main` tracking `origin/main`
* **Latest Remote Commit:** `ec2333d7fce25761194ebe9d0ebc9a07e430199a`
  - Message: *"Initial commit: RetinaFlow retinal disease classifier with dual ONNX models"*
* **GitHub Authentication & Credentials:**
  - Authenticated via GitHub CLI (`gh`) under account: **`markegyptian55-cloud`**
  - Git Operations Protocol: `https`
  - Active Token Scopes: `repo`, `workflow`, `read:org`, `gist`
* **Git LFS Status:**
  - Initialized and tracking `"*.onnx"` via `.gitattributes`.
  - Both large ONNX models (211 MB total) are 100% synced and stored in GitHub Git LFS media storage:
    - `models/resnet50_m5a.onnx` (SHA-256 pointer `658ebacd08...`, 98.17 MB)
    - `models/efficientnet_b5_m6.onnx` (SHA-256 pointer `b5d71c8402...`, 113.26 MB)
* **Where We Stopped:**
  - The repository has been fully audited from Google Colab research artifacts.
  - Both ONNX models are verified and functional on CPU inference.
  - The Gradio web app (`app.py`) with MCP server integration is complete.
  - Everything is committed and pushed cleanly to GitHub. Working tree is clean.

---

## 🎯 2. Clinical Mission & Diagnostic Scope

RetinaFlow is an 8-class diagnostic deep-learning computer vision system designed to detect and categorize retinal pathologies from fundus photographs.

### The 8 Target Diagnostic Classes (Strict Alphabetical Order)
1. **AMD** (Age-related Macular Degeneration)
2. **Cataract**
3. **Diabetic Retinopathy**
4. **Glaucoma**
5. **Hypertension** (Hypertensive Retinopathy)
6. **Myopia** (Pathological Myopia)
7. **Normal**
8. **Others** (drusen, epiretinal membrane, vascular occlusions, other lesions)

### Clinical Source Registries (11,839 Real Images)
Aggregated across 4 major clinical repositories into a single unified label schema:
- **ODIR-5K:** 7,000 images (59.13%)
- **APTOS-2019:** 3,484 images (29.43%)
- **ACRIMA:** 705 images (5.95%)
- **ORIGA:** 650 images (5.49%)

### The Extreme Imbalance Problem
The distribution of conditions in the real-world dataset is violently skewed:
- **Normal** (4,698 images) and **Diabetic Retinopathy** (4,113 images) dominate ~74.4% of the entire dataset.
- **Hypertension is ultra-rare:** Only **88 real images total** in the entire 11,839 image pool (**53.4× imbalance vs. Normal**).
- **The Untouched Held-Out Test Set:** Consists of **1,776 images** (100% real clinical images).
  - Only **13 Hypertension cases** exist in this test set.
  - *Statistical Reality:* Each single misclassified Hypertension image shifts the test recall by **~7.7 percentage points**. All Hypertension numbers must be interpreted directionally rather than with false precision.

---

## 🔬 3. The Research & Model Evolution (`EyeDisease.ipynb`)

The notebook contains 122 cells detailing 6 distinct milestones evaluated on the exact same 1,776-image real test set:

### Consolidated Test Set Benchmark

| Milestone | Architecture / Strategy | Accuracy | Macro-F1 | Weighted-F1 | Hypertension Recall | Key Diagnosis & Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **M1** | Scratch Custom CNN | 23.37% | 0.228 | 0.184 | 92.3% | Performance floor baseline; high recall was random prediction noise. |
| **M2** | Pretrained ResNet-50 (Standard) | 69.14% | 0.534 | 0.669 | 7.7% | **Silent Bug:** `class_weights` tensor out of scope at runtime (`weighted=False`), running unweighted cross-entropy. |
| **M3** | Sampler + Focal Loss | 25.73% | 0.298 | 0.257 | 69.2% | **Over-Correction:** Stacking `WeightedRandomSampler` AND active Focal Loss collapsed majority classes (Normal recall fell to 6.8%). |
| **M4** | OT-CFM Generative Augmentation | — | — | — | — | Velocity-field continuous normalizing flow trained for 41 epochs (loss 0.0235) to synthesize 1,600 images (+600 HT, +350 AMD, +350 Myopia, +300 Cataract). |
| **M5a** | ResNet-50 (Post-Flow-Matching) | **71.68%** | **0.600** | **0.710** | 15.4% | 🏆 **Champion #1 (General-Purpose):** Highest overall accuracy and balanced macro-F1 across all 8 classes. |
| **M5b** | Swin Transformer (Swin-T) | 72.24% | 0.573 | 0.711 | 7.7% | Higher raw accuracy, but worse Macro-F1 and poorer minority recall than ResNet-50. Architecture novelty did not outperform on this data scale. |
| **M6** | EfficientNet-B5 (Class-Weighted) | 62.61% | 0.570 | 0.653 | **38.5%** | 🎯 **Champion #2 (High-Sensitivity):** 2.5× better Hypertension recall (catches rare high-risk cases), with lower overall accuracy. |

---

### Per-Class Test Breakdown of Both Champion Models

#### Champion #1: M5a — ResNet-50 (General-Purpose)
- **Overall Accuracy:** 71.68% | **Macro-F1:** 0.600 | **Weighted-F1:** 0.710
- **Per-Class Metrics:**
  - `AMD`: Precision 0.613 | Recall 0.463 | F1 0.528 ($n=41$)
  - `Cataract`: Precision 0.587 | Recall 0.725 | F1 0.649 ($n=51$)
  - `Diabetic Retinopathy`: Precision 0.789 | Recall 0.726 | F1 0.756 ($n=617$)
  - `Glaucoma`: Precision 0.786 | Recall 0.579 | F1 0.667 ($n=140$)
  - `Hypertension`: Precision 0.333 | Recall **0.154** | F1 0.211 ($n=13$)
  - `Myopia`: Precision 0.804 | Recall 0.841 | F1 0.822 ($n=44$)
  - `Normal`: Precision 0.708 | Recall 0.840 | F1 0.768 ($n=705$)
  - `Others`: Precision 0.463 | Recall 0.345 | F1 0.396 ($n=165$)

#### Champion #2: M6 — EfficientNet-B5 (High-Sensitivity)
- **Overall Accuracy:** 62.61% | **Macro-F1:** 0.570 | **Weighted-F1:** 0.653
- **Per-Class Metrics:**
  - `AMD`: Precision 0.360 | Recall 0.590 | F1 0.440 ($n=41$)
  - `Cataract`: Precision 0.580 | Recall 0.820 | F1 0.680 ($n=51$)
  - `Diabetic Retinopathy`: Precision 0.890 | Recall 0.580 | F1 0.700 ($n=617$)
  - `Glaucoma`: Precision 0.510 | Recall 0.720 | F1 0.600 ($n=140$)
  - `Hypertension`: Precision 0.280 | Recall **0.385** | F1 0.320 ($n=13$)
  - `Myopia`: Precision 0.660 | Recall 0.890 | F1 0.760 ($n=44$)
  - `Normal`: Precision 0.790 | Recall 0.630 | F1 0.700 ($n=705$)
  - `Others`: Precision 0.250 | Recall 0.590 | F1 0.350 ($n=165$)

---

### The Section 19.0 Audit & Generative Recovery Finding
1. **Mode Collapse in Flow-Matching:** An empirical embedding audit using frozen ResNet-50 feature vectors revealed ~30% diversity collapse across synthetic samples (Collapse Ratio ≈ 0.67–0.71).
2. **Generative Recovery Test:** 15 epochs of targeted fine-tuning on the flow matching generator for weak classes did not reliably fix collapse (Myopia even regressed from 0.674 to 0.603).
3. **Decisive Strategic Decision:** Generative re-synthesis was cleanly terminated rather than wasting compute. Instead, the final system ships **both champion models side-by-side**, honestly exposing the clinical trade-off to the practitioner.

---

## 🏗️ 4. Codebase Architecture & Files

```
RetinaFlow/
├── AGENTS.md                    # Master context handoff file (THIS FILE)
├── README.md                    # Hugging Face Space card & comprehensive benchmark
├── requirements.txt             # Python dependencies
├── app.py                       # Gradio 4.44+ application with ONNX CPU runtime & MCP server
├── EyeDisease.ipynb             # Complete 122-cell end-to-end research notebook
├── .gitattributes               # Git LFS tracking rules (*.onnx)
├── .gitignore                   # Ignore patterns (pycache, env, checkpoints)
├── models/
│   ├── resnet50_m5a.onnx        # M5a (ResNet-50, 224x224 input, 98.17 MB)
│   └── efficientnet_b5_m6.onnx  # M6 (EfficientNet-B5, 300x300 input, 113.26 MB)
└── assets/
    └── samples/                 # Sample test fundus images (AMD, Cataract, DR, Glaucoma, HT, Myopia, Normal, Others)
```

### Key Technical Aspects in `app.py`:
- **ONNX Runtime CPU Inference:** Zero GPU requirements (`CPUExecutionProvider`).
- **Dynamic Preprocessing Resolution:**
  - ResNet-50 requires **$224 \times 224$**.
  - EfficientNet-B5 requires **$300 \times 300$**.
  - Normalization: ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- **Inference Caching:** Computes SHA-256 hash of image bytes; repeated uploads return instant cached logits.
- **Model Context Protocol (MCP):** Enabled via `mcp_server=True` on `demo.launch()`, exposing the `classify` endpoint as a callable tool for MCP clients:
  - Tool endpoint: `classify`
  - Inputs: `image` (PIL Image), `model_choice` (string: `"General-Purpose (ResNet-50)"` or `"High-Sensitivity (EfficientNet-B5)"`)
  - Output: JSON dict mapping all 8 class names to softmax probability floats.
- **UI Design System:** Custom "Quantum Black" dark-mode theme with an explicit model trade-off warning banner.

---

## 🛠️ 5. How to Run Locally

```bash
# 1. Ensure you are in the project directory
cd D:\projects\EyeDisease

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application
python app.py
```
App will serve on `http://127.0.0.1:7860`.

---

## ⚠️ 6. Strict Rules & Operational Guidelines for Any Future AI Agent

1. **User Confirmation Protocol:**
   - **NEVER** perform automated push, commit, or destructive file changes without presenting the plan and receiving explicit user confirmation (e.g., user says *"go do this"*).
2. **Git LFS File Size Protection:**
   - GitHub has a hard **100 MB** limit. `models/efficientnet_b5_m6.onnx` is **113.26 MB**.
   - Always verify Git LFS is active (`git lfs track "*.onnx"`) before staging or pushing any model weights.
3. **Medical Disclaimer Integrity:**
   - Any UI or documentation modification must preserve the research/educational disclaimer. The tool is **not** an FDA-approved clinical diagnostic device.
4. **Resolution Dispatch Preservation:**
   - If modifying `preprocess()` or inference logic in `app.py`, never unify the input resolution to a single size; ResNet-50 and EfficientNet-B5 must receive 224 and 300 respectively.

---

## 🔮 7. Backlog & Potential Next Steps

If the user requests new features or improvements:
- **Hugging Face Space Deployment:** Deploy `app.py` directly to Hugging Face Spaces (Gradio SDK 4.44.0) connected to this repository.
- **Grad-CAM Integration into UI:** Port the Grad-CAM visualization from Section 11.4 / 17.1 of the notebook into `app.py` so users see visual heatmaps alongside class probabilities.
- **FastAPI / REST Endpoint Decoupling:** Wrap ONNX Runtime sessions in a standalone FastAPI service for non-Gradio consumers.
- **Dockerization:** Add a multi-stage `Dockerfile` for standardized container deployment.
