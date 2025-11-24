# Video-Object-Detection-LAB-2-Assignment
This GitHub was made for the LAB 2 Assignment of the signature Deep Learning for Video Signal Processing. The main objective of this repository is to install correctly the official implementation MEGA approach on Pytorch for Video Object Detection in the laboratory computers.

The original GitHub repository used in this assignment is available at: [https://github.com/Scalsol/mega.pytorch](https://github.com/Scalsol/mega.pytorch)

---

## 1. Overview

The purpose of this repository is to provide:

* A **fully functional installation guide** adapted to current library compatibility requirements.
* A set of **fixes and patches** applied to the original MEGA codebase to support CPU execution and remove hard dependencies.
* Instructions to run both **BASE** and **MEGA** models using image-folder inference mode.
* A report summarizing the technical hurdles encountered and solutions applied.

---

## 2. Installation Guide (The Corrected Path)

To avoid the multiple errors encountered during the original installation process (incompatibility between PyTorch versions, syntax errors in Python 3.7, and CUDA compilation failures), please follow these specific steps directly.

### 2.1. Environment Setup

Create a clean Conda environment with Python 3.7. We use specific versions of PyTorch and TorchVision (1.6.0 / 0.7.0) instead of the older suggestions to ensure the C++ extensions compile correctly.

```bash
# 1. Create and activate environment
conda create --name MEGA -y python=3.7
source activate MEGA

# 2. Install basic dependencies
conda install ipython pip
pip install ninja yacs cython matplotlib tqdm opencv-python scipy

# 3. Install PyTorch 1.6.0 and TorchVision 0.7.0
# (Note: We use this version to resolve compilation errors found in 1.2/1.3)
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.0 -c pytorch
```

### 2.2. Install COCO API and Cityscapes Scripts

```bash
export INSTALL_DIR=$PWD

# Install COCO API
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# Install Cityscapes Scripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install
```

### 2.3. Install Apex

We use the standard build flags to avoid the specific CUDA extension failure found in the original instructions.

```bash
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py build_ext install
```

### 2.4. Clone and Patch MEGA Repository

Before installing MEGA, a syntax error in setup.py regarding type hinting must be fixed to support Python 3.7.

```python
# In mega.pytorch/setup.py
from typing import Union
# ...
parallel: Union[int, None] = None
```

Once fixed, install the library:

```bash
cd $INSTALL_DIR
git clone https://github.com/Scalsol/mega.pytorch.git
cd mega.pytorch
python setup.py build develop
pip install 'pillow<7.0.0'
```

---

## 3. Required Code Modifications

Even after installation, several runtime errors occur due to missing CUDA devices (if running on CPU) and strict OpenCV type requirements. The following modifications are required in your local files.

### 3.1. Make Apex/AMP Optional

Several MEGA modules assume Apex is always installed. To prevent crashes, we modified the following files to check for Apex presence dynamically:

* `mega_core/layers/nms.py`
* `mega_core/layers/roi_align.py`
* `mega_core/layers/roi_pool.py`

**Action:** Wrap imports in try-except blocks and only apply `@amp.float_function` if Apex is successfully loaded.

### 3.2. Fix OpenCV Coordinates

In `demo/predictor.py`, OpenCV throws errors if float coordinates are passed to drawing functions.

**Action:** In `overlay_class_names`, cast coordinates to integers:

```python
cv2.putText(img, class_str, (int(x), int(y)), ...)
```

### 3.3. CPU Configuration

If running without a GPU, modify the YAML configuration files to set:

```yaml
DEVICE: "cpu"
```

---

## 4. Model & Data Setup

Download the required resources and place them in the root of `mega.pytorch`.

* **Image Folder:** Download `image_folder.zip` (from Moodle), unzip it, and place the `image_folder` directory inside `mega.pytorch`.
* **Checkpoints:** Download the pre-trained models:

  * Single Frame Baseline (ResNet‑101): `R_101.pth`
  * MEGA (ResNet‑101): `MEGA_R_101.pth`
* **Configs:** Ensure you use the CPU‑adapted YAML files:

  * `configs/vid_R_101_C4_1x.yaml`
  * `configs/MEGA/vid_R_101_C4_MEGA_1x.yaml`

---

## 5. Running the Demo

Once the environment is set up and files are patched, run the inference on the image folder.

### 5.1. Run BASE Model

```bash
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth \
    --suffix ".JPEG" \
    --visualize-path image_folder \
    --output-folder outputs_base
```

### 5.2. Run MEGA Model

```bash
python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml MEGA_R_101.pth \
    --suffix ".JPEG" \
    --visualize-path image_folder \
    --output-folder outputs_mega
```

---

## 6. Report: Summary of Issues & Solutions

We started by following the official INSTALL.md. However, incompatible libraries and syntax errors forced us to deviate from the original instructions.

### 6.1. PyTorch Version Conflict

The assignment requested PyTorch 1.2, and the repo suggested 1.3. Both versions caused compilation errors with MEGA core extensions.

**Solution:** Upgraded to **PyTorch 1.6.0** and **TorchVision 0.7.0**.

### 6.2. Apex Installation

The instruction `python setup.py install --cuda_ext --cpp_ext` failed due to CUDA mismatches.

**Solution:** Installed with `python setup.py build_ext install`.

### 6.3. Python Syntax Error

The repository used the `| None` syntax, only valid in Python 3.10.

**Solution:** Replaced by `Union[int, None]`.

### 6.4. Apex Hard Dependency

Modules crashed when Apex wasn't available.

**Solution:** Modified MEGA modules to make Apex optional.

### 6.5. OpenCV Integer Coordinates

`cv2.putText` failed because floats were passed.

**Solution:** Cast all coordinates to `int()`.

---

## 7. Authors

This work was completed as part of the Deep Learning for Video Signal Processing course.

* **David Ruiz Simón**
* **Noelia Sierra Sánchez**

---

## 8. License

This repository follows the same licensing constraints as the original MEGA implementation. Please refer to the original license.

---

## 9. Reference

Chen et al., *Memory Enhanced Global-Local Aggregation for Video Object Detection*, CVPR 2020.
