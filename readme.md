# Project

This repository contains the two projects:

- **Project1**: DICOM loading and visualization
- **Project2**: 3D Image Segmentation

---

## Repository Structure

```
.
├── dataset/
│   ├── 10_AP_Ax2.50mm/                     # 207 CT DICOM slices
│   ├── 10_AP_Ax2.50mm_ManualROI_Liver.dcm  # Liver segmentation
│   └── 10_AP_Ax2.50mm_ManualROI_Tumor.dcm  # Tumor segmentation (multi-frame)
│
├── Project1/
│   ├── notebook.ipynb      # Main Python notebook
│   ├── utils.py            # Helper module
│   └── output/             # Generated figures, GIF and logs
│
└── Project2/
|   ├── notebook.ipynb          # Extended analysis notebook
├── ambiente.yml            # Conda environment specification
└── requirements.txt        # pip install requirements
```

---

## Setup Instructions

You can choose **Conda** (recommended) or **pip** to install the required dependencies.

### 1. Using Conda

1. **Create** the environment from `environment.yml`:

   ```bash
   conda env create -f Project2/environment.yml
   ```

2. **Activate** the environment:

   ```bash
   conda activate radct_env
   ```

3. (Optional) Verify installation:

   ```bash
   python -c "import pydicom, numpy, matplotlib, scipy, skimage; print('OK')"
   ```

### 2. Using pip

1. **Create** and **activate** a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```

2. **Install** dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. (Optional) Verify installation:

   ```bash
   python -c "import pydicom, numpy, matplotlib, scipy, skimage; print('OK')"
   ```

---

## Running the Notebooks

### Project1: Core Visualization & Animation

1. **Navigate** into the folder:

   ```bash
   cd Project1
   ```

2. **Start** Jupyter:

   ```bash
   jupyter notebook
   ```

3. **Open** `notebook.ipynb` and execute cells in order.  
   - It will load the CT series and segmentations (from `../dataset/`),  
   - Generate axial montages, overlays, MIPs, and a rotating MIP GIF.  
   - Outputs (PNG, GIF, logs) are saved under `output/`.
