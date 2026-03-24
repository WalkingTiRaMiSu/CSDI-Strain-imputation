# CSDI Strain Imputation

This project applies the CSDI (Conditional Score-based Diffusion Model)  
to reconstruct missing structural strain data from sensor measurements.

---

## Project Overview

- Dataset: Structural Health Monitoring (SHM) data
- Selected channels: strain-related channels (17–21)
- Data preprocessing:
  - Reduced data length to 1–10000
  - Artificial missing segment: 4000–7000

The model reconstructs missing values based on temporal patterns learned from the data.

---

## How to Run

### 1. Open Anaconda Prompt

### 2. Move to project folder
```bash
cd C:\Users\YOUR_USERNAME\Desktop\CSDI-Strain-imputation
3. Install required packages
pip install numpy pandas torch pyyaml matplotlib tqdm scikit-learn
4. Run the model
python exe_custom.py --device cuda --nsample 5
If GPU is not available
python exe_custom.py --device cpu --nsample 5
