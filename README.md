# GUIDE-CT: CT-Centered Risk Stratification for Guideline-Concerning IPMN

This repository contains the CT feature extraction code and the locked GUIDE-CT scoring model described in:

> **CT-Centered Risk Stratification for Guideline-Concerning Pancreatic Intraductal Papillary Mucinous Neoplasm**  
> *Radiology* (under review)

---

## Overview

Pancreatic intraductal papillary mucinous neoplasm (IPMN) referred after guideline concern creates a management gray zone: advanced neoplasia (high-grade dysplasia or invasive carcinoma, HGD/IC) should not be missed, but low-/intermediate-grade lesions may be over-escalated. GUIDE-CT is a ridge logistic-regression model that refines risk stratification **after** guideline concern using four CT-derived variables:

| Variable | Description |
|----------|-------------|
| **LD** | Lesion diameter (mm) |
| **MPD** | Main pancreatic duct diameter (mm) |
| **SVF** | Solid-volume fraction (%) |
| **ALARM** | Non-size alarm score (rescaled sum of binary guideline features) |

Key results across 1 222 patients (35 % HGD/IC):

- AUC: 0.927 (training) → 0.835–0.932 (test cohorts)
- Reader AUC: 0.696 → **0.846** with GUIDE-CT assistance (+0.150, *P* < .001)
- Sensitivity: 79 % → **100 %** in prospective tumor-board cohort; no HGD/IC de-escalation

---

## Repository structure

```
├── feature_extraction.py   # CT quantitative feature extraction from NIfTI segmentations
├── guide_ct_model.py       # Locked GUIDE-CT scoring model (Table S4)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-org>/guide-ct.git
cd guide-ct
pip install -r requirements.txt
```

Python 3.9+ is recommended.

---

## Usage

### 1. Score cases from a CSV

If you already have LD, MPD, SVF, and ALARM measurements:

```bash
python guide_ct_model.py \
    --input  cases.csv \
    --output scored.csv
```

Input CSV must contain columns `LD`, `MPD`, `SVF`, `ALARM` (column names are configurable via flags). Missing values are replaced by training-cohort medians before scoring.

**Python API:**

```python
from guide_ct_model import score_case, compute_alarm_score

alarm = compute_alarm_score(
    obstructive_jaundice=0,
    thickened_enhancing_wall=1,
    abrupt_duct_caliber_change=0,
    acute_pancreatitis=0,
    ca199_elevation=1,
    new_onset_diabetes=0,
)

result = score_case(LD=35.0, MPD=6.5, SVF=42.0, ALARM=alarm)
print(f"Probability of HGD/IC: {result.probability:.3f}")
print(f"High-risk (balanced):  {result.recommend_surgery_balanced}")
```

### 2. Extract features from NIfTI segmentations

```bash
python feature_extraction.py \
    --base_dir   /path/to/data \
    --output_dir /path/to/output \
    --workers    8
```

Expected directory layout under `--base_dir`:

```
base_dir/
├── C_image/<case>.nii.gz     unenhanced CT
├── C_mask/<case>.nii.gz
├── A_image/<case>.nii.gz     arterial phase CT
├── A_mask/<case>.nii.gz
├── P_image/<case>.nii.gz     portal venous phase CT  ← primary
├── P_mask/<case>.nii.gz
├── V_image/<case>.nii.gz     delayed phase CT
└── V_mask/<case>.nii.gz
```

**Segmentation label convention (multi-label mask):**

| Label | Structure |
|-------|-----------|
| 1 | Pancreatic parenchyma (gland) |
| 2 | Main pancreatic duct (MPD) |
| 3 | Solid component (mural nodule / solid burden) |
| 4 | Cystic component (target lesion / fluid) |

Each case produces a CSV in `--output_dir`; a combined `all_features.csv` is written at the end.

---

## Locked model equation

From Supplementary Table S4:

```
eta = -0.6223
      + 0.5204 × zLD
      + 0.5791 × zMPD
      + 1.4395 × zSVF
      + 1.4760 × zALARM

p = 1 / (1 + exp(-eta))
```

Standardization (training-cohort means ± SDs):

| Variable | Mean | SD | Median impute |
|----------|------|----|---------------|
| LD (mm) | 28.04 | 6.56 | 27.81 |
| MPD (mm) | 4.99 | 1.65 | 4.93 |
| SVF (%) | 35.01 | 8.95 | 34.83 |
| ALARM | 1.97 | 0.96 | 1.97 |

**Operating thresholds (locked in training):**

| Threshold | Value | Use |
|-----------|-------|-----|
| Balanced (primary) | 0.3925 | Main diagnostic performance table |
| Safety-prioritized | 0.1482 | High-sensitivity fixed-threshold analyses |

---

## Important limitations

- All model parameters, standardization constants, and thresholds were locked in the training cohort and **must not be refit** for external application.
- SVF was derived from a semi-automated, quality-controlled segmentation workflow; performance may differ with alternative segmentation pipelines.
- Calibration slopes varied across external cohorts; fixed-threshold results should be interpreted as observed operating points rather than evidence of universal threshold transportability.
- Prospective nonoperative safety has not been established. **Do not use for nonoperative management** outside a prospective safety validation study.
- The model was designed for use in patients **already referred after guideline concern**; it is not a surveillance screening tool.

---


---

## License

This code is released for research and educational use. See `LICENSE` for details.
