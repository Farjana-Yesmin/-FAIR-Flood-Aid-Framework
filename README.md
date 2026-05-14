# FAIR: Fairness-Aware AI for Post-Flood Aid Allocation in Bangladesh

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CCAI 2026](https://img.shields.io/badge/CCAI-2026%20IEEE-blue.svg)](https://arxiv.org/abs/2512.22210)

Official implementation of:

> Yesmin, F. & Akter, R. (2026).
> *Toward Equitable Recovery: A Fairness-Aware AI Framework for
> Prioritizing Post-Flood Aid in Bangladesh.*
> Accepted (oral), CCAI 2026 (IEEE), Nanjing, China.
> Preprint: [arXiv:2512.22210](https://arxiv.org/abs/2512.22210)

Part of the [FairHealth](https://github.com/Farjana-Yesmin/fairhealth) library —
`pip install fairhealth`

---

## The Problem

Post-disaster aid allocation in Bangladesh systematically underserves rural
Haor districts despite their higher flood vulnerability. Standard AI models
trained on historical allocation data learn and amplify these biases.

**The 2022 Bangladesh floods:** 7.2 million people affected, $405.5M in damages
across 11 districts — yet Sunamganj (94% inundated, 42.7% poverty) was
historically ranked 14th for aid priority.

---

## Results

### Predictive Performance

| Model | MSE | MAE | RMSE | R² |
|---|---|---|---|---|
| **Fair (ours)** | 12.47 | 2.89 | 3.53 | **0.784** |
| Baseline | 10.93 | 2.64 | 3.31 | 0.811 |

Only 2.7 percentage point R² cost for major fairness improvements.

### Fairness Improvements

| Metric | Fair Model | Baseline | Improvement |
|---|---|---|---|
| Statistical Parity Diff (USD M) | 3.82 | 6.54 | **41.6%** |
| Prediction Variance | 4.23 | 6.87 | **38.4%** |
| Regional Fairness Gap | 0.67 | 1.18 | **43.2%** |
| MAE Std (across districts) | 0.54 | 0.91 | **40.7%** |

### Priority Rankings — Key Finding

70.6% of upazilas receive significantly different rankings under the fair model.

| District | Baseline Rank | Fair Rank | Poverty | Inundation |
|---|---|---|---|---|
| **Sunamganj** | 14 | **6** | 42.7% | 94% |

5 high-poverty Haor upazilas move into top 20% priority tier.
Haor upazilas shift +3.8 positions on average.

### District Performance (Most Affected)

| District | Fair MAE | Baseline MAE | Improvement |
|---|---|---|---|
| Sunamganj | 2.73 | 3.89 | **29.8%** |
| Habiganj | 2.85 | 3.67 | **22.3%** |
| Sylhet | 2.91 | 3.54 | **17.8%** |

### Computational Efficiency

- Training: 8.7 min (fair) vs 7.2 min (baseline)
- Inference: <0.01 sec per upazila
- Parameters: 47,203 (fair) vs 45,089 (baseline)

---

## Architecture
Input Features (pre-flood vulnerability + flood exposure)
↓
Encoder Eθ: R^d → R^h
(3 FC layers, batch norm, ReLU, dropout p=0.3)
↓
┌────────────────────────────────────────────────┐
│  Task Predictor Pφ: R^h → R+                  │
│  (economic damage prediction, MSE loss)        │
│                                                │
│  Adversarial Predictor Aψ: R^h → R^K          │
│  (predicts district — tries to identify it)   │
│                                                │
│  Gradient Reversal Layer (GRL, λ=1.0)         │
│  (encoder learns to FOOL the adversary)        │
└────────────────────────────────────────────────┘
Total Loss: L = L_task − λ·L_adv
λ=1.0 → optimal fairness-accuracy balance (ablation: λ ∈ {0, 0.5, 1.0, 2.0})
**Priority score formula:**
Priority_i = 0.6 × norm(predicted_damage) + 0.4 × Vulnerability_i
Vulnerability = 0.30×poverty + 0.25×agriculture + 0.25×housing + 0.20×flood_extent
---

## Dataset

**Bangladesh 2022 Flood PDNA** — 87 upazila-level observations, 11 districts.

| File | Description |
|---|---|
| `bangladesh_floods_2022_upazila_level.csv` | ML-ready features, 87 upazilas |
| `bangladesh_floods_2022_district_level.csv` | District-level summary |
| `pdna_district_summary.csv` | Official PDNA district data |
| `pdna_human_impact.csv` | Population affected, displaced |
| `pdna_sector_summary.csv` | Sector damage breakdown |
| `fairness_metrics_summary.csv` | Fair vs baseline metric comparison |
| `district_performance_comparison.csv` | MAE by district |
| `model_predictions_comparison.csv` | Predicted vs actual damage |

Dataset on HuggingFace:
[fairhealth/bangladesh-flood-pdna-2022](https://huggingface.co/datasets/fairhealth/bangladesh-flood-pdna-2022)

**Data sources:** Ministry of Disaster Management and Relief (PDNA 2022),
Bangladesh Bureau of Statistics, World Bank, NASA SEDAC, EM-DAT.

---

## Quick Start

```bash
pip install torch pandas scikit-learn numpy
```

```python
# Run toward_equitable_recovery_ccai_2026.py
# Or use via FairHealth:
from fairhealth.equity.flood_aid import generate_priority_ranking

rankings = generate_priority_ranking(verbose=True)
# Rank 1: Sunamganj (priority=0.9428, Haor region, poverty=42.7%)
```

---

## Policy Implications

The fair model provides Bangladesh's Ministry of Disaster Management with:
1. Evidence that Sunamganj and Habiganj are chronically underserved
2. Inference time <0.01s — deployable for real-time prioritization
3. Alignment with National Plan for Disaster Management Priority 4: "Build Back Better"

---

## Citation

```bibtex
@inproceedings{yesmin2026ccai,
  author    = {Yesmin, Farjana and Akter, Romana},
  title     = {Toward Equitable Recovery: A Fairness-Aware AI Framework
               for Prioritizing Post-Flood Aid in Bangladesh},
  booktitle = {CCAI 2026, IEEE, Nanjing, China},
  year      = {2026},
  note      = {Oral presentation. Preprint: arXiv:2512.22210}
}
```

---

**Authors:** Farjana Yesmin · Romana Akter (Researcher, Dhaka, Bangladesh)
· MIT License
