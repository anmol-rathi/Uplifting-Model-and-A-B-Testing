# A/B Testing & Uplift Modeling — End-to-End Experimentation Portfolio Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

**An end-to-end experimentation pipeline going from power analysis → A/B testing → heterogeneous treatment effect estimation, validated against synthetic ground-truth data and applied to a real-world causal ML benchmark.**

</div>

---

## TL;DR

Most A/B testing portfolio projects stop at "p < 0.05, ship it." This one doesn't.

The **two-dataset strategy** is what makes this project different:

1. **Synthetic dataset** (100K users, known ITE) — proves the uplift pipeline actually recovers the correct answer against ground truth. Almost no portfolio project does this.
2. **Criteo dataset** (13M users, real randomized trial) — applies the identical pipeline to research-grade real data for the business narrative.

**Key result:** Targeting the top 20% of users by predicted uplift captures ~78% of total incremental conversions at 20% of rollout cost — **3.9× more efficient than blanket rollout.**

---

## What's In This Project

| Step | What it covers | Why it matters |
|---|---|---|
| **Power analysis** | MDE tradeoff curve, retrospective power | Shows you understand sample size math, not just p-values |
| **Validity checks** | SRM test, Love plot (covariate balance), novelty effects | The step every tutorial skips; interviewers ask about this |
| **Core A/B test** | z-test, bootstrap CIs, multiple comparisons (BH-FDR) | Rigor beyond "run a t-test" |
| **Statistical vs. practical significance** | Explicit MDE threshold, not just α | Shows business judgment |
| **CUPED** | ~18% variance reduction using pre-experiment data | Used at Microsoft, Uber, Booking.com; almost no portfolio project implements this |
| **Uplift modeling** | S/T/X-Learner meta-learners, Causal Forest | Goes from ATE to individual targeting |
| **Ground-truth validation** | Qini vs. known ITE, Spearman r | Uniquely possible with synthetic data |
| **Criteo pipeline** | Same code on 13M-row real dataset | External validity |
| **Decision memo** | Ship/no-ship recommendation | Shows you can make and defend a call |

---

## Project Structure

```
experimentation-uplift-project/
├── README.md
├── requirements.txt
├── 01_data_simulation.ipynb      ← Generate 100K synthetic users with known ITE
├── 02_ab_test_analysis.ipynb     ← SRM, power, z-test, CUPED, multiple comparisons
├── 03_uplift_modeling.ipynb      ← S/T/X-Learner, Causal Forest, Qini validation
├── 04_real_data_criteo.ipynb     ← Same pipeline on Criteo (13M real users)
├── 05_decision_memo.md           ← Ship/no-ship recommendation
├── data/
│   ├── synthetic_users_full.parquet      ← With ground-truth ITE
│   └── synthetic_users_blinded.parquet   ← As-if real experiment (no ITE)
├── models/
│   └── xlearner_final.joblib    ← Trained X-Learner for dashboard
├── figures/                     ← All generated charts
└── dashboard/
    └── app.py                   ← Interactive Streamlit dashboard
```

---

## Key Findings

### 1. Average treatment effect masks critical heterogeneity

| Segment | True ITE | Users |
|---|---|---|
| New mobile (< 90d tenure) | **+10pp** | 22% |
| New desktop (< 90d tenure) | +4pp | 12% |
| Mid-tenure mobile | +2pp | 18% |
| Mid-tenure desktop | +1pp | 11% |
| Tablet | **-0.5pp** | 10% |
| Long-tenure desktop | **-0.3pp** | 27% |

**Overall ATE: ~+2.4pp** — a blanket rollout decision based on this number misses that a quarter of users see no benefit or slight harm.

### 2. Uplift models recover the correct ranking

All three meta-learners and the Causal Forest achieve Spearman rank correlation > 0.7 with the true ITE ranking on the held-out test set — validated against ground truth that no real-data project can access.

### 3. Targeted rollout is 3.9× more efficient

| Strategy | Users exposed | Lift captured | Cost efficiency |
|---|---|---|---|
| Blanket rollout | 100% | 100% | 1.0× |
| Top 20% targeted | 20% | ~78% | **3.9×** |
| Top 30% targeted | 30% | ~88% | 2.9× |

### 4. CUPED reduced variance by 18%

Using `engagement_score` as the pre-experiment covariate, CUPED narrowed confidence intervals by 18% — equivalent to having a 22% larger sample at no additional data collection cost.

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

> **Note on Windows:** `econml` requires Microsoft C++ Build Tools. If install fails, the notebooks fall back to `causalml`'s `UpliftRandomForestClassifier`. The code handles this automatically.

### Run notebooks in order
```bash
jupyter notebook
```

Open and run: `01_data_simulation.ipynb` → `02_ab_test_analysis.ipynb` → `03_uplift_modeling.ipynb` → `04_real_data_criteo.ipynb`

### Criteo data (Notebook 04)
```bash
pip install kaggle
kaggle datasets download -d arashnic/criteo-uplift-prediction -p data/ --unzip
```
Or: download manually from [Kaggle](https://www.kaggle.com/datasets/arashnic/criteo-uplift-prediction) and place `criteo-uplift-v2.1.csv` in `data/`. If not found, Notebook 04 auto-generates a structural replica for code demonstration.

### Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Tooling (100% Free)

| Tool | Purpose |
|---|---|
| `statsmodels` | Power analysis, z/t-tests, SRM check |
| `scikit-learn` | Base ML models for meta-learners |
| `scikit-uplift` | Qini/AUUC metrics, uplift plots |
| `causalml` | S/T/X-Learner meta-learners, Uplift Random Forest |
| `econml` | CausalForestDML (optional, requires C++ build tools) |
| `streamlit` | Interactive dashboard |
| Kaggle Notebooks | Free GPU compute for full Criteo run |

---

## Resume Summary

> *Built an end-to-end experimentation pipeline (power analysis → A/B test → heterogeneous treatment effect estimation) validated against synthetic ground-truth data and applied to a 13M-row real-world causal ML benchmark (Criteo); used uplift modeling (S/T/X-learners, causal forests) to identify the user segment driving 78% of incremental conversions, informing a targeted rather than blanket rollout recommendation that is 3.9× more cost-efficient.*

---

## Further Reading

- [Netflix: Improving the Sensitivity of Online Experiments](https://netflixtechblog.com/improving-experimentation-efficiency-at-netflix-using-linear-mixed-models-and-the-netflix-prize-dataset-d1f1cfa50019) (CUPED context)
- [Netflix: Heterogeneous Treatment Effects](https://netflixtechblog.com/key-challenges-with-quasi-experiments-at-netflix-89b4f234b852)
- [Uber: Using Causal Inference to Improve User Experience](https://www.uber.com/blog/causal-inference-at-uber/)
- [Criteo AI Lab Uplift Dataset Paper](https://arxiv.org/abs/2106.00349)
- Kunzel et al. (2019): [Metalearners for Estimating Heterogeneous Treatment Effects](https://arxiv.org/abs/1706.03461) — the X-Learner paper

---

## License

MIT — use freely, attribution appreciated.
