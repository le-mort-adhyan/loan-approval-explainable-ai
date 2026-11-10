# loan-approval-explainable-ai
Mini Project Assignment 10.11.26

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=600&size=30&pause=1200&color=38BDF8&center=true&vCenter=true&width=920&height=70&lines=EXPLAINABLE+LOAN+APPROVAL;Decision+Trees+%7C+Feature+Selection+%7C+Explainable+ML" alt="Explainable Loan Approval" />

**A loan approval classifier that explains every decision**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classical%20ML-F7931E?logo=scikitlearn&logoColor=white)
![No Deep Learning](https://img.shields.io/badge/deep%20learning-none-22C55E)
![Status](https://img.shields.io/badge/status-complete-38BDF8)

[Project page](https://YOUR-USERNAME.github.io/loan-approval-explainable-ai/) · [📓 Notebook](notebooks/loan_approval.ipynb) · [📈 Results](results/) · [🚀 Quick start](#-quick-start)

</div>

---

##  Dashboard at a glance

|  Dataset |  Algorithms |  Best rejection recall |  Explainability |
|:---:|:---:|:---:|:---:|
| 614 applicants | 5 models + 3 feature-selection methods | `X.XX` | Tree rules + feature importance |

> Replace the `X.XX` values with your final numbers after running the notebook.

---

## Problem

Banks must decide whether to approve a loan. Manual review is slow, and customers and regulators expect a reason for each decision. This project predicts loan approval, compares feature subsets and algorithms, and explains each outcome with interpretable rules. It is just a mini project not a proper system workflow for banking sector .

## Pipeline

```mermaid
flowchart LR
    A[Raw loan data] --> B [Preprocess]
    B --> C [Feature engineering]
    C --> D [Feature selection]
    D --> E [Models + 5-fold CV]
    E --> F [Tree rules & importance]
    F --> G [predict.py]
```

## Feature selection comparison

| Method | Type | Syllabus module | Selected features | Recall | F1 |
|---|---|:---:|:---:|:---:|:---:|
| Full set (baseline) | none | — | all | `X.XX` | `X.XX` |
| Mutual information | Filter | 2 | `N` | `X.XX` | `X.XX` |
| RFE + Decision Tree | Wrapper | 2 | `N` | `X.XX` | `X.XX` |
| L1 (LASSO) logistic | Embedded | 3 | `N` | `X.XX` | `X.XX` |

## Model results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| KNN | `X.XX` | `X.XX` | `X.XX` | `X.XX` | `X.XX` |
| Naïve Bayes | `X.XX` | `X.XX` | `X.XX` | `X.XX` | `X.XX` |
| SVM | `X.XX` | `X.XX` | `X.XX` | `X.XX` | `X.XX` |
| Decision Tree | `X.XX` | `X.XX` | `X.XX` | `X.XX` | `X.XX` |
| Bagging / AdaBoost | `X.XX` | `X.XX` | `X.XX` | `X.XX` | `X.XX` |

> Accuracy alone is misleading here because about 69% of applications are approved. Recall on rejections is the key metric.

##  Why was this applicant rejected?

Each decision follows a path through the tree. Example format (replace with your output):

```text
Credit_History <= 0.5        → Rejected
  └─ LoanToIncome > 0.42     → Rejected
Reasons: no credit history; loan is large relative to income
```

## ⚡ Quick start

```bash
git clone https://github.com/YOUR-USERNAME/loan-approval-explainable-ai.git
cd loan-approval-explainable-ai
pip install -r requirements.txt
python src/train.py          # trains models, saves results/
python src/predict.py        # interactive single-applicant prediction
```

## Project structure

```text
loan-approval-explainable-ai/
├── index.html          # project page (GitHub Pages)
├── README.md
├── data/               # train.csv or download instructions
├── notebooks/loan_approval.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── results/            # plots and comparison table
└── requirements.txt
```

## Dataset

[Loan Prediction Problem Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset). Download `train.csv` and place it in `data/`. Check the link before submission.

## Limitations

- Small dataset (about 614 rows), so results vary between runs. Cross-validation is used to reduce this.
- Historical approvals may contain bias. A model trained on them can repeat it.
- This is a student project, not a production credit system.

## Author

**Shankhadip_Karmakar** · Roll no. `35530824067` · GKCIET, Malda , West Bengal · B.Tech CSE (AIML)

---

<div align="center">

 "If this project helped you, consider starring the repo. Please ."

</div>
