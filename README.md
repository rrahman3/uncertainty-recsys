# **Leveraging Uncertainty Quantification for Reducing Data for Recommender Systems**
📌 Presented at the **2023 IEEE International Conference on Big Data (BigData)**  

This project explores **uncertainty-aware data reduction strategies** for recommender systems, balancing **data minimization** and **recommendation utility** while considering **privacy regulations** such as the **California Consumer Privacy Act (CCPA)**.

## 🔍 Abstract  
The CCPA mandates limiting personal data while ensuring technical safeguards against user re-identification. However, operationalizing this in **recommender systems** remains a challenge.  

This study introduces **uncertainty quantification** as a metric for guiding **data reduction** while maintaining recommendation utility. We leverage two primary types of uncertainty in machine learning models:  

- **Aleatoric Uncertainty** – inherent noise and randomness in data  
- **Epistemic Uncertainty** – uncertainty due to a lack of knowledge in the model  

Using these concepts, we propose two **data reduction strategies**:  
1. **Within-user data reduction** – reducing individual user data points while controlling aleatoric uncertainty.  
2. **Between-user data reduction** – reducing certain types of users in training while balancing epistemic uncertainty.  

Our experiments on **MovieLens-1M** and **AmazonBooks** demonstrate that significant data reduction can be achieved with minimal accuracy loss. However, the impact varies across user groups, raising **fairness and transparency concerns** in AI models.

## 🚀 Key Contributions  
✔ **Introduced uncertainty-based data reduction** for recommender systems.  
✔ **Proposed two data reduction strategies** (within-user & between-user).  
✔ **Evaluated impact on uncertainty and accuracy** at aggregate and individual levels.  
✔ **Identified fairness concerns** in data reduction affecting different user types.  

---

## 📌 Methodology  

### 🔢 1️⃣ Uncertainty Quantification  
We estimate uncertainty using **Bayesian Neural Networks (BNN)** with **Monte Carlo (MC) dropout**:  
- **Aleatoric uncertainty**: modeled via probabilistic noise.  
- **Epistemic uncertainty**: measured using entropy of predictive probabilities.  

### 🔄 2️⃣ Data Reduction Strategies  
- **Within-user Data Reduction**: Removing per-user interactions and analyzing the impact on aleatoric uncertainty.  
- **Between-user Data Reduction**: Reducing training data for specific user groups and analyzing epistemic uncertainty.  

---

## ⚙️ Experimental Setup  
- **Datasets**:  
  - 📚 [MovieLens-1M](https://grouplens.org/datasets/movielens/)  
  - 📦 AmazonBooks (subset from [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html))  
- **Models**:  
  - 🤖 **BERT4Rec** (Transformer-based sequential recommender)  
  - 🧠 **NextItNet** (CNN-based sequential recommender)  
- **Metrics**:  
  - 🎯 **Hit Rate @20 (HR@20)** for accuracy  
  - 📊 **Aleatoric and Epistemic Uncertainty**  

---

## 🔬 Results  

📌 **Key Findings:**  
- ✅ **Recent interactions** are the most informative for within-user reduction.  
- ✅ **Removing some user groups (e.g., consistent raters)** has **minimal impact** on accuracy.  
- ⚠ **"Non-mainstream" users** (atypical behaviors) suffer more from data reduction.  
- 🔒 **Reducing user history** improves privacy but introduces potential biases.  

---

## 🔗 Citation
If you find this work useful, please cite:
```
@inproceedings{niu2023leveraging,
  author    = {Xi Niu, Ruhani Rahman, Xiangcheng Wu, Zhe Fu, Depeng Xu, Riyi Qiu},
  title     = {Leveraging Uncertainty Quantification for Reducing Data for Recommender Systems},
  booktitle = {IEEE International Conference on Big Data (BigData)},
  year      = {2023}
}
```
