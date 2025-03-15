Leveraging Uncertainty Quantification for Reducing Data for Recommender Systems
Overview
This repository contains the implementation and experiments related to the paper "Leveraging Uncertainty Quantification for Reducing Data for Recommender Systems", presented at the 2023 IEEE International Conference on Big Data (BigData). The paper explores uncertainty-aware data reduction strategies for recommender systems, balancing data minimization and recommendation utility while considering the California Consumer Privacy Act (CCPA).

Abstract
The recent CCPA regulation mandates limiting personal data to what is necessary for business operations while ensuring that re-identification of users is technically safeguarded. However, applying this requirement to recommender systems remains a challenge.

In this study, we introduce uncertainty quantification as a metric for guiding data reduction while maintaining recommendation utility. We leverage two primary types of uncertainty in machine learning models:

Aleatoric Uncertainty – inherent noise and randomness in data.
Epistemic Uncertainty – uncertainty due to a lack of knowledge in the model.
Using these concepts, we propose two data reduction strategies:

Within-user data reduction – reducing individual user data points while controlling aleatoric uncertainty.
Between-user data reduction – reducing certain types of users in training while balancing epistemic uncertainty.
Our analysis on datasets such as MovieLens-1M and AmazonBooks demonstrates that significant data reduction can be achieved with minimal accuracy loss, but the impact varies across user groups, raising fairness and transparency concerns in AI models.

Key Contributions
✅ Introduced uncertainty-based data reduction to recommender systems.
✅ Proposed two data reduction strategies (within-user & between-user).
✅ Evaluated the impact on both uncertainty and accuracy at aggregate and individual levels.
✅ Identified fairness concerns in data reduction affecting different user types.

Methodology
1️⃣ Uncertainty Quantification
We estimate uncertainty using Bayesian Neural Networks (BNN) with Monte Carlo (MC) dropout:

Aleatoric uncertainty is computed using a probabilistic noise term.
Epistemic uncertainty is measured via entropy of predictive probabilities.
2️⃣ Data Reduction Strategies
Within-user Data Reduction: Removing per-user interactions and analyzing the impact on aleatoric uncertainty.
Between-user Data Reduction: Reducing training data for specific user groups and analyzing epistemic uncertainty.
Experimental Setup
Datasets:
MovieLens-1M
AmazonBooks (subset from Amazon Reviews Dataset)
Models:
BERT4Rec (Transformer-based sequential recommender)
NextItNet (CNN-based sequential recommender)
Metrics:
Hit Rate @20 (HR@20) for accuracy
Aleatoric and Epistemic Uncertainty
Results
📌 Key Findings:

Recent interactions are the most informative for within-user reduction.
Removing some user groups (e.g., consistent raters) has minimal impact on accuracy.
"Non-mainstream" users (atypical behaviors) suffer more from data reduction.
Reducing user history improves privacy but introduces potential biases.
Installation
Prerequisites
Python 3.8+
PyTorch
NumPy, Pandas
scikit-learn
Matplotlib, Seaborn
Setup
Clone this repository and install dependencies:

bash
Copy
Edit
git clone https://github.com/your-repo/uncertainty-aware-recommender.git
cd uncertainty-aware-recommender
pip install -r requirements.txt
Usage
Train the Model
bash
Copy
Edit
python train.py --dataset movielens --model bert4rec --epochs 10
Evaluate Data Reduction Strategies
bash
Copy
Edit
python evaluate.py --strategy within-user --dataset amazonbooks
Citation
If you find this work useful, please cite:

csharp
Copy
Edit
@inproceedings{niu2023leveraging,
  author    = {Xi Niu, Ruhani Rahman, Xiangcheng Wu, Zhe Fu, Depeng Xu, Riyi Qiu},
  title     = {Leveraging Uncertainty Quantification for Reducing Data for Recommender Systems},
  booktitle = {IEEE International Conference on Big Data (BigData)},
  year      = {2023}
}
Acknowledgments
This research was supported by the National Science Foundation (NSF) (Award #1910696). We thank our collaborators and the research community for their valuable insights.

License
📜 This project is released under the MIT License.

