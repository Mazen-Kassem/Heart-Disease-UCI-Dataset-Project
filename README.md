# Heart-Disease-UCI-Dataset-Project
This project applies **Machine Learning** techniques to the **UCI Heart Disease dataset** to predict the presence of heart disease.  
It covers **Data Preprocessing, Dimensionality Reduction (PCA), Feature Selection, Supervised and Unsupervised Learning, Hyperparameter Tuning, and Deployment** of the final trained model.

## Project Overview
- Dataset: UCI Heart Disease Dataset (`id=45` from [ucimlrepo](https://pypi.org/project/ucimlrepo/))  
- Goal: Predict whether a patient has heart disease (`0 = no disease, 1 = disease`).  
- Techniques used:  
  - Data preprocessing & EDA  
  - PCA (dimensionality reduction)  
  - Feature importance & selection (Random Forest, RFE, Chi-Square)  
  - Supervised learning (Logistic Regression, Decision Tree, Random Forest, SVM)  
  - Unsupervised learning (KMeans, Hierarchical Clustering)  
  - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)  
  - Model deployment with full pipeline  

  # Repository Structure
 Heart-Disease-UCI-Dataset-Project/
├── deliverables/
│ ├── cleaned_data.csv
│ ├── pca_transformed_data.csv
│ ├── reduced_dataset.csv
│ ├── final_summary.txt
├── results/
│ ├── evaluation_metrics.txt
│ ├── roc_curves.png
│ ├── clustering.png
├── models/
│ └── final_model.pkl
├── README.md
├── requirements.txt
└── .gitignore


## Deliverables
- **2.1 – Data Preprocessing & EDA**  
  - Missing value imputation  
  - Standardization (StandardScaler)  
  - Histograms, Correlation Heatmap, Boxplots  

- **2.2 – PCA (Dimensionality Reduction)**  
  - Variance explained & cumulative plot  
  - Scatter plot of first 2 PCs  
  - PCA-transformed dataset (≥95% variance retained)  

- **2.3 – Feature Selection**  
  - Random Forest importance  
  - Recursive Feature Elimination (RFE)  
  - Chi-Square ranking  
  - Reduced dataset with consensus features  

- **2.4 – Supervised Learning**  
  - Logistic Regression, Decision Tree, Random Forest, SVM  
  - ROC curves & metrics (Accuracy, Precision, Recall, F1, AUC)  
  - Confusion matrix & classification report  

- **2.5 – Unsupervised Learning**  
  - KMeans (Elbow method, Silhouette scores)  
  - Hierarchical clustering dendrogram  
  - Cluster evaluation with ARI  

- **2.6 – Hyperparameter Tuning**  
  - RandomizedSearchCV for Random Forest  
  - GridSearchCV for SVM  
  - Best model chosen based on CV AUC  

- **2.7 – Model Deployment**  
  - Exported full pipeline (`models/final_model.pkl`)  
  - Includes imputation → scaling → trained best model  

- **2.10 – Final Packaging**  
  - Cleaned dataset  
  - PCA-transformed dataset  
  - Reduced dataset (feature selection)  
  - Final model pickle file  
  - Metrics report & summary  

---

## Best Model
- **Random Forest (tuned with RandomizedSearchCV)**  
- Test Set Results:  
  - **AUC = 0.92**  
  - Accuracy, Precision, Recall, F1 all balanced  

---

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/Mazen-Kassem/Heart-Disease-UCI-Dataset-Project
cd heart-disease-ml-project
