from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
heart_disease = fetch_ucirepo(id=45)

# Data
X = heart_disease.data.features
y = heart_disease.data.targets

# Check if we actually have categorical variables
print("Data types:\n", X.dtypes)

# Handle missing values with SimpleImputer (more robust)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Process target variable - convert to binary (0: no disease, 1: disease)
y_binary = (y > 0).astype(int)
print("\nTarget value distribution:")
print(y_binary.value_counts())

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
cleaned_data = pd.DataFrame(X_scaled, columns=X_imputed.columns)

# Add target to cleaned dataset for EDA
cleaned_data_with_target = cleaned_data.copy()
cleaned_data_with_target['target'] = y_binary.values

# --- EDA: Histograms ---
cleaned_data.hist(figsize=(15, 12), bins=20, edgecolor="black")
plt.suptitle("Histograms of Standardized Features", fontsize=16)
plt.show()

# --- EDA: Correlation Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(cleaned_data.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# --- EDA: Boxplots (all features) ---
plt.figure(figsize=(12, 6))
cleaned_data.plot(kind="box", rot=90)
plt.title("Boxplots of All Features")
plt.show()


# PCA Analysis
pca = PCA()
X_pca = pca.fit_transform(cleaned_data)

# Calculate explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Create subplots for better visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# --- 1. Variance Plot (Individual + Cumulative) ---
# Individual explained variance
ax1.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, alpha=0.7, label='Individual')
# Cumulative explained variance
ax1.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'ro-', label='Cumulative')
ax1.set_xlabel("Number of Principal Components")
ax1.set_ylabel("Explained Variance Ratio")
ax1.set_title("PCA - Explained Variance")
ax1.legend()
ax1.grid(True)

# Add markers for 80%, 90%, and 95% variance
for threshold in [0.8, 0.9, 0.95]:
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
    ax1.axvline(x=n_components, color='r', linestyle='--', alpha=0.7)
    ax1.text(n_components + 0.2, threshold - 0.05, f'{n_components} PCs', fontsize=10)

# --- 2. Scatter Plot of First 2 Principal Components ---
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_binary.values.ravel(), 
                     cmap="viridis", alpha=0.7, s=50)
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_title("PCA - Scatter Plot (First 2 Components)")
plt.colorbar(scatter, label="Heart Disease (0=No, 1=Yes)")
ax2.grid(True, alpha=0.3)

# Add variance explained information
variance_pc1 = explained_variance_ratio[0] * 100
variance_pc2 = explained_variance_ratio[1] * 100
ax2.text(0.05, 0.95, f'PC1: {variance_pc1:.1f}% variance\nPC2: {variance_pc2:.1f}% variance', 
         transform=ax2.transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Print optimal number of components for ≥95% variance
optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Optimal number of components to retain ≥95% variance: {optimal_components}")
print(f"Variance retained with {optimal_components} components: {cumulative_variance[optimal_components-1]:.4f}")

# --- 3. Deliverable: PCA-transformed dataset ---
pca_opt = PCA(n_components=optimal_components)
X_pca_final = pca_opt.fit_transform(cleaned_data)

# Convert to DataFrame
pca_columns = [f"PC{i+1}" for i in range(optimal_components)]
pca_df = pd.DataFrame(X_pca_final, columns=pca_columns)
pca_df["target"] = y_binary.values

# Show preview
print("\nPCA-transformed dataset shape:", pca_df.shape)
print("\nPCA-transformed dataset preview:")
print(pca_df.head())


# For chi-square, we need non-negative values, so let's use MinMaxScaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_imputed)
X_minmax_df = pd.DataFrame(X_minmax, columns=X_imputed.columns)

# 1. Feature Importance with Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(cleaned_data, y_binary.values.ravel())
importances = pd.Series(rf.feature_importances_, index=cleaned_data.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
importances.plot(kind="barh")
plt.title("Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

# 2. Recursive Feature Elimination (RFE)
logreg = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(logreg, n_features_to_select=10)
rfe.fit(cleaned_data, y_binary.values.ravel())

selected_rfe = cleaned_data.columns[rfe.support_]
print("\nSelected features by RFE:", selected_rfe.tolist())

rfe_ranking = pd.DataFrame({
    'feature': cleaned_data.columns,
    'ranking': rfe.ranking_
}).sort_values('ranking')

plt.figure(figsize=(10, 8))
sns.barplot(
    data=rfe_ranking.head(15), 
    x='ranking', y='feature', palette="magma_r"
)
plt.title("Top Features Selected by RFE", fontsize=14)
plt.xlabel("RFE Ranking (1 = Best)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# 3. Chi-Square Test (using MinMax scaled data)
chi2_selector = SelectKBest(chi2, k=10)
chi2_selector.fit(X_minmax_df, y_binary.values.ravel())
chi2_scores = pd.Series(chi2_selector.scores_, index=X_minmax_df.columns)
chi2_features = chi2_scores.sort_values(ascending=False).head(10).index.tolist()
print("\nTop 10 features by Chi-Square:", chi2_features)

chi2_ranking = pd.DataFrame({
    'feature': X_minmax_df.columns,
    'score': chi2_selector.scores_
}).sort_values('score', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=chi2_ranking.head(15), 
    x='score', y='feature', palette="cividis"
)
plt.title("Top 15 Features Ranked by Chi-Square Test", fontsize=14)
plt.xlabel("Chi-Square Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Compare results from different methods
feature_votes = {}
for feature in cleaned_data.columns:
    feature_votes[feature] = 0
    if feature in importances.head(10).index:
        feature_votes[feature] += 1
    if feature in selected_rfe:
        feature_votes[feature] += 1
    if feature in chi2_features:
        feature_votes[feature] += 1
    

# Select features that appear in at least 2 methods
final_selected_features = [feature for feature, votes in feature_votes.items() if votes >= 2]
print(f"\nFinal selected features (appear in ≥2 methods): {final_selected_features}")

# Create reduced dataset with selected features
reduced_dataset = cleaned_data[final_selected_features].copy()
reduced_dataset['target'] = y_binary.values

print(f"\nReduced dataset shape: {reduced_dataset.shape}")
print("\nReduced dataset preview:")
print(reduced_dataset.head())



# 2.4 Supervised Learning - Classification Models
# =====================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Use the reduced dataset (feature selection result) if available; otherwise cleaned_data
# reduced_dataset has 'target' appended already in your previous code
if 'reduced_dataset' in globals():
    X_model = reduced_dataset.drop(columns=['target']) # type: ignore
    y_model = reduced_dataset['target'].values.ravel() # type: ignore
else:
    X_model = cleaned_data
    y_model = y_binary.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_model, y_model, test_size=0.2, random_state=42, stratify=y_model
)

models = {
    "LogReg": LogisticRegression(max_iter=2000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

results = []

plt.figure(figsize=(8,6))
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    results.append([name, acc, prec, rec, f1, auc])

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Test Set)")
plt.legend()
plt.tight_layout()
plt.show()

results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1","AUC"])
print("2.4 Metrics (Test Set):")
print(results_df.sort_values("AUC", ascending=False))

# Confusion matrix & classification report for the best by AUC
best_name = results_df.sort_values("AUC", ascending=False).iloc[0]["Model"]
best_clf = models[best_name]
y_pred_best = best_clf.predict(X_test)
print(f"\nBest by AUC: {best_name}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_best, digits=3))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cbar=False)
plt.title(f"{best_name} - Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Use standardized features without target
X_unsup = cleaned_data.copy()  # (already scaled)

# --- K-Means: Elbow Method & Silhouette ---
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_unsup)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_unsup, labels))

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(list(K_range), inertias, marker='o')
ax[0].set_title("Elbow Method (KMeans)")
ax[0].set_xlabel("K"); ax[0].set_ylabel("Inertia")

ax[1].plot(list(K_range), sil_scores, marker='o')
ax[1].set_title("Silhouette Score (KMeans)")
ax[1].set_xlabel("K"); ax[1].set_ylabel("Silhouette")

plt.tight_layout(); plt.show()

# Choose K by best silhouette (or elbow); we’ll use silhouette here
best_k = list(K_range)[int(np.argmax(sil_scores))]
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
kmeans_labels = kmeans.fit_predict(X_unsup)

# Compare with true labels (note: unsupervised; ARI measures agreement ignoring label perms)
ari_kmeans = adjusted_rand_score(y_binary.values.ravel(), kmeans_labels)
print(f"KMeans: chosen K={best_k}, Silhouette={max(sil_scores):.3f}, ARI vs target={ari_kmeans:.3f}")

# --- Hierarchical Clustering (Agglomerative via linkage + dendrogram) ---
Z = linkage(X_unsup, method='ward')
plt.figure(figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=5, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram (truncated)")
plt.tight_layout(); plt.show()

# Cut dendrogram into 2 clusters (common for heart disease/no disease) and compare
hier_labels = fcluster(Z, t=2, criterion='maxclust') - 1  # make labels 0/1
ari_hier = adjusted_rand_score(y_binary.values.ravel(), hier_labels)
print(f"Hierarchical (2 clusters) ARI vs target={ari_hier:.3f}")



# 2.6 Hyperparameter Tuning
# =====================================
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = "roc_auc"

# --- Random Forest: RandomizedSearchCV ---
rf = RandomForestClassifier(random_state=42)
rf_distributions = {
    "n_estimators": randint(200, 600),
    "max_depth": randint(2, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None]
}
rf_search = RandomizedSearchCV(
    rf, rf_distributions, n_iter=40, cv=cv, scoring=scoring,
    random_state=42, n_jobs=-1, verbose=1
)
rf_search.fit(X_model, y_model)
print("\n[RF] Best AUC (CV):", rf_search.best_score_)
print("[RF] Best Params:", rf_search.best_params_)

# --- SVM: GridSearchCV (smaller grid) ---
svm = SVC(probability=True, random_state=42)
svm_param_grid = {
    "C": [0.1, 1, 5, 10],
    "gamma": ["scale", 0.1, 0.01, 0.001],
    "kernel": ["rbf"]
}
svm_search = GridSearchCV(
    svm, svm_param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1
)
svm_search.fit(X_model, y_model)
print("\n[SVM] Best AUC (CV):", svm_search.best_score_)
print("[SVM] Best Params:", svm_search.best_params_)

# Pick the overall best model by CV AUC
best_search = rf_search if rf_search.best_score_ >= svm_search.best_score_ else svm_search
best_model = best_search.best_estimator_
best_name = "RandomForest" if best_search is rf_search else "SVM"
print(f"\nChosen Best Model: {best_name} (CV AUC={best_search.best_score_:.4f})")

# Final test-set evaluation of the chosen tuned model
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)

print("\n[Tuned Best Model] Test Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("AUC:", roc_auc_score(y_test, y_proba))

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"{best_name} (AUC={roc_auc_score(y_test, y_proba):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - Tuned Best Model")
plt.legend(); plt.tight_layout(); plt.show()


# 2.7 Model Export & Deployment (save pipeline)
# =====================================
import os, joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Build a FULL pipeline that starts from raw features -> impute -> scale -> best model
# This makes the exported model reproducible end-to-end.
final_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", best_model)
])

# Fit on ALL data (recommended for final export after CV selection)
final_pipeline.fit(heart_disease.data.features, y_binary.values.ravel())

# Ensure folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Save model
model_path = "models/final_model.pkl"
joblib.dump(final_pipeline, model_path)

# Save a brief evaluation report
report_lines = []
report_lines.append("=== Supervised Models (Base) - Test Set ===\n")
report_lines.append(results_df.sort_values("AUC", ascending=False).to_string(index=False))
report_lines.append("\n\n=== Hyperparameter Tuning ===")
report_lines.append(f"\nRF best AUC (CV): {rf_search.best_score_:.4f}")
report_lines.append(f"\nRF best params: {rf_search.best_params_}")
report_lines.append(f"\nSVM best AUC (CV): {svm_search.best_score_:.4f}")
report_lines.append(f"\nSVM best params: {svm_search.best_params_}")
report_lines.append(f"\n\nChosen Best Model: {best_name}")
report_lines.append("\n(Tuned Best Model) Test metrics:")
report_lines.append(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
report_lines.append(f"\nPrecision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
report_lines.append(f"\nRecall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
report_lines.append(f"\nF1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
report_lines.append(f"\nAUC: {roc_auc_score(y_test, y_proba):.4f}")

with open("results/evaluation_metrics.txt", "w") as f:
    f.write("\n".join(report_lines))


    # 2.10 Final Deliverables Packaging
# =====================================

import os

# Ensure output directories exist
os.makedirs("deliverables", exist_ok=True)

# --- 1. Export Datasets ---
cleaned_data.to_csv("deliverables/cleaned_data.csv", index=False)
pca_df.to_csv("deliverables/pca_transformed_data.csv", index=False)
reduced_dataset.to_csv("deliverables/reduced_dataset.csv", index=False)

print(" Datasets exported: cleaned_data.csv, pca_transformed_data.csv, reduced_dataset.csv")

# --- 2. Export Best Model (already saved in models/final_model.pkl) ---
print(" Final trained model already saved at: models/final_model.pkl")

# --- 3. Export Metrics Report (already saved in results/evaluation_metrics.txt) ---
print(" Evaluation report saved at: results/evaluation_metrics.txt")

# --- 4. Create a Summary Report ---
summary_text = f"""
==========================
 Heart Disease Project
 Final Deliverables (2.10)
==========================

1. Cleaned Dataset: deliverables/cleaned_data.csv
   - Preprocessed with missing value imputation & scaling
   - {cleaned_data.shape[0]} samples, {cleaned_data.shape[1]} features

2. PCA Dataset: deliverables/pca_transformed_data.csv
   - Transformed with PCA (≥95% variance retained)
   - {pca_df.shape[1]-1} components + target

3. Reduced Dataset (Feature Selection): deliverables/reduced_dataset.csv
   - Selected features via RF, RFE, Chi-Square voting
   - {reduced_dataset.shape[1]-1} selected features + target

4. Trained Model:
   - File: models/final_model.pkl
   - Includes full pipeline (imputation → scaling → best tuned model)
   - Best Model: {best_name} with CV AUC={best_search.best_score_:.4f}

5. Evaluation Report:
   - File: results/evaluation_metrics.txt
   - Includes baseline models, hyperparameter tuning, final test performance
"""

with open("deliverables/final_summary.txt", "w") as f:
    f.write(summary_text)

print("Final summary report saved: deliverables/final_summary.txt")
