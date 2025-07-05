import arff
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. Đường dẫn ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "phishing.arff")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "..", "result")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== 2. Đọc dữ liệu ==========
with open(DATA_PATH, "r") as f:
    data = arff.load(f)

columns = [col[0] for col in data["attributes"]]
df = pd.DataFrame(data["data"], columns=columns).astype(int)

# ======= Phân bố trước SMOTE =======
before_counts = df["Result"].value_counts().sort_index()

# ========== 3. Tiền xử lý ==========
X = df.drop("Result", axis=1)
y = df["Result"]

# Rút trích đặc trưng
selector = SelectKBest(score_func=mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_columns)

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=selected_columns)

# Cân bằng dữ liệu với SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# ======= Phân bố sau SMOTE =======
after_counts = pd.Series(y).value_counts().sort_index()

# Vẽ biểu đồ so sánh phân bố nhãn trước/sau
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.barplot(x=before_counts.index, y=before_counts.values, ax=ax[0], palette="pastel")
ax[0].set_title("Before SMOTE")
ax[0].set_xlabel("Class Label")
ax[0].set_ylabel("Count")
ax[0].set_xticklabels(["-1", "1"])

sns.barplot(x=after_counts.index, y=after_counts.values, ax=ax[1], palette="coolwarm")
ax[1].set_title("After SMOTE")
ax[1].set_xlabel("Class Label")
ax[1].set_ylabel("Count")
ax[1].set_xticklabels(["-1", "1"])

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "label_distribution_comparison.png"))
plt.close()

# ========== 4. Tách tập train/test ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== 5. Khởi tạo mô hình ==========
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        bootstrap=True,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(),
    "Perceptron": Perceptron(max_iter=1000, tol=1e-3),
    "NaiveBayes": GaussianNB(),
}

results = []

# ========== 6. Huấn luyện & đánh giá ==========
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Lưu mô hình
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    # Ghi kết quả
    results.append(f"----- {name} -----\nAccuracy: {acc:.4f}\n{report}\n")

# ========== 7. KMeans (Clustering) ==========
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X)
sil_score = silhouette_score(X, clusters)
results.append(f"----- KMeans (Clustering) -----\nSilhouette Score: {sil_score:.4f}\n")

# ========== 8. Lưu kết quả ==========
with open(os.path.join(RESULT_DIR, "evaluation.txt"), "w") as f:
    f.writelines(results)

print("✅ Huấn luyện hoàn tất. Kết quả lưu tại: result/evaluation.txt")

# ========== 9. Biểu đồ Accuracy & Confusion Matrix ==========
model_names = list(models.keys())
accuracies = []
conf_matrices = []

for name in model_names:
    model = joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    accuracies.append(acc)
    conf_matrices.append(cm)

# Accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracies, y=model_names, palette="viridis")
plt.xlabel("Accuracy")
plt.title("Model Accuracies")
plt.xlim(0.5, 1.0)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "accuracy_plot.png"))
plt.close()

# Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, name in enumerate(model_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrices[idx], display_labels=["-1", "1"])
    disp.plot(ax=axes[idx], colorbar=False)
    axes[idx].set_title(f"{name} (Acc: {accuracies[idx]:.2f})")

plt.tight_layout()
plt.suptitle("Confusion Matrices for Models", fontsize=18, y=1.02)
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrices.png"))
plt.close()

# ========== 10. Precision / Recall / F1 ==========
precision_list = []
recall_list = []
f1_list = []

for name in model_names:
    model = joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    precision_list.append(report["weighted avg"]["precision"])
    recall_list.append(report["weighted avg"]["recall"])
    f1_list.append(report["weighted avg"]["f1-score"])

# Bar chart
x = np.arange(len(model_names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, precision_list, width, label='Precision')
plt.bar(x, recall_list, width, label='Recall')
plt.bar(x + width, f1_list, width, label='F1-score')

plt.xticks(x, model_names, rotation=45)
plt.ylim(0.5, 1.05)
plt.ylabel("Score")
plt.title("Precision / Recall / F1-score per Model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "prf_scores.png"))
plt.close()

# ========== 11. Feature Importance ==========
rf_model = joblib.load(os.path.join(MODEL_DIR, "RandomForest.pkl"))
importances = rf_model.feature_importances_
feature_names = selected_columns
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx], palette="crest")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "feature_importance_rf.png"))
plt.close()
