import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
)

# ==== Đường dẫn ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "result", "preprocessing", "processed_data.csv")
RESULT_DIR = os.path.join(BASE_DIR, "..", "result", "model_evaluation")
RESULT_DIR = os.path.abspath(RESULT_DIR)
os.makedirs(RESULT_DIR, exist_ok=True)

# ==== Đọc dữ liệu đã tiền xử lý ====
print(" Đang đọc dữ liệu đã tiền xử lý...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Result"])
y = df["Result"]

# ==== Phân phối nhãn ====
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="Set2")
plt.title("Phân phối nhãn sau tiền xử lý và SMOTE")
plt.xlabel("Label")
plt.ylabel("Số lượng mẫu")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "label_distribution.png"))
plt.close()
print("Đã lưu biểu đồ phân phối nhãn: label_distribution.png")

# ==== K-Fold Stratified ====
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==== Huấn luyện và đánh giá ====
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(),
    "Perceptron": Perceptron(),
    "NaiveBayes": GaussianNB(),
}

results = []

for name, model in models.items():
    print(f" Đang huấn luyện: {name}")
    accs, pres, recs, f1s = [], [], [], []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        m = copy.deepcopy(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        pres.append(precision_score(y_test, y_pred, average='weighted'))
        recs.append(recall_score(y_test, y_pred, average='weighted'))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))

    results.append({
        "Model": name,
        "Accuracy": np.mean(accs),
        "Precision": np.mean(pres),
        "Recall": np.mean(recs),
        "F1-score": np.mean(f1s),
    })

# ==== Lưu bảng kết quả ====
df_result = pd.DataFrame(results)
df_result.to_csv(os.path.join(RESULT_DIR, "score_comparison.csv"), index=False)

# ==== Vẽ biểu đồ tổng hợp ====
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
bar_width = 0.15
index = np.arange(len(df_result["Model"]))

plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.bar(index + i * bar_width, df_result[metric], width=bar_width, label=metric)
    for j, val in enumerate(df_result[metric]):
        plt.text(index[j] + i * bar_width, val + 0.005, f"{val:.2f}", ha='center', fontsize=8)

plt.xticks(index + bar_width * 1.5, df_result["Model"])
plt.ylim(0.5, 1.0)
plt.title("So sánh các mô hình (Cross Validation - K=5)")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "all_metrics_combined.png"))
plt.close()
print(" Đã vẽ biểu đồ tổng hợp: all_metrics_combined.png")

# ==== Heatmap tương quan ====
corr = df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Heatmap tương quan giữa các đặc trưng và nhãn")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "heatmap_correlation.png"))
plt.close()
print("Đã lưu heatmap: heatmap_correlation.png")

# ==== Boxplot 20 đặc trưng quan trọng nhất ====
selected_features = X.columns[:20]
boxplot_df = df[selected_features.tolist() + ["Result"]]

melted_df = pd.melt(boxplot_df, id_vars="Result", var_name="Feature", value_name="Value")
plt.figure(figsize=(16, 10))
sns.boxplot(data=melted_df, x="Feature", y="Value", hue="Result", palette="Set2")
plt.xticks(rotation=45, ha="right")
plt.title("Boxplot 20 đặc trưng phân theo nhãn Result")
plt.xlabel("Đặc trưng")
plt.ylabel("Giá trị")
plt.legend(title="Result", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "boxplot_20_features.png"))
plt.close()
print(" Đã lưu boxplot: boxplot_20_features.png")