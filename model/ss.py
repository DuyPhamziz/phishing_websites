import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ====  Đường dẫn ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRE_PATH = os.path.join(BASE_DIR, "..", "result", "preprocessing", "processed_data.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "..", "result", "outliers", "clean_data.csv")
RESULT_DIR = os.path.join(BASE_DIR, "..", "result", "compare_metrics")
os.makedirs(RESULT_DIR, exist_ok=True)
# ====  Danh sách mô hình ====
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(),
    "Perceptron": Perceptron(),
    "NaiveBayes": GaussianNB(),
}

def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
    return scores

# ====  Đọc và đánh giá dữ liệu gốc ====
df_raw = pd.read_csv(PRE_PATH)
X_raw = df_raw.drop(columns=["Result"])
y_raw = df_raw["Result"]
print(" Đánh giá dữ liệu gốc...")
scores_raw = evaluate_model(X_raw, y_raw)

# ==== Đọc và đánh giá dữ liệu đã loại outlier ====
df_clean = pd.read_csv(CLEAN_PATH)
X_clean = df_clean.drop(columns=["Result"])
y_clean = df_clean["Result"]
print("Đánh giá dữ liệu đã loại outlier...")
scores_clean = evaluate_model(X_clean, y_clean)

# ==== Tạo bảng so sánh ====
df_result = pd.DataFrame()
for model in models.keys():
    row = {
        "Model": model,
        "Accuracy (Before)": scores_raw[model]["Accuracy"],
        "Accuracy (After)": scores_clean[model]["Accuracy"],
        "Precision (Before)": scores_raw[model]["Precision"],
        "Precision (After)": scores_clean[model]["Precision"],
        "Recall (Before)": scores_raw[model]["Recall"],
        "Recall (After)": scores_clean[model]["Recall"],
        "F1-score (Before)": scores_raw[model]["F1-score"],
        "F1-score (After)": scores_clean[model]["F1-score"],
    }
    df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)

print("\n BẢNG SO SÁNH:")
print(df_result)

# ==== Vẽ biểu đồ tổng hợp ====
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
bar_width = 0.2
index = np.arange(len(df_result["Model"])) * 2

plt.figure(figsize=(14, 6))

for i, metric in enumerate(metrics):
    before = df_result[f"{metric} (Before)"]
    after = df_result[f"{metric} (After)"]

    plt.bar(index + i * bar_width, before, width=bar_width, label=f"{metric} - Before", alpha=0.7)
    plt.bar(index + i * bar_width + bar_width, after, width=bar_width, label=f"{metric} - After", alpha=0.9)

# Căn chỉnh trục và nhãn
plt.xticks(index + bar_width * 2, df_result["Model"])
plt.ylim(0.4, 1.05)
plt.xlabel("Mô hình")
plt.ylabel("Giá trị độ đo")
plt.title("So sánh độ đo các mô hình trước và sau khi loại outlier")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "compare_all_metrics.png"))
plt.close()

print("📊 Đã lưu biểu đồ tổng hợp: compare_all_metrics.png")

