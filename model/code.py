import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ==== Đường dẫn ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "result", "preprocessing", "cleaned_data.csv")
RESULT_DIR = os.path.join(BASE_DIR, "result", "model_with_outlier_removal")
os.makedirs(RESULT_DIR, exist_ok=True)

# ==== Hàm loại bỏ outlier ====
def remove_outliers_iqr(df, columns):
    df_cleaned = df.copy()
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)]
    return df_cleaned

# ==== Đọc dữ liệu ====
print("Đang đọc dữ liệu đã tiền xử lý...")
df = pd.read_csv(DATA_PATH)

features = df.drop(columns=["Result"]).columns.tolist()
df_no_outlier = remove_outliers_iqr(df, features)

print(f"Số mẫu ban đầu: {df.shape[0]} → Sau khi loại outlier: {df_no_outlier.shape[0]}")

X = df_no_outlier.drop(columns=["Result"])
y = df_no_outlier["Result"]

# ==== Phân phối nhãn sau xử lý outlier ====
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="Set2")
plt.title("Phân phối nhãn sau khi loại outlier")
plt.xlabel("Label")
plt.ylabel("Số lượng mẫu")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "label_distribution.png"))
plt.close()

# ==== Tách train/test ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
    print(f"Đang huấn luyện: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-score": report["weighted avg"]["f1-score"],
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Phishing (-1)", "Legitimate (1)"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"confusion_matrix_{name.lower()}.png"))
    plt.close()

# ====  Biểu đồ so sánh ====
df_result = pd.DataFrame(results)
df_result.to_csv(os.path.join(RESULT_DIR, "score_comparison.csv"), index=False)

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
plt.title("So sánh các mô hình sau khi loại outlier")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "all_metrics_combined.png"))
plt.close()

print("Hoàn tất! Đã lưu kết quả vào:", RESULT_DIR)
