import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# === Đường dẫn ===
BASE_DIR = os.getcwd()
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "phishing.arff")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "result", "preprocessing", "processed_data.csv")
RESULT_DIR = os.path.join(BASE_DIR, "result", "model_comparison")
os.makedirs(RESULT_DIR, exist_ok=True)

# === Đọc dữ liệu raw từ .arff ===
print("Đọc dữ liệu RAW từ .arff...")
data, meta = arff.loadarff(RAW_DATA_PATH)
df_raw = pd.DataFrame(data)
df_raw = df_raw.applymap(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x))

X_raw = df_raw.drop("Result", axis=1)
y_raw = df_raw["Result"]

# ===  Đọc dữ liệu đã tiền xử lý ===
print("Đọc dữ liệu đã tiền xử lý...")
df_processed = pd.read_csv(PROCESSED_DATA_PATH)
X_processed = df_processed.drop(columns=["Result"])
y_processed = df_processed["Result"]

# === Khởi tạo mô hình ===
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(),
    "Perceptron": Perceptron(),
    "NaiveBayes": GaussianNB(),
}

compare_scores = []
cm_processed_all = {}

for name, model in models.items():
    print(f"Đang xử lý mô hình: {name}")

    # Dữ liệu RAW
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )
    model_raw = copy.deepcopy(model)
    model_raw.fit(X_train_raw, y_train_raw)
    pred_raw = model_raw.predict(X_test_raw)
    acc_raw = accuracy_score(y_test_raw, pred_raw)
    report_raw = classification_report(y_test_raw, pred_raw, output_dict=True)

    # Dữ liệu PROCESSED
    X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )
    model_proc = copy.deepcopy(model)
    model_proc.fit(X_train_proc, y_train_proc)
    pred_proc = model_proc.predict(X_test_proc)
    acc_proc = accuracy_score(y_test_proc, pred_proc)
    report_proc = classification_report(y_test_proc, pred_proc, output_dict=True)

    compare_scores.append({
        "Model": name,
        "Accuracy (Raw)": acc_raw,
        "Accuracy (Processed)": acc_proc,
        "Precision (Raw)": report_raw["weighted avg"]["precision"],
        "Precision (Processed)": report_proc["weighted avg"]["precision"],
        "Recall (Raw)": report_raw["weighted avg"]["recall"],
        "Recall (Processed)": report_proc["weighted avg"]["recall"],
        "F1 (Raw)": report_raw["weighted avg"]["f1-score"],
        "F1 (Processed)": report_proc["weighted avg"]["f1-score"],
    })

    # === Ghi classification_report ===
    with open(os.path.join(RESULT_DIR, f"{name}_classification_report.txt"), "w", encoding="utf-8") as f_report:
        f_report.write(f"[{name}] Classification Report - RAW\n")
        f_report.write(classification_report(y_test_raw, pred_raw))
        f_report.write("\n\n")
        f_report.write(f"[{name}] Classification Report - PROCESSED\n")
        f_report.write(classification_report(y_test_proc, pred_proc))

    # === Lưu confusion matrix PROCESSED 
    cm_processed_all[name] = confusion_matrix(y_test_proc, pred_proc)

# === Lưu bảng kết quả ===
df_compare = pd.DataFrame(compare_scores)
df_compare.to_csv(os.path.join(RESULT_DIR, "score_comparison.csv"), index=False)

# === Biểu đồ tổng hợp các chỉ số (processed) ===
metrics = ["Accuracy", "Precision", "Recall", "F1"]
bar_width = 0.15
index = np.arange(len(df_compare["Model"]))

plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.bar(index + i * bar_width, df_compare[f"{metric} (Processed)"],
            width=bar_width, label=metric)
    for j, val in enumerate(df_compare[f"{metric} (Processed)"]):
        plt.text(index[j] + i * bar_width, val + 0.005, f"{val:.2f}",
                 ha='center', fontsize=8)

plt.xticks(index + bar_width * 1.5, df_compare["Model"])
plt.ylim(0.5, 1.0)
plt.title("So sánh các mô hình (dữ liệu đã tiền xử lý)")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "processed_models_comparison.png"))
plt.close()

# === Báo cáo chi tiết TXT ===
txt_path = os.path.join(RESULT_DIR, "score_comparison.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for row in compare_scores:
        f.write(f" Model: {row['Model']}\n")
        f.write(f"  - Accuracy:    {row['Accuracy (Raw)']:.4f} → {row['Accuracy (Processed)']:.4f}\n")
        f.write(f"  - Precision:   {row['Precision (Raw)']:.4f} → {row['Precision (Processed)']:.4f}\n")
        f.write(f"  - Recall:      {row['Recall (Raw)']:.4f} → {row['Recall (Processed)']:.4f}\n")
        f.write(f"  - F1-Score:    {row['F1 (Raw)']:.4f} → {row['F1 (Processed)']:.4f}\n")
        f.write("-" * 50 + "\n")

# === Vẽ nhiều confusion matrix (processed) trên 1 hình ===
n_models = len(cm_processed_all)
cols = 3
rows = (n_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for idx, (name, cm) in enumerate(cm_processed_all.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[idx], cmap="Blues", values_format="d", colorbar=False)
    axes[idx].set_title(f"{name} (Processed)")

# Tắt các subplot dư
for ax in axes[len(cm_processed_all):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "all_confusion_matrices_processed.png"))
plt.close()
print("Đã lưu confusion matrix tổng hợp: all_confusion_matrices_processed.png")
