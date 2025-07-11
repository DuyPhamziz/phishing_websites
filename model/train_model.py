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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import copy
# === 1. Đường dẫn ===
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing.arff")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# === 2. Đọc dữ liệu .arff ===
with open(DATA_PATH, "r") as f:
    data = arff.load(f)

columns = [col[0] for col in data["attributes"]]
df = pd.DataFrame(data["data"], columns=columns).astype(int)

# === 3. Tách dữ liệu ===
X_raw = df.drop("Result", axis=1)
y_raw = df["Result"]

# === 4. Tiền xử lý: chọn đặc trưng, chuẩn hóa, SMOTE ===
selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X_raw, y_raw)
selected_features = X_raw.columns[selector.get_support()]
X_filtered = pd.DataFrame(X_selected, columns=selected_features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)
X_processed = pd.DataFrame(X_scaled, columns=selected_features)

smote = SMOTE(random_state=42)
X_final, y_final = smote.fit_resample(X_processed, y_raw)

# === 5. Tách train/test ===
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# === 6. Mô hình và đánh giá ===
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50),
    "KNN": KNeighborsClassifier(),
    "Perceptron": Perceptron(),
    "NaiveBayes": GaussianNB()
}

compare_scores = []
for name, model in models.items():
    print(f"⏳ Đang xử lý mô hình: {name}")

    # Dùng bản sao riêng cho mỗi phần để tránh xung đột
    model_raw = copy.deepcopy(model)
    model_proc = copy.deepcopy(model)

    # Trước xử lý
    try:
        model_raw.fit(X_train_raw, y_train_raw)
        pred_raw = model_raw.predict(X_test_raw)
        acc_raw = accuracy_score(y_test_raw, pred_raw)
        report_raw = classification_report(y_test_raw, pred_raw, output_dict=True)
    except Exception as e:
        print(f"[!] Lỗi khi chạy mô hình {name} với dữ liệu raw: {e}")
        acc_raw = report_raw = None

    # Sau xử lý
    model_proc.fit(X_train, y_train)
    pred_proc = model_proc.predict(X_test)
    acc_proc = accuracy_score(y_test, pred_proc)
    report_proc = classification_report(y_test, pred_proc, output_dict=True)

    compare_scores.append({
        "Model": name,
        "Accuracy (Raw)": acc_raw if acc_raw else 0,
        "Accuracy (Processed)": acc_proc,
        "Precision (Raw)": report_raw["weighted avg"]["precision"] if report_raw else 0,
        "Precision (Processed)": report_proc["weighted avg"]["precision"],
        "Recall (Raw)": report_raw["weighted avg"]["recall"] if report_raw else 0,
        "Recall (Processed)": report_proc["weighted avg"]["recall"],
        "F1 (Raw)": report_raw["weighted avg"]["f1-score"] if report_raw else 0,
        "F1 (Processed)": report_proc["weighted avg"]["f1-score"],
    })


# === 7. Biểu đồ so sánh ===
df_compare = pd.DataFrame(compare_scores)
metrics = ["Accuracy", "Precision", "Recall", "F1"]
for metric in metrics:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_compare["Model"]))
    plt.bar(x - 0.2, df_compare[f"{metric} (Raw)"], width=0.4, label="Before")
    plt.bar(x + 0.2, df_compare[f"{metric} (Processed)"], width=0.4, label="After")
    plt.xticks(x, df_compare["Model"], rotation=45)
    plt.ylim(0.4, 1.0)
    plt.title(f"{metric} Comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{metric.lower()}_compare.png"))
    plt.close()

# === 8. Lưu bảng tổng hợp ===
df_compare.to_csv(os.path.join(RESULT_DIR, "score_comparison.csv"), index=False)
print("✅ Đã tạo xong bảng so sánh kết quả trước/sau xử lý.")
# === 9. Ghi ra file TXT dễ đọc ===
txt_path = os.path.join(RESULT_DIR, "score_comparison.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for row in compare_scores:
        f.write(f"📊 Model: {row['Model']}\n")
        f.write(f"  - Accuracy:    {row['Accuracy (Raw)']:.4f} → {row['Accuracy (Processed)']:.4f}\n")
        f.write(f"  - Precision:   {row['Precision (Raw)']:.4f} → {row['Precision (Processed)']:.4f}\n")
        f.write(f"  - Recall:      {row['Recall (Raw)']:.4f} → {row['Recall (Processed)']:.4f}\n")
        f.write(f"  - F1-Score:    {row['F1 (Raw)']:.4f} → {row['F1 (Processed)']:.4f}\n")
        f.write("-" * 50 + "\n")

print(f"📄 Đã lưu kết quả chi tiết tại: {txt_path}")
