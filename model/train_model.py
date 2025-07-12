import arff
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

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

# ✅ Loại bỏ các dòng có nhãn "Result = 0" (nghi ngờ)
df = df[df["Result"] != 0]

# === 3. Tách đặc trưng và nhãn ===
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

# === Biểu đồ phân bố nhãn trước và sau SMOTE ===
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=y_raw, palette="Set1")
plt.title("Trước SMOTE")
plt.xlabel("Label")
plt.ylabel("Số lượng mẫu")

plt.subplot(1, 2, 2)
sns.countplot(x=y_final, palette="Set2")
plt.title("Sau SMOTE")
plt.xlabel("Label")
plt.ylabel("Số lượng mẫu")

plt.suptitle("Phân bố nhãn trước và sau SMOTE", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(RESULT_DIR, "label_distribution_comparison.png"))
plt.close()

print("📊 Đã lưu biểu đồ phân bố nhãn trước/sau SMOTE tại: result/label_distribution_comparison.png")

# === 5. Tách train/test ===
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# === 6. Mô hình và đánh giá ===
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(),
    "Perceptron": Perceptron(),
    "NaiveBayes": GaussianNB(),
}

compare_scores = []
for name, model in models.items():
    print(f"⏳ Đang xử lý mô hình: {name}")

    model_raw = copy.deepcopy(model)
    model_proc = copy.deepcopy(model)

    try:
        model_raw.fit(X_train_raw, y_train_raw)
        pred_raw = model_raw.predict(X_test_raw)
        acc_raw = accuracy_score(y_test_raw, pred_raw)
        report_raw = classification_report(y_test_raw, pred_raw, output_dict=True)
    except Exception as e:
        print(f"[!] Lỗi khi chạy mô hình {name} với dữ liệu raw: {e}")
        acc_raw = report_raw = None

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

# === 7. Vẽ biểu đồ gộp 4 chỉ số (Processed data) ===
df_compare = pd.DataFrame(compare_scores)

metrics = ["Accuracy", "Precision", "Recall", "F1"]
bar_width = 0.15
index = np.arange(len(df_compare["Model"]))

fig, ax = plt.subplots(figsize=(14, 7))
for i, metric in enumerate(metrics):
    bars = ax.bar(index + i * bar_width, df_compare[f"{metric} (Processed)"], bar_width, label=metric)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.003, f"{yval * 100:.2f}%",
                ha='center', va='bottom', fontsize=8)

ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(df_compare["Model"], rotation=45)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Score")
ax.set_title("So sánh các chỉ số (Processed data)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "all_metrics_combined.png"))
plt.close()
print("📊 Đã vẽ hình tổng hợp các chỉ số tại: result/all_metrics_combined.png")

# === 8. Lưu bảng tổng hợp ===
df_compare.to_csv(os.path.join(RESULT_DIR, "score_comparison.csv"), index=False)

# === 9. Ghi TXT kết quả ===
txt_path = os.path.join(RESULT_DIR, "score_comparison.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for row in compare_scores:
        f.write(f"\n📊 Model: {row['Model']}\n")
        f.write(f"  - Accuracy:    {row['Accuracy (Raw)']:.4f} → {row['Accuracy (Processed)']:.4f}\n")
        f.write(f"  - Precision:   {row['Precision (Raw)']:.4f} → {row['Precision (Processed)']:.4f}\n")
        f.write(f"  - Recall:      {row['Recall (Raw)']:.4f} → {row['Recall (Processed)']:.4f}\n")
        f.write(f"  - F1-Score:    {row['F1 (Raw)']:.4f} → {row['F1 (Processed)']:.4f}\n")
        f.write("-" * 50 + "\n")

print(f"📄 Đã lưu kết quả chi tiết tại: {txt_path}")
# === Biểu đồ tương quan (heatmap) giữa đặc trưng và nhãn ===
heatmap_df = X_filtered.copy()
heatmap_df["Result"] = y_raw.reset_index(drop=True)

plt.figure(figsize=(14, 10))
corr = heatmap_df.corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Heatmap tương quan giữa các đặc trưng và nhãn Result")
plt.tight_layout()

# Lưu hình ảnh
plt.savefig(os.path.join(RESULT_DIR, "heatmap_correlation.png"))
plt.close()

print("📊 Đã lưu biểu đồ heatmap tại: result/heatmap_correlation.png")
# === Boxplot cho 20 đặc trưng quan trọng nhất ===

# Gộp dữ liệu với nhãn để dễ vẽ
boxplot_df = X_filtered.copy()
boxplot_df["Result"] = y_raw.reset_index(drop=True)

# Chuyển dữ liệu sang dạng "long" để dễ dùng seaborn.boxplot
melted_df = pd.melt(boxplot_df, id_vars="Result", var_name="Feature", value_name="Value")

plt.figure(figsize=(16, 10))
sns.boxplot(data=melted_df, x="Feature", y="Value", hue="Result", palette="Set2")
plt.xticks(rotation=45, ha="right")
plt.title("Boxplot phân phối của 20 đặc trưng quan trọng nhất theo nhãn Result")
plt.xlabel("Đặc trưng")
plt.ylabel("Giá trị (sau chuẩn hóa)")
plt.legend(title="Result", loc="upper right")
plt.tight_layout()

# Lưu hình
plt.savefig(os.path.join(RESULT_DIR, "boxplot_20_features.png"))
plt.close()

print("📊 Đã lưu biểu đồ boxplot các đặc trưng tại: result/boxplot_20_features.png")
# === 10. Lưu dữ liệu đã tiền xử lý ===
processed_df = pd.DataFrame(X_final, columns=selected_features)
processed_df["Result"] = y_final.values

processed_csv_path = os.path.join(RESULT_DIR, "processed_data.csv")
processed_df.to_csv(processed_csv_path, index=False)

print(f"✅ Đã lưu dữ liệu đã tiền xử lý tại: {processed_csv_path}")