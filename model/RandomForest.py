import pandas as pd
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree

# === 1. Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "result", "preprocessing", "processed_data.csv"))
RF_RESULT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "result", "RandomForest"))
os.makedirs(RF_RESULT_DIR, exist_ok=True)

# === 2. Đọc dữ liệu ===
print("📥 Đang đọc dữ liệu đã tiền xử lý...")
if not os.path.exists(DATA_PATH):
    print(f"❌ Không tìm thấy file: {DATA_PATH}")
    exit()

data = pd.read_csv(DATA_PATH)
print("✅ Đọc dữ liệu thành công!")

# === 3. Đặc trưng và nhãn ===
X = data.drop(columns=['Result'])
y = data['Result']
feature_names = X.columns

# === 4. Thực hiện K-Fold CV ===
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

depths = [3, 5, 7, 9, 11, None]
results = []
all_models = {}

for depth in depths:
    accs, pres, recs, f1s = [], [], [], []
    best_model = None
    best_f1 = -1

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=100, max_depth=depth, criterion='gini', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        accs.append(acc)
        pres.append(pre)
        recs.append(rec)
        f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_model = (copy.deepcopy(model), X_test, y_test, y_pred)

    results.append({
        "Depth": str(depth) if depth is not None else "None",
        "Accuracy": np.mean(accs),
        "Precision": np.mean(pres),
        "Recall": np.mean(recs),
        "F1-score": np.mean(f1s)
    })

    if str(depth) == "11":
        all_models['depth_11'] = best_model

# === 5. Vẽ biểu đồ thực nghiệm ===
df_result = pd.DataFrame(results)
df_result.set_index("Depth", inplace=True)
df_result_percent = df_result * 100
labels = df_result_percent.index.tolist()
x = np.arange(len(labels))
width = 0.18
colors = ["#66c2a5", "#8da0cb", "#ffd92f", "#bdbdbd"]
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

fig, ax = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(metrics):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, df_result_percent[metric], width, label=metric, color=colors[i])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                f"{height:.1f}%", ha='center', va='bottom', fontsize=6)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(70, 100)
ax.set_xlabel("Độ sâu của cây")
ax.set_ylabel("Giá trị (%)")
ax.set_title("Ảnh hưởng của độ sâu (max_depth) đến các độ đo (K-Fold = 5)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "depth_vs_metrics_kfold.png"), dpi=300)
plt.close()
print("✅ Đã lưu biểu đồ thực nghiệm các độ đo theo độ sâu với K-Fold!")

# === 6. Đánh giá mô hình depth=11 chi tiết ===
best_rf, X_test, y_test, y_pred = all_models['depth_11']

report = classification_report(y_test, y_pred, digits=4)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

with open(os.path.join(RF_RESULT_DIR, "rf_metrics_depth11.txt"), "w", encoding="utf-8") as f:
    f.write("📊 Classification Report:\n")
    f.write(report + "\n")
    f.write(f"Criterion:  {best_rf.criterion}\n")
    f.write(f"Max depth:  {best_rf.max_depth}\n")
    f.write(f"Accuracy:   {acc:.4f}\n")
    f.write(f"Precision:  {precision:.4f}\n")
    f.write(f"Recall:     {recall:.4f}\n")
    f.write(f"F1-score:   {f1:.4f}\n")

# Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (depth=11)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "confusion_matrix_depth11.png"), dpi=300)
plt.close()

# Feature importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance (depth=11)")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='steelblue')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel("Đặc trưng")
plt.ylabel("Tầm quan trọng")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "feature_importance_depth11.png"), dpi=300)
plt.close()

# Plot tree
plt.figure(figsize=(20, 10))
plot_tree(best_rf.estimators_[0],
          filled=True,
          feature_names=feature_names,
          class_names=["Clean", "Phishing"],
          max_depth=3,
          fontsize=9,
          impurity=True)
plt.title("Minh họa một cây trong Random Forest (depth=11)")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "example_tree_depth11.png"), dpi=300)
plt.close()

# Tổng hợp độ đo dạng cột
metrics = [acc, precision, recall, f1]
labels = ["Accuracy", "Precision", "Recall", "F1-score"]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, metrics, color="skyblue")
plt.ylim(0.5, 1.0)
plt.title("Các độ đo mô hình Random Forest (depth=11, K-Fold)")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "rf_metrics_barplot_depth11.png"), dpi=300)
plt.close()

print("✅ Hoàn tất toàn bộ quá trình đánh giá mô hình Random Forest với K-Fold!")
# === 7. Huấn luyện mô hình cuối cùng với toàn bộ tập dữ liệu ===
from sklearn.model_selection import train_test_split

print("🚀 Huấn luyện mô hình cuối cùng với max_depth = 11...")

# Chia tập huấn luyện và kiểm tra
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Huấn luyện mô hình
final_model = RandomForestClassifier(
    n_estimators=100, max_depth=11, criterion="gini", random_state=42)
final_model.fit(X_train_final, y_train_final)

# Dự đoán
y_pred_final = final_model.predict(X_test_final)

# Đánh giá
acc_final = accuracy_score(y_test_final, y_pred_final)
prec_final = precision_score(y_test_final, y_pred_final, average="weighted")
rec_final = recall_score(y_test_final, y_pred_final, average="weighted")
f1_final = f1_score(y_test_final, y_pred_final, average="weighted")

report_final = classification_report(y_test_final, y_pred_final, digits=4)

# Lưu classification report
with open(os.path.join(RF_RESULT_DIR, "rf_final_model_report.txt"), "w", encoding="utf-8") as f:
    f.write("📊 Classification Report (Final Model):\n")
    f.write(report_final + "\n")
    f.write(f"Accuracy:   {acc_final:.4f}\n")
    f.write(f"Precision:  {prec_final:.4f}\n")
    f.write(f"Recall:     {rec_final:.4f}\n")
    f.write(f"F1-score:   {f1_final:.4f}\n")

print("✅ Đã huấn luyện và đánh giá mô hình cuối cùng!")

# === 8. Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_final, y_pred_final), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Final Model)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "confusion_matrix_final_model.png"), dpi=300)
plt.close()

# === 9. Feature Importance
importances_final = final_model.feature_importances_
indices_final = np.argsort(importances_final)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importance (Final Model)")
plt.bar(range(X.shape[1]), importances_final[indices_final], align="center", color='mediumseagreen')
plt.xticks(range(X.shape[1]), feature_names[indices_final], rotation=90)
plt.xlabel("Đặc trưng")
plt.ylabel("Tầm quan trọng")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "feature_importance_final_model.png"), dpi=300)
plt.close()

# === 10. Plot tree (minh họa một cây)
plt.figure(figsize=(20, 10))
plot_tree(final_model.estimators_[0],
          filled=True,
          feature_names=feature_names,
          class_names=["Clean", "Phishing"],
          max_depth=3,
          fontsize=9,
          impurity=True)
plt.title("Minh họa một cây trong Random Forest (Final Model)")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "example_tree_final_model.png"), dpi=300)
plt.close()

# === 11. Barplot các độ đo
metrics_final = [acc_final, prec_final, rec_final, f1_final]
labels_final = ["Accuracy", "Precision", "Recall", "F1-score"]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels_final, metrics_final, color="salmon")
plt.ylim(0.5, 1.0)
plt.title("Các độ đo mô hình Random Forest (Final Model)")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "rf_metrics_barplot_final_model.png"), dpi=300)
plt.close()

# === 12. Lưu mô hình (tuỳ chọn)
import joblib
joblib.dump(final_model, os.path.join(RF_RESULT_DIR, "final_random_forest_model.pkl"))
print("📦 Mô hình cuối đã được lưu dưới dạng .pkl")
