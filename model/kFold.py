import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree

# === Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "result", "preprocessing", "processed_data.csv"))
RF_RESULT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "result", "RandomForest"))
os.makedirs(RF_RESULT_DIR, exist_ok=True)

# ===  Đọc dữ liệu ===
print("\U0001F4E5 Đang đọc dữ liệu đã tiền xử lý...")
if not os.path.exists(DATA_PATH):
    print(f"❌ Không tìm thấy file: {DATA_PATH}")
    exit()

data = pd.read_csv(DATA_PATH)
print(" Đọc dữ liệu thành công!")

# === Đặc trưng và nhãn ===
X = data.drop(columns=['Result'])
y = data['Result']
feature_names = X.columns

# ===  Thực hiện K-Fold CV ===
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

depths = [3, 5, 7, 9, 11, None]
results = []

for depth in depths:
    accs, pres, recs, f1s = [], [], [], []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=100, max_depth=depth, criterion='gini', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        pres.append(precision_score(y_test, y_pred, average='weighted'))
        recs.append(recall_score(y_test, y_pred, average='weighted'))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))

    results.append({
        "Depth": str(depth) if depth is not None else "None",
        "Accuracy": np.mean(accs),
        "Precision": np.mean(pres),
        "Recall": np.mean(recs),
        "F1-score": np.mean(f1s)
    })

# === Vẽ biểu đồ thực nghiệm ===
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
print("Đã lưu biểu đồ thực nghiệm các độ đo theo độ sâu với K-Fold!")
