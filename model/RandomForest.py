import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree

# === 1. Đường dẫn ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dữ liệu đầu vào
DATA_PATH = os.path.join(BASE_DIR, "result", "processed_data.csv")

# Kết quả Random Forest lưu tại thư mục riêng
RF_RESULT_DIR = os.path.join(BASE_DIR, "result", "randomforest")
os.makedirs(RF_RESULT_DIR, exist_ok=True)

# === 2. Đọc dữ liệu đã tiền xử lý ===
if not os.path.exists(DATA_PATH):
    print(f"❌ Không tìm thấy file: {DATA_PATH}")
    exit()

data = pd.read_csv(DATA_PATH)
print("✅ Đọc dữ liệu thành công:")
print(data.head())

# === 3. Tách đặc trưng và nhãn ===
X = data.drop(columns=['Result'])
y = data['Result']

# === 4. Tách train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Huấn luyện Random Forest ===
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# === 6. Dự đoán và đánh giá ===
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# === In ra màn hình ===
print("\n📊 Classification Report:\n")
print(report)
print(f"📌 Accuracy:  {acc:.4f}")
print(f"📌 Precision: {precision:.4f}")
print(f"📌 Recall:    {recall:.4f}")
print(f"📌 F1-score:  {f1:.4f}")

# === 7. Ghi file độ đo ===
with open(os.path.join(RF_RESULT_DIR, "rf_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("📊 Classification Report:\n\n")
    f.write(report + "\n")
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n")
print("✅ Đã lưu báo cáo độ đo tại: rf_metrics.txt")

# === 8. Confusion Matrix ===
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "confusion_matrix.png"))
plt.close()
print("✅ Đã lưu hình confusion matrix.")

# === 9. Feature Importance ===
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45, ha='right')
plt.xlabel("Đặc trưng")
plt.ylabel("Tầm quan trọng")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "feature_importance.png"))
plt.close()
print("✅ Đã lưu hình feature importance.")

# === 10. Minh họa 1 cây trong Random Forest ===
plt.figure(figsize=(20,10))
plot_tree(rf.estimators_[0], 
          filled=True, 
          feature_names=feature_names, 
          class_names=["Clean", "Phishing"],
          max_depth=3,
          fontsize=10)
plt.title("Minh họa một cây trong Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(RF_RESULT_DIR, "example_tree.png"))
plt.close()
print("✅ Đã lưu hình cây quyết định minh họa.")
