import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
# ========== 1. Đường dẫn ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Đường dẫn đến dữ liệu đã xử lý
DATA_PATH = os.path.join(BASE_DIR, "..", "result", "preprocessing", "processed_data.csv")

# 🔥 Tạo thư mục lưu kết quả riêng cho KNN
RESULT_DIR = os.path.join(BASE_DIR, "..", "result", "KNN")
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== 2. Đọc dữ liệu đã tiền xử lý ==========
print(f"[+] Đang đọc dữ liệu từ: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df.drop("Result", axis=1)
y = df["Result"]

# ========== 3. Chia tập huấn luyện / kiểm tra ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 4. Huấn luyện KNN ==========
print("[+] Đang huấn luyện mô hình KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ========== 5. Đánh giá ==========
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n[+] Accuracy: {accuracy:.4f}")
print("[+] Báo cáo phân loại:\n", report)

# ========== 6. Ghi kết quả vào file ==========
output_path = os.path.join(RESULT_DIR, "knn_results.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"📊 Mô hình: K-Nearest Neighbors (k=5)\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print(f"[+] Kết quả đã lưu tại: {output_path}")
# === 7. Chọn k tối ưu bằng cross-validation ===
print("[+] Đang tìm giá trị k tối ưu bằng cross-validation...")

k_range = range(1, 20, 2)  # k lẻ từ 1 đến 19
accuracies = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X, y, cv=5)
    mean_score = scores.mean()
    accuracies.append(mean_score)
    print(f"k = {k}: Accuracy = {mean_score:.4f}")

# === 8. Vẽ và lưu biểu đồ ===
plt.figure(figsize=(8, 5))
plt.plot(k_range, accuracies, marker='o', linestyle='-', color='blue')
plt.title("Độ chính xác theo giá trị k (Cross-validation)")
plt.xlabel("Số láng giềng gần nhất (k)")
plt.ylabel("Độ chính xác trung bình")
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(RESULT_DIR, "knn_k_selection.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"[+] Đã lưu biểu đồ tại: {plot_path}")
