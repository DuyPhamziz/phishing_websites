import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# ========== 1. Đường dẫn ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "phishing.arff")

# 🔥 Tạo thư mục lưu kết quả riêng cho KNN
RESULT_DIR = os.path.join(BASE_DIR, "..", "result", "KNN")
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== 2. Đọc và xử lý dữ liệu ==========
print(f"[+] Đang đọc dữ liệu từ : {DATA_PATH}")
data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

# Decode byte → int
df_cleaned = df.apply(lambda col: col.map(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x)))
X = df_cleaned.drop("Result", axis=1)
y = df_cleaned["Result"]

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
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(report)

print(f"[+] Kết quả đã lưu tại: {output_path}")
