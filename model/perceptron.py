import os
import pandas as pd
from scipy.io import arff
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. Đường dẫn ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "phishing.arff")

# 🔥 Tạo thư mục riêng cho Perceptron
RESULT_DIR = os.path.join(BASE_DIR, "..", "result", "Perceptron")
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== 2. Đọc dữ liệu ==========
print(f"[+] Đang đọc dữ liệu từ: {DATA_PATH}")
data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

# ========== 3. Tiền xử lý ==========
df_cleaned = df.apply(lambda col: col.map(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x)))
X = df_cleaned.drop("Result", axis=1)
y = df_cleaned["Result"]

# ========== 4. Tách dữ liệu ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 5. Huấn luyện mô hình ==========
print("[+] Huấn luyện mô hình Perceptron...")
clf = Perceptron()
clf.fit(X_train, y_train)

# ========== 6. Dự đoán và đánh giá ==========
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n[+] Accuracy: {accuracy:.4f}")
print("[+] Báo cáo phân loại:\n", report)

# ========== 7. Ghi kết quả ==========
output_path = os.path.join(RESULT_DIR, "perceptron_report.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(report)

print(f"[+] Kết quả đã lưu tại: {output_path}")
