import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# ========== Đường dẫn ==========

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tới file đã tiền xử lý
DATA_PATH = os.path.join(BASE_DIR, "..", "result", "preprocessing", "processed_data.csv")

# Thư mục lưu kết quả riêng
RESULT_DIR = os.path.join(BASE_DIR, "..", "result", "NaiveBayes")
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== Đọc dữ liệu đã xử lý ==========
print(f"[+] Đang đọc dữ liệu từ: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df.drop("Result", axis=1)
y = df["Result"]

# ========== Tách train/test ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Huấn luyện mô hình ==========
print("[+] Đang huấn luyện mô hình Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)

# ========== Đánh giá ==========
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n[+] Accuracy: {accuracy:.4f}")
print("[+] Báo cáo phân loại:\n", report)

# ========== Lưu kết quả ==========
output_path = os.path.join(RESULT_DIR, "naive_bayes_report.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("Mô hình: Naive Bayes\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print(f"[+] Kết quả đã lưu tại: {output_path}")
