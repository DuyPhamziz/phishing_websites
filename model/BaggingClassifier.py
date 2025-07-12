import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ========== 1. Đường dẫn ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "phishing.arff")
RESULT_DIR = os.path.join(BASE_DIR, "..", "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== 2. Đọc dữ liệu ==========
print(f"[+] Đang đọc dữ liệu từ: {DATA_PATH}")
data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

# ========== 3. Tiền xử lý ==========
df_cleaned = df.applymap(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x))
X = df_cleaned.drop("Result", axis=1)
y = df_cleaned["Result"]

# ========== 4. Tách dữ liệu ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 5. Mô hình Bagging ==========
print("[+] Huấn luyện mô hình BaggingClassifier...")
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train, y_train)

# ========== 6. Đánh giá ==========
y_pred = bagging.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n[+] Accuracy: {accuracy:.4f}")
print("[+] Báo cáo phân loại:\n", report)

# ========== 7. Lưu kết quả ==========
output_path = os.path.join(RESULT_DIR, "bagging_report.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(report)

print(f"[+] Kết quả đã lưu tại: {output_path}")
