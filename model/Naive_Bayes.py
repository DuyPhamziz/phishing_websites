import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
#đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "phishing.arff")
RESULT_DIR = os.path.join(BASE_DIR, "..", "result")
os.makedirs(RESULT_DIR,exist_ok=True)
# đọc dự liệu
print(f"[+] Đang đọc dữ liệu từ: {DATA_PATH}")
data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)
#tiền sử lý
df_cleaned = df.applymap(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x))
x = df_cleaned.drop("Result", axis=1)
y = df_cleaned["Result"]
# tách train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#khởi tạo và huấn luyện mô hình
print("[+] Huấn luyện mô hình Naive Bayes...")
nb = GaussianNB()
nb.fit(x_train, y_train)
#dự đoán và đánh giá
y_pred = nb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"\n[+] Accuracy: {accuracy:.4f}")
print("[+] Báo cáo phân loại:\n", report)
#ghi kết quả ra file
output_path = os.path.join(RESULT_DIR, "naive_bayes_report.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(report)
print(f"[+] Kết quả đã lưu tại: {output_path}")