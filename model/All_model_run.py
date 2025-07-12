import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB

# ========== 1. Đường dẫn ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "phishing.arff")
RESULT_BASE = os.path.join(BASE_DIR, "..", "result")
os.makedirs(RESULT_BASE, exist_ok=True)

# ========== 2. Đọc và tiền xử lý dữ liệu ==========
print(f"[+] Đang đọc dữ liệu từ: {DATA_PATH}")
data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

# Chuyển byte → int
df_cleaned = df.apply(lambda col: col.map(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x)))

X = df_cleaned.drop("Result", axis=1)
y = df_cleaned["Result"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 3. Khởi tạo mô hình ==========
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Perceptron": Perceptron(),
    "NaiveBayes": GaussianNB(),
}

# ========== 4. Huấn luyện và ghi kết quả ==========
for name, model in models.items():
    print(f"\n[+] Huấn luyện mô hình: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"    - Accuracy: {acc:.4f}")
    
    # Tạo thư mục riêng cho mô hình
    model_result_dir = os.path.join(RESULT_BASE, name)
    os.makedirs(model_result_dir, exist_ok=True)

    # Ghi kết quả ra file
    report_path = os.path.join(model_result_dir, f"{name.lower()}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)

    print(f"    - Kết quả lưu tại: {report_path}")
