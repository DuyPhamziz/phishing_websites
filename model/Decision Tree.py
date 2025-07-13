import os
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ========== 1. Đường dẫn ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "phishing.arff")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "..", "result", "DecisionTree")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ========== 2. Đọc dữ liệu ==========
print(f"[+] Đang đọc dữ liệu từ: {DATA_PATH}")
data, meta = arff.loadarff(DATA_PATH)

# ========== 3. Tiền xử lý ==========
print("[+] Tiền xử lý dữ liệu...")
df = pd.DataFrame(data)
df_cleaned = df.apply(lambda col: col.map(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else int(x)))
# ========== 4. Tách đặc trưng và nhãn ==========
X = df_cleaned.drop("Result", axis=1)
y = df_cleaned["Result"]

# ========== 5. Chia tập huấn luyện / kiểm tra ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 6. Huấn luyện mô hình ==========
print("[+] Huấn luyện Decision Tree...")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# ========== 7. Dự đoán và đánh giá ==========
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n[+] Accuracy: {accuracy:.4f}")
print("[+] Báo cáo phân loại:\n", report)

# ========== 8. Lưu rule cây ==========
rules = export_text(clf, feature_names=list(X.columns))
rules_path = os.path.join(RESULT_DIR, "tree_rules.txt")
with open(rules_path, "w", encoding="utf-8") as f:
    f.write(rules)
print(f"[+] Rule cây quyết định đã lưu tại: {rules_path}")

# ========== 9. Lưu hình ảnh cây ==========
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns,
          class_names=["Legitimate (-1)", "Phishing (1)"],
          filled=True, rounded=True, max_depth=3, fontsize=10)
plt.title("Cây quyết định phân loại phishing website (depth ≤ 3)")
plot_path = os.path.join(RESULT_DIR, "decision_tree.png")
plt.savefig(plot_path)
plt.close()
print(f"[+] Đã lưu hình ảnh cây quyết định tại: {plot_path}")
