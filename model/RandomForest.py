from scipy.io import arff
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os

# Tạo thư mục lưu kết quả
output_dir = 'result/random forest'
os.makedirs(output_dir, exist_ok=True)

# Bước 1: Đọc dữ liệu
data, meta = arff.loadarff('C:/phishing_websites/data/phishing.arff')
df = pd.DataFrame(data)

# Bước 2: Giải mã các cột dạng byte
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')

# Bước 3: Tách dữ liệu
X = df.drop('Result', axis=1)
y = df['Result']

# Bước 4: Chia và huấn luyện mô hình với 30 thuộc tính
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Bước 5: Dự đoán và đánh giá
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
recall = report_dict['-1']['recall']

# Bước 6: Ghi kết quả vào file
with open(os.path.join(output_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("📊 ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST (30 đặc trưng)\n\n")
    f.write(f"✅ Accuracy: {accuracy:.4f}\n")
    f.write(f"✅ Recall lớp -1 (URL nguy hiểm): {recall:.4f}\n\n")
    f.write("📌 Classification Report:\n")
    f.write(report)

# Bước 7: Lưu biểu đồ đặc trưng quan trọng
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
plt.title("Top 10 đặc trưng quan trọng (dựa trên 30 đặc trưng)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), [features[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# In ra console
print(f"✅ Accuracy (30 features): {accuracy:.4f}")
print(f"✅ Recall lớp -1 (URL nguy hiểm): {recall:.4f}")
