import pandas as pd
import os
from sklearn.ensemble import IsolationForest

# ==== Đường dẫn ==== 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "result", "preprocessing", "processed_data.csv")
OUTLIER_DIR = os.path.join(BASE_DIR, "..", "result", "outliers")
os.makedirs(OUTLIER_DIR, exist_ok=True)

OUTLIER_CSV = os.path.join(OUTLIER_DIR, "outliers_detected.csv")
CLEAN_CSV = os.path.join(OUTLIER_DIR, "clean_data.csv")

# ==== Đọc dữ liệu ==== 
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Result"])
y = df["Result"]

# ==== Isolation Forest để phát hiện outlier ==== 
iso = IsolationForest(contamination=0.05, random_state=42)
y_pred = iso.fit_predict(X)  # -1: outlier, 1: normal

# ==== Phân loại dòng outlier và dòng sạch ====
df["is_outlier"] = y_pred
df_outliers = df[df["is_outlier"] == -1].drop(columns=["is_outlier"])
df_clean = df[df["is_outlier"] == 1].drop(columns=["is_outlier"])

# ==== Lưu kết quả ==== 
df_outliers.to_csv(OUTLIER_CSV, index=False)
df_clean.to_csv(CLEAN_CSV, index=False)

# ==== In thông tin ====
print(f"Tổng số dòng ban đầu: {len(df)}")
print(f"Số dòng bị phát hiện là outlier: {len(df_outliers)}")
print(f"Số dòng còn lại sau khi loại: {len(df_clean)}")
print(f" Đã lưu outliers vào: {OUTLIER_CSV}")
print(f"Đã lưu clean data vào: {CLEAN_CSV}")
