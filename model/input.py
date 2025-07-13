import os
import arff
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# === Đường dẫn ===
BASE_DIR = os.getcwd()
RESULT_DIR = os.path.join(BASE_DIR, "result")
PREPROC_DIR = os.path.join(RESULT_DIR, "preprocessing")
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing.arff")
os.makedirs(PREPROC_DIR, exist_ok=True)

# ===  Đọc dữ liệu ARFF ===
with open(DATA_PATH, "r") as f:
    data = arff.load(f)
columns = [col[0] for col in data["attributes"]]
df = pd.DataFrame(data["data"], columns=columns).astype(int)
round_digits = 4

# === Lưu file raw_data.csv và bảng ảnh 10 dòng đầu ===
df.to_csv(os.path.join(PREPROC_DIR, "raw_data.csv"), index=False)
df_raw10 = df.head(10).copy()
df_raw10.insert(0, "STT", range(1, 11))
wrapped_columns = ["STT"] + [textwrap.fill(col, width=12) for col in df_raw10.columns if col != "STT"]
fig, ax = plt.subplots(figsize=(30, 8))
ax.axis('off')
table = ax.table(
    cellText=df_raw10.values,
    colLabels=wrapped_columns,
    loc='center',
    cellLoc='center',
    colLoc='center',
    bbox=[0.05, 0, 0.9, 1]
)
table.auto_set_font_size(False)
table.scale(1.2, 2.3)
for pos, cell in table.get_celld().items():
    row, col = pos
    if row == 0:
        cell.set_fontsize(9)
        cell.set_text_props(weight='bold')
    else:
        cell.set_fontsize(17)
plt.tight_layout()
plt.savefig(os.path.join(PREPROC_DIR, "df_head_raw.png"), dpi=300)
plt.close()
print(" Đã lưu raw_data.csv và df_head_raw.png")

# === Chuẩn hóa dữ liệu ===
X_raw = df.drop("Result", axis=1)
y_raw = df["Result"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# === SMOTE ===
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_raw)

# === Chọn 20 đặc trưng bằng SelectKBest ===
selector = SelectKBest(mutual_info_classif, k=20)
selector.fit(X_resampled, y_resampled)
selected_columns = X_raw.columns[selector.get_support()]
X_selected = pd.DataFrame(X_resampled, columns=X_raw.columns)[selected_columns]
df_processed = X_selected.copy()
df_processed["Result"] = y_resampled.values
df_processed_rounded = df_processed.round(round_digits)

# == Lưu processed_data.csv (20 cột + nhãn) và bảng ảnh ===
df_processed_rounded.to_csv(os.path.join(PREPROC_DIR, "processed_data.csv"), index=False)
df_proc10 = df_processed_rounded.head(10).copy()
df_proc10.insert(0, "STT", range(1, 11))
wrapped_proc_cols = ["STT"] + [textwrap.fill(col, width=14) for col in df_proc10.columns if col != "STT"]
fig, ax = plt.subplots(figsize=(30, 8))
ax.axis('off')
table = ax.table(
    cellText=df_proc10.values,
    colLabels=wrapped_proc_cols,
    loc='center',
    cellLoc='center',
    colLoc='center',
    bbox=[0.05, 0, 0.9, 1]
)
table.auto_set_font_size(False)
table.scale(1.2, 2.3)
for pos, cell in table.get_celld().items():
    row, col = pos
    if row == 0:
        cell.set_fontsize(8)
        cell.set_text_props(weight='bold')
    else:
        cell.set_fontsize(10)
plt.tight_layout()
plt.savefig(os.path.join(PREPROC_DIR, "df_head_processed.png"), dpi=300)
plt.close()
print(" Đã lưu processed_data.csv và df_head_processed.png")

# === Biểu đồ phân bố nhãn trước/sau SMOTE ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x=y_raw, hue=y_raw, palette="Set2", legend=False)
plt.title("Trước SMOTE")
plt.xlabel("Result")
plt.ylabel("Số lượng")
plt.xticks([0, 1], ["-1", "1"])
plt.subplot(1, 2, 2)
sns.countplot(x=y_resampled, hue=y_resampled, palette="Set2", legend=False)
plt.title("Sau SMOTE")
plt.xlabel("Result")
plt.ylabel("Số lượng")
plt.xticks([0, 1], ["-1", "1"])
plt.suptitle("Phân bố nhãn Result trước và sau SMOTE", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(PREPROC_DIR, "class_distribution.png"), dpi=300)
plt.close()
print("Đã lưu biểu đồ phân bố nhãn")

# === Lưu danh sách đặc trưng ===
with open(os.path.join(PREPROC_DIR, "selected_features.txt"), "w", encoding="utf-8") as f:
    f.write("20 đặc trưng được chọn bởi SelectKBest:\n\n")
    for i, col in enumerate(selected_columns, 1):
        f.write(f"{i:2d}. {col}\n")
print(" Đã lưu danh sách đặc trưng")

# === Boxplot ===
plt.figure(figsize=(20, 8))
sns.boxplot(data=df_processed[selected_columns], orient="h", palette="Set2")
plt.title("Boxplot của 20 đặc trưng quan trọng nhất", fontsize=14)
plt.xlabel("Giá trị (sau chuẩn hóa)")
plt.ylabel("Đặc trưng")
plt.tight_layout()
plt.savefig(os.path.join(PREPROC_DIR, "boxplot_20_features.png"), dpi=300)
plt.close()
print(" Đã lưu boxplot 20 đặc trưng")

# ===  Heatmap tương quan ===
plt.figure(figsize=(12, 10))
sns.heatmap(df_processed[selected_columns].corr(), cmap="coolwarm", annot=True, fmt=".2f", square=True)
plt.title("Heatmap tương quan giữa 20 đặc trưng quan trọng")
plt.tight_layout()
plt.savefig(os.path.join(PREPROC_DIR, "heatmap_corr_20_features.png"), dpi=300)
plt.close()
print(" Đã lưu heatmap tương quan")

# === 12. In ra console (5 dòng) ===
print("\n DỮ LIỆU GỐC:")
print(f"Số dòng: {df.shape[0]}, Số cột: {df.shape[1]}")
print(df.head(5))

print("\n DỮ LIỆU SAU XỬ LÝ (20 đặc trưng + Result):")
print(f"Số dòng: {df_processed.shape[0]}, Số cột: {df_processed.shape[1]}")
print(df_processed.head(5))
