import os
import arff
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# === 1. Đường dẫn ===
BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing.arff")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# === 2. Đọc dữ liệu ===
with open(DATA_PATH, "r") as f:
    data = arff.load(f)
columns = [col[0] for col in data["attributes"]]
df = pd.DataFrame(data["data"], columns=columns).astype(int)

# === 3. Hiển thị 10 dòng đầu và lưu bảng ===
df_10 = df.head(10).copy()
df_10.insert(0, "STT", range(1, 11))
wrapped_columns = ["STT"] + [textwrap.fill(col, width=12) for col in columns]

fig, ax = plt.subplots(figsize=(30, 8))
ax.axis('off')
table = ax.table(
    cellText=df_10.values,
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
        cell.set_fontsize(8)
        cell.set_text_props(weight='bold')
    else:
        cell.set_fontsize(10)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "df_head_input.png"), dpi=300)
plt.close()
print("✅ Đã lưu bảng dữ liệu mẫu.")

# === 4. Tách dữ liệu và chuẩn hoá ===
X_raw = df.drop("Result", axis=1)
y_raw = df["Result"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# === 5. Áp dụng SMOTE ===
smote = SMOTE(random_state=42)
X_final, y_final = smote.fit_resample(X_scaled, y_raw)

# === 6. Biểu đồ phân bố nhãn ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x=y_raw, hue=y_raw, palette="Set2", legend=False)
plt.title("Trước SMOTE")
plt.xlabel("Result")
plt.ylabel("Số lượng")
plt.xticks([0, 1], ["-1", "1"])

plt.subplot(1, 2, 2)
sns.countplot(x=y_final, hue=y_final, palette="Set2", legend=False)
plt.title("Sau SMOTE")
plt.xlabel("Result")
plt.ylabel("Số lượng")
plt.xticks([0, 1], ["-1", "1"])

plt.suptitle("Phân bố nhãn Result trước và sau SMOTE", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(RESULT_DIR, "class_distribution.png"), dpi=300)
plt.close()
print("✅ Đã lưu hình phân bố nhãn.")

# === 7. Chọn 20 đặc trưng bằng SelectKBest ===
selector = SelectKBest(mutual_info_classif, k=20)
selector.fit(X_raw, y_raw)
selected_columns = X_raw.columns[selector.get_support()]

# === 8. Lưu danh sách đặc trưng ===
with open(os.path.join(RESULT_DIR, "selected_features.txt"), "w", encoding="utf-8") as f:
    f.write("20 đặc trưng được chọn bởi SelectKBest:\n\n")
    for i, col in enumerate(selected_columns, 1):
        f.write(f"{i:2d}. {col}\n")
print("✅ Đã lưu danh sách đặc trưng vào selected_features.txt")

# === 9. Vẽ Boxplot cho 20 đặc trưng trên cùng một biểu đồ ===
plt.figure(figsize=(20, 8))
sns.boxplot(data=df[selected_columns], orient="h", palette="Set2")  # Vẽ ngang
plt.title("Boxplot của 20 đặc trưng quan trọng nhất", fontsize=14)
plt.xlabel("Giá trị")
plt.ylabel("Thuộc tính")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "boxplot_20_features.png"), dpi=300)
plt.close()

print("✅ Đã lưu hình boxplot 20 đặc trưng (gộp).")

# === 10. Heatmap tương quan ===
plt.figure(figsize=(12, 10))
sns.heatmap(df[selected_columns].corr(), cmap="coolwarm", annot=False, fmt=".2f", square=True)
plt.title("Heatmap tương quan giữa 20 đặc trưng quan trọng")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "heatmap_corr_20_features.png"), dpi=300)
plt.close()
print("✅ Đã lưu heatmap tương quan giữa 20 đặc trưng.")
