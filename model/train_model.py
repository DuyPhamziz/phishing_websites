import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Tính đường dẫn tuyệt đối an toàn từ chính file này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "phishing.arff")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "..", "result")

# Tạo thư mục nếu chưa có
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Tải dữ liệu từ file .arff
with open(DATA_PATH, "r") as f:
    data = arff.load(f)
columns = [col[0] for col in data["attributes"]]
df = pd.DataFrame(data["data"], columns=columns).astype(int)

# Tách dữ liệu
X = df.drop("Result", axis=1)
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

# Huấn luyện & đánh giá từng mô hình
results = []
os.makedirs("../model/models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Lưu mô hình
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))


    # Ghi kết quả
    results.append(f"----- {name} -----\nAccuracy: {acc:.4f}\n{report}\n")

# Voting Classifier (Ensemble)
voting_model = VotingClassifier(estimators=[
    ("dt", models["DecisionTree"]),
    ("rf", models["RandomForest"]),
    ("lr", models["LogisticRegression"]),
], voting="hard")

voting_model.fit(X_train, y_train)
y_pred = voting_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
joblib.dump(voting_model, "../model/models/VotingClassifier.pkl")
results.append(f"----- VotingClassifier -----\nAccuracy: {acc:.4f}\n{report}\n")

# Ghi toàn bộ vào file
os.makedirs("../result", exist_ok=True)
with open(os.path.join(RESULT_DIR, "evaluation.txt"), "w") as f:

    f.writelines(results)

print("Huấn luyện xong. Kết quả lưu tại result/evaluation.txt")
