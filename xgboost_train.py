import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier, plot_importance
import sys

# === Add utils folder to path and import helpers ===
sys.path.append("Utils")
from tools import clean_grade, stratified_splitting

# === Step 1: Load preprocessed CSV ===
df = pd.read_csv("data/cancer_data_preprocessed.csv")
print(f"Loaded clinical data: {df.shape}")

# === Step 2: Clean tumor grade ===
df["Tumour Grade"] = df["Tumour Grade"].apply(clean_grade)

# === Step 3: Load aligned ESM2 embeddings ===
esm2 = torch.load("esm2_embeddings.pt")
X_embed = esm2["X"].numpy()
assert len(df) == X_embed.shape[0], "Mismatch between CSV and ESM2 embeddings row count!"

# === Step 4: Process clinical features ===
categorical_cols = ["Donor Sex", "Tumour Grade"]
numeric_cols = ["Donor Age at Diagnosis"]

ct = ColumnTransformer([
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

X_clinical = ct.fit_transform(df[categorical_cols + numeric_cols])

# === Step 5: Combine features ===
X = np.hstack([X_clinical, X_embed])
y = df["Cancer Stage"].astype(int) - 1  # Convert stages 1–4 to 0–3

# === Step 6: Split data ===
dataset, labels = stratified_splitting(X, y)
X_train, y_train = np.array(dataset['train']), np.array(labels['train'])
X_test, y_test = np.array(dataset['test']), np.array(labels['test'])

# === Step 7: Train XGBoost ===
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# === Step 8: Predict and evaluate ===
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1-score: {f1_score(y_test, y_pred, average='macro'):.4f}")

# === Step 9: ROC Curve ===
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
y_score = model.predict_proba(X_test)
fpr, tpr, roc_auc = {}, {}, {}

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(4):
    plt.plot(fpr[i], tpr[i], label=f"Stage {i+1} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# === Step 10: Feature Importance Plot ===
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=15, height=0.4)
plt.title("Top XGBoost Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
