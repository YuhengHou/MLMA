import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

# === CONFIG ===
DATA_PATH = "data/cancer_data_preprocessed.csv"
OUTPUT_MODEL_PATH = "outputs/clinical_classifier.pt"
EPOCHS = 50
BATCH_SIZE = 32
LR = 2e-4

# === Load data ===
print("📄 Loading clinical data...")
df = pd.read_csv(DATA_PATH)
df = df[["Donor Age at Diagnosis", "Donor Sex", "Tumour Grade", "Cancer Stage"]].dropna()

# === Encode labels ===
le = LabelEncoder()
df["label"] = le.fit_transform(df["Cancer Stage"])

# === Process clinical features ===
categorical_cols = ["Donor Sex", "Tumour Grade"]
numeric_cols = ["Donor Age at Diagnosis"]

# One-hot encode categorical
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_features = ohe.fit_transform(df[categorical_cols])

# Standardize numeric
scaler = StandardScaler()
num_features = scaler.fit_transform(df[numeric_cols])

# Combine features
import numpy as np
X = torch.tensor(np.hstack([num_features, cat_features]), dtype=torch.float32)
y = torch.tensor(df["label"].values)

# === Train/val/test split ===
print("🔀 Splitting dataset...")
X_np, y_np = X.numpy(), y.numpy()
X_train, X_temp, y_train, y_temp = train_test_split(X_np, y_np, test_size=0.3, stratify=y_np, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# === Classifier ===
class ClinicalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# === Train ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClinicalClassifier(X.shape[1], len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

print("🚀 Training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# === Evaluate ===
print("🧪 Evaluating...")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).argmax(dim=1).cpu()
        all_preds.extend(preds)
        all_labels.extend(yb)

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
print(f"✅ Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

# === Save ===
os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
print(f"✅ Model saved to {OUTPUT_MODEL_PATH}")
