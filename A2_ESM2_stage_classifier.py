import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# === CONFIG ===
ESM2_MODEL_PATH = "model/esm2_model"
DATA_PATH = "data/cancer_data_preprocessed.csv"
EMBEDDING_CACHE = "data/esm2_embeddings.pt"
OUTPUT_MODEL_PATH = "outputs/esm2_classifier.pt"
EPOCHS = 100
BATCH_SIZE = 32
LR = 2e-4

# === Load model ===
print("📦 Loading ESM2 model and tokenizer from local path...")
tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(ESM2_MODEL_PATH, local_files_only=True)
model.eval()

# === Encode function ===
def encode_batch(seqs, batch_size=32):
    embeddings = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        cleaned_batch = [''.join([aa for aa in s if aa in tokenizer.get_vocab()]) for s in batch]
        inputs = tokenizer(cleaned_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embs = outputs.last_hidden_state.mean(dim=1)
        embeddings.append(batch_embs.cpu())
    return torch.cat(embeddings, dim=0)

# === Load data ===
print("📄 Loading CSV data...")
df = pd.read_csv(DATA_PATH)
df = df[["mutated_protein", "Cancer Stage"]].dropna()

le = LabelEncoder()
df["label"] = le.fit_transform(df["Cancer Stage"])

# === Load or compute embeddings ===
if os.path.exists(EMBEDDING_CACHE):
    print("📂 Found cache. Loading encoded embeddings from disk...")
    cache = torch.load(EMBEDDING_CACHE)
    X, y = cache['X'], cache['y']
else:
    print("🧬 Encoding protein sequences with ESM2 and caching...")
    X = encode_batch(df["mutated_protein"].tolist())
    y = torch.tensor(df["label"].tolist())
    torch.save({'X': X, 'y': y}, EMBEDDING_CACHE)
    print(f"✅ Saved embedding cache to {EMBEDDING_CACHE}")

# === Train/val/test split ===
print("🔀 Splitting dataset into train/val/test...")
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
class ProteinClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

# === Train ===
print("🚀 Starting training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls = ProteinClassifier(X.shape[1], len(le.classes_)).to(device)
optimizer = torch.optim.AdamW(model_cls.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model_cls.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model_cls(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model_cls.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model_cls(xb)
            loss = loss_fn(out, yb)
            val_loss += loss.item()

    print(f"📚 Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# === Evaluate ===
print("🧪 Evaluating on test set...")
model_cls.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model_cls(xb).argmax(dim=1).cpu()
        all_preds.extend(preds)
        all_labels.extend(yb)

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
print(f"✅ Test Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

# === Save model ===
print("💾 Saving model...")
os.makedirs("outputs", exist_ok=True)
torch.save(model_cls.state_dict(), OUTPUT_MODEL_PATH)
print(f"✅ Model saved to {OUTPUT_MODEL_PATH}")