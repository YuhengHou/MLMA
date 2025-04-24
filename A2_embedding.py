# generate_mutated_wildtype_embeddings.py

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import gc

# === CONFIG ===
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

ESM2_MODEL_PATH = "model/esm2_t33_650M_UR50D"
DATA_PATH = "data/cancer_data_preprocessed.csv"
SAVE_MUT_ONLY = "data/esm2_embeddings.pt"
SAVE_DELTA = "data/esm2_delta_embeddings.pt"
SAVE_DELTA_FINETUNED = "data/finetuned_delta_embeddings.pt"
MAX_LEN = 4096
BATCH_SIZE = 4  # Safest default
EPOCHS = 5
LR = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
print("📦 Loading ESM2 model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(ESM2_MODEL_PATH, local_files_only=True).to(device)
model.eval()

# === Chunked embed for long sequences ===
def chunked_embed(text):
    ids = tokenizer(text, return_tensors='pt', truncation=False)["input_ids"].squeeze(0)
    chunks = [ids[i:i+MAX_LEN] for i in range(0, len(ids), MAX_LEN)]
    outs = []
    for chunk in chunks:
        inputs = {"input_ids": chunk.unsqueeze(0).to(device)}
        with torch.no_grad():
            out = model(**inputs).last_hidden_state.mean(dim=1)
        outs.append(out.cpu())
        torch.cuda.empty_cache()
    return torch.mean(torch.stack(outs), dim=0)

# === Load data ===
df = pd.read_csv(DATA_PATH)
df = df[["mutated_protein", "wildtype_protein", "Cancer Stage"]].dropna()
le = LabelEncoder()
df["label"] = le.fit_transform(df["Cancer Stage"])

class DeltaProteinDataset(Dataset):
    def __init__(self, df):
        self.mut = df["mutated_protein"].tolist()
        self.wt = df["wildtype_protein"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mut[idx], self.wt[idx], torch.tensor(self.labels[idx])

loader = DataLoader(DeltaProteinDataset(df), batch_size=BATCH_SIZE)

# === 1. mutated_protein only ===
def export_mutated_only():
    print("🔹 Exporting mutated_protein embeddings...")
    embeddings, labels = [], []
    for mut_seq, _, label in tqdm(loader):
        for m, l in zip(mut_seq, label):
            emb = chunked_embed(m)
            embeddings.append(emb)
            labels.append(l)
    X = torch.stack(embeddings)
    y = torch.tensor(labels)
    torch.save({'X': X, 'y': y}, SAVE_MUT_ONLY)

# === 2. delta embedding only ===
def export_delta():
    print("🔹 Exporting delta (mut - wt) embeddings...")
    embeddings, labels = [], []
    for mut_seq, wt_seq, label in tqdm(loader):
        for m, w, l in zip(mut_seq, wt_seq, label):
            mut_emb = chunked_embed(m)
            wt_emb = chunked_embed(w)
            delta = mut_emb - wt_emb
            embeddings.append(delta)
            labels.append(l)
    X = torch.stack(embeddings)
    y = torch.tensor(labels)
    torch.save({'X': X, 'y': y}, SAVE_DELTA)

# === 3. delta + finetune for stage ===
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def export_delta_finetuned():
    print("🔹 Fine-tuning delta embedding with MLP classifier...")

    model_cls = SimpleMLP(model.config.hidden_size, len(le.classes_)).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(model_cls.parameters()), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for mut_seq, wt_seq, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            for m, w, l in zip(mut_seq, wt_seq, labels):
                mut_emb = chunked_embed(m)
                wt_emb = chunked_embed(w)
                delta = mut_emb - wt_emb

                # ensure label is LongTensor on correct device with shape [1]
                if isinstance(l, torch.Tensor):
                    l_tensor = l.view(1).long().to(device)
                else:
                    l_tensor = torch.tensor([l], dtype=torch.long, device=device)

                logits = model_cls(delta.unsqueeze(0).to(device))
                loss = loss_fn(logits, l_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                torch.cuda.empty_cache()

        print(f"✅ Epoch {epoch+1} - Loss: {total_loss:.4f}")

    # === Save finetuned embeddings ===
    model.eval(); model_cls.eval()
    all_delta, all_labels = [], []
    for mut_seq, wt_seq, labels in tqdm(loader, desc="Saving"):
        for m, w, l in zip(mut_seq, wt_seq, labels):
            mut_emb = chunked_embed(m)
            wt_emb = chunked_embed(w)
            delta = mut_emb - wt_emb
            all_delta.append(delta)
            all_labels.append(int(l))  # convert to raw int

    X = torch.stack(all_delta)
    y = torch.tensor(all_labels, dtype=torch.long)
    torch.save({'X': X, 'y': y}, SAVE_DELTA_FINETUNED)
    print(f"✅ Saved to {SAVE_DELTA_FINETUNED}")

# === Run All ===
# export_mutated_only()
# torch.cuda.empty_cache(); gc.collect()

# export_delta()
# torch.cuda.empty_cache(); gc.collect()

export_delta_finetuned()
torch.cuda.empty_cache(); gc.collect()
