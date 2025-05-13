import pandas as pd
import torch
import sklearn
from packaging import version
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load your preprocessed data
df = pd.read_csv("data/cancer_data_preprocessed.csv")

# Choose correct encoder argument depending on sklearn version
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
else:
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

# Define clinical columns
categorical_cols = ["Donor Sex", "Tumour Grade"]
numeric_cols = ["Donor Age at Diagnosis"]

# Fit column transformer to clinical features
ct = ColumnTransformer([
    ("cat", encoder, categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])
ct.fit(df[categorical_cols + numeric_cols])

# Get full list of feature names
cat_names = ct.named_transformers_["cat"].get_feature_names_out(categorical_cols)
num_names = numeric_cols
clinical_feature_names = list(cat_names) + num_names
total_feature_names = clinical_feature_names + [f"ESM2_{i}" for i in range(320)]

# Show what f7 refers to
target_index = 7
print(f"🔍 f{target_index} → {total_feature_names[target_index]}")
