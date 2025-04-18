import pandas as pd
import numpy as np

def stage_mapping(df):

    stage_mapping = {
        "I": "1", "1": "1", "1a": "1", "1b": "1", "T1aN0M0": "1", "T1bN0M0": "1", "T1bNXMX": "1", "A": "1",
        "II": "2", "2": "2", "2a": "2", "T2N0MX": "2", "T2N1MX": "2", "T2aNXMX": "2", "B": "2",
        "III": "3", "3": "3", "3b": "3", "3c": "3", "C": "3",
        "T3N0MX": "3", "T3N1MX": "3", "T3aNXMX": "3", "T3aN0MX": "3", "T3N1bMX": "3", "T3NXMX": "3",
        "T3aN0M0": "3", "T3bNXMX": "3", "T3N1aMX": "3", "T3N1M0": "3",
        "IV": "4", "4": "4", "T4N1M1": "4", "T4N1bM1": "4", "T3N1M1": "4", "T2N1M1": "4",
        "unknown": "unknown"
    }
    df["Mapped Cancer Stage"] = df["Cancer Stage"].apply(lambda s: stage_mapping.get(str(s).strip(), "unknown"))

    required_columns = [
        "mutated_protein", "wildtype_protein",
        "Donor Age at Diagnosis", "Donor Sex", "Tumour Grade",
        "Donor Vital Status", "Donor Survival Time",
        "Cancer Type", "Histology Abbreviation",
        "Mapped Cancer Stage"
    ]

    df_fusion = df[required_columns].dropna()
    df_fusion = df_fusion[df_fusion["Mapped Cancer Stage"].isin(["1", "2", "3", "4"])]
    df_fusion = df_fusion[df_fusion["mutated_protein"].str.len() > 0]
    df_fusion = df_fusion[df_fusion["wildtype_protein"].str.len() > 0]
    return df_fusion

def clean_grade(s):
    s = str(s).strip().upper()
    if "WELL" in s or s.startswith("1"):
        return "Grade 1"
    elif "MODERATE" in s or s.startswith("2"):
        return "Grade 2"
    elif "POOR" in s or s.startswith("3"):
        return "Grade 3"
    elif "UNDIFFERENTIATED" in s or s.startswith("4"):
        return "Grade 4"
    elif "I-II" in s:
        return "Grade 1-2"
    elif "II-III" in s:
        return "Grade 2-3"
    elif s == "I":
        return "Grade 1"
    elif s == "II":
        return "Grade 2"
    elif s == "III":
        return "Grade 3"
    else:
        return s

def stratified_splitting(full_dataset, label_array, classes=[0,1,2,3],
                         train_size=0.7, val_size=0.1, test_size=0.2, random_state=None):    
    if random_state is not None:
        np.random.seed(random_state)

    train_idx, val_idx, test_idx = [], [], []

    for label in classes:
        class_indices = np.where(label_array == label)[0]
        np.random.shuffle(class_indices)

        n_total = len(class_indices)
        n_train = int(n_total * train_size)
        n_val = int(n_total * val_size)
        n_test = n_total - n_train - n_val

        train_idx.extend(class_indices[:n_train])
        val_idx.extend(class_indices[n_train:n_train + n_val])
        test_idx.extend(class_indices[n_train + n_val:])

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    dataset_subsets = {
        'train': [full_dataset[i] for i in train_idx],
        'val': [full_dataset[i] for i in val_idx],
        'test': [full_dataset[i] for i in test_idx]
    }

    label_subsets = {
        'train': [label_array[i] for i in train_idx],
        'val': [label_array[i] for i in val_idx],
        'test': [label_array[i] for i in test_idx]
    }

    return dataset_subsets, label_subsets
