import pandas as pd
from datasets import load_dataset

class EasyData():
  def __init__(self, embedding, method = ["a1", "a2", "a3"]):
    '''
    method: {"a1": only embedding, "a2": only clinical data, "a3": both embedding and clinical data}
    embedding: embedding of protein sequences
    '''
    self.embedding = pd.DataFrame(embedding, columns=[f"emb_{i}" for i in range(embedding.shape[1])])
    self.method = method

  def stage_mapping(self, df):

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

  def load_clinical_data(self, file_path = "seq-to-pheno/TCGA-Cancer-Variant-and-Clinical-Data"):
        df = load_dataset(file_path)
        df = df['train'].to_pandas()
        df_mapped = self.stage_mapping(df)

        clinical_feature = df_mapped[["Donor Age at Diagnosis", "Donor Sex",  "Cancer Type", "Histology Abbreviation", "Mapped Cancer Stage"]]
        clinical_feature = clinical_feature.reset_index(drop=True)
        return clinical_feature

  def prepare_data(self):
      final_dataset = {}

      for i in self.method:
        if i == 'a1':
          final_dataset["a1"] = self.embedding

        elif i == 'a2':
          final_dataset["a2"] = self.load_clinical_data()
        
        elif i == 'a3':
          final_dataset["a3"] = pd.concat([self.embedding, self.load_clinical_data()], axis=1)
      
      return final_dataset