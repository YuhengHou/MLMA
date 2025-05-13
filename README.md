# MLMA Final Project: Cancer Stage Classification with Genetic and Clinical Data

## Project Structure

- `A2_ESM2*` and `A2_ProtBert.ipynb`: Preprocessing and embedding generation using ESM2 and ProtBert models
- `A3_baseline.ipynb`, `A3_MLP.ipynb`, `A3_XGBoost.ipynb`: Classification experiments (baseline, MLP, XGBoost)
- `DNN_Downstream_classifier.ipynb`: Full training and evaluation pipeline for DNNs using embeddings
- `DNN_Embedding_Visualization.ipynb`: DNN t-SNE or PCA plots for visualizing learned embeddings
- `DNN_Final_Plot_Result.ipynb`: DNN Final performance summaries and visual plots
- `MLMA_Final_proj_TreeModels.ipynb`: Tree-based classification pipeline (XGBoost) with comparison
- `preprocess_protein_data.py`: Protein sequence and clinical metadata preprocessing script
- `esm2_embeddings.pt`, `esm2_delta_embeddings.pt`: Precomputed embeddings from ESM2 models
- `xgboost_train.py`: Training pipeline using XGBoost on processed features
- `xgboost_feature_importance.png`, `xgboost_roc_curve.png`: Visualizations of tree-based model outputs
-  `Result_plot.py`: Final Notebook scripts to generate results and figures
- `data_analyse.ipynb`: Exploratory data analysis and feature review


## How to Run

1. Ensure `esm2_embeddings.pt` and `esm2_delta_embeddings.pt` are generated
2. Run `preprocess_protein_data.py` to prepare the input dataset
3. Choose a model training notebook (`A3_XGBoost.ipynb`, `DNN_Downstream_classifier.ipynb`, etc.)
4. Use `Result_plot.py` or `DNN_Final_Plot_Result.ipynb` to generate final figures and evaluation metrics

## Dependencies

- Python 3.8+
- PyTorch
- scikit-learn
- XGBoost
- matplotlib / seaborn
- pandas, numpy

## Output

- Performance plots (ROC curves, feature importance)
- Classification reports (accuracy, F1 score)
- Embedded representations for interpretability

## Contributors

- Eric Wang
- Xunyu Shen
- Yuheng Hou
- Yuxuan Sun
