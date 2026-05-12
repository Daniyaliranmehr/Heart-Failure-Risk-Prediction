# Heart Failure Prediction using MLP Neural Network

## Overview

This project uses a Multi-Layer Perceptron (MLP) neural network to predict heart failure survival outcomes based on clinical records.

The objective is to apply deep learning techniques to healthcare data and evaluate the effectiveness of MLP models for medical risk prediction.

## Dataset

The dataset used in this project contains clinical records of heart failure patients.  
It includes multiple medical features that can help predict patient survival outcomes.

### Dataset Information

- Number of instances: 249
- Number of features: 12
- Target variable: `DEATH_EVENT`
- Task type: Binary Classification

### Features

| Feature | Description |
|---|---|
| age | Age of the patient |
| anaemia | Decrease of red blood cells or hemoglobin |
| creatinine_phosphokinase | Level of CPK enzyme in the blood |
| diabetes | Whether the patient has diabetes |
| ejection_fraction | Percentage of blood leaving the heart at each contraction |
| high_blood_pressure | Whether the patient has hypertension |
| platelets | Platelet count in the blood |
| serum_creatinine | Level of serum creatinine in the blood |
| serum_sodium | Level of serum sodium in the blood |
| sex | Gender of the patient |
| smoking | Whether the patient smokes |
| time | Follow-up period (days) |

### Target Variable

| Target | Meaning |
|---|---|
| 0 | Patient survived |
| 1 | Patient died |

## Project Structure

```text
heart-failure-risk-prediction/
├─ dataset/
│  ├─ test.csv
│  ├─ test_with_predictions.csv
│  ├─ train.csv
├─ experiments/
│  ├─ activation_function_experiment.ipynb
│  ├─ batch_size_experiment.ipynb
│  ├─ layer_experiment.ipynb
│  ├─ learning_rate_experiment.ipynb
├─ images
├─ saved_values
│  ├─ model.pth
│  ├─ variables.pt
├─ .gitignore
├─ dataset.zip  
├─ test.ipynb
├─ train.ipynb
```

### File Descriptions

| File / Folder | Description |
|---|---|
| `dataset/` | Contains dataset files used for training and testing |
| `dataset/train.csv` | Training dataset used for model learning |
| `dataset/test.csv` | Unlabeled dataset |
| `dataset/test_with_predictions.csv` | test.csv with labels |
| `experiments/` | Jupyter notebooks for hyperparameter experiments |
| `experiments/activation_function_experiment.ipynb` |  |
| `experiments/batch_size_experiment.ipynb` |  |
| `experiments/layer_experiment.ipynb` |  |
| `experiments/learning_rate_experiment.ipynb` |  |
| `images/` | Designed images for the project |
| `saved_values/` | Saved model and frequently used variables shared across project files |
| `saved_values/model.pth` | Trained model |
| `saved_values/variables.pt` | Saved variables |
| `train.ipynb` | Main notebook for training the MLP model |
| `test.ipynb` | Notebook for predictions |
| `dataset.zip` | Compressed version of the dataset folder |
