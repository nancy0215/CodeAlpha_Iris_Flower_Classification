# CodeAlpha - Iris Flower Classification

**Data Science Internship - Task 1**

## Overview
This project trains a machine learning model to classify Iris flowers into three species
(setosa, versicolor, virginica) based on four measurements: sepal length, sepal width,
petal length, and petal width.

## Dataset
The classic Iris dataset (Fisher, 1936) — 150 flower samples, 50 of each species,
loaded via `scikit-learn`.

## Approach
1. **Data Exploration** (`01_load_explore.py`) — loaded the dataset and examined summary
   statistics and class balance.
2. **Visualization** (`02_visualize.py`) — plotted pairwise relationships between features
   by species to understand separability.
3. **Model Training** (`03_train_model.py`) — split data 80/20 into train/test sets,
   trained a Random Forest Classifier, and evaluated performance.

## Results
- **Test accuracy: 90%**
- Setosa was classified perfectly (10/10) — it's linearly separable from the other species
  based on petal measurements.
- Versicolor and virginica showed minor overlap (a few misclassifications between them),
  consistent with their closer feature distributions observed during visualization.

## Files
| File | Description |
|---|---|
| `01_load_explore.py` | Loads dataset, prints summary stats |
| `02_visualize.py` | Generates pairplot visualization |
| `03_train_model.py` | Trains and evaluates the classifier |
| `iris.csv` | Dataset used |
| `iris_pairplot.png` | Feature relationship visualization |
| `iris_model.pkl` | Saved trained model |
| `label_encoder.pkl` | Saved label encoder for species names |

## Tools Used
Python, Pandas, Scikit-learn, Seaborn, Matplotlib

## How to Run
```bash
pip install pandas scikit-learn seaborn matplotlib joblib
python 01_load_explore.py
python 02_visualize.py
python 03_train_model.py
```

---
*Part of the CodeAlpha Data Science Internship*
