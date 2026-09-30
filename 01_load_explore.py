import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.to_csv('iris.csv', index=False)

print("Shape of dataset (rows, columns):", df.shape)
print(df.head())
print(df['species'].value_counts())
print(df.describe())
