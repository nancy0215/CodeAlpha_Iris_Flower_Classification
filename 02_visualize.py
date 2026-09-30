import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
sns.pairplot(df, hue='species', height=1.8)
plt.savefig('iris_pairplot.png', dpi=120, bbox_inches='tight')
print("Saved iris_pairplot.png")
