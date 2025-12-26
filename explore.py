import pandas as pd

df = pd.read_csv("dataset.csv", encoding="latin1")

print(df.head())
print("\nColumns:", df.columns)
print("\nLabel Distribution:")
print(df['labels'].value_counts())
