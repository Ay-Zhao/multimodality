import pandas as pd

df = pd.read_csv("bace.csv")

label_col = "TARGET"   # change this to your label column name

print(df[label_col].value_counts())
print("\nPercentage distribution:")
print(df[label_col].value_counts(normalize=True) * 100)
