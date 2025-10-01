import pandas as pd

df = pd.read_csv(r"C:\Users\Ali\PycharmProjects\PythonProject43\diamonds (cleaned).csv")

print(df.shape)
print(df.isnull().sum())