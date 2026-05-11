import pandas as pd

df = pd.read_csv("lab4/Titanic-Dataset.csv")
df_new = df[['Pclass', 'Sex', 'Age']]
df_new.to_csv('lab4/titanic_new.csv', index=False)