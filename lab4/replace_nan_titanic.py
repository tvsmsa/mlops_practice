import pandas as pd

df = pd.read_csv("lab4/titanic_new.csv")
df_age_mean = df['Age'].mean()
df['Age'] = df['Age'].fillna(df_age_mean)
df.to_csv('lab4/titanic_new.csv', index=False)