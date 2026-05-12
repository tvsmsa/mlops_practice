import pandas as pd
df = pd.read_csv("lab4/titanic_new.csv")
one_hot = pd.get_dummies(df['Sex'])
df = df.drop('Sex',axis = 1)
df = df.join(one_hot)
df.to_csv('lab4/titanic_new.csv', index=False)