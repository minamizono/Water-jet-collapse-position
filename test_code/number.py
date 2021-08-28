import pandas as pd
df = pd.DataFrame({'salary_1':[230,345,222],'salary_2':[235,375,292],'salary_3':[210,385,260]})
print(df.describe())

df2 = df.T
df3=df2.describe().T

df3.to_csv('test.csv')