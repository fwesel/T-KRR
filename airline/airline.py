import pandas as pd
import numpy as np

df = pd.read_pickle('airline.pickle')
df = df.drop('Year',axis=1)
df = df.drop('Unnamed: 0',axis=1)
temp = df.pop('ArrDelay')
df['ArrDelay'] = temp
df['ArrTime'] = 60*np.floor(df['ArrTime']/100)+np.mod(df['ArrTime'], 100)
df['DepTime'] = 60*np.floor(df['DepTime']/100)+np.mod(df['DepTime'], 100)
df = df.astype(int)
df.to_csv("airline.csv",index=False,header=True)