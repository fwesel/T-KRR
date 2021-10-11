import pandas as pd
import numpy as np

columns = ['AGE','WORKCLASS','FNLWGT','EDUCATION','EDUCATION-NUM','MARITAL-STATUS','OCCUPATION','RELATIONSHIP','RACE','SEX','CAPITAL-GAIN','CAPITAL-LOSS','HOURS-PER-WEEK','NATIVE-COUNTRY','TARGET']
df = pd.read_csv('adultTrain.csv',delimiter=';',names=columns,na_values=['?',' ?'])
df = pd.concat([df,pd.read_csv('adultTest.csv',delimiter=';',names=columns,na_values=['?',' ?'])],ignore_index=True,sort=False,axis=0)
df['TARGET'] = df['TARGET'].str.rstrip('.')
df = df.dropna()
continousLabels = ['AGE','FNLWGT','EDUCATION-NUM','CAPITAL-GAIN','CAPITAL-LOSS','HOURS-PER-WEEK']
categoricalLabels = list(set(set(columns)-set(continousLabels)))
categoricalLabels.append(categoricalLabels.pop(categoricalLabels.index('TARGET')))
df = pd.get_dummies(df,prefix=categoricalLabels,columns=categoricalLabels,drop_first=True)
df.to_csv('adult.csv',header=True,index=False)