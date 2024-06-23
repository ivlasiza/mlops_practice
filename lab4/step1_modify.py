import numpy as np
import pandas as pd

df_train = pd.read_csv('datasets/titanic-train.csv')
df_test = pd.read_csv('datasets/titanic-test.csv')

print('START: fill empty age column')

train_mean_age = int(np.mean(df_train['Age']))
df_train['Age'] = df_train['Age'].fillna(train_mean_age)

test_mean_age = int(np.mean(df_test['Age']))
df_test['Age'] = df_test['Age'].fillna(test_mean_age)

df_train.to_csv('datasets/titanic-train.csv')
df_test.to_csv('datasets/titanic-test.csv')

print('FINISH: ill empty age column')

