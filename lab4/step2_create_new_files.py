import pandas as pd

print("START: one hot encoding for Sex column")

df_train = pd.read_csv('datasets/titanic-train.csv')
df_test = pd.read_csv('datasets/titanic-test.csv')

df_train_ext = pd.get_dummies(df_train, columns=['Sex'], drop_first=True)
df_test_ext = pd.get_dummies(df_test, columns=['Sex'], drop_first=True)

df_train_ext.to_csv('datasets/titanic-train-ext.csv')
df_test_ext.to_csv('datasets/titanic-test-ext.csv')

print("FINISH: one hot encoding for Sex column")
