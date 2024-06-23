import pandas as pd
from catboost import datasets

all_datasets = datasets.titanic()
df_train = pd.DataFrame(all_datasets[0])
df_test = pd.DataFrame(all_datasets[1])
df_train.to_csv('datasets/titanic-train.csv')
df_test.to_csv('datasets/titanic-test.csv')
