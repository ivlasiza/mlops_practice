import pandas as pd
from catboost import datasets

# Загрузка данных Titanic из catboost.datasets
all_datasets = datasets.titanic()

# Преобразование тренировочного и тестового наборов данных в DataFrame
df_train = pd.DataFrame(all_datasets[0])
df_test = pd.DataFrame(all_datasets[1])

# Сохранение тренировочного набора данных в файл 'titanic-train.csv'
df_train.to_csv('datasets/titanic-train.csv', index=False)

# Сохранение тестового набора данных в файл 'titanic-test.csv'
df_test.to_csv('datasets/titanic-test.csv', index=False)
