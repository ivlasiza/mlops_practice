import numpy as np
import pandas as pd

# Загрузка тренировочного и тестового наборов данных из CSV файлов
df_train = pd.read_csv('datasets/titanic-train.csv')
df_test = pd.read_csv('datasets/titanic-test.csv')

print('START: fill empty age column')

# Вычисление среднего возраста для тренировочного набора данных
train_mean_age = int(np.mean(df_train['Age']))
# Заполнение пропущенных значений в колонке 'Age' среднего возраста в тренировочном наборе данных
df_train['Age'] = df_train['Age'].fillna(train_mean_age)

# Вычисление среднего возраста для тестового набора данных
test_mean_age = int(np.mean(df_test['Age']))
# Заполнение пропущенных значений в колонке 'Age' среднего возраста в тестовом наборе данных
df_test['Age'] = df_test['Age'].fillna(test_mean_age)

# Сохранение обновленных тренировочного и тестового наборов данных обратно в CSV файлы
df_train.to_csv('datasets/titanic-train.csv', index=False)
df_test.to_csv('datasets/titanic-test.csv', index=False)

print('FINISH: fill empty age column')

