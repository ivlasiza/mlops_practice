import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Начало подготовки модели
print('Start: model preparation')

# Загрузка тренировочных данных из CSV файла
df_train = pd.read_csv('train/train_result.csv')

# Выделение признаков (температура) и целевой переменной (наличие шума)
X_train = df_train[['temp']]
y_train = df_train['is_noize']

# Создание и обучение логистической регрессии на тренировочных данных
model = LogisticRegression()
model.fit(X_train, y_train)

# Сохранение обученной модели в файл с помощью pickle
pickle.dump(model, open('models/model.pkl', 'wb'))

# Завершение подготовки модели
print('Finish: model preparation')
