from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle

# Загрузка обученной модели из файла
model = pickle.load(open('models/model.pkl', 'rb'))

# Загрузка тестовых данных из CSV файла
df_test = pd.read_csv('test/test_result.csv')

# Выделение признаков (температура) и целевой переменной (наличие шума) из тестового набора данных
X_test = df_test[['temp']]
y_test = df_test['is_noize']

# Предсказание меток классов для тестового набора данных
predict = model.predict(X_test)

# Вычисление и вывод метрик модели
print("Metrics:")
print(f"Accuracy = {accuracy_score(y_test, predict)}")
print(f"Precision = {precision_score(y_test, predict)}")
print(f"Recall = {recall_score(y_test, predict)}")
print(f"F1-score = {f1_score(y_test, predict)}")

# Завершение работы пайплайна
print("****** Pipeline finished ******")
