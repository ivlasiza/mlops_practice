import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_handler(train_path, test_path):
    """
    Функция для предварительной обработки данных.
    
    Параметры:
    - train_path: путь к файлу с тренировочными данными
    - test_path: путь к файлу с тестовыми данными
    
    Функция выполняет стандартизацию столбца 'temp' и сохраняет результаты в новые файлы.
    """
    # Загрузка тренировочных данных из CSV файла
    train_df = pd.read_csv(train_path)
    # Загрузка тестовых данных из CSV файла
    test_df = pd.read_csv(test_path)

    # Создание объекта StandardScaler и обучение на тренировочных данных
    scaler = StandardScaler()
    scaler.fit(train_df[['temp']])

    # Применение стандартизации к тренировочным данным
    train_scaled_data = scaler.transform(train_df[['temp']])
    # Применение стандартизации к тестовым данным
    test_scaled_data = scaler.transform(test_df[['temp']])

    # Замена оригинальных данных на стандартизированные в тренировочном наборе данных
    train_df['temp'] = train_scaled_data
    # Сохранение стандартизированных тренировочных данных в новый файл
    train_df.to_csv('train/train_result.csv', index=False)

    # Замена оригинальных данных на стандартизированные в тестовом наборе данных
    test_df['temp'] = test_scaled_data
    # Сохранение стандартизированных тестовых данных в новый файл
    test_df.to_csv('test/test_result.csv', index=False)

# Начало процесса предварительной обработки данных
print('start: preprocess data')
preprocess_handler(train_path='train/train_data.csv', test_path='test/test_data.csv')
# Завершение процесса предварительной обработки данных
print('finish: preprocess data')
