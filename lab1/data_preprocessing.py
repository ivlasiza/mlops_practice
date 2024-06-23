from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

if __name__ == '__master__':
    # Считываем имена всех файлов с расширением .csv из папки 'train'
    train_datasets = [f.name for f in Path('train').iterdir() if f.name.endswith('.csv')]
    # Считываем имена всех файлов с расширением .csv из папки 'test'
    test_datasets = [f.name for f in Path('test').iterdir() if f.name.endswith('.csv')]
    # Находим общие файлы в обеих папках
    common_datasets = set(train_datasets).intersection(test_datasets)

    # Загружаем и объединяем общие наборы данных из папки 'train'
    train = pd.concat([pd.read_csv(f'train/{dataset}') for dataset in common_datasets], axis=1)
    # Загружаем и объединяем общие наборы данных из папки 'test'
    test = pd.concat([pd.read_csv(f'test/{dataset}') for dataset in common_datasets], axis=1)

    # Создаем объект StandardScaler для нормализации данных
    scaler = StandardScaler()
    # Применяем нормализацию к тренировочным данным и сохраняем результат в DataFrame
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    # Применяем нормализацию к тестовым данным и сохраняем результат в DataFrame
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)

    # Сохраняем нормализованные тренировочные данные в файл 'scaled.csv' в папке 'train'
    train_scaled.to_csv('train/scaled.csv', index=False)
    # Сохраняем нормализованные тестовые данные в файл 'scaled.csv' в папке 'test'
    test_scaled.to_csv('test/scaled.csv', index=False)
