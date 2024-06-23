import numpy as np
import pandas as pd
import os

def generate_data(items_cnt=1000, base_value=15, noize_coeff=0.2):
    """
    Функция генерирует данные с шумом.
    
    Параметры:
    - items_cnt: количество основных элементов данных
    - base_value: базовое значение для нормального распределения
    - noize_coeff: коэффициент шума
    
    Возвращает:
    - result_data: словарь с двумя ключами 'temp' и 'is_noize', содержащими значения температуры и метки шума соответственно
    """
    # Генерация основных данных на основе нормального распределения
    base_data = np.random.normal(loc=base_value, size=items_cnt)
    # Генерация шумовых данных
    noize_data = np.random.normal(loc=base_value*3, scale=noize_coeff*100, size=(int(items_cnt*noize_coeff)))
    
    # Создание массива для меток шума (0 - основной данные, 1 - шумовые данные)
    is_noize = np.zeros(len(base_data) + len(noize_data), dtype=int)
    is_noize[items_cnt:] = 1

    # Объединение основных и шумовых данных в один словарь
    result_data = {
        'temp': np.concatenate((base_data, noize_data)),
        'is_noize': is_noize,
    }

    return result_data

# Создание директорий 'train' и 'test', если они не существуют
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Генерация и сохранение тренировочных данных
print('start: create train data')
train_data = generate_data(items_cnt=1000)
df_train = pd.DataFrame(train_data)
df_train.to_csv('train/train_data.csv', index=False)
print('finish: create train data')

# Генерация и сохранение тестовых данных
print('start: create test data')
test_data = generate_data(items_cnt=300)
df_test = pd.DataFrame(test_data)
df_test.to_csv('test/test_data.csv', index=False)
print('finish: create test data')

