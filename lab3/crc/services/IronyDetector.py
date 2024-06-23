from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax

class IronyDetector:
    """
    Класс, в котором инкапсулирована логика по настройке модели и анализу текста на наличие иронии.
    """

    @classmethod
    def init(self):
        """
        Метод для инициализации токенизатора и модели для обнаружения иронии.
        """
        model_id = "cardiffnlp/twitter-roberta-base-irony"  # ID модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)  # Загрузка токенизатора
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)  # Загрузка модели

    @classmethod
    def analyze(self, text_to_detect: str):
        """
        Метод для анализа текста на наличие иронии.
        
        Параметры:
        - text_to_detect: строка текста для анализа.
        
        Возвращает:
        - rslt: словарь с вероятностями для меток "irony" и "non-irony".
        """
        labels = ["non-irony", "irony"]  # Метки для классификации
        encoded_input = self.tokenizer(text_to_detect, return_tensors="pt")  # Токенизация входного текста
        output = self.model(**encoded_input)  # Получение предсказаний модели
        scores = output[0][0].detach().numpy()  # Извлечение результатов предсказания
        scores = softmax(scores)  # Применение функции softmax для получения вероятностей

        # Сортировка меток по вероятностям в порядке убывания
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        # Формирование результирующего словаря с метками и соответствующими вероятностями
        rslt = {}
        for i in range(scores.shape[0]):
            lb = labels[ranking[i]]
            sc = scores[ranking[i]]
            rslt[lb] = np.round(float(sc), 3)
        
        return rslt
