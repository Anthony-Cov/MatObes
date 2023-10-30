import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

PATH = '../data/finance.csv'


def coin_flip(money, probability=0.51):
    """Моделирует игру в монетку с заданной вероятностью выигрыша.

    Parameters:
        money (float): Начальная сумма денег, которую игрок ставит
        probability (float, optional): Вероятность выигрыша в диапазоне от 0 до 1
            По умолчанию установлена вероятность 0.51

    Returns:
        argument:  float: Новая сумма денег после игры
    """
    assert 0 < probability < 1, 'Вероятность должна быть в диапазоне (0, 1)'
    p = random.random()
    coefficient = 1.01 if p >= probability else 0.99
    return money * coefficient


class MoneyGame:
    """Моделирует серию игр с депозитом

   Parameters:
       money (float): Начальная сумма

   Attributes:
       history (numpy.ndarray): История изменения депозита

   Methods:
       game(num_iterations=1000): Игра с указанным числом итераций
   """

    def __init__(self, money):
        self.history = [money]

    def game(self, num_iterations=1000):
        for _ in range(num_iterations):
            self.history.append(coin_flip(self.history[-1]))
        self.history = np.array(self.history)


def get_time_series_data():
    """Загружает финансовый и случайный временные ряды для анализа"""
    random.seed(42)

    df = pd.read_csv('../data/finance.csv')
    window = 10
    finance_time_series = df['Сумма'].rolling(window=window).mean()[window-1:][:851]
    finance_time_series = MinMaxScaler().fit_transform(finance_time_series.values.reshape(-1, 1)).flatten()
    dates = pd.to_datetime(df['Время'], format='%d.%m.%Y').values[window-1:][:851]

    deposit = MoneyGame(1000)
    deposit.game(num_iterations=851 - window - 1 )
    random_time_series = MinMaxScaler().fit_transform(deposit.history.reshape(-1, 1)).flatten()

    return finance_time_series, random_time_series, dates
