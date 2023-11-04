import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import MinMaxScaler
import random

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
    finance_time_series = df['Сумма'].rolling(window=window).mean()[window - 1:][:851]
    finance_time_series = MinMaxScaler().fit_transform(finance_time_series.values.reshape(-1, 1)).flatten()
    dates = pd.to_datetime(df['Время'], format='%d.%m.%Y').values[window - 1:][:851]

    deposit = MoneyGame(1000)
    deposit.game(num_iterations=851 - window - 1)
    random_time_series = MinMaxScaler().fit_transform(deposit.history.reshape(-1, 1)).flatten()

    return finance_time_series, random_time_series, dates


def generate_random_walk(steps):
    """Случайное блуждание"""
    position = 0
    time_series = [position]

    for _ in range(steps):
        step = random.choice([-1, 1])
        position += step
        time_series.append(position)
    return np.array(time_series)


def generate_state_vectors(time_series, embedding_dimension, time_delay):
    """
    Генерирует фазовые векторы из временного ряда.

    Parameters:
        time_series (numpy.ndarray): Временной ряд для анализа.
        embedding_dimension (int): Размерность фазового вложения (количество компонентов вектора).
        time_delay (int): Задержка времени между компонентами фазового вектора.

    Returns:
        numpy.ndarray: Массив, содержащий фазовые векторы, представляющие состояния системы.
    """
    N = len(time_series)
    state_vectors = []

    for i in range(N - (embedding_dimension - 1) * time_delay):
        state_vector = [time_series[i + j * time_delay] for j in range(embedding_dimension)]
        state_vectors.append(state_vector)

    return np.array(state_vectors)


def calculate_correlation_dimension(state_vectors, epsilon):
    """
  Рассчитывает размерность корреляции системы на основе фазовых векторов.

  Parameters:
      state_vectors (numpy.ndarray): Массив фазовых векторов представляющих состояния системы.
      epsilon (float): Параметр, задающий расстояние для оценки корреляционной размерности.

  Returns:
     float: Оценка корреляционной размерности системы.
  """
    num_points = state_vectors.shape[0]
    distances = squareform(pdist(state_vectors, 'euclidean'))
    correlation_sum = 0

    for i in range(num_points):
        num_points_within_epsilon = np.sum(distances[i] < epsilon) - 1  # исключаем точку саму с собой
        correlation_sum += num_points_within_epsilon
    return (2 / (num_points * (num_points - 1))) * correlation_sum
