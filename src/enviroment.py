import gymnasium as gym 
from gymnasium import spaces
import csv
import numpy as np 
from pathlib import Path 
import os
from collections import deque
import heapq


'''
Для корректной последующей совместимости с фреймворком Stable Baselines3 (SB3) необходима реализация в классе окружения
следующих атрибутов:
1. __init__

    Конструктор среды.
    self.action_space (например, spaces.Discrete(3))
    self.observation_space (например, spaces.Box(...))

2. reset(self)

    Сброс среды в начальное состояние.
    Возвращает:
        Начальное наблюдение (observation)

3. step(self, action)

    Применяет действие агента, обновляет состояние среды.
    Возвращает кортеж:
        observation (новое наблюдение)
        reward (награда за переход)
        done или terminated, truncated (bool, эпизод завершён или нет)
        info (словарь с дополнительной отладочной инфой, можно пустым {})
    

4. render(self, mode="human")

    (Опционально) Визуализация текущего состояния среды.
    Если не нужен визуальный вывод, можно реализовать как пустой метод.

5. close(self)

    (Опционально) Завершение работы среды, очистка ресурсов.

6. seed(self, seed=None)

    (Необязательный, но полезный) — фиксирует seed для воспроизводимости (Gym уже это не требует явно).
'''

class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, scaled_data, time_data, df_index=None,
                  timestamps=None, starting_cash=100000, starting_shares=0,
                 window_size=10, price_column=3, num_actions=3):
        super().__init__()

        self.data = np.asarray(data, dtype=np.float32)
        self.scaled_data = np.asarray(scaled_data, dtype=np.float32)
        self.time_data = np.asarray(time_data, dtype=np.float32)
        self.starting_cash = starting_cash
        self.starting_shares = starting_shares
        self.window_size = window_size
        self.price_column = price_column
        self.num_actions = num_actions

        #self.df_index = df_index if df_index is not None else np.arange(len(data))
        self.timestamps = timestamps
        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares
        self.current_step = self.window_size
        self.action_space = spaces.Discrete(self.num_actions)  # пространство действий (для начала: 0-ничего не делаем, 1-лонг, 2-шорт)

        # Формируем размерность observation space (скользящее окно по признакам)
        self.feature_size = scaled_data.shape[1] + time_data.shape[1]
        obs_shape = (self.window_size, self.feature_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # Переменные состояния
        self.data_deque = deque(maxlen=self.window_size)
        self.scaled_data_deque = deque(maxlen=self.window_size)
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares
        self.done = False

        # Заполняем деки начальными window_size наблюдениями
        self.data_deque.clear()
        self.scaled_data_deque.clear()
        for i in range(self.window_size):
            self.data_deque.append(self.data[i])
            s = np.concatenate((self.scaled_data[i], self.time_data[i]))
            self.scaled_data_deque.append(s)
        # Вычисляем цену на текущем шаге
        self.current_price = self.data[self.current_step, self.price_column]
        self.current_fraction = 0.0

        return self._get_state(), {}

    def _get_state(self):
        # Вернуть np.array shape (window_size, features)
        return np.array(self.scaled_data_deque, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        # 1) если эпизод закончен
        if self.done:
            return self._get_state(), 0.0, True, False, {}

        # --- Расчёт целевого веса ---
        step_fraction = 2.0 / (self.num_actions - 1) if self.num_actions > 1 else 0
        target_fraction = -1.0 + step_fraction * action
        # лимит изменения
        max_delta = 0.6
        target_fraction = np.clip(target_fraction,
                                self.current_fraction - max_delta,
                                self.current_fraction + max_delta)

        # --- Сколько было в портфеле до сделки ---
        old_portfolio = self.current_cash + self.current_shares * self.current_price

        # --- Сколько хотим держать после ---
        target_position_value = target_fraction * old_portfolio
        if self.current_price > 0:
            target_shares = int(target_position_value / self.current_price)
        else:
            target_shares = self.current_shares

        # ограничение абсолютной позиции 90% портфеля
        max_pos = 0.8 * old_portfolio
        if abs(target_position_value) > max_pos:
            target_shares = int(np.sign(target_position_value) * max_pos / self.current_price)

        # разница акций
        delta_shares = target_shares - self.current_shares
        trade_value = delta_shares * self.current_price

        # комиссия 0.01%
        commission = abs(trade_value) * 0.0001

        # проверка на кэш (только для лонга)
        if delta_shares > 0 and (self.current_cash - trade_value - commission) < 0:
            max_affordable = (self.current_cash - commission) / self.current_price
            delta_shares = int(max_affordable)
            trade_value = delta_shares * self.current_price
            commission = abs(trade_value) * 0.0001

        #commission = 0

        # обновляем баланс и позицию
        self.current_cash   -= (trade_value + commission)
        self.current_shares += delta_shares
        self.current_fraction = target_fraction

        # шаг вперёд
        self.current_step += 1
        self.done = (self.current_step >= len(self.data) - 1)

        # обновляем цену и окно наблюдений
        if not self.done:
            self.current_price = self.data[self.current_step, self.price_column]
            s = np.concatenate((self.scaled_data[self.current_step],
                                self.time_data[self.current_step]))
            self.scaled_data_deque.append(s)

        # --- Расчёт награды ---
        new_portfolio = self.current_cash + self.current_shares * self.current_price
        returns = (new_portfolio - old_portfolio) / old_portfolio

        prev_price = self.data[self.current_step - 1, self.price_column]
        market_return = (self.current_price - prev_price) / prev_price
        reward = returns - market_return
        
        info = {
            "step": self.current_step,
            "old_portfolio": old_portfolio,
            "new_portfolio": new_portfolio,
            "cash": self.current_cash,
            "shares": self.current_shares,
            "fraction": self.current_fraction,
            "current_price": self.current_price,
            "commission": commission,
            "timestamp": (self.timestamps[self.current_step]
                        if self.current_step < len(self.timestamps) else None)
        }

        return self._get_state(), reward, self.done, False, info



    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Portfolio Value: {self.current_cash + self.current_shares * self.current_price:.2f}, "
              f"Cash: {self.current_cash:.2f}, Shares: {self.current_shares}, Current Price: {self.current_price:.2f}")

    # def seed(self, seed=None):
    #     np.random.seed(seed)



class StockTradingEnvDSR(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, scaled_data, time_data, df_index=None,
                 timestamps=None, starting_cash=100000, starting_shares=0,
                 window_size=10, price_column=3, num_actions=3, 
                 commission_rate=0.00003): 
        super().__init__()
        
        self.data = np.asarray(data, dtype=np.float32)
        self.scaled_data = np.asarray(scaled_data, dtype=np.float32)
        self.time_data = np.asarray(time_data, dtype=np.float32)
        self.starting_cash = starting_cash
        self.starting_shares = starting_shares
        self.window_size = window_size
        self.price_column = price_column
        self.num_actions = num_actions
        self.commission_rate = commission_rate  # Инициализация комиссии
        self.timestamps = timestamps

        # DSR параметры
        self.A_prev = 0
        self.B_prev = 0
        self.eta = 1/252
        self.portfolio_value_prev = starting_cash

        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares
        self.current_step = self.window_size
        self.action_space = spaces.Discrete(self.num_actions)

        # Формирование observation space
        self.feature_size = scaled_data.shape[1] + time_data.shape[1]
        obs_shape = (self.window_size, self.feature_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        # Состояние
        self.data_deque = deque(maxlen=self.window_size)
        self.scaled_data_deque = deque(maxlen=self.window_size)
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.current_cash = self.starting_cash
        self.current_shares = self.starting_shares
        self.done = False

        # Сброс DSR параметров
        self.A_prev = 0
        self.B_prev = 0

        # Заполнение деков
        self.data_deque.clear()
        self.scaled_data_deque.clear()
        for i in range(self.window_size):
            self.data_deque.append(self.data[i])
            s = np.concatenate((self.scaled_data[i], self.time_data[i]))
            self.scaled_data_deque.append(s)
        
        # Установка начальной цены
        self.current_price = self.data[self.current_step, self.price_column]
        
        # Корректная инициализация стоимости портфеля
        self.portfolio_value_prev = self.current_cash + self.current_shares * self.current_price
        self.current_fraction = 0.0

        return self._get_state(), {}

    def _get_state(self):
        return np.array(self.scaled_data_deque, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        if self.done:
            return self._get_state(), 0.0, True, False, {}

        # Переход к новому шагу
        self.current_step += 1
        self.done = (self.current_step >= len(self.data) - 1)

        # Новая цена актива
        self.current_price = self.data[self.current_step, self.price_column]
        
        # Стоимость портфеля ДО ребалансировки
        prev_portfolio = self.current_cash + self.current_shares * self.current_price
        
        # Расчет доходности
        if self.portfolio_value_prev > 1e-8:
            R_t = (prev_portfolio - self.portfolio_value_prev) / self.portfolio_value_prev
        else:
            R_t = 0.0

        # Расчет DSR
        delta_A = R_t - self.A_prev
        delta_B = R_t ** 2 - self.B_prev
        variance = self.B_prev - self.A_prev ** 2
        
        if variance > 1e-8:
            numerator = self.B_prev * delta_A - 0.5 * self.A_prev * delta_B
            D_t = numerator / variance ** 1.5
        else:
            D_t = 0.0
        
        # Обновление DSR параметров
        self.A_prev += self.eta * delta_A
        self.B_prev += self.eta * delta_B

        # Ребалансировка портфеля
        step_fraction = 2.0 / (self.num_actions - 1) if self.num_actions > 1 else 0
        target_fraction = -1.0 + step_fraction * action
        target_value = target_fraction * prev_portfolio
        
        # Целевое количество акций
        if self.current_price > 1e-8:
            target_shares = int(target_value / self.current_price)
        else:
            target_shares = self.current_shares
        
        delta_shares = target_shares - self.current_shares
        trade_value = delta_shares * self.current_price
        commission = abs(trade_value) * self.commission_rate  # Всегда положительная

        # Проверка доступности средств (покупка)
        if delta_shares > 0:
            required_cash = trade_value + commission
            if required_cash > self.current_cash:
                max_affordable = (self.current_cash - commission) / self.current_price
                delta_shares = int(max_affordable)
                trade_value = delta_shares * self.current_price
                commission = abs(trade_value) * self.commission_rate
        
        # Проверка доступности акций (продажа)
        if delta_shares < 0:
            if self.current_shares + delta_shares < 0:
                delta_shares = -self.current_shares
                trade_value = delta_shares * self.current_price
                commission = abs(trade_value) * self.commission_rate

        # Исполнение сделки
        self.current_shares += delta_shares
        self.current_cash -= trade_value + commission

        # Обновление состояния портфеля
        new_portfolio = self.current_cash + self.current_shares * self.current_price
        
        # Критически важно для следующего шага!
        self.portfolio_value_prev = new_portfolio
        
        if new_portfolio > 1e-8:
            self.current_fraction = (self.current_shares * self.current_price) / new_portfolio
        else:
            self.current_fraction = 0.0

        # Обновление наблюдений
        if not self.done:
            s = np.concatenate((
                self.scaled_data[self.current_step],
                self.time_data[self.current_step]
            ))
            self.scaled_data_deque.append(s)
        
        reward = D_t

        info = {
            "step": self.current_step,
            "portfolio_value": new_portfolio,
            "cash": self.current_cash,
            "shares": self.current_shares,
            "fraction": self.current_fraction,
            "current_price": self.current_price,
            "return": R_t,
            "dsr": D_t,
            "commission": commission,
            "timestamp": (self.timestamps[self.current_step]
                        if self.current_step < len(self.timestamps) else None)
        }
        
        return self._get_state(), reward, self.done, False, info

    def render(self, mode="human"):
        portfolio_value = self.current_cash + self.current_shares * self.current_price
        print(f"Step: {self.current_step}, Portfolio Value: {portfolio_value:.2f}, "
              f"Cash: {self.current_cash:.2f}, Shares: {self.current_shares}, "
              f"Price: {self.current_price:.2f}, Fraction: {self.current_fraction:.4f}")