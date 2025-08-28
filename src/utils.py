from pathlib import Path
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_and_save_data(ticker, period="max", interval="5m"):
    PROJECT_PATH = Path(os.getcwd(), 'data')
    #PROJECT_PATH = Path(Path(os.getcwd()).parent, 'data')
    PROJECT_PATH.mkdir(parents=True, exist_ok=True)
    FILE_PATH = PROJECT_PATH / f"{ticker}.csv"
    print(FILE_PATH)
    if FILE_PATH.exists():
        print(f"Данные для {ticker} уже существуют. Загрузка из кэша.")
        data = pd.read_csv(FILE_PATH, sep=',', index_col='timestamp', parse_dates=True)
        data[~data.index.duplicated(keep='first')]
        return data
    
    print(f"Загрузка данных для {ticker} с Yahoo Finance...")
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True  # Используем скорректированные цены
    )
    
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns) or set(map(lambda x:x.lower(), required_columns)).issubset(df.columns):
        # missing = required_columns - set(df.columns)
        raise ValueError(f"Отсутствуют обязательные колонки")
    
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    if 'log_ret' not in df.columns:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    df.to_csv(FILE_PATH)
    print(f"Данные сохранены в {FILE_PATH}")
    
    return df

def get_data(ticker, period="max", interval="B"):
    """
    Единая функция для получения данных
    """
    # Пытаемся загрузить из локального кэша
    try:
        PROJECT_PATH = Path(Path(os.getcwd()).parent, 'data', f"{ticker}.csv")
        # print('GDGDGDGD', PROJECT_PATH)
        if PROJECT_PATH.exists():
            data = pd.read_csv(PROJECT_PATH, sep=',', index_col='timestamp', parse_dates=True)
            if 'log_ret' not in data.columns:
                data['log_ret'] = np.log(data['close'] / data['close'].shift(1))

            data[~data.index.duplicated(keep='first')]
            data.index = pd.to_datetime(data.index)
            data.index = data.index.tz_localize(None)
            data = data.resample(interval).last()
            print(f'Shape of data: {data.shape}', data.index[0], data.index[-1])
            return data.dropna()
    except Exception as e:
        print(f"Ошибка при чтении локальных данных: {e}")
    
    # Если локальные данные недоступны - загружаем через Yahoo Finance
    return fetch_and_save_data(ticker, period, interval)


def get_scaled_data(data):
    features = pd.DataFrame(index=data.index)
    
    # 1. Расчет волатильности
    features['absRet'] = np.abs(data['log_ret'])
    features['EmaAbsRet'] = features['absRet'].ewm(span=20, adjust=False).mean().shift(1) * np.sqrt(20)
    features['vol'] = features['EmaAbsRet'].ewm(span=200, adjust=False).mean().shift(1) * np.sqrt(200) + 1e-9
    
    # 2. Нормализованная доходность
    features['normalized_log_ret'] = data['log_ret'].shift(1) / features['vol'].shift(1)
    
    # 3. Создание фичей с префиксом f_
    periods = np.logspace(1.0, 3.0, num=10, base=10).astype(int)
    # for period in periods:
    #     # Фильтрация по префиксу f_
    #     features[f'f_{period}'] = (
    #         np.sqrt(period) * 
    #         features['normalized_log_ret'].rolling(
    #             window=period, 
    #             win_type='hann',
    #             min_periods=1
    #         ).mean().shift(1)
    #     )

    features = pd.concat([
        features['normalized_log_ret'].rolling(window=p).mean().shift(1) * np.sqrt(p)
        for p in periods
    ], axis=1, keys=[f'f_{p}' for p in periods])
    # 4. Оставляем ТОЛЬКО фичи с префиксом f_
    # features['f_vp'] = features.groupby(pd.Grouper(freq=period))['volume'].transform('sum')
    f_columns = [col for col in features.columns if col.startswith('f_')]
    features = features.fillna(method='bfill').fillna(0)
    
    return features


def get_time_data(data_index):
    """
    Генерирует цикличные временные признаки из временного индекса
    
    :param data_index: pd.DatetimeIndex из основного датафрейма
    :return: DataFrame с временными признаками
    """
    time_features = pd.DataFrame(index=data_index)
    
    # Извлекаем компоненты времени
    time_features['hour'] = data_index.hour
    time_features['day_of_week'] = data_index.dayofweek  # Пн=0, Вс=6
    time_features['day_of_month'] = data_index.day
    time_features['month'] = data_index.month
    
    # Преобразуем в цикличные фичи
    for col, max_val in [('hour', 24), 
                         ('day_of_week', 7), 
                         ('day_of_month', 31), 
                         ('month', 12)]:
        time_features[f'{col}_sin'] = np.sin(2 * np.pi * time_features[col] / max_val)
        time_features[f'{col}_cos'] = np.cos(2 * np.pi * time_features[col] / max_val)
    
    # Удаляем исходные столбцы
    time_features = time_features.drop(columns=['hour', 'day_of_week', 'day_of_month', 'month'])
    
    return time_features

def get_scaled_data_new(data):
    """
    Только основные фичи.
    """
    features = pd.DataFrame(index=data.index)
    
    # Рассчитываем логарифмическую доходность
    features['f_log_ret'] = np.log(data['close'] / data['close'].shift(1)).shift(1)

    # Рассчитываем волатильность рынка и отношение краткосрочной волатильности к долгосрочной
    features['f_vol20'] = features['f_log_ret'].rolling(window=20, min_periods=20).std()
    features['f_vol60'] = features['f_log_ret'].rolling(window=60, min_periods=60).std()
    features['f_vol20/vol60'] = features['f_vol20'] / features['f_vol60']

    # Вычисляем простое скользящее среднее доходности за 20 периодов
    features['f_ema_10'] = data['close'].ewm(span=10, adjust=False).mean().shift(1)
    features['f_ema_90'] = data['close'].ewm(span=90, adjust=False).mean().shift(1)

    # RSI и MACD (функции ниже)
    features['f_rsi'] = calculate_rsi(data['close'], window=14).shift(1)
    macd_line, macd_signal = calculate_macd(data['close'])
    features['f_macd_line'] = macd_line.shift(1)
    features['f_macd_signal'] = macd_signal.shift(1)
    features['f_macd_hist'] = features['f_macd_line'] - features['f_macd_signal'] 

    # Нормализация признаков, так как DQN чувствительная к масштабу
    scaler = StandardScaler()
    features_norm = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )
    #columns_to_drop = ['f_vol20', 'f_vol60', 'f_macd_line', 'f_macd_signal']
    columns_to_drop = ['f_vol20', 'f_macd_line', 'f_macd_signal']
    columns_to_drop = ['f_macd_hist']
    features_norm = features_norm.drop(columns_to_drop, axis=1)
    return features_norm.dropna()
    

def get_time_data_new(data_index):
    """
    Генерирует цикличные временные признаки из временного индекса для ежедневных данных
    
    :param data_index: pd.DatetimeIndex из основного датафрейма
    :return: DataFrame с цикличными временными признаками
    """
    time_features = pd.DataFrame(index=data_index)
    
    # Извлекаем компоненты времени (только для ежедневных данных)
    time_features['day_of_week'] = data_index.dayofweek  # Пн=0, Вс=6
    time_features['day_of_month'] = data_index.day
    time_features['day_of_year'] = data_index.dayofyear
    time_features['week_of_year'] = data_index.isocalendar().week
    time_features['month'] = data_index.month
    time_features['quarter'] = data_index.quarter
    time_features['year'] = data_index.year
    
    # Преобразуем в цикличные фичи
    cyclic_features = []
    for col, max_val in [
        ('day_of_week', 7), 
        ('day_of_month', 31), 
        ('day_of_year', 366),  # Учет високосного года
        ('week_of_year', 53),  # ISO weeks can have 53 weeks
        ('month', 12),
        ('quarter', 4)
    ]:
        time_features[f'{col}_sin'] = np.sin(2 * np.pi * time_features[col] / max_val)
        time_features[f'{col}_cos'] = np.cos(2 * np.pi * time_features[col] / max_val)
        cyclic_features.extend([f'{col}_sin', f'{col}_cos'])
    
    # Добавляем бинарный признак конца месяца
    time_features['is_month_end'] = data_index.is_month_end.astype(int)
    
    # Сохраняем год как отдельный признак (без цикличности)
    final_features = cyclic_features + ['is_month_end']
    
    return time_features[final_features]

def calculate_rsi(series, window = 14):
    """
    Расчет RSI (Relative Strength Index)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    Расчет MACD:
    Возвращает (macd_line, macd_signal)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line, macd_signal

def prepare_aligned_data(ticker, config, interval='B', out_of_sample=False):
    """Возвращает синхронизированные данные"""
    raw_data = get_data(ticker, interval=interval)
    
    # Вычисляем признаки для ВСЕХ данных
    scaled_data = get_scaled_data_new(raw_data)
    time_data = get_time_data_new(scaled_data.index)
    
    raw_data = raw_data.loc[scaled_data.index]
    print('feature finance:', scaled_data.columns)
    print('feature time:', time_data.columns)
    # print(raw_data.shape, scaled_data.shape, time_data.shape)
    # Фильтруем по датам из конфига
    train_start, train_end = config["train_period"]
    val_start, val_end = config["val_period"]
    full_mask = (raw_data.index >= train_start) & (raw_data.index <= val_end)
    
    # Применяем маску ко всем данным
    if not out_of_sample:
        filtered_raw = raw_data.loc[full_mask]
        filtered_scaled = scaled_data.loc[full_mask]
        filtered_time = time_data.loc[full_mask]
    else:
        full_mask = (raw_data.index >= train_start) & (raw_data.index <= train_end)
        filtered_raw = raw_data.loc[full_mask]
        filtered_scaled = scaled_data.loc[full_mask]
        filtered_time = time_data.loc[full_mask] 

    # Синхронизация индексов после dropna
    common_idx = filtered_scaled.index.intersection(filtered_time.index)
    filtered_raw = filtered_raw.loc[common_idx]
    filtered_scaled = filtered_scaled.loc[common_idx]
    filtered_time = filtered_time.loc[common_idx]
    
    return (
        filtered_raw,
        filtered_scaled,
        filtered_time,
        common_idx
    )


def load_and_prepare_data(ticker):
    raw_data = get_data(ticker)

    scaled_data = get_scaled_data(data)

    time_data = get_time_data(scaled_data.index)
    aligned_data = raw_data.loc[scaled_data.index]

    data_array = aligned_data[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
    scaled_array = scaled_data.values.astype(np.float32)
    time_array = time_data.values.astype(np.float32)

    return data_array, scaled_array, time_array



def Return(rets):
    """
    Annual return estimate

    :rets: daily returns of the strategy
    """
    days_in_year = 365.25
    return np.mean(rets)*days_in_year


def Volatility(rets):
    """
    Estimation of annual volatility

    :rets: daily returns of the strategy
    """
    days_in_year = 365.25
    return np.std(rets)*np.sqrt(days_in_year)


def SharpeRatio(rets):
    """
    Estimating the annual Sharpe ratio

    :rets: daily returns of the strategy
    """
    volatility = Volatility(rets)
    if (volatility>0):
        return Return(rets)/volatility
    else:
        return float('NaN')

def statistics_calc(rets, bh, name='_', save_path=Path(os.getcwd(), 'save', 'models'),
                   plot=False, mode='train', ticker='none', 
                   positions=None):
    """
    Draws a graph of portfolio equity, calculates performance metrics and visualizes trading positions
    
    :param rets: daily returns of the strategy
    :param positions: array of trading positions (float values: positive-long, negative-short, 0-flat)
    """
    sharpe = SharpeRatio(rets)
    ret = Return(rets)
    vol = Volatility(rets)

    bh_sharpe = SharpeRatio(bh)
    bh_ret = Return(bh)
    bh_vol = Volatility(bh)

    if plot:
        all_stats = {}
        plt.figure(figsize=(8, 5))
        ax1 = plt.gca()
        
        # Plot equity curves
        ax1.plot(rets.cumsum(), label='strategy', linewidth=2)
        ax1.plot(bh.cumsum(), label='buy & hold', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        plt.title(f'{ticker} Equity Curve on {mode}')
        plt.legend()
        plt.tight_layout()
        #plt.savefig(os.path.join(save_path, f'equity_curve_{name}.png'))
        plt.show()

        # Plot positions if provided
        if positions is not None:
            plt.figure(figsize=(8,5))
            ax2 = plt.gca()
            x = range(len(positions))
            ax2.plot(x, positions, marker='*', linestyle='-', linewidth=1.5, markersize=8, label='positions')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Size of position')
            ax2.legend()
            plt.title('Positions')
            plt.show()

        print(f'Metrics on {mode}:')
        print(f'Sharpe ratio: {sharpe:.4f}, annual return: {ret:.4f}, volatility: {vol:.4f}')
        
        
        all_stats[name] = {
            "Sharpe ratio": float(sharpe),
            "Annual return": float(ret),
            "Volatility": float(vol),
            "bh Sharpe ratio": float(bh_sharpe),
            "bh Annual return": float(bh_ret),
            "bh Volatility": float(bh_vol)
        }

        # with open(json_path, 'w') as f:
        #     json.dump(all_stats, f, indent=4)

    return pd.DataFrame(
        [[sharpe, ret, vol, bh_sharpe, bh_ret, bh_vol]],
        columns=['Sharpe ratio', 'Annual return', 'Volatility', 
                 'bh Sharpe ratio', 'bh Annual return', 'bh Volatility'],
        index=[name]
    )


def daily_portfolio_return(env, model, interval='B'):
    obs = env.reset()
    done = [False]
    portfolio_values = []
    timestamps = []
    b_h = []
    positions = []
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        positions.append(action)
        obs, rewards, done, info = env.step(action)

        portfolio_values.append(info[0]['portfolio_value'])
        timestamps.append(info[0].get('timestamp', len(timestamps)))
        b_h.append(info[0]['current_price'])

    df = pd.DataFrame({
        "portfolio_value": portfolio_values,
        "current_price": b_h
    }, index=pd.to_datetime(timestamps))

    daily = df.resample(interval).last()
    
    # Рассчитываем дневную доходность
    daily['log_return'] = np.log(daily['portfolio_value'] / daily['portfolio_value'].shift(1))
    daily['log_return_b_h'] = np.log(daily['current_price'] / daily['current_price'].shift(1))
    return daily['log_return'].dropna(), daily['log_return_b_h'].dropna(), positions

def evaluate_daily_sharpe(model, env, n_eval_episodes=1):
    """Оценка стратегии по коэффициенту Шарпа"""
    episode_sharpes = []
    
    for _ in range(n_eval_episodes):
        returns, b_h_ret = daily_portfolio_return(env, model)
        if len(returns) > 1:
            sharpe = SharpeRatio(returns.values)
            episode_sharpes.append(sharpe)
    
    return np.mean(episode_sharpes), returns, b_h_ret if episode_sharpes else 0, returns, b_h_ret