from pathlib import Path
import os
import sys
import optuna
from optuna.samplers import TPESampler
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import ProgressBarCallback
import numpy as np
import torch
import random
import yaml
import datetime
import pandas as pd

# path_to_src = Path(os.getcwd(), 'src')
#sys.path.append(str(path_to_src))
from enviroment import StockTradingEnvDSR
from agents import LinearFeatureExtractor, RNNFeatureExtractor, LSTMExtractor, MLPFeatureExtractor
from utils import get_data, get_scaled_data, get_time_data, prepare_aligned_data, evaluate_daily_sharpe, statistics_calc, daily_portfolio_return

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_env(data, scaled_data, time_data, config, mode='train', n_envs=1, period_override=None):
    """Создает окружение с возможностью переопределения периодов"""
    if period_override:
        start, end = period_override
    else:
        if mode == 'train':
            start, end = config.get("train_period", (None, None))
        elif mode == 'val':
            start, end = config.get("val_period", (None, None))
        elif mode == 'test':
            start, end = config.get("test_period", (None, None))
    
    if start is None or end is None:
        raise ValueError(f"Для режима {mode} не заданы периоды")
    
    mask = (data.index >= start) & (data.index <= end)
    
    # Применяем маску
    data_segment = data[mask]
    scaled_segment = scaled_data[mask]
    time_segment = time_data[mask]
    
    env_kwargs = {
        "data": data_segment[['open', 'high', 'low', 'close']].values.astype(np.float32),
        "scaled_data": scaled_segment.values.astype(np.float32),
        "time_data": time_segment.values.astype(np.float32),
        "window_size": config["window_size"],
        "price_column": config["price_column"],
        "num_actions": config["num_actions"], 
        "timestamps": data_segment.index.values
    }
    
    if n_envs > 1:
        return make_vec_env(
            lambda: StockTradingEnvDSR(**env_kwargs),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
            seed=config["seed"]
        )
    else:
        return DummyVecEnv([lambda: StockTradingEnvDSR(**env_kwargs)])

def get_model_config(model_type, trial=None):
    base_config = {
        "MLP": {
            "features_extractor_class": MLPFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": trial.suggest_int('linear_dim', 32, 256) if trial else 64,
                "hidden_layers": [256, 128],
                "dropout": trial.suggest_float('dropout', 0.1, 0.8) if trial else 0.1  # Добавлено условие
            }
        },
        "Linear": {
            "features_extractor_class": LinearFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": trial.suggest_int('linear_dim', 16, 128) if trial else 32,
            }
        },
        "RNN": {
            "features_extractor_class": RNNFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": trial.suggest_int('linear_dim', 32, 256) if trial else 64,
                "hidden_size": trial.suggest_int('hidden_dim', 64, 512) if trial else 128  # Добавлено условие
            }
        },
        "LSTM": {
            "features_extractor_class": LSTMExtractor,
            "features_extractor_kwargs": {
                "features_dim": trial.suggest_int('linear_dim', 32, 256) if trial else 64,
                "hidden_size": trial.suggest_int('hidden_dim', 64, 512) if trial else 128  # Добавлено условие
            }
        }
    }
    config = base_config.get(model_type, base_config["Linear"])
    
    # Если это не MLP, удаляем dropout и hidden_layers из параметров
    if model_type != "MLP":
        if "dropout" in config["features_extractor_kwargs"]:
            del config["features_extractor_kwargs"]["dropout"]
        if "hidden_layers" in config["features_extractor_kwargs"]:
            del config["features_extractor_kwargs"]["hidden_layers"]
    
    return config

def train_single_model(config, interval='B', trial=None):
    set_seeds(config["seed"])
    
    raw_data, scaled_data, time_data, _ = prepare_aligned_data(
        config["ticker"], config, interval
    )
    
    # Создаем окружения с использованием периодов из конфига
    env_train = create_env(
        raw_data, scaled_data, time_data, config,
        mode='train', n_envs=5
    )
    env_val = create_env(
        raw_data, scaled_data, time_data, config,
        mode='val', n_envs=1
    )

    # Получаем конфиг экстрактора
    policy_kwargs = get_model_config(config["model_type"], trial)
    
    # Обновляем параметры PPO
    ppo_params = config["ppo_params"].copy()
    ppo_params["policy_kwargs"] = policy_kwargs
    
    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        **ppo_params
    )
    
    model.learn(total_timesteps=config["total_timesteps"], log_interval=100, callback=ProgressBarCallback())
    
    returns, b_h_ret, positions = daily_portfolio_return(env_val, model)
    positions = [-1 + 2.0 / (config['num_actions'] - 1) * pos for pos in positions]
    metrics = statistics_calc(returns, b_h_ret, plot=True, mode='validation', positions=positions)
    
    return metrics, returns, b_h_ret, model

def sliding_window_training(config, interval='B', trial=None, out_of_sample=False):
    set_seeds(config["seed"])
    window_cfg = config["sliding_window_config"]
    best_agent = None
    all_results = []

    raw_data, scaled_data, time_data, _ = prepare_aligned_data(
        config["ticker"], config, interval, out_of_sample
    )
    start_year = window_cfg["start_year"]
    n_windows = window_cfg["n_windows"]
    # print(raw_data.index[-1])
    for window_idx in range(n_windows):
        current_year = start_year + window_idx
        print(f"\nОкно {window_idx+1}/{n_windows} (Год: {current_year})")
        
        # Расчет периодов для текущего окна
        train_start = datetime.datetime(current_year, 1, 1)
        train_end = train_start + datetime.timedelta(days=365 * window_cfg["train_years"])
        val_start = train_end
        val_end = val_start + datetime.timedelta(days=365 * window_cfg["val_years"])
        test_start = val_end
        test_end = test_start + datetime.timedelta(days=365 * window_cfg["test_years"])
        # print('test end', test_end)
        # Создание окружений
        env_train = create_env(
            raw_data, scaled_data, time_data, config, 
            n_envs=window_cfg["n_envs"],
            period_override=(train_start, train_end)
        )
        
        env_val = create_env(
            raw_data, scaled_data, time_data, config,
            mode='val', n_envs=1,
            period_override=(val_start, val_end)
        )
        
        best_agent_val = None
        best_val_reward = -np.inf
        
        for agent_idx in range(window_cfg["n_agents"]):
            print(f"Обучение агента {agent_idx+1}/{window_cfg['n_agents']}")
            set_seeds(config["seed"] + agent_idx)
            
            # Получаем конфиг экстрактора
            policy_kwargs = get_model_config(config["model_type"], trial)
            
            # Инициализация модели
            if best_agent and window_idx > 0:
                model = PPO(
                    "MlpPolicy",
                    env_train,
                    **config["ppo_params"],
                    policy_kwargs=policy_kwargs
                )
                model.set_parameters(best_agent.get_parameters())
            else:
                model = PPO(
                    "MlpPolicy",
                    env_train,
                    **config["ppo_params"],
                    policy_kwargs=policy_kwargs
                )
            
            # Обучение
            model.learn(total_timesteps=config["total_timesteps"], callback=ProgressBarCallback())
            
            # Оценка
            mean_reward, _ = evaluate_policy(
                model, env_val, n_eval_episodes=1, deterministic=True
            )
            print(f"Агент {agent_idx+1} Средняя награда: {mean_reward:.4f}")
            
            if mean_reward > best_val_reward:
                best_val_reward = mean_reward
                best_agent_val = model
        
        # Тестирование лучшего агента
        env_test = create_env(
            raw_data, scaled_data, time_data, config,
            mode='test', n_envs=1,
            period_override=(test_start, test_end)
        )
        
        returns, b_h_ret, positions = daily_portfolio_return(env_test, best_agent_val)
        positions = [-1 + 2.0 / (config['num_actions'] - 1) * pos for pos in positions]
        metrics = statistics_calc(
            returns, b_h_ret, 
            name=f"Window_{window_idx}", 
            plot=False,
            mode='test',
            ticker=config["ticker"],
            positions=positions
        )
        
        all_results.append(metrics)
        best_agent = best_agent_val  # Для следующего окна
        
        # Очистка
        del env_train, env_val, env_test
        torch.cuda.empty_cache()
    
    return pd.concat(all_results), best_agent

def objective_ppo(trial):
    """Целевая функция для оптимизации гиперпараметров"""
    config = CONFIG.copy()
    config["window_size"] = trial.suggest_categorical("window_size", [30, 60, 90, 120, 200])
    config["num_actions"] = trial.suggest_categorical("num_actions", [3, 5, 11, 19, 27])
    config["total_timesteps"] = trial.suggest_categorical("total_timesteps", [30000, 50000, 100000])
    config["ppo_params"]["n_steps"] = trial.suggest_int("n_steps", 128, 2048, step=128)
    # Размер батча для обучения (должен делиться на n_steps)
    config["ppo_params"]["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048])
    # Число проходов (эпох) по собранному батчу
    config["ppo_params"]["n_epochs"] = trial.suggest_int("n_epochs", 3, 30)
    # Дисконт-фактор
    config["ppo_params"]["gamma"] = trial.suggest_float("gamma", 0.90, 0.9999, step=0.001)
    # Лямбда для GAE
    config["ppo_params"]["gae_lambda"] = trial.suggest_float("gae_lambda", 0.8, 1.0, step=0.01)
    # Диапазон “обрезки” вероятностей
    config["ppo_params"]["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4, step=0.05)
    # Скорость обучения
    config["ppo_params"]["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    # Коэффициент энтропийного бонуса
    config["ppo_params"]["ent_coef"] = trial.suggest_loguniform("ent_coef", 1e-5, 1e-1)
    # Коэффициент потерь value-функции
    config["ppo_params"]["vf_coef"] = trial.suggest_float("vf_coef", 0.1, 1.0, step=0.1)
    # Максимальная норма градиента
    config["ppo_params"]["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.1, 1.0)
    # config["model_type"] = trial.suggest_categorical("model_type", ["Linear", "RNN", "LSTM"])
                                                                
    min_acceptable_return = 0.05
    all_results = sliding_window_training(config, trial=trial)
    annual_return = all_results["Annual return"].mean()
    sharpe = all_results["Sharpe ratio"].mean()
    vol = all_results["Volatility"].mean()
    if annual_return < min_acceptable_return:
        return annual_return - 1.0
    return 0.7 * all_results["Sharpe ratio"].mean() + 0.3 * all_results["Annual return"].mean() - 0.1 * vol

def optimize_config(config, objective, out_of_sample=False):
    if config["optuna"]["n_trials"] > 0:
        # Оптимизация гиперпараметров с Optuna
        study = optuna.create_study(
            direction=config["optuna"]["direction"],
            sampler=TPESampler(seed=config["seed"])
        )

        study.optimize(
            objective,
            n_trials=config["optuna"]["n_trials"],
            timeout=config["optuna"]["timeout"],
            show_progress_bar=True
        )

        best_params = study.best_params
        
        with open("best_params.yaml", "w") as f:
            yaml.dump(best_params, f)
            
        print(f"Лучшие параметры: {best_params}")
        print(f"Лучшая награда: {study.best_value}")
        return best_params
    else:
        # Простое обучение без оптимизации
        if config['training_mode'] == 'single':
            metrics, ret_cur, b_h_ret_cur, model = train_single_model(config)
        else:
            metrics, model = sliding_window_training(config, out_of_sample=out_of_sample)
        return metrics, model