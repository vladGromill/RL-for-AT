# RL-for-AT
Reinforcement Learning for Algorithmic Trading

## Project Structure

- **`agents.py`**: Contains neural network architectures for extracting features from data (Linear, RNN, LSTM).
-  **`enviroment.py`**: Implements a trading environment in the Gymnasium format, necessary for training RL agents, which is compatible with methods from the SB3 library.
-  **`utils.py`**: Contains auxiliary functions for working with data, preprocessing and evaluating results.
-  **`train_sb3.py`**: Basic script for training and optimizing trading strategies (DQN).
