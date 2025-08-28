import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LinearFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0] * observation_space.shape[1] # window_size * features
        self.linear = nn.Sequential(
            nn.Linear(n_input, features_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.linear(x.flatten(start_dim=1))

class MLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, 
                 hidden_layers=[256, 128], dropout=0.2):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0] * observation_space.shape[1]
        
        layers = []
        prev_size = n_input
        
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.LayerNorm(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = layer_size
        
        layers.append(nn.Linear(prev_size, features_dim))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # Выравниваем вход (batch_size, window_size, features) -> (batch_size, window_size*features)
        x = x.flatten(start_dim=1)
        return self.net(x)

class RNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, hidden_size=128):
        super().__init__(observation_space, features_dim)
        self.rnn = nn.RNN(
            input_size=observation_space.shape[1],  # number of features
            hidden_size=hidden_size,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, features_dim)
    
    def forward(self, observations):
        _, hidden = self.rnn(observations)
        return self.linear(hidden.squeeze(0))

class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=32, hidden_size=64):
        super().__init__(observation_space, features_dim)
        self.lstm = nn.LSTM(
            input_size=observation_space.shape[1],  # number of features
            hidden_size=hidden_size,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, features_dim)
    
    def forward(self, observations):
        _, (hidden, _) = self.lstm(observations)
        return self.linear(hidden.squeeze(0))