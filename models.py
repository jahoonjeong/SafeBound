import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, condition):
        gamma, beta = torch.chunk(self.linear(condition), 2, dim=-1)
        return gamma, beta


class TSMixerBlock(nn.Module):
    def __init__(self, hidden_dim, seq_len, dropout):
        super().__init__()
        self.norm_time = nn.LayerNorm(hidden_dim)
        self.time_linear = nn.Linear(seq_len, seq_len)
        self.norm_feature = nn.LayerNorm(hidden_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.norm_time(x).transpose(1, 2)
        x = self.time_linear(x).transpose(1, 2) + residual
        residual = x
        x = self.norm_feature(x)
        return self.feature_mlp(x) + residual


class SafeBound(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        seq_len,
        r_min,
        r_max,
        kappa,
        hidden_dim=32,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        if r_min <= 0 or r_max <= r_min:
            raise ValueError("Expected 0 < r_min < r_max.")

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.film_generator = FiLMGenerator(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TSMixerBlock(hidden_dim, seq_len, dropout) for _ in range(num_layers)]
        )
        self.step_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.slack_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.dual_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

        self.kappa = float(kappa)
        self.register_buffer("r_min", torch.tensor(float(r_min)))
        self.register_buffer("r_max", torch.tensor(float(r_max)))
        nn.init.constant_(self.slack_head[-2].bias, -2.0)

    def forward(self, x_sensors, x_condition):
        h = self.embedding(x_sensors)
        gamma, beta = self.film_generator(x_condition)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        h = (1.0 + gamma) * h + beta

        for block in self.blocks:
            h = block(h)

        delta = self.step_head(h).squeeze(-1)
        health = torch.exp(-torch.cumsum(delta, dim=1) / self.kappa)

        e_t = h[:, -1, :]
        rate = torch.sigmoid(self.rate_head(e_t).squeeze(-1))
        rate = rate * (self.r_max - self.r_min) + self.r_min

        lower_bound = health[:, -1] / rate
        slack = self.slack_head(e_t).squeeze(-1)
        prediction = lower_bound + slack
        dual = self.dual_head(e_t).squeeze(-1)

        return prediction, lower_bound, slack, dual
