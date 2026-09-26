import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class KKTLoss(nn.Module):
    def __init__(self, epsilon, tau, alpha, beta, gamma=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, prediction, lower_bound, target, slack, dual):
        target = target.reshape(-1)
        prediction = prediction.reshape(-1)
        lower_bound = lower_bound.reshape(-1)
        dual = dual.reshape(-1)

        loss_pred = torch.mean((prediction - target) ** 2)
        g_t = lower_bound - target
        violation = F.relu(g_t)
        loss_feas = torch.mean(violation ** 2)
        loss_tight = -torch.mean(lower_bound)
        gate = torch.sigmoid((self.epsilon - torch.abs(g_t)) / self.tau)
        loss_comp = torch.mean(gate * (dual * violation) ** 2)

        total = (
            loss_pred
            + self.gamma * loss_comp
            + self.alpha * loss_feas
            + self.beta * loss_tight
        )
        return total, {
            "pred": loss_pred.item(),
            "comp": loss_comp.item(),
            "feas": loss_feas.item(),
            "tight": loss_tight.item(),
        }


def validation_metrics(prediction, lower_bound, target):
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    violation_magnitude = float(np.mean(np.maximum(0.0, lower_bound - target)))
    return rmse, violation_magnitude


def benchmark_metrics(prediction, lower_bound, target, battery=False):
    prediction = np.asarray(prediction).reshape(-1)
    lower_bound = np.asarray(lower_bound).reshape(-1)
    target = np.asarray(target).reshape(-1)

    error = prediction - target
    rmse = float(np.sqrt(np.mean(error ** 2)))
    over = float(np.mean(prediction > target) * 100.0)
    mono = float(abs(spearmanr(prediction, target).statistic))
    rmse_lb = float(np.sqrt(np.mean((lower_bound - target) ** 2)))
    margin = float(np.mean(target - lower_bound))
    violation = float(np.mean(lower_bound > target) * 100.0)

    over_mask = prediction > target
    mitigated = over_mask & (lower_bound <= target)
    mitigation = float(np.mean(mitigated[over_mask]) * 100.0) if np.any(over_mask) else 0.0

    metrics = {
        "RMSE": rmse,
        "Over%": over,
        "Mono": mono,
        "RMSE(LB)": rmse_lb,
        "Marg": margin,
        "Vio%": violation,
        "Mitig%": mitigation,
    }

    if battery:
        metrics["MAE"] = float(np.mean(np.abs(error)))
    else:
        gamma = np.where(error >= 0.0, 1.0 / 10.0, 1.0 / 13.0)
        metrics["Score"] = float(np.sum(np.exp(gamma * np.abs(error)) - 1.0))

    return metrics
