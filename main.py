import argparse
import os

import numpy as np

from config import BatteryConfig, CMAPSSConfig, NCMAPSSConfig
from data_loaders import BatteryLoader, CMAPSSLoader, NCMAPSSLoader
from models import SafeBound
from train import Trainer
from utils import KKTLoss, set_seed


def _model(config, train_loader, rate_bounds):
    sensors, condition, _ = next(iter(train_loader))
    r_min, r_max = rate_bounds
    return SafeBound(
        input_dim=sensors.shape[-1],
        cond_dim=condition.shape[-1],
        seq_len=config.window_size,
        r_min=r_min,
        r_max=r_max,
        kappa=config.kappa,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)


def _criterion(config):
    return KKTLoss(
        epsilon=config.epsilon,
        tau=config.tau,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
    )


def _mean_metrics(metrics):
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}


def _print_metrics(label, metrics):
    values = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"{label} | {values}")


def run_cmapss(subset):
    config = CMAPSSConfig()
    runs = []
    for seed in config.seeds:
        set_seed(seed)
        loader = CMAPSSLoader(config, seed)
        train_loader, validation_loader, test_loader, rate_bounds = loader.get_loaders(subset)
        model = _model(config, train_loader, rate_bounds)
        trainer = Trainer(model, config, _criterion(config))
        save_path = os.path.join(config.data_dir, f"best_model_{subset}_seed{seed}.pt")
        trainer.train(train_loader, validation_loader, save_path)
        metrics = trainer.evaluate(test_loader, battery=False)
        _print_metrics(f"{subset} seed={seed}", metrics)
        runs.append(metrics)
    _print_metrics(f"{subset} mean(5 runs)", _mean_metrics(runs))


def run_ncmapss(subset):
    config = NCMAPSSConfig()
    runs = []
    for seed in config.seeds:
        set_seed(seed)
        loader = NCMAPSSLoader(config, seed)
        train_loader, validation_loader, test_loader, rate_bounds = loader.get_loaders(subset)
        model = _model(config, train_loader, rate_bounds)
        trainer = Trainer(model, config, _criterion(config))
        save_path = os.path.join(config.data_dir, f"best_model_{subset}_seed{seed}.pt")
        trainer.train(train_loader, validation_loader, save_path)
        metrics = trainer.evaluate(test_loader, battery=False)
        _print_metrics(f"{subset} seed={seed}", metrics)
        runs.append(metrics)
    _print_metrics(f"{subset} mean(5 runs)", _mean_metrics(runs))


def run_battery():
    config = BatteryConfig()
    for test_id in config.battery_ids:
        runs = []
        for seed in config.seeds:
            set_seed(seed)
            loader = BatteryLoader(config, seed)
            train_loader, validation_loader, test_loader, rate_bounds = loader.load_loo(test_id)
            model = _model(config, train_loader, rate_bounds)
            trainer = Trainer(model, config, _criterion(config))
            save_path = os.path.join(config.data_dir, f"best_model_{test_id}_seed{seed}.pt")
            trainer.train(train_loader, validation_loader, save_path)
            metrics = trainer.evaluate(test_loader, battery=True)
            _print_metrics(f"{test_id} seed={seed}", metrics)
            runs.append(metrics)
        _print_metrics(f"{test_id} mean(5 runs)", _mean_metrics(runs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("CMAPSS", "NCMAPSS", "Battery"))
    parser.add_argument("--subset")
    args = parser.parse_args()

    if args.dataset == "CMAPSS":
        for subset in ([args.subset] if args.subset else CMAPSSConfig.subsets):
            run_cmapss(subset)
    elif args.dataset == "NCMAPSS":
        for subset in ([args.subset] if args.subset else NCMAPSSConfig.subsets):
            run_ncmapss(subset)
    else:
        run_battery()
