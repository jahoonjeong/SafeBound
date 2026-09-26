import copy

import numpy as np
import torch
import torch.optim as optim

from utils import benchmark_metrics, validation_metrics


class Trainer:
    def __init__(self, model, config, criterion):
        self.model = model
        self.config = config
        self.criterion = criterion
        self.device = config.device

        dual_ids = {id(parameter) for parameter in model.dual_head.parameters()}
        primal = [parameter for parameter in model.parameters() if id(parameter) not in dual_ids]
        dual = list(model.dual_head.parameters())

        self.optimizer = optim.AdamW(
            [
                {"params": primal, "lr": config.lr},
                {"params": dual, "lr": config.lr * config.dual_lr_ratio},
            ],
            weight_decay=config.weight_decay,
        )

    def train(self, train_loader, validation_loader, save_path):
        best_score = float("inf")
        best_state = None

        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0

            for sensors, condition, target in train_loader:
                sensors = sensors.to(self.device)
                condition = condition.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                prediction, lower_bound, slack, dual = self.model(sensors, condition)
                loss, _ = self.criterion(prediction, lower_bound, target, slack, dual)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            rmse, violation_magnitude = self.validate(validation_loader)
            score = rmse + self.config.risk_omega * violation_magnitude
            print(
                f"Epoch {epoch + 1:02d}/{self.config.epochs} | "
                f"loss={running_loss / len(train_loader):.4f} | "
                f"val_rmse={rmse:.4f} | val_vio_mag={violation_magnitude:.4f} | "
                f"score={score:.4f}"
            )

            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, save_path)

        if best_state is None:
            raise RuntimeError("No checkpoint was produced.")

        self.model.load_state_dict(best_state)
        return best_score

    def _predict(self, loader):
        self.model.eval()
        predictions, lower_bounds, targets = [], [], []

        with torch.no_grad():
            for sensors, condition, target in loader:
                sensors = sensors.to(self.device)
                condition = condition.to(self.device)
                prediction, lower_bound, _, _ = self.model(sensors, condition)
                predictions.append(prediction.cpu().numpy())
                lower_bounds.append(lower_bound.cpu().numpy())
                targets.append(target.numpy())

        return (
            np.concatenate(predictions).reshape(-1),
            np.concatenate(lower_bounds).reshape(-1),
            np.concatenate(targets).reshape(-1),
        )

    def validate(self, loader):
        prediction, lower_bound, target = self._predict(loader)
        return validation_metrics(prediction, lower_bound, target)

    def evaluate(self, loader, battery=False):
        prediction, lower_bound, target = self._predict(loader)
        return benchmark_metrics(prediction, lower_bound, target, battery=battery)
